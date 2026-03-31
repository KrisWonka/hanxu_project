"""
Speech-to-Text engine.

Supports:
  - local_whisper:  Local Whisper model (free, no API key, runs on device)
  - openai_whisper: OpenAI Whisper API (best quality, needs OPENAI_API_KEY)
  - dashscope:      Alibaba DashScope Paraformer API (good Chinese, domestic)

All accept WAV bytes as input and return recognized text.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


WHISPER_HALLUCINATIONS = {
    "字幕", "索兰亚", "感谢观看", "谢谢观看", "请不吝点赞",
    "订阅", "小铃铛", "下次再见", "字幕由", "字幕制作",
    "amara.org", "subtitles", "subscribe", "thank you for watching",
    "the end", "see you next time",
}


class STTEngine:
    """Unified speech-to-text interface."""

    def __init__(self, provider: str = "local_whisper", language: str = "zh", whisper_model: str = "medium"):
        self.provider = provider
        self.language = language
        self._whisper_model_name = whisper_model
        self._whisper_model = None

    def recognize(self, wav_bytes: bytes) -> str:
        """Convert WAV audio bytes to text. Returns empty string on failure."""
        if not wav_bytes:
            return ""

        if self._is_silence(wav_bytes):
            logger.info("STT: audio too quiet, skipping")
            return ""

        try:
            if self.provider == "local_whisper":
                text = self._recognize_local_whisper(wav_bytes)
            elif self.provider == "openai_whisper":
                text = self._recognize_openai(wav_bytes)
            elif self.provider == "dashscope":
                text = self._recognize_dashscope(wav_bytes)
            else:
                raise ValueError(f"Unknown STT provider: {self.provider}")

            if self._is_hallucination(text):
                logger.info("STT: filtered hallucination: %s", text)
                return ""
            return text

        except Exception as e:
            logger.error("STT recognition failed [%s]: %s", self.provider, e)
            return ""

    @staticmethod
    def _is_silence(wav_bytes: bytes, threshold: int = 300) -> bool:
        """Check if audio is effectively silent."""
        import numpy as np
        import wave

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
                return rms < threshold
        except Exception:
            return False

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Detect common Whisper hallucination patterns on silent input."""
        if not text:
            return False
        t = text.lower().strip()
        return any(h in t for h in WHISPER_HALLUCINATIONS)

    def _get_whisper_model(self):
        if self._whisper_model is None:
            import whisper  # type: ignore

            model_name = os.environ.get("WHISPER_MODEL", self._whisper_model_name)
            logger.info("Loading local Whisper model (%s)...", model_name)
            self._whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded")
        return self._whisper_model

    def _recognize_local_whisper(self, wav_bytes: bytes) -> str:
        model = self._get_whisper_model()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            result = model.transcribe(
                tmp_path,
                language=self.language,
                fp16=False,
            )
            text = result["text"].strip()
            logger.info("STT [local_whisper]: %s", text)
            return text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _recognize_openai(self, wav_bytes: bytes) -> str:
        from openai import OpenAI

        client = OpenAI()

        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "recording.wav"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=self.language,
        )

        text = transcript.text.strip()
        logger.info("STT [openai]: %s", text)
        return text

    def _recognize_dashscope(self, wav_bytes: bytes) -> str:
        import dashscope  # type: ignore
        from dashscope.audio.asr import Recognition  # type: ignore

        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            recognition = Recognition(
                model="paraformer-realtime-v2",
                format="wav",
                sample_rate=16000,
                language_hints=["zh"],
                callback=None,
            )
            result = recognition.call(tmp_path)
            logger.debug("DashScope raw status: %s", result.status_code)

            sentences = result.get_sentence()
            logger.debug("DashScope raw sentences: %s", sentences)

            if isinstance(sentences, list):
                text = "".join(
                    s.get("text", "") if isinstance(s, dict) else str(s)
                    for s in sentences
                ).strip()
            elif isinstance(sentences, dict):
                text = sentences.get("text", "").strip()
            else:
                text = str(sentences).strip() if sentences else ""

            logger.info("STT [dashscope]: %s", text)
            return text
        finally:
            Path(tmp_path).unlink(missing_ok=True)

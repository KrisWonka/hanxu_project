"""
Text-to-Speech module.

Uses edge-tts (free, high-quality Chinese voices) as default.
Falls back to pyttsx3 for offline use.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class TTS:
    """Text-to-Speech with edge-tts primary and pyttsx3 fallback."""

    def __init__(
        self,
        provider: str = "edge",
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
    ):
        self.provider = provider
        self.voice = voice
        self.rate = rate

    def speak(self, text: str) -> None:
        """Synthesize text and play through speakers."""
        if not text or not text.strip():
            return

        logger.info("TTS [%s]: %s", self.provider, text)

        if self.provider == "edge":
            try:
                self._speak_edge(text)
                return
            except Exception as e:
                logger.warning("edge-tts failed: %s — trying pyttsx3", e)

        self._speak_pyttsx3(text)

    def _speak_edge(self, text: str) -> None:
        import edge_tts  # type: ignore

        async def _synthesize_and_play():
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name

            await communicate.save(tmp_path)
            self._play_file(tmp_path)

            Path(tmp_path).unlink(missing_ok=True)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_synthesize_and_play())).result()
        else:
            asyncio.run(_synthesize_and_play())

    def _speak_pyttsx3(self, text: str) -> None:
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error("pyttsx3 also failed: %s — printing instead", e)
            print(f"🔊 [TTS]: {text}")

    @staticmethod
    def _play_file(path: str) -> None:
        """Play an audio file through the default output device."""
        try:
            from pydub import AudioSegment  # type: ignore
            from pydub.playback import play  # type: ignore

            audio = AudioSegment.from_file(path)
            play(audio)
        except ImportError:
            import subprocess
            import sys

            if sys.platform == "darwin":
                subprocess.run(["afplay", path], check=True)
            else:
                subprocess.run(
                    ["aplay", "-q", path],
                    check=False,
                    stderr=subprocess.DEVNULL,
                )

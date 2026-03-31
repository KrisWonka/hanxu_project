"""
Microphone recording module.

Dev mode:  records from default system mic using sounddevice.
Prod mode: same (USB mic shows up as ALSA device on Orange Pi).
"""

from __future__ import annotations

import io
import wave
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Recorder:
    """Records audio from the default microphone."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        silence_timeout: float = 2.0,
        silence_threshold: int = 500,
        input_device: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_timeout = silence_timeout
        self.silence_threshold = silence_threshold
        self.input_device = input_device
        self._frames: list[np.ndarray] = []
        self._recording = False

    def record_until_stopped(self) -> bytes:
        """Record audio until stop() is called from another thread. Returns WAV bytes."""
        import sounddevice as sd  # type: ignore

        self._frames = []
        self._recording = True

        def _callback(indata, frame_count, time_info, status):
            if status:
                logger.warning("Audio status: %s", status)
            if self._recording:
                self._frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=_callback,
            blocksize=1024,
            device=self.input_device,
        ):
            while self._recording:
                sd.sleep(100)

        return self._to_wav()

    def record_with_silence_detection(self) -> bytes:
        """Record audio, auto-stop after silence_timeout seconds of silence."""
        import sounddevice as sd  # type: ignore

        self._frames = []
        self._recording = True
        silent_chunks = 0
        chunks_per_second = self.sample_rate / 1024

        def _callback(indata, frame_count, time_info, status):
            nonlocal silent_chunks
            if status:
                logger.warning("Audio status: %s", status)
            if not self._recording:
                return

            self._frames.append(indata.copy())

            amplitude = np.abs(indata).mean()
            if amplitude < self.silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if (
                self.silence_timeout > 0
                and len(self._frames) > chunks_per_second  # at least 1s recorded
                and silent_chunks > chunks_per_second * self.silence_timeout
            ):
                self._recording = False

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=_callback,
            blocksize=1024,
            device=self.input_device,
        ):
            while self._recording:
                sd.sleep(100)

        return self._to_wav()

    def stop(self):
        """Stop an ongoing recording."""
        self._recording = False

    def _to_wav(self) -> bytes:
        """Convert recorded frames to WAV bytes in memory."""
        if not self._frames:
            logger.warning("No audio frames captured")
            return b""

        audio_data = np.concatenate(self._frames, axis=0)
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        return buf.getvalue()

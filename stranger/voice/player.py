"""Play audio through the speakers (sounddevice / PortAudio)."""

from __future__ import annotations

import time

import numpy as np
import sounddevice as sd

from .tts import SpokenAudio


def list_output_devices() -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]


def resolve_output_device(device: int | str | None) -> int | None:
    if device is None or isinstance(device, int):
        return device
    if device.strip().isdigit():
        return int(device)
    needle = device.lower()
    for index, name in list_output_devices():
        if needle in name.lower():
            return index
    raise ValueError(f"No output device matching {device!r}. Available: "
                     + "; ".join(f"[{i}] {n}" for i, n in list_output_devices()))


class AudioPlayer:
    """Non-blocking playback with stop(), so a reply can be cut short."""

    def __init__(self, device: int | str | None = None, volume: float = 1.0):
        self.device_index = resolve_output_device(device)
        self.volume = volume
        self._stream: sd.OutputStream | None = None
        self._pos = 0
        self._data: np.ndarray | None = None

    @property
    def device_name(self) -> str:
        return str(sd.query_devices(self.device_index, "output")["name"])

    def play(self, audio: SpokenAudio, block: bool = True) -> None:
        self.stop()
        self._data = np.clip(audio.samples * self.volume, -1.0, 1.0).astype(np.float32)
        self._pos = 0

        def callback(outdata, frames, time_info, status):  # noqa: ARG001
            chunk = self._data[self._pos:self._pos + frames]
            outdata[:len(chunk), 0] = chunk
            if len(chunk) < frames:
                outdata[len(chunk):, 0] = 0.0
                self._pos = len(self._data)
                raise sd.CallbackStop
            self._pos += frames

        self._stream = sd.OutputStream(samplerate=audio.sample_rate, channels=1, dtype="float32",
                                       device=self.device_index, callback=callback)
        self._stream.start()
        if block:
            self.wait()

    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def wait(self, poll_seconds: float = 0.05) -> None:
        while self.is_playing():
            time.sleep(poll_seconds)
        self._close()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.abort()
            finally:
                self._close()

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

"""Microphone capture via sounddevice (PortAudio).

Audio is returned as mono float32 numpy arrays at 16 kHz, ready for Whisper.
The device is opened at 16 kHz when it supports that, otherwise at its native
rate and resampled.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import WHISPER_SAMPLE_RATE


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} ({self.channels} ch, {self.default_sample_rate} Hz)"


def list_input_devices() -> list[InputDevice]:
    """All devices that can record audio."""
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append(
                InputDevice(index, info["name"], info["max_input_channels"], int(info["default_samplerate"]))
            )
    return devices


def resolve_device(device: int | str | None) -> int | None:
    """Turn a device spec (None, index, or name substring) into a PortAudio index."""
    if device is None:
        return None
    if isinstance(device, int):
        return device
    if device.strip().isdigit():
        return int(device)
    needle = device.lower()
    matches = [d for d in list_input_devices() if needle in d.name.lower()]
    if not matches:
        raise ValueError(f"No input device matching {device!r}. Available: "
                         + "; ".join(str(d) for d in list_input_devices()))
    return matches[0].index


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio. Uses scipy if present, linear interpolation otherwise."""
    if src_rate == dst_rate:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    except ImportError:
        n_out = int(round(len(audio) * dst_rate / src_rate))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def rms_level(audio: np.ndarray) -> float:
    """Root-mean-square level of the signal, 0.0 for silence, up to 1.0."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class Microphone:
    """Record from one input device as 16 kHz mono float32."""

    def __init__(self, device: int | str | None = None, sample_rate: int = WHISPER_SAMPLE_RATE):
        self.device_index = resolve_device(device)
        self.target_rate = sample_rate
        self.capture_rate = self._pick_capture_rate()

    @property
    def device_name(self) -> str:
        info = sd.query_devices(self.device_index, "input")
        return str(info["name"])

    def _pick_capture_rate(self) -> int:
        try:
            sd.check_input_settings(device=self.device_index, channels=1, samplerate=self.target_rate)
            return self.target_rate
        except Exception:
            info = sd.query_devices(self.device_index, "input")
            return int(info["default_samplerate"])

    def _to_target(self, frames: np.ndarray) -> np.ndarray:
        mono = frames[:, 0] if frames.ndim == 2 else frames
        return resample(np.ascontiguousarray(mono, dtype=np.float32), self.capture_rate, self.target_rate)

    def record(self, seconds: float) -> np.ndarray:
        """Block for `seconds` and return the captured audio."""
        n_frames = int(seconds * self.capture_rate)
        frames = sd.rec(n_frames, samplerate=self.capture_rate, channels=1,
                        dtype="float32", device=self.device_index)
        sd.wait()
        return self._to_target(frames)

    def chunks(self, chunk_seconds: float) -> Iterator[np.ndarray]:
        """Yield consecutive audio chunks forever (until the consumer stops iterating)."""
        q: queue.Queue[np.ndarray] = queue.Queue()
        frames_per_chunk = int(chunk_seconds * self.capture_rate)

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            q.put(indata.copy())

        with sd.InputStream(samplerate=self.capture_rate, channels=1, dtype="float32",
                            device=self.device_index, blocksize=frames_per_chunk, callback=callback):
            while True:
                yield self._to_target(q.get())

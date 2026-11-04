"""Speech output. The Piper clip test needs piper-tts and a voice; it is skipped otherwise.

    STRANGER_PIPER_VOICES=~/.local/share/stranger/piper-voices python -m unittest tests.test_voice
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stranger.voice.tts import SpokenAudio, VoiceConfig, pcm16_to_float


class HelperTests(unittest.TestCase):
    def test_pcm16_to_float_and_duration(self) -> None:
        samples = pcm16_to_float(np.array([0, 16384, -32768], dtype="<i2").tobytes())
        self.assertAlmostEqual(samples[1], 0.5, places=3)
        self.assertEqual(SpokenAudio(np.zeros(22050, dtype=np.float32), 22050, "x", 0.0).duration, 1.0)


class PiperClipTest(unittest.TestCase):
    def test_short_clip_is_written(self) -> None:
        try:
            from stranger.voice.piper_tts import DEFAULT_VOICES_DIR, PiperTTS, list_voices
        except ImportError:
            self.skipTest("piper-tts not installed")
        if not list_voices(DEFAULT_VOICES_DIR):
            self.skipTest(f"no Piper voice in {DEFAULT_VOICES_DIR}")
        tts = PiperTTS(VoiceConfig(backend="piper"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            tts.synthesize_to_wav("The shape waits, and it does not mind.", path)
            self.assertGreater(path.stat().st_size, 10_000)
        audio = tts.synthesize("Hello.")
        self.assertGreater(audio.duration, 0.2)


if __name__ == "__main__":
    unittest.main()

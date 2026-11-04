"""Speech input. The hardware test records 3 s and is run only with STRANGER_MIC_TEST=1."""

from __future__ import annotations

import os
import unittest

import numpy as np

from stranger.speech import SpeechConfig, Transcript, utterances
from stranger.speech.microphone import resample, rms_level


class HelperTests(unittest.TestCase):
    def test_resample_and_rms(self) -> None:
        audio = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)
        self.assertEqual(len(resample(audio, 48000, 16000)), 16000)
        self.assertGreater(rms_level(audio), 0.5)
        self.assertEqual(rms_level(np.zeros(10, dtype=np.float32)), 0.0)

    def test_utterances_group_until_silence(self) -> None:
        def t(text):
            return Transcript(text=text, segments=[], language="en", audio_seconds=2.0, elapsed_seconds=0.1)
        stream = [t("good"), t("evening"), t(""), t("one two three four"), t("five"), t("six"), t("seven"), t("")]
        self.assertEqual(list(utterances(iter(stream), max_chunks=3)),
                         ["good evening", "one two three four five six", "seven"])
        muted = {"on": True}
        self.assertEqual(list(utterances(iter(stream), 3, lambda: muted["on"])), [])


@unittest.skipUnless(os.environ.get("STRANGER_MIC_TEST") == "1", "set STRANGER_MIC_TEST=1 to record from the microphone")
class MicrophoneTest(unittest.TestCase):
    def test_record_and_transcribe(self) -> None:
        from stranger.speech import Microphone, Transcriber
        mic = Microphone(os.environ.get("STRANGER_MIC_DEVICE"))
        audio = mic.record(3.0)
        print(f"\nrecorded 3 s on {mic.device_name}, level {rms_level(audio):.4f}")
        transcriber = Transcriber(SpeechConfig(model_size=os.environ.get("STRANGER_WHISPER", "small")))
        transcriber.load()
        print("heard:", transcriber.transcribe(audio).text or "(nothing)")


if __name__ == "__main__":
    unittest.main()

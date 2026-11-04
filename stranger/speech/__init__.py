"""Speech-to-text input: microphone capture plus local Whisper transcription.

This package knows nothing about the agent. It turns microphone audio into
text (`transcribe_stream`) and groups it into utterances (`utterances`).
"""

from .config import WHISPER_SAMPLE_RATE, SpeechConfig
from .microphone import InputDevice, Microphone, list_input_devices, resolve_device
from .stream import transcribe_stream, utterances
from .transcriber import Transcriber, Transcript

__all__ = ["WHISPER_SAMPLE_RATE", "SpeechConfig", "InputDevice", "Microphone", "list_input_devices",
           "resolve_device", "Transcriber", "Transcript", "transcribe_stream", "utterances"]

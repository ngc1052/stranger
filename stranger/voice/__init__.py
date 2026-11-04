"""Voice output: text-to-speech synthesis and speaker playback."""

from .player import AudioPlayer, list_output_devices, resolve_output_device
from .tts import SpokenAudio, TextToSpeech, VoiceConfig, make_tts

__all__ = ["AudioPlayer", "list_output_devices", "resolve_output_device",
           "SpokenAudio", "TextToSpeech", "VoiceConfig", "make_tts"]

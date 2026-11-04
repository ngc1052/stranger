# Speech output

`--speak` reads each reply aloud. Thinking is never spoken. Press **x** in
the terminal or **Stop speech** on the page to cut a reply off.

## Piper (offline, default)

```bash
pip install piper-tts
python -m piper.download_voices en_GB-alan-medium --download-dir ~/.local/share/stranger/piper-voices
python -m stranger serve --speak
```

A voice is a pair of files, `<name>.onnx` and `<name>.onnx.json`. The
voices directory is `~/.local/share/stranger/piper-voices` or whatever
`STRANGER_PIPER_VOICES` points to. Pick a voice with `--voice en_US-lessac-medium`
(a name in the directory) or `--voice /path/to/voice.onnx`. The list of
available voices: `python -m piper.download_voices --help`, or
https://github.com/OHF-Voice/piper1-gpl.

`python -m stranger check` lists the voices it finds, and the output
devices for `--output-device`.

## Other backends

| `--tts` | Needs | Notes |
|---|---|---|
| `piper` | piper-tts + a voice | offline neural voice, default |
| `espeak` | the `espeak-ng` binary | offline, robotic, no download |
| `elevenlabs` | `ELEVENLABS_API_KEY` | `--voice` is a voice id |
| `openai` | `OPENAI_API_KEY` | `--voice` is a voice name such as `onyx` |

## Checking it works

```bash
STRANGER_PIPER_VOICES=~/.local/share/stranger/piper-voices python -m unittest tests.test_voice
```

The clip test writes a short WAV with the installed voice; it is skipped
when no voice is found.

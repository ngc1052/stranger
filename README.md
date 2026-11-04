# The Stranger

A spoken-conversation agent that runs on a local language model. You talk;
it listens to everything, and answers when you say its name. Its replies
stream to a web page and to the terminal, with the model's thinking kept
apart from what it says, and with the telemetry you need to run it live:
token counts, time to first token, total time, how full the context window
is, what state the agent is in, and whether the model server is reachable.

Everything runs on your own machine. No API keys are needed. Speech
recognition is local (Whisper), the model is local (Ollama) or on a server
you control (any OpenAI-compatible endpoint such as vLLM through an SSH
tunnel), and the voice is local (Piper).

```
microphone ─► Whisper ─► utterance ─► Agent ──► model server (Ollama / vLLM)
                                        │            thinking ┊ text (streamed)
                                        ├─► events ─► web page (SSE) and terminal
                                        └─► Piper ─► speakers
```

## Quick start

```bash
git clone https://github.com/ngc1052/stranger.git && cd stranger
pip install -r requirements.txt            # or: pip install -e ".[all]"
ollama pull qwen3:1.7b                     # any Ollama model works; this one fits in 2 GB
python -m stranger check                   # endpoint, models, microphones, speakers, voices
python -m stranger serve --input text      # open http://127.0.0.1:8766 and type
```

Say or type something containing the word **Stranger** and it answers.
Anything else is remembered but not answered. To use the microphone and
hear the replies:

```bash
python -m stranger serve --mic-device "Digital Microphone" --speak
```

`python -m stranger chat` gives the same agent in the terminal only.

## What the page shows

| Panel | Contents |
|---|---|
| Transcript | your lines (left, blue) and the agent's (right, sand). Lines that were heard but not answered are faded and say why. |
| Thinking | the model's reasoning stream, grey, in its own panel, never mixed into the reply. If a model returns no separate thinking, the panel says so. |
| Last reply | input tokens, output tokens, time to first token, total time. The total ticks live while a reply streams. |
| Context window | a bar and a number: how many tokens of the window the next request would use, and whether that is measured or estimated. |
| Status | listening, deciding, thinking, speaking. |
| Model endpoint | online / offline, checked every five seconds. |

The header has a switch for the reply rule (when named / always), and the
footer a text box that feeds the agent exactly like transcribed speech,
plus **Stop speech** and **Clear**.

## The reply rule

Every utterance is appended to the conversation, so the model can follow a
discussion it is not part of. In the default `named` mode the agent only
generates a reply when its trigger word is said as a whole word, in any
case ("Stranger," yes; "strangers" no). `always` mode answers everything.
Switch with `--reply always`, the header control, the `/always` and `/named`
commands in the terminal chat, or `POST /api/settings`. The trigger word
defaults to the last word of the profile's display name (`--trigger-name`
overrides it).

## Profiles: who the agent is

A profile is a directory with a `constitution.md` and an optional
`profile.toml`. The constitution is the persona, in plain language, and is
always the first thing in the system prompt. A context file (any text) can
follow it. Two profiles ship: `stranger` and a plain `assistant`.

```bash
python -m stranger profiles
python -m stranger serve --profile assistant
python -m stranger serve --profile ~/personas/gardener     # any directory with a constitution.md
```

See [docs/profiles.md](docs/profiles.md) for the fields and an example.

## The model endpoint

```bash
python -m stranger serve --host http://localhost:11434 --model qwen3:1.7b            # Ollama (default)
ssh -N -L 8000:localhost:8000 cluster-login                                           # tunnel to vLLM
python -m stranger serve --api openai --host http://localhost:8000 --model Qwen/Qwen3-4B --num-ctx 32768
```

Both protocols are translated into the same stream of events, so thinking
and text stay separate and token counts arrive the same way. Other
OpenAI-compatible servers (llama.cpp, LM Studio) work with `--api openai`.
Generation options: `--max-tokens`, `--max-sentences` (replies are cut
after this many complete sentences; 0 disables), `--num-ctx`,
`--no-thinking`.

## Speech in and out

Speech recognition uses faster-whisper on the CPU, in chunks
(`--chunk-seconds`, default 2 s). An utterance ends at the first silent
chunk or after `--max-chunks` chunks. The microphone is ignored while the
agent is deciding, thinking or speaking, so it does not hear itself.
`--mic-device` takes an index or a name substring from `python -m stranger check`;
on some laptops the system default microphone records silence and a named
device must be chosen.

Speech output is off by default. `--speak` reads each reply with an offline
Piper voice (never the thinking); press **x** in the terminal or **Stop
speech** on the page to cut it off. Installing a voice is two commands, see
[docs/voice.md](docs/voice.md). `--tts espeak` works without any download.

## Telemetry, precisely

- **Token counts** come from the server (`eval_count` / `prompt_eval_count`
  on Ollama, `usage` on OpenAI-style servers). When a reply is cut at the
  sentence limit the connection is closed early and Ollama sends no counts;
  the page then shows `~N` estimated from streamed chunks and says so, and
  the input count reads "not returned".
- **Time to first token** is measured at the client from the request to the
  first streamed chunk, thinking included. **Total time** is the whole call.
- **Context meter**: after a reply with server counts, measured input +
  output tokens plus a characters/4 estimate of anything appended since;
  before that, a characters/4 estimate of the whole prompt. The source is
  always shown.

## HTTP API and events

`GET /events` is a Server-Sent Events stream: a `snapshot` first, then
every agent event (`status`, `transcript`, `reply_start`, `thinking_delta`,
`text_delta`, `stats`, `reply_end`, `meter`, `decision`, `connection`,
`notice`, `settings`, `cleared`, `speech`). `GET /api/state` returns the
snapshot. `POST /api/utterance`, `/api/settings`, `/api/command` take JSON.
Details in [docs/http-api.md](docs/http-api.md); the architecture and the
state machine in [docs/architecture.md](docs/architecture.md).

## Tests

```bash
python -m unittest discover -s tests -t .
```

The suite runs offline: the LLM client is tested against fake Ollama and
OpenAI-compatible servers, the agent and the web server against a scripted
client. Two tests are hardware-gated: the Piper clip test runs when a voice
is installed, the microphone test with `STRANGER_MIC_TEST=1`.

## Layout

```
stranger/
  __main__.py      CLI: serve | chat | check | profiles
  profiles.py      constitution + profile.toml + context -> system prompt
  llm/             LLMClient (ollama | openai protocols), ThinkRouter, messages
  agent/           Agent (reply rule, status, streamed generation), EventBus, ContextMeter
  speech/          microphone capture, Whisper transcription, utterance grouping
  voice/           Piper / espeak / ElevenLabs / OpenAI synthesis, speaker playback
  web/             FastAPI server (SSE) and the page
  chat.py          terminal view of the same event stream
  keys.py          q / x keys without blocking
profiles/          stranger, assistant
tests/             offline suite with fake servers
docs/              architecture, profiles, HTTP API, voice setup
```

## Requirements

Python 3.11 or newer. The core (`stranger.agent`, `stranger.llm`,
`stranger.profiles`) uses the standard library only. The web GUI needs
FastAPI and uvicorn; speech in needs faster-whisper, sounddevice, numpy and
scipy; speech out needs piper-tts and sounddevice. Ollama or another model
server must be running.

License: MIT.

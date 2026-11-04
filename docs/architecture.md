# Architecture

## One agent, several windows on it

The `Agent` (`stranger/agent/core.py`) is the only thing that holds state:
the conversation history, the transcript, the status, the last reply's
statistics, the context meter and the endpoint connection. It is plain
Python with no web or audio dependencies.

Everything else is either an input that feeds it or an output that watches
it:

| Component | Role |
|---|---|
| `speech/` | microphone -> Whisper -> `utterances()` -> `agent.submit(text, "mic")` |
| `web/server.py` | `POST /api/utterance` -> `agent.submit(text, "typed")`; `GET /events` relays the event bus |
| `chat.py` `TerminalView` | subscribes to the same event bus and prints |
| `voice/` | the optional `speak(text)` hook the agent calls after a reply |
| `llm/` | the model call the agent makes |

Because the page and the terminal only read events, closing the page
loses nothing: on reconnect it receives a `snapshot` (the full state) and
then continues live.

## The turn

```
submit(text) ──► queue ──► worker thread ──► handle_utterance(text)
                                                 │
                        status: deciding         ├─ append Turn(user, text) to history
                                                 ├─ named? / always?  ── no ──► Decision("silent")
                                                 │ yes
                        status: thinking         ├─ client.stream(system_prompt, messages)
                        status: speaking         │     on_thinking -> event thinking_delta
                                                 │     on_text     -> event text_delta, stats(live)
                                                 ├─ append Turn(agent, reply) ; meter.record(usage)
                                                 ├─ events reply_end, stats, meter
                        status: speaking         ├─ speak(reply)  (optional, blocking)
                        status: listening        └─ Decision("reply")
```

Utterances are processed one at a time, in order. While the agent is not
listening (`agent.busy`), the microphone feed discards what it hears, so
the agent does not transcribe its own voice.

## Status

| Status | Meaning |
|---|---|
| `listening` | idle, waiting for an utterance |
| `deciding` | an utterance arrived; the reply rule is being applied |
| `thinking` | the model's thinking stream is arriving |
| `speaking` | the reply text is streaming, or the voice is playing it |

## Events

The `EventBus` keeps every event in `log` (handy in tests) and fans new
ones out to subscribers (the SSE handler, the terminal view). Each event is
a dict with `seq`, `type`, `t` and its own fields; `format_sse` renders it
as a Server-Sent Event.

| Type | Fields |
|---|---|
| `status` | `status` |
| `transcript` | `entry` {n, role: human/agent, speaker, text, t, source, named, answered, thinking} |
| `reply_start` | `speaker` |
| `thinking_delta`, `text_delta` | `delta` |
| `stats` | `live` (true while streaming: output_tokens so far, total_seconds) or the final set |
| `reply_end` | `speaker, text, had_thinking, stats, error` |
| `meter` | `used, window, fraction, percent, source` |
| `decision` | `action` (reply/silent/ignored), `reason` |
| `connection` | `online, detail, checked_at` |
| `notice` | `level` (info/warning/error), `text` |
| `settings` | `reply_mode` |
| `cleared`, `speech` | `speech` has `state` start/end |

## The model client

`LLMClient.stream()` takes a system prompt and chat messages and returns a
`Reply` (text, thinking, `Usage`, elapsed time, model, error). Internally
both protocols are reduced to a stream of `{thinking, content}` deltas plus
a final `{done, input_tokens, output_tokens}` event, which `ThinkRouter`
splits into the two callbacks. The router also understands inline
`<think>...</think>` tags, even when a tag is split across chunks, and
enforces the sentence limit by closing the connection after the last
allowed sentence. `warm_up()` sends the system prompt once with a one-token
limit so the server caches the evaluated prefix before the first real turn.

## The context meter

`ContextMeter` answers "how full is the window?" without a tokenizer. After
a reply with server counts, `used = input_tokens + output_tokens` of that
request plus characters/4 of anything appended since, labelled `measured`.
Before any reply, characters/4 of the whole prompt, labelled `estimated`.
The window is `--num-ctx`.

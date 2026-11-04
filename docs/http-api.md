# HTTP API

The server binds to `127.0.0.1:8766` by default (`--bind`, `--port`).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/` | | the page |
| GET | `/events` | | Server-Sent Events: a `snapshot` event, then every agent event |
| GET | `/api/state` | | the snapshot as JSON |
| POST | `/api/utterance` | `{"text": "...", "source": "typed"}` | `{"queued": true}`; 400 when text is empty |
| POST | `/api/settings` | `{"reply_mode": "named" \| "always"}` | `{"reply_mode": ...}` |
| POST | `/api/command` | `{"command": "stop_speech" \| "clear"}` | `{"ok": bool}` |

## The snapshot

```json
{
  "status": "listening",
  "settings": {"user_name": "Visitor", "display_name": "The Stranger", "speaker_tag": "THE STRANGER",
               "trigger_name": "Stranger", "reply_mode": "named", "speak": false},
  "model": "qwen3:1.7b", "endpoint": "http://localhost:11434",
  "transcript": [ {"n": 1, "role": "human", "speaker": "Visitor", "text": "...", "named": true, "answered": true, ...} ],
  "context_messages": 2,
  "stats": {"input_tokens": 1380, "output_tokens": 412, "output_tokens_estimated": null, "counts_source": "server",
            "total_seconds": 9.6, "first_token_seconds": 1.7, "truncated": false, "model": "qwen3:1.7b"},
  "meter": {"used": 1792, "window": 8192, "fraction": 0.2188, "percent": 21.9, "source": "measured"},
  "connection": {"online": true, "detail": "Ollama 0.30.11", "checked_at": 1793803519.5}
}
```

## Server-Sent Events

Each event is sent as

```
id: <seq>
event: <type>
data: {"seq": ..., "type": "...", "t": <unix time>, ...fields}
```

with a `: keep-alive` comment every two seconds when idle. Event types and
fields are listed in [architecture.md](architecture.md#events). A minimal
client:

```js
const es = new EventSource("/events");
es.addEventListener("text_delta", (m) => process.stdout.write(JSON.parse(m.data).delta));
```

```bash
curl -N http://127.0.0.1:8766/events
curl -X POST -H 'Content-Type: application/json' -d '{"text":"Stranger, hello"}' http://127.0.0.1:8766/api/utterance
```

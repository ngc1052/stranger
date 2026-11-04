# Configuration

All settings live in one TOML file, `stranger.toml`. The copy at the root
of the repository is the editable default; `python -m stranger config`
shows what is in effect and where every value came from.

## Where settings come from

Each layer overrides the one before it:

| Layer | Example |
|---|---|
| defaults | the field defaults in `stranger/config.py` |
| file | `stranger.toml`: `--config PATH`, else `$STRANGER_CONFIG`, else `./stranger.toml`, else `~/.config/stranger/stranger.toml` |
| preset | `--preset cluster` overlays `[presets.cluster.*]` from the same file |
| environment | `STRANGER_MODEL_HOST=http://localhost:8000`, `STRANGER_VOICE_ENABLED=true` |
| flags | `--model`, `--host`, `--port`, `--speak`, ... |

Unknown sections and keys are errors that name them; a value of the
wrong type is an error that names the key and the expected type. Lists in
the environment are comma-separated; booleans accept true/false, yes/no,
on/off, 1/0.

## Sections

The keys mirror the dataclasses one to one.

**[model]** (`LLMConfig`): `host`, `api` (ollama | openai), `model`,
`num_ctx`, `max_tokens`, `max_sentences`, `temperature`, `repeat_penalty`,
`stop` (a list), `think`, `keep_alive`, `timeout_seconds`.

**[agent]**: `profile`, `user_name`, `trigger_name` (default: last word of
the profile's display name), `reply_mode` (named | always), `warm_up`.

**[speech]**: `input` (mic | text), `device`, `model_size`,
`chunk_seconds`, `max_chunks`, `language`, `compute_device`,
`compute_type`, `beam_size`.

**[voice]**: `enabled`, `backend` (piper | espeak | elevenlabs | openai),
`voice`, `output_device`, `speed`.

**[web]**: `bind`, `port`, `quiet`.

## Presets

A preset is a named overlay for the sections it mentions. The default file
ships three:

```toml
[presets.cluster.model]        # vLLM behind `ssh -N -L 8000:localhost:8000 cluster-login`
host = "http://localhost:8000"
api = "openai"
model = "Qwen/Qwen3-4B"
num_ctx = 32768

[presets.fast.model]           # no thinking, short replies
think = false
max_tokens = 120

[presets.typed.speech]         # no microphone
input = "text"
```

```bash
python -m stranger serve --preset cluster
python -m stranger config --preset fast
```

## Live settings

Some settings take effect on the next reply without a restart:
`agent.reply_mode`, `agent.trigger_name`, `model.temperature`,
`model.max_sentences`, `model.max_tokens`, `model.think`. Change them from
the **settings** drawer on the page, the reply-rule switch in the header,
or the API:

```bash
curl -X POST -H 'Content-Type: application/json' \
     -d '{"model.temperature": 0.8, "agent.reply_mode": "always"}' http://127.0.0.1:8766/api/settings
```

Everything else (model, endpoint, microphone, voice, port) is read at
start, so edit the file and restart. The `config` command marks the live
ones.

## Profiles are not configuration

A profile says who the agent is; the configuration says how it runs. The
only crossover is `stop_words` in `profile.toml`, which are added to
`model.stop`. See [profiles.md](profiles.md).

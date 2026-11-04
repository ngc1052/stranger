"""Settings: one TOML file, layered overrides, the dataclasses as the single source of truth.

Resolution order, each layer overriding the one before:

  1. defaults      the field defaults of the dataclasses below
  2. file          stranger.toml: --config PATH, else $STRANGER_CONFIG, else ./stranger.toml,
                   else ~/.config/stranger/stranger.toml
  3. preset        [presets.NAME] in the same file, overlaid on the sections it names
  4. environment   STRANGER_<SECTION>_<KEY>, e.g. STRANGER_MODEL_HOST=http://localhost:8000
  5. flags         command-line options

`load_settings()` returns the merged Settings plus a table saying which layer
each value came from, which `python -m stranger config` prints. Unknown keys
are errors that name the section and key; values are coerced to the field's type.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .llm.client import LLMConfig

DEFAULT_FILE_NAME = "stranger.toml"
ENV_PREFIX = "STRANGER_"
LIVE_KEYS = {"model.temperature", "model.max_sentences", "model.max_tokens", "model.think",
             "agent.reply_mode", "agent.trigger_name"}        # changeable while the server runs


class ConfigError(ValueError):
    pass


@dataclass
class AgentOptions:
    profile: str = "stranger"               # profile name under profiles/, or a directory path
    user_name: str = "Visitor"              # the human's name, as the model sees it
    trigger_name: str | None = None         # word that makes the agent answer; None = last word of the display name
    reply_mode: str = "named"               # "named" | "always"
    warm_up: bool = True                    # pre-evaluate the system prompt at start


@dataclass
class SpeechOptions:
    input: str = "mic"                      # "mic" | "text"
    device: str | int | None = None         # microphone index or name substring; None = system default
    model_size: str = "small"               # faster-whisper model
    chunk_seconds: float = 2.0
    max_chunks: int = 6                     # close an utterance after this many speech chunks
    language: str | None = "en"
    compute_device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1


@dataclass
class VoiceOptions:
    enabled: bool = False                   # read replies aloud
    backend: str = "piper"                  # piper | espeak | elevenlabs | openai
    voice: str | None = None                # voice name or path; None = backend default
    output_device: str | int | None = None
    speed: float = 1.0


@dataclass
class WebOptions:
    bind: str = "127.0.0.1"
    port: int = 8766
    quiet: bool = False                     # no terminal echo of the conversation


@dataclass
class Settings:
    model: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentOptions = field(default_factory=AgentOptions)
    speech: SpeechOptions = field(default_factory=SpeechOptions)
    voice: VoiceOptions = field(default_factory=VoiceOptions)
    web: WebOptions = field(default_factory=WebOptions)
    source_file: Path | None = None         # the file that was read, if any
    preset: str | None = None

    SECTIONS = ("model", "agent", "speech", "voice", "web")

    def as_dict(self) -> dict:
        return {name: dataclasses.asdict(getattr(self, name)) for name in self.SECTIONS}

    def speech_config(self):
        from .speech import SpeechConfig
        s = self.speech
        return SpeechConfig(model_size=s.model_size, device=s.device, chunk_seconds=s.chunk_seconds,
                            compute_device=s.compute_device, compute_type=s.compute_type,
                            language=s.language, beam_size=s.beam_size)

    def voice_config(self):
        from .voice import VoiceConfig
        v = self.voice
        return VoiceConfig(backend=v.backend, voice=v.voice, output_device=v.output_device, speed=v.speed)


# -- locating and reading the file ---------------------------------------------------------

def find_config_file(explicit: str | Path | None = None, env: dict = os.environ) -> Path | None:
    candidates = [explicit, env.get(ENV_PREFIX + "CONFIG"), Path.cwd() / DEFAULT_FILE_NAME,
                  Path.home() / ".config" / "stranger" / DEFAULT_FILE_NAME]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
        if c is not None and c is explicit:
            raise ConfigError(f"config file not found: {c}")
    return None


def read_config_file(path: Path) -> dict:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    for section in data:
        if section not in Settings.SECTIONS and section != "presets":
            raise ConfigError(f"{path}: unknown section [{section}]; expected one of "
                              f"{', '.join(Settings.SECTIONS)} or [presets.NAME]")
    return data


# -- coercion -------------------------------------------------------------------------------

def _field_types(section: str) -> dict[str, Any]:
    cls = type(getattr(Settings(), section))
    return {f.name: f.type for f in fields(cls)}


def coerce(value: Any, type_name: str, where: str) -> Any:
    """Coerce a string (from the environment or a flag) or a TOML value to the field's annotated type."""
    t = str(type_name)
    optional = "None" in t
    if value is None or (isinstance(value, str) and value.lower() in ("none", "null", "") and optional):
        if optional:
            return None
        raise ConfigError(f"{where}: a value is required")
    base = t.replace(" | None", "").replace("None | ", "")
    try:
        if base == "bool":
            if isinstance(value, bool):
                return value
            s = str(value).lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            raise ValueError
        if base == "int":
            return int(value)
        if base == "float":
            return float(value)
        if base.startswith("tuple"):
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            return tuple(str(v) for v in value)
        if base == "str | int":
            if isinstance(value, int):
                return value
            s = str(value)
            return int(s) if s.isdigit() else s
        return str(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: cannot read {value!r} as {base}") from None


# -- merging --------------------------------------------------------------------------------

def _apply(settings: Settings, section: str, values: dict, layer: str, sources: dict, where: str) -> None:
    if section not in Settings.SECTIONS:
        raise ConfigError(f"{where}: unknown section [{section}]")
    types = _field_types(section)
    target = getattr(settings, section)
    for key, value in values.items():
        if key not in types:
            raise ConfigError(f"{where}: unknown key {key!r} in [{section}]; known keys: {', '.join(types)}")
        setattr(target, key, coerce(value, types[key], f"{where} [{section}] {key}"))
        sources[f"{section}.{key}"] = layer


def env_overrides(env: dict = os.environ) -> dict[str, dict]:
    """STRANGER_MODEL_HOST -> {"model": {"host": ...}}. Keys may contain underscores (MAX_TOKENS)."""
    out: dict[str, dict] = {}
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX) or name == ENV_PREFIX + "CONFIG" or name == ENV_PREFIX + "PIPER_VOICES":
            continue
        rest = name[len(ENV_PREFIX):].lower()
        section, _, key = rest.partition("_")
        if section in Settings.SECTIONS and key:
            out.setdefault(section, {})[key] = value
    return out


def load_settings(config_path: str | Path | None = None, preset: str | None = None,
                  flags: dict[str, dict] | None = None, env: dict = os.environ) -> tuple[Settings, dict[str, str]]:
    """Merge defaults, file, preset, environment and flags. Returns (settings, sources)."""
    settings = Settings()
    sources = {f"{s}.{f.name}": "default" for s in Settings.SECTIONS for f in fields(type(getattr(settings, s)))}
    path = find_config_file(config_path, env)
    data = read_config_file(path) if path else {}
    settings.source_file = path
    for section in Settings.SECTIONS:
        if section in data:
            _apply(settings, section, data[section], "file", sources, str(path))
    if preset:
        presets = data.get("presets", {})
        if preset not in presets:
            raise ConfigError(f"no preset {preset!r} in {path or 'any config file'}; available: "
                              f"{', '.join(presets) or 'none'}")
        settings.preset = preset
        for section, values in presets[preset].items():
            _apply(settings, section, values, f"preset {preset}", sources, f"{path} [presets.{preset}]")
    for section, values in env_overrides(env).items():
        _apply(settings, section, values, "environment", sources, "environment")
    for section, values in (flags or {}).items():
        _apply(settings, section, {k: v for k, v in values.items() if v is not None}, "flag", sources, "command line")
    if settings.agent.reply_mode not in ("named", "always"):
        raise ConfigError("agent.reply_mode must be 'named' or 'always'")
    if settings.model.api not in ("ollama", "openai"):
        raise ConfigError("model.api must be 'ollama' or 'openai'")
    if settings.speech.input not in ("mic", "text"):
        raise ConfigError("speech.input must be 'mic' or 'text'")
    return settings, sources


def format_settings(settings: Settings, sources: dict[str, str]) -> str:
    """A table of every setting, its value, where it came from, and whether it can change live."""
    lines = [f"config file: {settings.source_file or 'none found'}" + (f"   preset: {settings.preset}" if settings.preset else "")]
    for section in Settings.SECTIONS:
        lines.append(f"\n[{section}]")
        for key, value in dataclasses.asdict(getattr(settings, section)).items():
            dotted = f"{section}.{key}"
            shown = ", ".join(value) if isinstance(value, (tuple, list)) else repr(value) if value is None else str(value)
            live = "  (live)" if dotted in LIVE_KEYS else ""
            lines.append(f"  {key:<16} = {shown:<36} {sources.get(dotted, 'default')}{live}")
    return "\n".join(lines)

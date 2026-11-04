"""Profiles: who the agent is.

A profile is a directory holding a `constitution.md` (the persona, in plain
language, always first in the system prompt) and optionally a `profile.toml`
(display name, speaker tag, preamble, context file). The context file is
any text placed after the constitution, verbatim, under a label.

    profile = load_profile("stranger")              # built-in, under profiles/
    profile = load_profile("~/my-persona")          # any directory with a constitution.md
    prompt = build_system_prompt(profile, user_name="Robin")
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
CONSTITUTION_FILE = "constitution.md"
PROFILE_FILE = "profile.toml"


class ProfileError(ValueError):
    pass


@dataclass
class Profile:
    name: str
    path: Path
    constitution: str
    display_name: str                        # shown on screen, e.g. "The Stranger"
    speaker_tag: str                         # used in the conversation history, e.g. "THE STRANGER"
    preamble: str = ""                       # added after the constitution; may contain {name}
    context_text: str | None = None          # optional notes, verbatim
    context_label: str = "CONTEXT"
    context_file: Path | None = None
    stop_words: tuple[str, ...] = field(default_factory=tuple)   # extra stop sequences for the model

    def __str__(self) -> str:
        ctx = f", context {self.context_file.name}" if self.context_file else ", no context"
        return f"{self.name} ({self.display_name}{ctx})"


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    if not profiles_dir.is_dir():
        return []
    return sorted(p.name for p in profiles_dir.iterdir() if (p / CONSTITUTION_FILE).is_file())


def _resolve(name_or_path: str, profiles_dir: Path) -> Path:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir() and (candidate / CONSTITUTION_FILE).is_file():
        return candidate.resolve()
    built_in = profiles_dir / name_or_path
    if (built_in / CONSTITUTION_FILE).is_file():
        return built_in
    raise ProfileError(f"No profile named {name_or_path!r}. Available: {', '.join(list_profiles(profiles_dir)) or 'none'}. "
                       f"A profile is a directory with a {CONSTITUTION_FILE} (see docs/profiles.md).")


def load_profile(name_or_path: str, profiles_dir: Path = PROFILES_DIR) -> Profile:
    path = _resolve(name_or_path, profiles_dir)
    constitution = (path / CONSTITUTION_FILE).read_text(encoding="utf-8").strip()
    if not constitution:
        raise ProfileError(f"{path / CONSTITUTION_FILE} is empty")
    meta: dict = {}
    if (path / PROFILE_FILE).is_file():
        try:
            meta = tomllib.loads((path / PROFILE_FILE).read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise ProfileError(f"{path / PROFILE_FILE}: {e}") from e
    display_name = str(meta.get("display_name", path.name.replace("_", " ").title()))
    speaker_tag = str(meta.get("speaker_tag", display_name.upper()))
    context_file = context_text = None
    if meta.get("context_file"):
        context_file = (path / str(meta["context_file"])).resolve()
        if not context_file.is_file():
            raise ProfileError(f"context_file {context_file} not found (from {path / PROFILE_FILE})")
        context_text = context_file.read_text(encoding="utf-8").strip()
    return Profile(name=path.name, path=path, constitution=constitution, display_name=display_name,
                   speaker_tag=speaker_tag, preamble=str(meta.get("preamble", "")),
                   context_text=context_text, context_label=str(meta.get("context_label", "CONTEXT")),
                   context_file=context_file, stop_words=tuple(meta.get("stop_words", ())))


def build_system_prompt(profile: Profile, user_name: str | None = None) -> str:
    """Constitution first, then the preamble (with the user's name), then the context block."""
    parts = [profile.constitution]
    if profile.preamble:
        parts.append(profile.preamble.format(name=user_name or "the person you are talking to"))
    if profile.context_text:
        parts.append(f"{profile.context_label}:\n---\n{profile.context_text}\n---")
    return "\n\n".join(parts)

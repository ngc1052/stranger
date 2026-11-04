"""Routing of a streamed response into thinking and text, with a sentence limit."""

from __future__ import annotations

import re
import time
from collections.abc import Callable

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def count_sentences(text: str) -> int:
    text = text.strip()
    return len([p for p in _SENTENCE_END_RE.split(text) if p.strip()]) if text else 0


def trim_to_sentences(text: str, max_sentences: int) -> str:
    """Keep at most `max_sentences` complete sentences; drop a trailing unfinished fragment."""
    text = text.strip()
    if not text or not max_sentences:
        return text
    parts = _SENTENCE_END_RE.split(text)
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    last = parts[-1].strip()
    if last and last[-1] in ".!?\"')":
        complete.append(last)
    if not complete:                  # one unfinished sentence: better than nothing
        return last
    return " ".join(complete[:max_sentences])


class ThinkRouter:
    """Split a streamed response into thinking and text, forwarding each as it arrives.

    Handles both a separate thinking field and inline <think>...</think> tags
    (which may be split across chunks). Stops forwarding text after
    `max_sentences` complete sentences and sets `finished`, so the caller can
    close the stream: what was shown is exactly the reply kept.
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self, on_thinking: Callable[[str], None], on_text: Callable[[str], None], max_sentences: int = 0):
        self.on_thinking, self.on_text, self.max_sentences = on_thinking, on_text, max_sentences
        self.thinking_out, self.text_out = "", ""
        self.pending = ""             # held back in case a tag is split across chunks
        self.in_think = False
        self.finished = False
        self.first_token_at: float | None = None

    def thinking(self, delta: str) -> None:
        self.first_token_at = self.first_token_at or time.perf_counter()
        self.thinking_out += delta
        self.on_thinking(delta)

    def content(self, delta: str) -> None:
        self.first_token_at = self.first_token_at or time.perf_counter()
        self.pending += delta
        while self.pending and not self.finished:
            tag = self._CLOSE if self.in_think else self._OPEN
            idx = self.pending.find(tag)
            if idx >= 0:
                self._emit(self.pending[:idx])
                self.pending = self.pending[idx + len(tag):]
                self.in_think = not self.in_think
                continue
            keep = 0                  # a partial tag at the end ("<thi") waits for the next chunk
            for n in range(min(len(tag) - 1, len(self.pending)), 0, -1):
                if tag.startswith(self.pending[-n:]):
                    keep = n
                    break
            self._emit(self.pending[:len(self.pending) - keep] if keep else self.pending)
            self.pending = self.pending[len(self.pending) - keep:] if keep else ""
            break

    def flush(self) -> None:
        if self.pending and not self.finished:
            self._emit(self.pending)
            self.pending = ""

    def _emit(self, piece: str) -> None:
        if not piece:
            return
        if self.in_think:
            self.thinking_out += piece
            self.on_thinking(piece)
            return
        if self.max_sentences:
            for i, ch in enumerate(piece):
                if ch in ".!?" and count_sentences(self.text_out + piece[:i + 1]) >= self.max_sentences:
                    cut = piece[:i + 1]
                    self.text_out += cut
                    self.on_text(cut)
                    self.finished = True
                    return
        self.text_out += piece
        self.on_text(piece)

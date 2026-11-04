"""Context-size meter: how much of the model's context window the next request would fill.

Two sources, always labelled:
  measured    the server's own input + output token counts from the last reply,
              plus an estimate for anything appended to the conversation since
  estimated   no reply yet (or the server sent no counts): characters / 4
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..llm.messages import Turn

CHARS_PER_TOKEN = 4.0          # a fair average for English prose on common tokenizers
PER_MESSAGE_OVERHEAD = 6       # role tags and separators in the chat template


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN + 0.5)


@dataclass(frozen=True)
class MeterReading:
    used: int
    window: int
    source: str                 # "measured" or "estimated"

    @property
    def fraction(self) -> float:
        return min(1.0, self.used / self.window) if self.window else 0.0

    def as_dict(self) -> dict:
        return {"used": self.used, "window": self.window, "fraction": round(self.fraction, 4),
                "percent": round(100 * self.fraction, 1), "source": self.source}


class ContextMeter:
    def __init__(self, window: int, system_prompt: str = ""):
        self.window = window
        self.system_prompt = system_prompt
        self._measured_tokens: int | None = None
        self._measured_chars = 0

    def _chars(self, history: Sequence[Turn]) -> int:
        chars = len(self.system_prompt)
        for turn in history:
            chars += len(turn.speaker) + len(turn.text) + 2 + int(PER_MESSAGE_OVERHEAD * CHARS_PER_TOKEN)
        return chars

    def record(self, input_tokens: int | None, output_tokens: int | None, history: Sequence[Turn]) -> None:
        """Remember the server's counts. `history` includes the reply they produced."""
        if input_tokens is None:
            return
        self._measured_tokens = int(input_tokens) + int(output_tokens or 0)
        self._measured_chars = self._chars(history)

    def reading(self, history: Sequence[Turn]) -> MeterReading:
        chars = self._chars(history)
        if self._measured_tokens is None:
            return MeterReading(estimate_tokens(" " * chars), self.window, "estimated")
        added = max(0, chars - self._measured_chars)
        return MeterReading(self._measured_tokens + estimate_tokens(" " * added), self.window, "measured")

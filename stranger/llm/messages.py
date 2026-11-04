"""Conversation history and its translation into chat messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    speaker: str        # a tag such as "ROBIN" or "THE STRANGER"
    text: str


def build_messages(history: Sequence[Turn], assistant_tag: str) -> list[dict]:
    """Turn the history into alternating chat messages.

    The agent's own turns become assistant messages. Everyone else's become
    user messages prefixed with their tag, and consecutive user messages are
    merged into one (several things were said before the agent answered).
    The list always starts with a user message, as chat templates expect.
    """
    messages: list[dict] = []
    for turn in history:
        if turn.speaker == assistant_tag:
            if not messages:
                messages.append({"role": "user", "content": "[The conversation begins.]"})
            messages.append({"role": "assistant", "content": turn.text})
        else:
            line = f"{turn.speaker}: {turn.text}"
            if messages and messages[-1]["role"] == "user":
                messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n" + line}
            else:
                messages.append({"role": "user", "content": line})
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": f"[Your turn, {assistant_tag}.]"})
    return messages

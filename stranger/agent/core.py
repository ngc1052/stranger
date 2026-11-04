"""The Stranger's turn-taking.

Every utterance is appended to the conversation so the model can follow it.
Whether the agent answers is decided by `reply_mode`:

  named    answer only when the agent's name is said in the utterance (default)
  always   answer every utterance

Status, shown by the GUI and the terminal: listening, deciding, thinking, speaking.

The agent is plain Python: no web, no audio. The web server and the terminal
chat wrap it and relay its events; the speech hooks are optional callables.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

from ..llm import LLMClient, Reply, Turn, build_messages
from .events import EventBus
from .meter import ContextMeter


class Status(str, Enum):
    LISTENING = "listening"
    DECIDING = "deciding"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class AgentSettings:
    user_name: str = "Visitor"              # label for the human
    display_name: str = "The Stranger"      # label for the agent
    speaker_tag: str = "THE STRANGER"       # the agent's tag in the history
    trigger_name: str = "Stranger"          # the word that makes it answer in "named" mode
    reply_mode: str = "named"               # "named" | "always"


_NAME_CACHE: dict[str, re.Pattern] = {}


def mentions_name(text: str, name: str = "Stranger") -> bool:
    """True when `name` appears as a whole word, any case ("stranger," yes, "strangers" no)."""
    pat = _NAME_CACHE.get(name)
    if pat is None:
        pat = _NAME_CACHE[name] = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.I)
    return bool(pat.search(text))


@dataclass
class Decision:
    action: str                 # reply | silent | ignored
    reason: str
    reply: Reply | None = None

    def as_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason}


class Agent:
    def __init__(self, client: LLMClient, system_prompt: str, settings: AgentSettings | None = None,
                 events: EventBus | None = None, speak=None, stop_speech=None):
        self.client = client
        self.system_prompt = system_prompt
        self.settings = settings or AgentSettings()
        self.events = events or EventBus()
        self.history: list[Turn] = []
        self.transcript: list[dict] = []
        self.status = Status.LISTENING
        self.last_stats: dict | None = None
        self.connection: dict = {"online": None, "detail": "not checked yet", "checked_at": None}
        self.meter = ContextMeter(client.config.num_ctx, system_prompt)
        self.speak, self.stop_speech = speak, stop_speech     # optional: speak(text) blocks; stop_speech()
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        self.events.publish("status", status=self.status.value)

    # -- state for a page that connects late -------------------------------------------

    @property
    def busy(self) -> bool:
        return self.status is not Status.LISTENING or not self._queue.empty()

    def state(self) -> dict:
        with self._lock:
            s = self.settings
            return {"status": self.status.value,
                    "settings": {"user_name": s.user_name, "display_name": s.display_name, "speaker_tag": s.speaker_tag,
                                 "trigger_name": s.trigger_name, "reply_mode": s.reply_mode, "speak": self.speak is not None},
                    "model": self.client.config.model, "endpoint": self.client.config.host,
                    "transcript": list(self.transcript), "context_messages": len(self.history),
                    "stats": self.last_stats, "meter": self.meter.reading(self.history).as_dict(),
                    "connection": dict(self.connection)}

    # -- background worker: utterances are handled one at a time, in order ------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, daemon=True, name="agent")
            self._worker.start()

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)

    def submit(self, text: str, source: str = "typed") -> None:
        self._queue.put((text, source))

    def _run(self) -> None:
        while not self._stopping.is_set():
            item = self._queue.get()
            if item is None:
                break
            try:
                self.handle_utterance(*item)
            except Exception as e:                       # keep serving; show the error
                self.events.publish("notice", level="error", text=f"turn failed: {e}")
                self._set_status(Status.LISTENING)

    # -- the rule ----------------------------------------------------------------------

    def handle_utterance(self, text: str, source: str = "typed") -> Decision:
        text = text.strip()
        if not text:
            return Decision("ignored", "empty utterance")
        with self._lock:
            self._set_status(Status.DECIDING)
            try:
                self.history.append(Turn(self.settings.user_name.upper(), text))
                named = mentions_name(text, self.settings.trigger_name)
                answer = self.settings.reply_mode == "always" or named
                self._add_transcript("human", self.settings.user_name, text, source=source, named=named, answered=answer)
                if not answer:
                    decision = Decision("silent", f"'{self.settings.trigger_name}' not mentioned; added to context "
                                        f"({len(self.history)} messages), no reply")
                else:
                    reply = self._generate()
                    why = "always answers" if self.settings.reply_mode == "always" else f"'{self.settings.trigger_name}' mentioned"
                    decision = (Decision("reply", f"{why}; replied", reply) if reply.text
                                else Decision("silent", f"{why}, but no reply: {reply.error}", reply))
            finally:
                self._set_status(Status.LISTENING)
            self.events.publish("decision", **decision.as_dict())
            self._publish_meter()
            return decision

    def _generate(self) -> Reply:
        label = self.settings.display_name
        started = time.perf_counter()
        counts = {"chunks": 0}
        self.events.publish("reply_start", speaker=label)

        def on_thinking(delta: str) -> None:
            if self.status is not Status.THINKING:
                self._set_status(Status.THINKING)
            self.events.publish("thinking_delta", delta=delta)

        def on_text(delta: str) -> None:
            counts["chunks"] += 1
            if self.status is not Status.SPEAKING:
                self._set_status(Status.SPEAKING)
            self.events.publish("text_delta", delta=delta)
            self.events.publish("stats", live=True, output_tokens=counts["chunks"],
                                total_seconds=round(time.perf_counter() - started, 2))

        messages = build_messages(self.history, self.settings.speaker_tag)
        reply = self.client.stream(self.system_prompt, messages, on_thinking=on_thinking, on_text=on_text)
        if reply.error:
            self.events.publish("notice", level="warning", text=f"model: {reply.error}")
        u = reply.usage
        stats = {"live": False, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "output_tokens_estimated": u.output_tokens_estimated, "counts_source": u.counts_source,
                 "total_seconds": round(reply.elapsed_seconds, 2), "first_token_seconds": u.first_token_seconds,
                 "truncated": u.truncated, "model": reply.model}
        if reply.text:
            self.history.append(Turn(self.settings.speaker_tag, reply.text))
            self._add_transcript("agent", label, reply.text, thinking=reply.thinking)
            self.meter.record(u.input_tokens, u.output_tokens, self.history)
        self.last_stats = stats
        self.events.publish("reply_end", speaker=label, text=reply.text, had_thinking=bool(reply.thinking),
                            stats=stats, error=reply.error)
        self.events.publish("stats", **stats)
        self._publish_meter()
        if reply.text:
            self._speak_aloud(reply.text)
        return reply

    def _speak_aloud(self, text: str) -> None:
        if self.speak is None:
            return
        self._set_status(Status.SPEAKING)
        self.events.publish("speech", state="start")
        try:
            self.speak(text)
        except Exception as e:
            self.events.publish("notice", level="warning", text=f"speech failed: {e}")
        finally:
            self.events.publish("speech", state="end")

    # -- commands ----------------------------------------------------------------------

    def set_reply_mode(self, mode: str) -> str:
        if mode not in ("named", "always"):
            raise ValueError("reply_mode must be 'named' or 'always'")
        with self._lock:
            self.settings.reply_mode = mode
            self.events.publish("settings", reply_mode=mode)
            return mode

    def clear(self) -> None:
        """Forget the conversation (the constitution and context stay)."""
        with self._lock:
            self.history.clear()
            self.transcript.clear()
            self.last_stats = None
            self.meter = ContextMeter(self.client.config.num_ctx, self.system_prompt)
            self.events.publish("cleared")
            self._publish_meter()

    def halt_speech(self) -> bool:
        if self.stop_speech is None:
            return False
        self.stop_speech()
        return True

    def set_connection(self, online: bool, detail: str) -> None:
        changed = self.connection.get("online") != online or self.connection.get("detail") != detail
        self.connection = {"online": online, "detail": detail, "checked_at": time.time()}
        if changed:
            self.events.publish("connection", **self.connection)

    # -- helpers -------------------------------------------------------------------------

    def _set_status(self, status: Status) -> None:
        if status is not self.status:
            self.status = status
            self.events.publish("status", status=status.value)

    def _add_transcript(self, role: str, speaker: str, text: str, **extra) -> None:
        entry = {"n": len(self.transcript) + 1, "role": role, "speaker": speaker, "text": text, "t": time.time(), **extra}
        self.transcript.append(entry)
        self.events.publish("transcript", entry=entry)

    def _publish_meter(self) -> None:
        self.events.publish("meter", **self.meter.reading(self.history).as_dict())

"""Terminal view of the agent: your lines, the thinking in grey, the reply as it streams, the stats.

    python -m stranger chat                    # microphone in
    python -m stranger chat --input text       # typed in
"""

from __future__ import annotations

import sys
import threading
import time

from .agent import Agent, EventBus
from .keys import Key, KeyListener

DIM, RESET = "\033[2;90m", "\033[0m"


def format_stats(stats: dict) -> str:
    """E.g. "412 output tokens, 1,380 input tokens, 9.6 seconds total, 1.7 seconds to first token"."""
    parts = []
    if stats.get("output_tokens") is not None:
        parts.append(f"{stats['output_tokens']:,} output tokens")
    elif stats.get("output_tokens_estimated"):
        parts.append(f"about {stats['output_tokens_estimated']:,} output tokens (estimated from the stream; "
                     "the server returned no count because the reply was stopped at the sentence limit)")
    else:
        parts.append("output token count not returned")
    parts.append(f"{stats['input_tokens']:,} input tokens" if stats.get("input_tokens") is not None
                 else "input token count not returned")
    parts.append(f"{stats['total_seconds']:.1f} seconds total" if stats.get("total_seconds") is not None
                 else "total time not recorded")
    parts.append(f"{stats['first_token_seconds']:.1f} seconds to first token" if stats.get("first_token_seconds") is not None
                 else "time to first token not recorded")
    return ", ".join(parts)


class TerminalView:
    """Subscribes to the agent's events and prints them: a second window on the same stream the page shows."""

    SPINNER = "|/-\\"

    def __init__(self, agent: Agent, out=None, colour: bool | None = None, show_thinking: bool = True):
        self.agent, self.out = agent, out or sys.stdout
        self.colour = colour if colour is not None else self._isatty()
        self.show_thinking = show_thinking
        self.dim, self.reset = (DIM, RESET) if self.colour else ("", "")
        self._streaming: str | None = None
        self._spinner_stop = threading.Event()
        self._spinner: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._queue = agent.events.subscribe()

    def _isatty(self) -> bool:
        try:
            return self.out.isatty()
        except Exception:
            return False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="terminal-view")
        self._thread.start()

    def drain(self, timeout: float = 0.5) -> None:
        """Process queued events synchronously (tests and text mode)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._handle(self._queue.get(timeout=0.05))
            except Exception:
                if self._queue.empty():
                    return

    def _run(self) -> None:
        while True:
            self._handle(self._queue.get())

    def write(self, s: str) -> None:
        print(s, end="", file=self.out, flush=True)

    def line(self, s: str = "") -> None:
        if self._streaming:
            self.write("\n")
            self._streaming = None
        print(s, file=self.out, flush=True)

    # -- spinner while waiting for the first token ------------------------------------------

    def _spin(self, label: str) -> None:
        i = 0
        while not self._spinner_stop.is_set():
            self.write(f"\r{self.SPINNER[i % 4]} {label} ")
            i += 1
            self._spinner_stop.wait(0.12)
        self.write("\r" + " " * (len(label) + 4) + "\r")

    def _start_spinner(self, label: str) -> None:
        if not self.colour:
            self.line("...")
            return
        self._spinner_stop.clear()
        self._spinner = threading.Thread(target=self._spin, args=(label,), daemon=True)
        self._spinner.start()

    def _stop_spinner(self) -> None:
        if self._spinner:
            self._spinner_stop.set()
            self._spinner.join(timeout=1)
            self._spinner = None

    # -- event handling -----------------------------------------------------------------------

    def _handle(self, e: dict) -> None:
        kind, s = e["type"], self.agent.settings
        if kind == "transcript" and e["entry"]["role"] == "human":
            en = e["entry"]
            self.line(f"{en['speaker']}: {en['text']}")
        elif kind == "reply_start":
            self._start_spinner(f"{s.display_name} is thinking")
        elif kind == "thinking_delta":
            self._stop_spinner()
            if not self.show_thinking:
                return
            if self._streaming != "thinking":
                self.line(f"{self.dim}[THINKING]")
                self.write("    ")
                self._streaming = "thinking"
            self.write(self.dim + e["delta"].replace("\n", "\n    ") + self.reset)
        elif kind == "text_delta":
            self._stop_spinner()
            if self._streaming != "text":
                if self._streaming == "thinking":
                    self.line(f"\n{self.dim}[end of thinking]{self.reset}\n")
                elif self.show_thinking:
                    self.line(f"{self.dim}[THINKING: the model returned no separate thinking block]{self.reset}\n")
                self.write(f"{s.display_name}: ")
                self._streaming = "text"
            self.write(e["delta"])
        elif kind == "reply_end":
            self._stop_spinner()
            if self._streaming == "thinking":
                self.line(f"\n{self.dim}[end of thinking]{self.reset}\n")
            if not e["text"]:
                if not e["had_thinking"] and self.show_thinking:
                    self.line(f"{self.dim}[THINKING: the model returned no separate thinking block]{self.reset}\n")
                self.line(f"{s.display_name}: [no reply]")
            self.line(f"{self.dim}  [{format_stats(e['stats'])}]{self.reset}\n")
        elif kind == "decision" and e["action"] == "silent":
            self.line(f"{self.dim}  [{e['reason']}]{self.reset}\n")
        elif kind == "meter":
            pass
        elif kind == "notice":
            self.line(f"[{e['level']}] {e['text']}")
        elif kind == "connection":
            self.line(f"[model endpoint {'online: ' + e['detail'] if e['online'] else 'OFFLINE'}]")
        elif kind == "speech" and e["state"] == "start":
            self.line(f"{self.dim}  [speaking; press x to stop]{self.reset}")


def run_text(agent: Agent, view: TerminalView) -> None:
    view.line(f"Type as {agent.settings.user_name}; /quit to leave. Replies: {agent.settings.reply_mode}.\n")
    view.start()
    while True:
        try:
            typed = input(f"{agent.settings.user_name}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if typed.lower() in ("/quit", "/q", "/exit"):
            break
        if typed.lower() == "/always":
            agent.set_reply_mode("always")
            continue
        if typed.lower() == "/named":
            agent.set_reply_mode("named")
            continue
        if typed:
            with KeyListener() as keys:
                agent.handle_utterance(typed, source="typed")
                for key in keys.drain():
                    if key is Key.STOP_SPEECH:
                        agent.halt_speech()
            time.sleep(0.1)
    view.line("\n[bye]")


def run_mic(agent: Agent, view: TerminalView, speech_config, max_chunks: int) -> None:
    from .speech import Microphone, Transcriber, transcribe_stream, utterances

    mic = Microphone(speech_config.device)
    transcriber = Transcriber(speech_config)
    view.line(f"Loading Whisper '{speech_config.model_size}'...")
    transcriber.load()
    view.line(f"Listening on: {mic.device_name}. Press q to quit" +
              (", x to stop speech." if agent.speak else "."))
    view.start()
    agent.start()
    stop = threading.Event()

    def listen() -> None:
        try:
            for text in utterances(transcribe_stream(speech_config, mic, transcriber), max_chunks, lambda: agent.busy):
                if stop.is_set():
                    return
                agent.submit(text, source="mic")
        except Exception as e:
            agent.events.publish("notice", level="error", text=f"microphone: {e}")
            stop.set()

    threading.Thread(target=listen, daemon=True, name="listener").start()
    with KeyListener() as keys:
        while not stop.is_set():
            for key in keys.drain():
                if key is Key.QUIT:
                    stop.set()
                elif key is Key.STOP_SPEECH:
                    agent.halt_speech()
            time.sleep(0.05)
    agent.stop()
    view.line("\n[bye]")

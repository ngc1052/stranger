"""Single-key input from the terminal without blocking: q to quit, x to stop speech."""

from __future__ import annotations

import queue
import sys
import threading
from enum import Enum


class Key(Enum):
    QUIT = "quit"
    STOP_SPEECH = "stop_speech"


KEYMAP = {"q": Key.QUIT, "\x03": Key.QUIT, "x": Key.STOP_SPEECH}


class KeyListener:
    """Background keyboard reader. Use as a context manager so the terminal is restored."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._queue: queue.Queue[Key] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._saved = None

    def __enter__(self) -> "KeyListener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread:
            return
        self._enter_cbreak()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="key-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._restore()

    def drain(self) -> list[Key]:
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys

    def push(self, key: Key) -> None:
        self._queue.put(key)

    def _is_tty(self) -> bool:
        try:
            return self.stream.isatty()
        except Exception:
            return False

    def _enter_cbreak(self) -> None:
        if not self._is_tty():
            return
        try:
            import termios
            import tty
            fd = self.stream.fileno()
            self._saved = (fd, termios.tcgetattr(fd))
            tty.setcbreak(fd)
        except Exception:
            self._saved = None

    def _restore(self) -> None:
        if self._saved:
            import termios
            fd, attrs = self._saved
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            self._saved = None

    def _read_loop(self) -> None:
        import select
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self.stream], [], [], 0.1)
            except Exception:
                return
            if not ready:
                continue
            if self._is_tty():
                ch = self.stream.read(1)
            else:
                line = self.stream.readline()
                if not line:
                    return
                ch = line.strip()[:1]
            key = KEYMAP.get(ch)
            if key:
                self._queue.put(key)

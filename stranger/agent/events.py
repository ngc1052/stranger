"""A tiny publish/subscribe bus: the agent publishes, each SSE connection subscribes."""

from __future__ import annotations

import itertools
import json
import queue
import threading
import time
from typing import Any


class EventBus:
    def __init__(self, max_queue: int = 2000):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self.max_queue = max_queue
        self.log: list[dict] = []          # every event so far (tests and debugging)

    def publish(self, kind: str, **data: Any) -> dict:
        event = {"seq": next(self._seq), "type": kind, "t": time.time(), **data}
        with self._lock:
            self.log.append(event)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass                        # a stalled browser tab loses events, not the server
        return event

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def of_type(self, kind: str) -> list[dict]:
        return [e for e in self.log if e["type"] == kind]

    @staticmethod
    def format_sse(event: dict) -> str:
        return f"id: {event['seq']}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"

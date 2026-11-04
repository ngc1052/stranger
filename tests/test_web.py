"""The HTTP endpoints and the SSE stream, with a fake client behind the agent."""

from __future__ import annotations

import json
import time
import unittest

from stranger.agent import Agent, AgentSettings
from tests.fakes import FakeClient


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient
        from stranger.web.server import create_app
        self.agent = Agent(FakeClient(), "C" * 100, AgentSettings(user_name="Robin"))
        self.app = create_app(self.agent, endpoint_check=lambda: (True, "fake endpoint"), check_seconds=0.05)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def _wait_for(self, kind: str, count: int = 1, timeout: float = 3.0) -> list[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            found = self.agent.events.of_type(kind)
            if len(found) >= count:
                return found
            time.sleep(0.02)
        self.fail(f"no {kind} event within {timeout}s")

    def test_page_state_and_typed_utterance(self) -> None:
        self.assertIn("<title>The Stranger</title>", self.client.get("/").text)
        state = self.client.get("/api/state").json()
        self.assertEqual(state["settings"]["reply_mode"], "named")
        self.assertIn("meter", state)
        self.assertTrue(self.client.post("/api/utterance", json={"text": "Stranger, hello."}).json()["queued"])
        self.assertEqual(self._wait_for("decision")[-1]["action"], "reply")
        self._wait_for("connection")
        state = self.client.get("/api/state").json()
        self.assertTrue(state["connection"]["online"])
        self.assertEqual(state["stats"]["output_tokens"], 40)
        self.assertEqual(state["transcript"][-1]["role"], "agent")

    def test_settings_and_commands(self) -> None:
        self.assertEqual(self.client.post("/api/settings", json={"reply_mode": "always"}).json()["reply_mode"], "always")
        self.assertEqual(self.client.post("/api/settings", json={"reply_mode": "nope"}).status_code, 400)
        self.assertEqual(self.client.post("/api/utterance", json={"text": "  "}).status_code, 400)
        self.assertFalse(self.client.post("/api/command", json={"command": "stop_speech"}).json()["ok"])
        self.assertTrue(self.client.post("/api/command", json={"command": "clear"}).json()["ok"])
        self.assertEqual(self.client.post("/api/command", json={"command": "dance"}).status_code, 400)

    def test_sse_sends_snapshot_then_events(self) -> None:
        with self.client.stream("GET", "/events") as resp:
            self.assertTrue(resp.headers["content-type"].startswith("text/event-stream"))
            lines = resp.iter_lines()
            first = [next(lines) for _ in range(3)]
            self.assertEqual(first[1], "event: snapshot")
            self.assertEqual(json.loads(first[2][len("data: "):])["status"], "listening")
            self.agent.set_reply_mode("always")
            seen = ""
            for line in lines:
                seen += line + "\n"
                if '"type": "settings"' in line:
                    break
            self.assertIn("event: settings", seen)

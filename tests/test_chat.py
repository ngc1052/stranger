"""The terminal view: chunks in order, thinking apart from the reply, the stats line."""

from __future__ import annotations

import io
import unittest

from stranger.agent import Agent, AgentSettings
from stranger.chat import TerminalView, format_stats
from tests.fakes import FakeClient


class TerminalViewTests(unittest.TestCase):
    def _run(self, client: FakeClient, text: str, colour=False, show_thinking=True):
        out = io.StringIO()
        agent = Agent(client, "C", AgentSettings(user_name="Robin"))
        view = TerminalView(agent, out, colour=colour, show_thinking=show_thinking)
        agent.handle_utterance(text)
        view.drain()
        return agent, out.getvalue()

    def test_order_and_separation(self) -> None:
        client = FakeClient(text="A joke, like the diagonal, is found. Not made.", thinking="Robin asks about jokes.")
        agent, text = self._run(client, "Stranger, is a joke found or made?")
        self.assertIn("Robin: Stranger, is a joke found or made?", text)
        self.assertLess(text.index("[THINKING]"), text.index("Robin asks about jokes."))
        self.assertLess(text.index("[end of thinking]"), text.index("The Stranger: A joke, like the diagonal, is found. Not made."))
        self.assertIn("[40 output tokens, 900 input tokens, 1.2 seconds total, 0.4 seconds to first token]", text)

    def test_no_thinking_block_is_noted(self) -> None:
        _, text = self._run(FakeClient(text="Yes.", thinking=""), "Stranger?")
        self.assertIn("the model returned no separate thinking block", text)
        self.assertIn("The Stranger: Yes.", text)

    def test_silence_is_explained(self) -> None:
        _, text = self._run(FakeClient(), "nobody named")
        self.assertIn("not mentioned", text)
        self.assertNotIn("The Stranger:", text)

    def test_dim_codes_only_on_a_tty(self) -> None:
        _, plain = self._run(FakeClient(), "Stranger?")
        self.assertNotIn("\033[", plain)
        _, coloured = self._run(FakeClient(), "Stranger?", colour=True)
        self.assertIn("\033[2;90m", coloured)

    def test_stats_without_server_counts(self) -> None:
        line = format_stats({"output_tokens": None, "output_tokens_estimated": 57, "input_tokens": None,
                             "total_seconds": 3.0, "first_token_seconds": None})
        self.assertIn("about 57 output tokens (estimated", line)
        self.assertIn("input token count not returned", line)
        self.assertIn("time to first token not recorded", line)

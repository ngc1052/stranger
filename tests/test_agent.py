"""The reply rule, the status sequence, the event stream, the context meter."""

from __future__ import annotations

import unittest

from stranger.agent import Agent, AgentSettings, ContextMeter, Status, mentions_name
from stranger.llm import Turn
from tests.fakes import FakeClient


def agent_for(client=None, **settings) -> Agent:
    return Agent(client or FakeClient(), "CONSTITUTION " * 50, AgentSettings(user_name="Robin", **settings))


class ReplyRuleTests(unittest.TestCase):
    def test_silent_when_name_not_mentioned(self) -> None:
        a = agent_for()
        d = a.handle_utterance("What do you make of the garden, then?")
        self.assertEqual(d.action, "silent")
        self.assertEqual(a.client.calls, [])
        self.assertEqual(a.history, [Turn("ROBIN", "What do you make of the garden, then?")])
        self.assertEqual(a.transcript[-1]["answered"], False)
        self.assertEqual(a.events.of_type("text_delta"), [])

    def test_reply_when_named(self) -> None:
        a = agent_for()
        d = a.handle_utterance("Stranger, was the garden there before the gardener?")
        self.assertEqual(d.action, "reply")
        self.assertEqual(len(a.client.calls), 1)
        system_prompt, messages = a.client.calls[0]
        self.assertTrue(system_prompt.startswith("CONSTITUTION"))
        self.assertEqual(messages, [{"role": "user", "content": "ROBIN: Stranger, was the garden there before the gardener?"}])
        self.assertEqual(a.history[-1], Turn("THE STRANGER", a.client.text))
        self.assertEqual(a.transcript[-1]["role"], "agent")

    def test_always_mode_answers_everything(self) -> None:
        a = agent_for(reply_mode="always")
        self.assertEqual(a.handle_utterance("No name here.").action, "reply")
        a.set_reply_mode("named")
        self.assertEqual(a.handle_utterance("Still no name.").action, "silent")
        with self.assertRaises(ValueError):
            a.set_reply_mode("sometimes")

    def test_name_detection(self) -> None:
        self.assertTrue(mentions_name("Well, stranger, what now?"))
        self.assertTrue(mentions_name("STRANGER!"))
        self.assertFalse(mentions_name("strangers are everywhere"))
        self.assertFalse(mentions_name("a strange idea"))
        self.assertTrue(mentions_name("Hey Ada", "Ada"))

    def test_context_keeps_growing_while_silent(self) -> None:
        a = agent_for()
        readings = [a.meter.reading(a.history).used]
        for text in ["First we talked about the courtyard.", "Then about the gardener and his chain.",
                     "And finally about whether shapes wait for us."]:
            self.assertEqual(a.handle_utterance(text).action, "silent")
            readings.append(a.meter.reading(a.history).used)
        self.assertEqual(len(a.history), 3)
        self.assertTrue(all(b > a_ for a_, b in zip(readings, readings[1:])), readings)
        a.handle_utterance("Stranger, do you agree?")
        _, messages = a.client.calls[0]
        self.assertEqual(len(messages), 1)                      # merged into one user message...
        self.assertEqual(messages[0]["content"].count("\n"), 3)  # ...holding all four lines

    def test_model_failure_is_reported_not_raised(self) -> None:
        a = agent_for(FakeClient(error="model call failed: boom"))
        d = a.handle_utterance("Stranger?")
        self.assertEqual(d.action, "silent")
        self.assertIn("boom", d.reason)
        self.assertEqual(a.status, Status.LISTENING)
        self.assertEqual(a.events.of_type("notice")[-1]["level"], "warning")

    def test_clear_forgets_the_conversation(self) -> None:
        a = agent_for()
        a.handle_utterance("Stranger, hello.")
        a.clear()
        self.assertEqual(a.history, [])
        self.assertEqual(a.transcript, [])
        self.assertEqual(a.state()["meter"]["source"], "estimated")


class EventTests(unittest.TestCase):
    def test_status_sequence_and_separate_streams(self) -> None:
        a = agent_for()
        a.handle_utterance("Stranger, a word?")
        statuses = [e["status"] for e in a.events.of_type("status")]
        self.assertEqual(statuses[-4:], ["deciding", "thinking", "speaking", "listening"])
        kinds = [e["type"] for e in a.events.log if e["type"] in ("thinking_delta", "text_delta")]
        self.assertEqual(kinds[0], "thinking_delta")
        self.assertEqual(kinds, sorted(kinds, key=lambda k: 0 if k == "thinking_delta" else 1))
        self.assertEqual("".join(e["delta"] for e in a.events.of_type("text_delta")).strip(), a.client.text)
        end = a.events.of_type("reply_end")[-1]
        self.assertTrue(end["had_thinking"])
        self.assertEqual(end["stats"]["input_tokens"], 900)
        self.assertEqual(end["stats"]["output_tokens"], 40)
        self.assertEqual(end["stats"]["first_token_seconds"], 0.4)
        self.assertEqual(end["stats"]["total_seconds"], 1.2)
        self.assertTrue(any(e.get("live") for e in a.events.of_type("stats")))

    def test_speech_hook_runs_after_the_reply_and_can_be_stopped(self) -> None:
        spoken, stopped = [], []
        a = Agent(FakeClient(), "C", AgentSettings(), speak=spoken.append, stop_speech=lambda: stopped.append(1))
        a.handle_utterance("Stranger, speak.")
        self.assertEqual(spoken, [a.client.text])
        self.assertEqual([e["state"] for e in a.events.of_type("speech")], ["start", "end"])
        self.assertTrue(a.halt_speech())
        self.assertEqual(stopped, [1])
        self.assertFalse(Agent(FakeClient(), "C").halt_speech())

    def test_worker_processes_submissions_in_order(self) -> None:
        a = agent_for()
        a.start()
        a.submit("one"); a.submit("Stranger, two"); a.submit("three")
        import time
        for _ in range(100):
            if len(a.events.of_type("decision")) == 3:
                break
            time.sleep(0.02)
        a.stop()
        self.assertEqual([e["action"] for e in a.events.of_type("decision")], ["silent", "reply", "silent"])
        self.assertEqual([t.text for t in a.history if t.speaker == "ROBIN"], ["one", "Stranger, two", "three"])


class MeterTests(unittest.TestCase):
    def test_value_before_and_after_a_reply(self) -> None:
        a = agent_for()
        before = a.state()["meter"]
        self.assertEqual(before["window"], 4096)
        self.assertGreater(before["used"], 0)
        self.assertEqual(before["source"], "estimated")
        a.handle_utterance("Stranger, a word?")
        after = a.events.of_type("meter")[-1]
        self.assertEqual((after["used"], after["source"]), (940, "measured"))
        self.assertAlmostEqual(after["fraction"], 940 / 4096, places=3)
        a.handle_utterance("More talk, no name.")
        later = a.events.of_type("meter")[-1]
        self.assertGreater(later["used"], 940)
        self.assertEqual(later["source"], "measured")

    def test_estimate_only_without_server_counts(self) -> None:
        meter = ContextMeter(2048, "x" * 400)
        self.assertEqual(meter.reading([]).used, 100)
        meter.record(None, None, [])
        self.assertEqual(meter.reading([Turn("ROBIN", "y" * 40)]).source, "estimated")
        self.assertEqual(ContextMeter(0).reading([]).fraction, 0.0)

"""The LLM client against fake Ollama and OpenAI-compatible servers; the stream router; messages."""

from __future__ import annotations

import unittest

from stranger.llm import (LLMClient, LLMConfig, ThinkRouter, Turn, build_messages, check_endpoint,
                          count_sentences, trim_to_sentences)
from tests.fakes import FakeModelServer


class RouterTests(unittest.TestCase):
    def test_inline_think_tags_split_across_chunks(self) -> None:
        th, tx = [], []
        r = ThinkRouter(th.append, tx.append)
        for chunk in ["<thi", "nk>plan it</th", "ink>Hello ", "there."]:
            r.content(chunk)
        r.flush()
        self.assertEqual("".join(th), "plan it")
        self.assertEqual("".join(tx), "Hello there.")

    def test_sentence_limit_closes_the_stream(self) -> None:
        tx = []
        r = ThinkRouter(lambda s: None, tx.append, max_sentences=2)
        r.content("One. Two. Three. Four.")
        self.assertTrue(r.finished)
        self.assertEqual("".join(tx), "One. Two.")

    def test_sentence_helpers(self) -> None:
        self.assertEqual(count_sentences("A. B! C?"), 3)
        self.assertEqual(trim_to_sentences("A. B. C unfinished", 2), "A. B.")
        self.assertEqual(trim_to_sentences("Only a fragment", 2), "Only a fragment")


class MessageTests(unittest.TestCase):
    def test_history_becomes_alternating_messages(self) -> None:
        history = [Turn("ROBIN", "Hello."), Turn("ROBIN", "Anyone there?"), Turn("THE STRANGER", "I am here."),
                   Turn("ROBIN", "Good.")]
        msgs = build_messages(history, "THE STRANGER")
        self.assertEqual(msgs, [{"role": "user", "content": "ROBIN: Hello.\nROBIN: Anyone there?"},
                                {"role": "assistant", "content": "I am here."},
                                {"role": "user", "content": "ROBIN: Good."}])

    def test_starts_and_ends_with_a_user_message(self) -> None:
        msgs = build_messages([Turn("THE STRANGER", "I spoke first.")], "THE STRANGER")
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[-1]["role"], "user")


class ClientTests(unittest.TestCase):
    def test_ollama_protocol(self) -> None:
        with FakeModelServer() as srv:
            self.assertEqual(check_endpoint(srv.host, "ollama"), (True, "Ollama 0.0-fake"))
            client = LLMClient(LLMConfig(model="fake-model", host=srv.host, api="ollama", think=True))
            th, tx = [], []
            reply = client.stream("Be brief.", [{"role": "user", "content": "V: hello"}], th.append, tx.append)
            self.assertEqual("".join(th), "weighing it carefully")
            self.assertEqual(reply.text, "The shape waits. It does not mind.")   # markdown stripped
            self.assertEqual(reply.usage.input_tokens, 123)
            self.assertEqual(reply.usage.output_tokens, 9)
            self.assertEqual(reply.usage.counts_source, "server")
            self.assertEqual(reply.usage.prompt_eval_seconds, 0.5)
            self.assertIsNotNone(reply.usage.first_token_seconds)
            self.assertTrue(reply.ok)
            path, req = srv.requests[0]
            self.assertEqual(path, "/api/chat")
            self.assertEqual(req["messages"][0], {"role": "system", "content": "Be brief."})
            self.assertTrue(req["think"])
            self.assertEqual(req["options"]["num_ctx"], 8192)
            self.assertAlmostEqual(client.warm_up("Be brief."), 0, delta=5)
            self.assertEqual(srv.requests[-1][1]["options"]["num_predict"], 1)

    def test_sentence_limit_means_estimated_counts(self) -> None:
        with FakeModelServer() as srv:
            client = LLMClient(LLMConfig(model="fake-model", host=srv.host, max_sentences=1))
            reply = client.stream("x", [{"role": "user", "content": "V: hi"}])
            self.assertEqual(reply.text, "The shape waits.")
            self.assertTrue(reply.usage.truncated)
            self.assertEqual(reply.usage.counts_source, "estimated")
            self.assertIsNone(reply.usage.output_tokens)
            self.assertGreater(reply.usage.output_tokens_estimated, 0)

    def test_openai_protocol(self) -> None:
        with FakeModelServer() as srv:
            self.assertEqual(check_endpoint(srv.host, "openai"), (True, "OpenAI-compatible server, models: fake-model"))
            client = LLMClient(LLMConfig(model="fake-model", host=srv.host, api="openai", think=True))
            th, tx = [], []
            reply = client.stream("Be brief.", [{"role": "user", "content": "V: hello"}], th.append, tx.append)
            self.assertEqual("".join(th), "weighing it carefully")
            self.assertEqual(reply.text, "The shape waits.")
            self.assertEqual(reply.usage.input_tokens, 123)
            self.assertEqual(reply.usage.output_tokens, 7)
            path, req = srv.requests[0]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(req["stream_options"], {"include_usage": True})

    def test_unreachable_endpoint_is_an_error_not_an_exception(self) -> None:
        client = LLMClient(LLMConfig(host="http://127.0.0.1:9", timeout_seconds=2))
        self.assertFalse(client.check()[0])
        reply = client.stream("x", [{"role": "user", "content": "V: hi"}])
        self.assertFalse(reply.ok)
        self.assertIn("model call failed", reply.error)

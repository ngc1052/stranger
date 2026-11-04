"""Test doubles: a scripted LLM client, and fake Ollama / OpenAI-compatible HTTP servers."""

from __future__ import annotations

import http.server
import json
import threading

from stranger.llm import LLMConfig, Reply, Usage


class FakeClient:
    """Streams a fixed thinking block then a fixed reply, word by word, and records every call."""

    def __init__(self, text="I would say the question is older than the courtyard.", thinking="They ask about gardens.",
                 usage: Usage | None = None, num_ctx: int = 4096, error: str | None = None):
        self.text, self.thinking, self.error = text, thinking, error
        self.usage = usage or Usage(input_tokens=900, output_tokens=40, first_token_seconds=0.4, counts_source="server")
        self.config = LLMConfig(num_ctx=num_ctx)
        self.calls: list[tuple[str, list[dict]]] = []

    def check(self):
        return True, "fake endpoint"

    def warm_up(self, system_prompt):
        return 0.0

    def stream(self, system_prompt, messages, on_thinking=None, on_text=None) -> Reply:
        self.calls.append((system_prompt, list(messages)))
        if self.error:
            return Reply(text="", error=self.error, elapsed_seconds=0.1)
        for piece in self.thinking.split(" ") if self.thinking else ():
            on_thinking(piece + " ")
        for piece in self.text.split(" "):
            on_text(piece + " ")
        return Reply(text=self.text, thinking=self.thinking or None, usage=self.usage, elapsed_seconds=1.2, model="fake")


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            return self._json({"version": "0.0-fake"})
        if self.path == "/api/tags":
            return self._json({"models": [{"name": "fake-model"}]})
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "fake-model"}]})
        self.send_error(404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, req))
        if self.path == "/api/chat":
            return self._ollama(req)
        if self.path == "/v1/chat/completions":
            return self._openai(req)
        self.send_error(404)

    def _ollama(self, req):
        if not req.get("stream"):
            return self._json({"model": req["model"], "message": {"role": "assistant", "content": ""}, "done": True})
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        events = [{"message": {"role": "assistant", "thinking": "weighing it "}},
                  {"message": {"role": "assistant", "thinking": "carefully"}},
                  {"message": {"role": "assistant", "content": "The shape "}},
                  {"message": {"role": "assistant", "content": "waits. "}},
                  {"message": {"role": "assistant", "content": "It does **not** mind."}},
                  {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 123,
                   "eval_count": 9, "prompt_eval_duration": 5e8, "eval_duration": 1.5e9}]
        try:
            for e in events:
                self.wfile.write((json.dumps({"model": req["model"], **e}) + "\n").encode())
                self.wfile.flush()
        except BrokenPipeError:          # the client closed the stream early (sentence limit): expected
            pass

    def _openai(self, req):
        if not req.get("stream"):
            return self._json({"model": req["model"], "choices": [{"message": {"role": "assistant", "content": ""}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 0}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "weighing it "}}]},
                  {"choices": [{"delta": {"reasoning_content": "carefully"}}]},
                  {"choices": [{"delta": {"content": "The shape "}}]},
                  {"choices": [{"delta": {"content": "waits."}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 7}}]
        for c in chunks:
            self.wfile.write(f"data: {json.dumps({'model': req['model'], **c})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class FakeModelServer:
    """A tiny HTTP server speaking both protocols; `requests` records what it received."""

    def __init__(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.requests = []
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def requests(self):
        return self.server.requests

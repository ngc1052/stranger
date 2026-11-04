"""One client for a local or remote chat model, streaming, with the standard library only.

Two wire protocols, chosen by `LLMConfig.api`:

  ollama   Ollama's native API (/api/chat, NDJSON stream). Default: http://localhost:11434.
  openai   The OpenAI-compatible protocol served by vLLM, llama.cpp, LM Studio and
           others (/v1/chat/completions, SSE stream). Point `host` at the server,
           for example an SSH tunnel to a cluster node.

Both are translated into the same stream of events, so everything above this
module (thinking/text routing, usage, timings) is protocol-agnostic.

    client = LLMClient(LLMConfig(model="qwen3:1.7b"))
    reply = client.stream(system_prompt, messages, on_thinking=print, on_text=print)
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from .stream import ThinkRouter

DEFAULT_MODEL = "qwen3:1.7b"
INSTALL_HELP = """No model server found. To run one locally with Ollama:
  Linux:   curl -fsSL https://ollama.com/install.sh | sh     then: ollama serve   (or systemctl start ollama)
  macOS:   brew install ollama && brew services start ollama
  Windows: https://ollama.com/download
Then pull a model:  ollama pull qwen3:1.7b"""

_MARKDOWN_RE = re.compile(r"[*_`#]+")


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    host: str = "http://localhost:11434"
    api: str = "ollama"                 # "ollama" | "openai"
    max_tokens: int = 600               # cap on generated tokens (thinking included)
    temperature: float = 0.6
    repeat_penalty: float = 1.15        # small models loop without it
    stop: tuple[str, ...] = ()          # extra stop sequences
    max_sentences: int = 3              # keep the first N complete sentences; 0 = no limit
    num_ctx: int = 8192                 # context window to request (Ollama) and to meter against
    think: bool | None = True           # ask for a separate thinking stream; None = server default
    keep_alive: str = "30m"             # Ollama: keep the model loaded between turns
    timeout_seconds: float = 120.0


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_estimated: int | None = None   # streamed chunks, when the server sent no count
    counts_source: str = "missing"               # "server" | "estimated" | "missing"
    first_token_seconds: float | None = None
    prompt_eval_seconds: float | None = None
    generate_seconds: float | None = None
    truncated: bool = False                      # stopped at the sentence limit

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Reply:
    text: str
    thinking: str | None = None
    usage: Usage = field(default_factory=Usage)
    elapsed_seconds: float = 0.0
    model: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and self.error is None


# -- HTTP helpers --------------------------------------------------------------------

def _get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_endpoint(host: str = "http://localhost:11434", api: str = "ollama") -> tuple[bool, str]:
    """(reachable, description). Ollama answers /api/version; OpenAI-style servers /v1/models."""
    try:
        if api == "ollama":
            return True, f"Ollama {_get(f'{host}/api/version').get('version', '?')}"
        names = [m.get("id", "?") for m in _get(f"{host}/v1/models").get("data", [])]
        return True, "OpenAI-compatible server, models: " + (", ".join(names) or "none listed")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return False, f"{host} unreachable ({e}). " + (INSTALL_HELP if api == "ollama" else "")


def installed_models(host: str = "http://localhost:11434", api: str = "ollama") -> list[str]:
    if api == "ollama":
        return [m["name"] for m in _get(f"{host}/api/tags").get("models", [])]
    return [m.get("id", "?") for m in _get(f"{host}/v1/models").get("data", [])]


# -- the client ------------------------------------------------------------------------

class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    # -- requests, per protocol, all shaped like Ollama's stream events ------------------

    def _ollama_body(self, system_prompt: str, messages: list[dict], stream: bool) -> dict:
        c = self.config
        body = {"model": c.model, "messages": [{"role": "system", "content": system_prompt}, *messages],
                "stream": stream, "keep_alive": c.keep_alive,
                "options": {"num_predict": c.max_tokens, "temperature": c.temperature, "num_ctx": c.num_ctx,
                            "repeat_penalty": c.repeat_penalty, "stop": list(c.stop)}}
        if c.think is not None:
            body["think"] = c.think
        return body

    def _openai_body(self, system_prompt: str, messages: list[dict], stream: bool) -> dict:
        c = self.config
        body = {"model": c.model, "messages": [{"role": "system", "content": system_prompt}, *messages],
                "stream": stream, "max_tokens": c.max_tokens, "temperature": c.temperature,
                "repetition_penalty": c.repeat_penalty, "stop": list(c.stop)}
        if stream:
            body["stream_options"] = {"include_usage": True}
        if c.think is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(c.think)}
        return body

    def _open(self, path: str, body: dict):
        req = urllib.request.Request(f"{self.config.host}{path}", data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "*/*"})
        try:
            return urllib.request.urlopen(req, timeout=self.config.timeout_seconds)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"model server error {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e

    def _events(self, system_prompt: str, messages: list[dict]) -> Iterator[dict]:
        """Yield {"thinking", "content"} deltas and a final {"done": True, ...counts} event."""
        if self.config.api == "openai":
            yield from self._events_openai(system_prompt, messages)
            return
        with self._open("/api/chat", self._ollama_body(system_prompt, messages, stream=True)) as resp:
            for raw in resp:
                if not raw.strip():
                    continue
                event = json.loads(raw)
                if "error" in event:
                    raise RuntimeError(event["error"])
                msg = event.get("message", {})
                if msg.get("thinking") or msg.get("content"):
                    yield {"thinking": msg.get("thinking", ""), "content": msg.get("content", "")}
                if event.get("done"):
                    yield {"done": True, "model": event.get("model", self.config.model),
                           "input_tokens": event.get("prompt_eval_count"), "output_tokens": event.get("eval_count"),
                           "prompt_eval_seconds": event.get("prompt_eval_duration", 0) / 1e9,
                           "generate_seconds": event.get("eval_duration", 0) / 1e9}

    def _events_openai(self, system_prompt: str, messages: list[dict]) -> Iterator[dict]:
        with self._open("/v1/chat/completions", self._openai_body(system_prompt, messages, stream=True)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"]))
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    content = delta.get("content") or ""
                    if thinking or content:
                        yield {"thinking": thinking, "content": content}
                usage = chunk.get("usage")
                if usage:
                    yield {"done": True, "model": chunk.get("model", self.config.model),
                           "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}

    # -- public API ------------------------------------------------------------------------

    def check(self) -> tuple[bool, str]:
        return check_endpoint(self.config.host, self.config.api)

    def warm_up(self, system_prompt: str) -> float:
        """Load the model and pre-evaluate the system prompt; returns seconds taken.

        Servers keep the evaluated prefix in their cache, so the constitution
        and context are paid for once here instead of on the first real turn.
        """
        t0 = time.perf_counter()
        messages = [{"role": "user", "content": "[The conversation begins.]"}]
        if self.config.api == "openai":
            body = self._openai_body(system_prompt, messages, stream=False)
            body["max_tokens"] = 1
            _post(f"{self.config.host}/v1/chat/completions", body, self.config.timeout_seconds)
        else:
            body = self._ollama_body(system_prompt, messages, stream=False)
            body["options"]["num_predict"] = 1
            try:
                _post(f"{self.config.host}/api/chat", body, self.config.timeout_seconds)
            except urllib.error.HTTPError:            # a model without a thinking switch
                body.pop("think", None)
                _post(f"{self.config.host}/api/chat", body, self.config.timeout_seconds)
        return time.perf_counter() - t0

    def stream(self, system_prompt: str, messages: list[dict], on_thinking: Callable[[str], None] | None = None,
               on_text: Callable[[str], None] | None = None) -> Reply:
        """Stream one reply. Thinking and text are delivered to separate callbacks as they arrive.

        Text stops being forwarded after `max_sentences` complete sentences and the
        connection is closed (which cancels generation), so the returned text is
        exactly what the callbacks received. Markdown emphasis is stripped from the
        final text because small models leak it and it is meant to be spoken.
        """
        router = ThinkRouter(on_thinking or (lambda s: None), on_text or (lambda s: None), self.config.max_sentences)
        started = time.perf_counter()
        done: dict = {}
        chunks = 0
        error = None
        try:
            for event in self._events(system_prompt, messages):
                if event.get("done"):
                    done = event
                    continue
                chunks += 1
                if event["thinking"]:
                    router.thinking(event["thinking"])
                if event["content"]:
                    router.content(event["content"])
                if router.finished:
                    break
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
            error = f"model call failed: {e}"
        router.flush()
        elapsed = time.perf_counter() - started
        text = _MARKDOWN_RE.sub("", router.text_out).strip()
        thinking = router.thinking_out.strip() or None
        if not text and error is None:
            error = "thinking used the whole token budget" if thinking else "the model returned no text"
        usage = Usage(
            input_tokens=done.get("input_tokens"), output_tokens=done.get("output_tokens"),
            output_tokens_estimated=chunks if done.get("output_tokens") is None and chunks else None,
            counts_source="server" if done.get("output_tokens") is not None else ("estimated" if chunks else "missing"),
            first_token_seconds=round(router.first_token_at - started, 2) if router.first_token_at else None,
            prompt_eval_seconds=round(done["prompt_eval_seconds"], 2) if "prompt_eval_seconds" in done else None,
            generate_seconds=round(done["generate_seconds"], 2) if "generate_seconds" in done else None,
            truncated=router.finished,
        )
        return Reply(text=text, thinking=thinking, usage=usage, elapsed_seconds=elapsed,
                     model=done.get("model", self.config.model), error=error)

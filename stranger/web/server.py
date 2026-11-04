"""FastAPI app around the Agent: SSE event stream, state snapshot, commands, microphone feed.

    GET  /               the page
    GET  /events         Server-Sent Events: a `snapshot` first, then every agent event
    GET  /api/state      the same snapshot as JSON
    POST /api/utterance  {"text": "..."}: a typed utterance, queued like a transcribed one
    POST /api/settings   {"reply_mode": "named" | "always"}
    POST /api/command    {"command": "stop_speech" | "clear"}
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..agent import Agent, EventBus

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(agent: Agent, endpoint_check=None, check_seconds: float = 5.0, mic_feed=None) -> FastAPI:
    """`endpoint_check() -> (online, detail)` is polled every `check_seconds`;
    `mic_feed(agent, stop_event)` runs in a thread for the life of the app when given."""
    stop = threading.Event()

    def monitor() -> None:
        while not stop.is_set():
            try:
                online, detail = endpoint_check()
            except Exception as e:
                online, detail = False, str(e)
            agent.set_connection(online, detail)
            stop.wait(check_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        agent.start()
        if endpoint_check is not None:
            threading.Thread(target=monitor, daemon=True, name="endpoint-monitor").start()
        if mic_feed is not None:
            threading.Thread(target=mic_feed, args=(agent, stop), daemon=True, name="mic-feed").start()
        yield
        stop.set()
        agent.stop()

    app = FastAPI(title="The Stranger", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.agent = agent
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/state")
    def state():
        return agent.state()

    @app.post("/api/utterance")
    def utterance(body: dict):
        text = str(body.get("text", "")).strip()
        if not text:
            raise HTTPException(400, "text is required")
        agent.submit(text, source=str(body.get("source", "typed")))
        return {"queued": True}

    @app.post("/api/settings")
    def settings(body: dict):
        try:
            return {"reply_mode": agent.set_reply_mode(str(body.get("reply_mode", "")))}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/command")
    def command(body: dict):
        name = str(body.get("command", ""))
        if name == "stop_speech":
            return {"ok": agent.halt_speech()}
        if name == "clear":
            agent.clear()
            return {"ok": True}
        raise HTTPException(400, f"unknown command {name!r}")

    @app.get("/events")
    async def events():
        bus: EventBus = agent.events
        q = bus.subscribe()

        async def gen():
            try:
                yield EventBus.format_sse({"seq": 0, "type": "snapshot", "t": time.time(), **agent.state()})
                while True:
                    try:
                        event = await asyncio.to_thread(q.get, True, 2.0)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield EventBus.format_sse(event)
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def mic_feed_factory(speech_config, max_chunks: int = 6):
    """Return a mic_feed(agent, stop) that transcribes the microphone into the agent."""
    def feed(agent: Agent, stop: threading.Event) -> None:
        from ..speech import Microphone, Transcriber, transcribe_stream, utterances
        try:
            mic = Microphone(speech_config.device)
            transcriber = Transcriber(speech_config)
            agent.events.publish("notice", level="info", text=f"Loading Whisper '{speech_config.model_size}'...")
            transcriber.load()
            agent.events.publish("notice", level="info", text=f"Listening on {mic.device_name}.")
            stream = transcribe_stream(speech_config, mic, transcriber)
            for text in utterances(stream, max_chunks, lambda: agent.busy):   # deaf while answering
                if stop.is_set():
                    return
                agent.submit(text, source="mic")
        except Exception as e:
            agent.events.publish("notice", level="error",
                                 text=f"microphone unavailable: {e}. Type utterances in the box instead.")
    return feed

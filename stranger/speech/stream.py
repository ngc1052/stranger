"""Continuous capture: microphone chunks in, transcripts out, utterances assembled."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .config import SpeechConfig
from .microphone import Microphone
from .transcriber import Transcriber, Transcript


def transcribe_stream(config: SpeechConfig | None = None, microphone: Microphone | None = None,
                      transcriber: Transcriber | None = None) -> Iterator[Transcript]:
    """Yield one Transcript per audio chunk, indefinitely.

    Silent chunks yield an empty Transcript so the caller can keep time.
    Stop by breaking out of the loop.
    """
    config = config or SpeechConfig()
    microphone = microphone or Microphone(config.device)
    transcriber = transcriber or Transcriber(config)
    transcriber.load()
    for chunk in microphone.chunks(config.chunk_seconds):
        yield transcriber.transcribe(chunk)


def utterances(transcripts: Iterator[Transcript], max_chunks: int = 6,
               is_muted: Callable[[], bool] = lambda: False) -> Iterator[str]:
    """Group consecutive speech chunks into one utterance, ended by a silent chunk.

    An utterance is also closed after `max_chunks` consecutive speech chunks so a
    long monologue still reaches the agent. While `is_muted()` is true, chunks are
    discarded (used while the agent is answering so it does not hear itself).
    """
    buffer: list[str] = []
    for t in transcripts:
        if is_muted():
            buffer.clear()
            continue
        if t.is_empty:
            if buffer:
                yield " ".join(buffer)
                buffer.clear()
            continue
        buffer.append(t.text)
        if len(buffer) >= max_chunks:
            yield " ".join(buffer)
            buffer.clear()

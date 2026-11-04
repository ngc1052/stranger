"""Language-model access: one client, two wire protocols, streamed thinking and text."""

from .client import LLMClient, LLMConfig, Reply, Usage, check_endpoint, installed_models
from .messages import Turn, build_messages
from .stream import ThinkRouter, count_sentences, trim_to_sentences

__all__ = ["LLMClient", "LLMConfig", "Reply", "Usage", "check_endpoint", "installed_models", "Turn", "build_messages",
           "ThinkRouter", "count_sentences", "trim_to_sentences"]

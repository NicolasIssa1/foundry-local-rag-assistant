"""
Tests for default model aliases — src/llm/client.py.

FoundryRuntime itself requires a real Foundry Local SDK installation to
construct, so it isn't instantiated here; these tests just pin down the
default aliases so an accidental change is caught by the test suite.
"""
from src.llm.client import DEFAULT_CHAT_ALIAS, DEFAULT_EMBED_ALIAS


def test_default_chat_alias_is_qwen3_1_7b():
    assert DEFAULT_CHAT_ALIAS == "qwen3-1.7b"


def test_default_embed_alias_is_unchanged():
    assert DEFAULT_EMBED_ALIAS == "qwen3-embedding-0.6b"

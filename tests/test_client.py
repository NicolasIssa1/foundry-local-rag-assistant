"""
Tests for src/llm/client.py — default model aliases and lazy chat-model
loading (FoundryRuntime(load_chat=False), used by index-only commands).

The real FoundryLocalManager/Configuration require a Foundry Local
installation, so they're patched out with mocks; these tests verify our
own loading logic (which aliases get requested, in what order, and that
the chat model is skipped when load_chat=False) without touching the SDK.
"""
from unittest.mock import MagicMock, patch

from src.llm.client import DEFAULT_CHAT_ALIAS, DEFAULT_EMBED_ALIAS, FoundryRuntime


def _mock_manager() -> MagicMock:
    """A MagicMock standing in for FoundryLocalManager.instance."""
    manager = MagicMock()
    manager.catalog.get_model.return_value = MagicMock()
    return manager


def _make_runtime(manager: MagicMock, **kwargs) -> FoundryRuntime:
    with patch("src.llm.client.Configuration"), \
         patch("src.llm.client.FoundryLocalManager") as mock_manager_cls:
        mock_manager_cls.instance = manager
        return FoundryRuntime(verbose=False, **kwargs)


def _requested_aliases(manager: MagicMock) -> list[str]:
    return [call.args[0] for call in manager.catalog.get_model.call_args_list]


# ── Default aliases ────────────────────────────────────────────────────────────

def test_default_chat_alias_is_qwen3_1_7b():
    assert DEFAULT_CHAT_ALIAS == "qwen3-1.7b"


def test_default_embed_alias_is_unchanged():
    assert DEFAULT_EMBED_ALIAS == "qwen3-embedding-0.6b"


# ── Lazy chat-model loading ──────────────────────────────────────────────────────

def test_load_chat_false_only_requests_the_embedding_model():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=False)
    aliases = _requested_aliases(manager)
    assert aliases == [DEFAULT_EMBED_ALIAS]
    runtime.close()


def test_load_chat_true_requests_both_models():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    aliases = _requested_aliases(manager)
    assert aliases == [DEFAULT_EMBED_ALIAS, DEFAULT_CHAT_ALIAS]
    runtime.close()


def test_load_chat_defaults_to_true():
    manager = _mock_manager()
    runtime = _make_runtime(manager)
    aliases = _requested_aliases(manager)
    assert DEFAULT_CHAT_ALIAS in aliases
    runtime.close()


def test_load_chat_false_close_does_not_unload_a_chat_model():
    """close() must not blow up or call unload() on a chat model that was
    never loaded — the embedding model must still be unloaded normally."""
    manager = _mock_manager()
    embed_model = MagicMock()
    manager.catalog.get_model.return_value = embed_model
    runtime = _make_runtime(manager, load_chat=False)
    runtime.close()
    embed_model.unload.assert_called_once()


def test_load_chat_false_chat_raises_clear_error():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=False)
    try:
        runtime.chat([{"role": "user", "content": "hi"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    finally:
        runtime.close()


def test_load_chat_false_stream_chat_raises_clear_error():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=False)
    try:
        list(runtime.stream_chat([{"role": "user", "content": "hi"}]))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    finally:
        runtime.close()

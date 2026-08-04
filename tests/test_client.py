"""
Tests for src/llm/client.py — default model aliases and lazy chat-model
loading (FoundryRuntime(load_chat=False), used by index-only commands).

The real FoundryLocalManager/Configuration require a Foundry Local
installation, so they're patched out with mocks; these tests verify our
own loading logic (which aliases get requested, in what order, and that
the chat model is skipped when load_chat=False) without touching the SDK.
"""
from unittest.mock import MagicMock, patch

from foundry_local_sdk.openai.chat_client import ChatClientSettings

from src.llm.client import (
    DEFAULT_CHAT_ALIAS,
    DEFAULT_EMBED_ALIAS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEMPERATURE,
    FoundryRuntime,
)


def _mock_manager() -> MagicMock:
    """A MagicMock standing in for FoundryLocalManager.instance.

    The mocked chat client's .settings is a REAL ChatClientSettings
    instance (not a further mock) so unset fields correctly default to
    None, matching production behaviour — needed to test that we do NOT
    set frequency_penalty.
    """
    manager = MagicMock()
    model = MagicMock()
    model.get_chat_client.return_value.settings = ChatClientSettings()
    manager.catalog.get_model.return_value = model
    return manager


def _make_runtime(manager: MagicMock, verbose: bool = False, **kwargs) -> FoundryRuntime:
    with patch("src.llm.client.Configuration"), \
         patch("src.llm.client.FoundryLocalManager") as mock_manager_cls:
        mock_manager_cls.instance = manager
        return FoundryRuntime(verbose=verbose, **kwargs)


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


# ── Regression: no progress_callback crosses the native/background-thread
# boundary (observed SIGSEGV in Microsoft.AI.Foundry.Local.Core.dylib when
# the SDK invokes a Python callback from a ThreadPool worker thread via
# ctypes/libffi — see SESSION_SUMMARY.md). Covers both verbose modes since
# the crash-prone code path was only reachable when verbose=True. ──────────

def test_download_and_register_eps_never_receives_a_progress_callback_verbose():
    manager = _mock_manager()
    runtime = _make_runtime(manager, verbose=True)
    manager.download_and_register_eps.assert_called_once_with()
    runtime.close()


def test_download_and_register_eps_never_receives_a_progress_callback_quiet():
    manager = _mock_manager()
    runtime = _make_runtime(manager, verbose=False)
    manager.download_and_register_eps.assert_called_once_with()
    runtime.close()


def test_model_download_never_receives_a_progress_callback_verbose():
    manager = _mock_manager()
    embed_model = MagicMock()
    manager.catalog.get_model.return_value = embed_model
    runtime = _make_runtime(manager, verbose=True, load_chat=False)
    embed_model.download.assert_called_once_with()
    runtime.close()


def test_model_download_never_receives_a_progress_callback_quiet():
    manager = _mock_manager()
    embed_model = MagicMock()
    manager.catalog.get_model.return_value = embed_model
    runtime = _make_runtime(manager, verbose=False, load_chat=False)
    embed_model.download.assert_called_once_with()
    runtime.close()


def test_verbose_index_only_load_still_downloads_loads_and_logs(capsys):
    """The verbose index-only path (main.py index) must still complete
    download + load and print progress lines, just without a native
    progress callback."""
    manager = _mock_manager()
    embed_model = MagicMock()
    manager.catalog.get_model.return_value = embed_model
    runtime = _make_runtime(manager, verbose=True, load_chat=False)

    embed_model.download.assert_called_once_with()
    embed_model.load.assert_called_once_with()

    out = capsys.readouterr().out
    assert "Downloading embedding model" in out
    assert "Loading embedding model" in out
    assert f"{DEFAULT_EMBED_ALIAS} ready." in out

    runtime.close()


# ── Generation safeguards (Phase 3): max_tokens + presence_penalty applied
# to the real chat client whenever it's loaded, to prevent uncontrolled or
# repetitive generation (observed during evaluation: extreme repetition,
# a leaked closing </think> tag, and fabricated citation numbers repeated
# verbatim three times). frequency_penalty was tried too but is NOT set —
# see the comment on DEFAULT_MAX_TOKENS in client.py: it made this backend
# return zero tokens for every generation, verified empirically. ────────────

def test_chat_client_gets_max_tokens_configured():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    assert runtime._chat_client.settings.max_tokens == DEFAULT_MAX_TOKENS
    runtime.close()


def test_chat_client_gets_presence_penalty_configured():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    assert runtime._chat_client.settings.presence_penalty == DEFAULT_PRESENCE_PENALTY
    runtime.close()


def test_chat_client_frequency_penalty_is_left_unset():
    """frequency_penalty must stay untouched (None) — setting it broke
    generation entirely on the real backend (empty stream every time)."""
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    assert runtime._chat_client.settings.frequency_penalty is None
    runtime.close()


def test_max_tokens_is_a_reasonable_positive_limit():
    """Guards against a future accidental 0/None/unbounded regression."""
    assert isinstance(DEFAULT_MAX_TOKENS, int)
    assert 0 < DEFAULT_MAX_TOKENS <= 2000


def test_load_chat_false_does_not_touch_chat_settings():
    """No chat client exists when load_chat=False, so there is nothing to
    configure — must not raise."""
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=False)
    assert runtime._chat_client is None
    runtime.close()


# ── Deterministic decoding (Phase 4): temperature=0 + a fixed random_seed,
# added after repeated evaluation runs showed run-to-run variance in
# length/repetition/factual claims with identical prompts. Both fields are
# real ChatClientSettings parameters on the installed SDK (verified by
# reading foundry_local_sdk/openai/chat_client.py), not guessed. ──────────

def test_chat_client_gets_temperature_configured():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    assert runtime._chat_client.settings.temperature == DEFAULT_TEMPERATURE
    runtime.close()


def test_temperature_is_low_and_nonzero():
    """Guards the empirical finding in client.py: temperature=0.0 silently
    disables presence_penalty on this backend and causes a degenerate
    <think>-block repetition loop; a small nonzero value is required."""
    assert 0.0 < DEFAULT_TEMPERATURE <= 0.3


def test_chat_client_gets_random_seed_configured():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=True)
    assert runtime._chat_client.settings.random_seed == DEFAULT_RANDOM_SEED
    runtime.close()


def test_load_chat_false_does_not_touch_temperature_or_seed():
    manager = _mock_manager()
    runtime = _make_runtime(manager, load_chat=False)
    assert runtime._chat_client is None
    runtime.close()

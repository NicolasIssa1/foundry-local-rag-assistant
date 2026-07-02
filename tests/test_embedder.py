"""
Tests for the Embedder class.

All tests that touch the network use unittest.mock to replace the OpenAI
client, so they run without Foundry Local running. The single integration
test at the bottom is automatically skipped when the server is unreachable.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.embeddings.embedder import Embedder, DEFAULT_MODEL, _MAX_BATCH_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_embedding(index: int, dim: int = 4) -> MagicMock:
    """Return a mock embedding object as the OpenAI SDK would produce."""
    obj = MagicMock()
    obj.index = index
    obj.embedding = [float(index) * 0.1] * dim
    return obj


def _mock_response(n: int, dim: int = 4) -> MagicMock:
    """Return a mock embeddings response with n embedding objects."""
    response = MagicMock()
    response.data = [_fake_embedding(i, dim) for i in range(n)]
    return response


def _embedder_with_mock(mock_response) -> tuple[Embedder, MagicMock]:
    """Return an Embedder whose internal OpenAI client is fully mocked."""
    embedder = Embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    embedder._client = mock_client
    return embedder, mock_client


# ── Construction ──────────────────────────────────────────────────────────────

def test_embedder_default_model():
    embedder = Embedder()
    assert embedder.model == DEFAULT_MODEL


def test_embedder_custom_model():
    embedder = Embedder(model="phi-3.5-mini")
    assert embedder.model == "phi-3.5-mini"


def test_embedder_model_property_is_read_only():
    embedder = Embedder()
    with pytest.raises(AttributeError):
        embedder.model = "other-model"


# ── embed() — return shape ────────────────────────────────────────────────────

def test_embed_returns_list():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    result = embedder.embed(["hello"])
    assert isinstance(result, list)


def test_embed_returns_one_vector_per_text():
    n = 3
    embedder, _ = _embedder_with_mock(_mock_response(n))
    result = embedder.embed(["a", "b", "c"])
    assert len(result) == n


def test_embed_each_vector_is_a_list_of_floats():
    embedder, _ = _embedder_with_mock(_mock_response(2, dim=4))
    result = embedder.embed(["x", "y"])
    for vec in result:
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


def test_embed_vector_dimension_matches_model_output():
    dim = 384
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=dim))
    result = embedder.embed(["hello"])
    assert len(result[0]) == dim


# ── embed() — API call behaviour ──────────────────────────────────────────────

def test_embed_calls_openai_create_once_for_small_batch():
    embedder, mock_client = _embedder_with_mock(_mock_response(2))
    embedder.embed(["a", "b"])
    assert mock_client.embeddings.create.call_count == 1


def test_embed_passes_correct_model_to_api():
    embedder, mock_client = _embedder_with_mock(_mock_response(1))
    embedder.embed(["test"])
    call_kwargs = mock_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL


def test_embed_passes_texts_as_input_to_api():
    texts = ["hello", "world"]
    embedder, mock_client = _embedder_with_mock(_mock_response(2))
    embedder.embed(texts)
    call_kwargs = mock_client.embeddings.create.call_args.kwargs
    assert call_kwargs["input"] == texts


def test_embed_batches_large_input():
    """Texts exceeding _MAX_BATCH_SIZE must be split into multiple API calls."""
    n = _MAX_BATCH_SIZE + 5
    texts = [f"text {i}" for i in range(n)]

    # Each API call can return different sized mocks
    first_batch_response = _mock_response(_MAX_BATCH_SIZE)
    second_batch_response = _mock_response(5)

    embedder = Embedder()
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        first_batch_response,
        second_batch_response,
    ]
    embedder._client = mock_client

    result = embedder.embed(texts)

    assert mock_client.embeddings.create.call_count == 2
    assert len(result) == n


def test_embed_preserves_order_even_if_api_returns_unordered():
    """embed() sorts by index, so shuffled API responses stay in order."""
    dim = 2
    response = MagicMock()
    # Return embeddings in reverse index order
    response.data = [_fake_embedding(i, dim) for i in reversed(range(3))]

    embedder, _ = _embedder_with_mock(response)
    result = embedder.embed(["a", "b", "c"])

    # Index 0 should be first regardless of API response order
    assert result[0] == [0.0 * 0.1] * dim
    assert result[1] == [1.0 * 0.1] * dim
    assert result[2] == [2.0 * 0.1] * dim


# ── embed() — error handling ──────────────────────────────────────────────────

def test_embed_raises_on_empty_list():
    embedder = Embedder()
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed([])


# ── embed_one() ───────────────────────────────────────────────────────────────

def test_embed_one_returns_single_vector():
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    result = embedder.embed_one("hello world")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_one_returns_flat_list_not_nested():
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    result = embedder.embed_one("hello")
    # Must be a list of floats, not a list of lists
    assert not isinstance(result[0], list)


def test_embed_one_raises_on_empty_string():
    embedder = Embedder()
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed_one("")


def test_embed_one_delegates_to_embed():
    """embed_one must call embed() internally (keeps batching logic in one place)."""
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    with patch.object(embedder, "embed", wraps=embedder.embed) as spy:
        embedder.embed_one("test")
        spy.assert_called_once_with(["test"])


# ── Live integration test (skipped if Foundry Local is not running) ────────────

def _foundry_local_is_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5272/v1/models", timeout=1)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _foundry_local_is_running(),
    reason="Foundry Local not running on localhost:5272",
)
def test_live_embed_returns_non_zero_vector():
    embedder = Embedder()
    vec = embedder.embed_one("Microsoft Foundry Local RAG assistant")
    assert len(vec) > 0
    assert any(v != 0.0 for v in vec)

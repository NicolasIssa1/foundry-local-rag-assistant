"""
Tests for FoundryEmbedder — the Foundry Local OpenAI-compatible embedding backend.

All tests mock the openai.OpenAI client so no Foundry Local installation
is required. The mock interface matches what the real client returns:
  client.embeddings.create(model=..., input=...) -> response
  response.data  →  list of objects each with a .embedding attribute.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.embeddings.foundry_embedder import FoundryEmbedder, _MAX_BATCH_SIZE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_data_item(embedding: list[float]) -> MagicMock:
    """One item from response.data with a .embedding attribute."""
    obj = MagicMock()
    obj.embedding = embedding
    return obj


def _mock_response(n: int, dim: int = 4) -> MagicMock:
    """Fake SDK response with n embedding objects of the given dimension."""
    response = MagicMock()
    response.data = [_fake_data_item([float(i) * 0.1] * dim) for i in range(n)]
    return response


def _embedder_with_mock(mock_response) -> tuple[FoundryEmbedder, MagicMock]:
    """Return a FoundryEmbedder whose internal openai client is fully mocked."""
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response
    return FoundryEmbedder(client=mock_client, model="qwen3-embedding-0.6b"), mock_client


# ── Construction ──────────────────────────────────────────────────────────────

def test_foundry_embedder_model_property():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    assert embedder.model == "qwen3-embedding-0.6b"


def test_foundry_embedder_custom_model():
    client = MagicMock()
    embedder = FoundryEmbedder(client=client, model="custom-embed-model")
    assert embedder.model == "custom-embed-model"


def test_foundry_embedder_model_property_is_read_only():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    with pytest.raises(AttributeError):
        embedder.model = "other"


# ── embed() — return shape ────────────────────────────────────────────────────

def test_embed_returns_list():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    result = embedder.embed(["hello"])
    assert isinstance(result, list)


def test_embed_returns_one_vector_per_text():
    embedder, _ = _embedder_with_mock(_mock_response(3))
    result = embedder.embed(["a", "b", "c"])
    assert len(result) == 3


def test_embed_each_vector_is_a_list_of_floats():
    embedder, _ = _embedder_with_mock(_mock_response(2, dim=4))
    result = embedder.embed(["x", "y"])
    for vec in result:
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


def test_embed_vector_dimension_matches_model_output():
    dim = 1024
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=dim))
    result = embedder.embed(["hello"])
    assert len(result[0]) == dim


# ── embed() — SDK call behaviour ──────────────────────────────────────────────

def test_embed_calls_embeddings_create_once_for_small_batch():
    embedder, mock_client = _embedder_with_mock(_mock_response(2))
    embedder.embed(["a", "b"])
    assert mock_client.embeddings.create.call_count == 1


def test_embed_passes_texts_to_embeddings_create():
    texts = ["hello", "world"]
    embedder, mock_client = _embedder_with_mock(_mock_response(2))
    embedder.embed(texts)
    mock_client.embeddings.create.assert_called_once_with(model="qwen3-embedding-0.6b", input=texts)


def test_embed_batches_large_input():
    """Texts exceeding _MAX_BATCH_SIZE must trigger multiple SDK calls."""
    n = _MAX_BATCH_SIZE + 5

    first_response = _mock_response(_MAX_BATCH_SIZE)
    second_response = _mock_response(5)

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [first_response, second_response]

    embedder = FoundryEmbedder(client=mock_client)
    result = embedder.embed([f"text {i}" for i in range(n)])

    assert mock_client.embeddings.create.call_count == 2
    assert len(result) == n


def test_embed_second_batch_receives_remaining_texts():
    """Verify the second call gets only the overflow texts, not a full batch."""
    n = _MAX_BATCH_SIZE + 3
    texts = [f"t{i}" for i in range(n)]

    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = [
        _mock_response(_MAX_BATCH_SIZE),
        _mock_response(3),
    ]

    FoundryEmbedder(client=mock_client).embed(texts)

    first_call_texts = mock_client.embeddings.create.call_args_list[0].kwargs["input"]
    second_call_texts = mock_client.embeddings.create.call_args_list[1].kwargs["input"]
    assert first_call_texts == texts[:_MAX_BATCH_SIZE]
    assert second_call_texts == texts[_MAX_BATCH_SIZE:]


def test_embed_preserves_insertion_order():
    """Vectors must come back in the same order as the input texts."""
    dim = 2
    response = _mock_response(3, dim=dim)
    embedder, _ = _embedder_with_mock(response)
    result = embedder.embed(["a", "b", "c"])

    assert result[0] == response.data[0].embedding
    assert result[1] == response.data[1].embedding
    assert result[2] == response.data[2].embedding


# ── embed() — error handling ──────────────────────────────────────────────────

def test_embed_raises_on_empty_list():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed([])


def test_embed_does_not_call_sdk_on_empty_list():
    embedder, mock_client = _embedder_with_mock(_mock_response(1))
    with pytest.raises(ValueError):
        embedder.embed([])
    mock_client.embeddings.create.assert_not_called()


# ── embed_one() ───────────────────────────────────────────────────────────────

def test_embed_one_returns_single_vector():
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    result = embedder.embed_one("hello world")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_one_returns_flat_list_not_nested():
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    result = embedder.embed_one("hello")
    assert not isinstance(result[0], list)


def test_embed_one_raises_on_empty_string():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed_one("")


def test_embed_one_delegates_to_embed():
    """embed_one must call embed() so batching logic stays in one place."""
    embedder, _ = _embedder_with_mock(_mock_response(1, dim=4))
    with patch.object(embedder, "embed", wraps=embedder.embed) as spy:
        embedder.embed_one("test")
        spy.assert_called_once_with(["test"])


def test_embed_one_returns_first_element_of_embed():
    """embed_one("x") == embed(["x"])[0]."""
    dim = 8
    response = _mock_response(1, dim=dim)
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = response
    embedder = FoundryEmbedder(client=mock_client)

    single = embedder.embed_one("hello")
    batch = embedder.embed(["hello"])
    assert single == batch[0]


# ── Interface parity with Embedder (OpenAI backend) ───────────────────────────

def test_foundry_embedder_has_embed_method():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    assert callable(getattr(embedder, "embed", None))


def test_foundry_embedder_has_embed_one_method():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    assert callable(getattr(embedder, "embed_one", None))


def test_foundry_embedder_has_model_property():
    embedder, _ = _embedder_with_mock(_mock_response(1))
    assert isinstance(embedder.model, str)

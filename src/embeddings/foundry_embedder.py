from __future__ import annotations

# Maximum texts per SDK call. Matches the ceiling used by the OpenAI backend.
_MAX_BATCH_SIZE = 64


class FoundryEmbedder:
    """Converts text strings into embedding vectors via Foundry Local's
    OpenAI-compatible REST API.

    foundry-local-sdk 0.5.1 is a control-plane SDK only (no native embedding
    client), so this wraps the openai.OpenAI client obtained from FoundryRuntime:

        with FoundryRuntime() as runtime:
            embedder = FoundryEmbedder(runtime.get_embedding_client(), runtime.embed_model_id)

    Public interface is intentionally identical to the OpenAI-backed Embedder
    so that Retriever and any other consumer can use either backend without
    modification. This is the Foundry Local provider; additional providers
    (Ollama, LM Studio, Azure OpenAI) would follow the same pattern.
    """

    def __init__(self, client, model: str = "") -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one float vector per text.

        Batches requests so no single call exceeds _MAX_BATCH_SIZE texts.
        Raises ValueError for empty input to prevent silent no-ops.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _MAX_BATCH_SIZE]
            response = self._client.embeddings.create(model=self._model, input=batch)
            # The API returns embeddings in the same order as the input.
            # We iterate .data directly without re-sorting.
            vectors.extend(item.embedding for item in response.data)

        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for embedding a single string."""
        if not text:
            raise ValueError("text must be a non-empty string")
        return self.embed([text])[0]

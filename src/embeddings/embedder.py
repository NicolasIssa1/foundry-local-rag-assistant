from __future__ import annotations

from openai import OpenAI

# Default model served by Foundry Local for embeddings.
# Can be overridden at construction time to match whatever model is loaded.
DEFAULT_MODEL = "all-minilm-l6-v2"
DEFAULT_BASE_URL = "http://localhost:5272/v1"
# Foundry Local ignores this value but the OpenAI SDK requires a non-empty string.
_PLACEHOLDER_API_KEY = "foundry-local"

# Maximum texts per API call. Foundry Local (and OpenAI) impose a batch limit;
# 64 is a safe, conservative ceiling for local models.
_MAX_BATCH_SIZE = 64


class Embedder:
    """Converts text strings into embedding vectors via Foundry Local.

    Uses the OpenAI-compatible REST API at base_url, so the same code
    works against any endpoint that speaks the /v1/embeddings protocol.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = _PLACEHOLDER_API_KEY,
    ) -> None:
        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one float vector per text.

        Batches requests so no single API call exceeds _MAX_BATCH_SIZE texts.
        Raises ValueError for empty input to prevent silent no-ops.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _MAX_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # The API returns embeddings sorted by their index field,
            # but we sort explicitly to guarantee order regardless of server behaviour.
            sorted_data = sorted(response.data, key=lambda d: d.index)
            vectors.extend(item.embedding for item in sorted_data)

        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Convenience method for embedding a single string."""
        if not text:
            raise ValueError("text must be a non-empty string")
        return self.embed([text])[0]

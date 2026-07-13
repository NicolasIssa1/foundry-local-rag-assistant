from __future__ import annotations

from typing import Iterator

from foundry_local import FoundryLocalManager
from openai import OpenAI

DEFAULT_EMBED_ALIAS = "qwen3-embedding-0.6b"
DEFAULT_CHAT_ALIAS = "qwen2.5-0.5b"


class FoundryRuntime:
    """Manages the Foundry Local SDK lifecycle for embedding and chat models.

    foundry-local-sdk 0.5.1 is a control-plane SDK only: FoundryLocalManager
    starts/queries the local Foundry service and downloads/loads models, but
    exposes no embedding or chat inference methods itself. Actual inference
    goes through Foundry Local's OpenAI-compatible REST API, so this class
    pairs FoundryLocalManager with an openai.OpenAI client pointed at the
    manager's endpoint.

    On construction the runtime:
      1. Starts the Foundry Local service (requires the 'foundry' CLI).
      2. Downloads and loads the embedding model (first run only, then cached).
      3. Downloads and loads the chat model (first run only, then cached).

    Use as a context manager so models are unloaded cleanly on exit:

        with FoundryRuntime() as runtime:
            embedder = FoundryEmbedder(runtime.get_embedding_client(), runtime.embed_model_id)
            answer = runtime.chat([{"role": "user", "content": prompt}])
    """

    def __init__(
        self,
        embed_alias: str = DEFAULT_EMBED_ALIAS,
        chat_alias: str = DEFAULT_CHAT_ALIAS,
        verbose: bool = True,
    ) -> None:
        self._embed_alias = embed_alias
        self._chat_alias = chat_alias
        self._verbose = verbose

        self._manager = FoundryLocalManager(bootstrap=True)
        self._embed_model_id = self._load_model(embed_alias, kind="embedding")
        self._chat_model_id = self._load_model(chat_alias, kind="chat")
        self._client = OpenAI(base_url=self._manager.endpoint, api_key=self._manager.api_key)

    # ── Internal setup helpers ────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    def _load_model(self, alias: str, kind: str) -> str:
        """Download (if needed) and load a model by its catalog alias. Returns its model id."""
        self._log(f"[foundry] Loading {kind} model: {alias}")
        model_info = self._manager.download_model(alias)
        self._manager.load_model(alias)
        self._log(f"[foundry] {alias} ready.")
        return model_info.id

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def embed_model_id(self) -> str:
        """Resolved catalog model id for the embedding model."""
        return self._embed_model_id

    @property
    def chat_model_id(self) -> str:
        """Resolved catalog model id for the chat model."""
        return self._chat_model_id

    def get_embedding_client(self) -> OpenAI:
        """Return the OpenAI-compatible client for use by FoundryEmbedder."""
        return self._client

    def chat(self, messages: list[dict]) -> str:
        """Send a messages list and return the complete response string."""
        response = self._client.chat.completions.create(model=self._chat_model_id, messages=messages)
        return response.choices[0].message.content

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Yield response tokens one at a time for streaming output."""
        stream = self._client.chat.completions.create(
            model=self._chat_model_id, messages=messages, stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Unload both models and release resources."""
        self._manager.unload_model(self._embed_alias)
        self._manager.unload_model(self._chat_alias)

    def __enter__(self) -> FoundryRuntime:
        return self

    def __exit__(self, *_) -> None:
        self.close()

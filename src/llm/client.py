from __future__ import annotations

from typing import Callable, Iterator

from foundry_local_sdk import Configuration, FoundryLocalManager

DEFAULT_EMBED_ALIAS = "qwen3-embedding-0.6b"
DEFAULT_CHAT_ALIAS = "qwen2.5-0.5b"


class FoundryRuntime:
    """Manages the Foundry Local SDK lifecycle for embedding and chat models.

    On construction the runtime:
      1. Initialises the SDK singleton.
      2. Downloads execution providers on first run (~200 MB, permanently cached).
      3. Downloads and loads the embedding model (first run only, then cached).
      4. Downloads and loads the chat model (first run only, then cached).

    Use as a context manager so models are unloaded cleanly on exit:

        with FoundryRuntime() as runtime:
            embedder = FoundryEmbedder(runtime.get_embedding_client())
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

        self._manager = self._init_sdk()
        self._download_eps()
        self._embed_model = self._load_model(embed_alias, kind="embedding")
        self._chat_model = self._load_model(chat_alias, kind="chat")
        self._embed_client = self._embed_model.get_embedding_client()
        self._chat_client = self._chat_model.get_chat_client()

    # ── Internal setup helpers ────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg)

    def _init_sdk(self):
        config = Configuration(app_name="foundry_local_rag_assistant")
        FoundryLocalManager.initialize(config)
        return FoundryLocalManager.instance

    def _download_eps(self) -> None:
        """Download and register execution providers (idempotent after first run)."""
        if not self._verbose:
            self._manager.download_and_register_eps()
            return

        # Track the current EP name so we can print a newline before switching.
        current: list[str] = [""]

        def _progress(ep_name: str, percent: float) -> None:
            if ep_name != current[0]:
                if current[0]:
                    print()
                current[0] = ep_name
            print(f"\r  {ep_name:<40}  {percent:5.1f}%", end="", flush=True)

        self._manager.download_and_register_eps(progress_callback=_progress)
        if current[0]:
            print()

    def _load_model(self, alias: str, kind: str):
        """Download (if needed) and load a model by its catalog alias."""
        self._log(f"[foundry] Loading {kind} model: {alias}")
        model = self._manager.catalog.get_model(alias)

        if self._verbose:
            model.download(
                lambda p: print(
                    f"\r  Downloading {alias}: {p:.1f}%", end="", flush=True
                )
            )
            print()
        else:
            model.download()

        model.load()
        self._log(f"[foundry] {alias} ready.")
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def get_embedding_client(self):
        """Return the native embedding client for use by FoundryEmbedder."""
        return self._embed_client

    def chat(self, messages: list[dict]) -> str:
        """Send a messages list and return the complete response string."""
        response = self._chat_client.complete_chat(messages)
        return response.choices[0].message.content

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Yield response tokens one at a time for streaming output."""
        for chunk in self._chat_client.complete_streaming_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Unload both models and release native resources."""
        self._embed_model.unload()
        self._chat_model.unload()

    def __enter__(self) -> FoundryRuntime:
        return self

    def __exit__(self, *_) -> None:
        self.close()

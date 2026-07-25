from __future__ import annotations

from typing import Iterator

from foundry_local_sdk import Configuration, FoundryLocalManager

DEFAULT_EMBED_ALIAS = "qwen3-embedding-0.6b"
DEFAULT_CHAT_ALIAS = "qwen3-1.7b"


class FoundryRuntime:
    """Manages the Foundry Local SDK lifecycle for embedding and chat models.

    On construction the runtime:
      1. Initialises the SDK singleton.
      2. Downloads execution providers on first run (~200 MB, permanently cached).
      3. Downloads and loads the embedding model (first run only, then cached).
      4. Downloads and loads the chat model (first run only, then cached) —
         unless load_chat=False, e.g. for index-only commands that never
         call chat()/stream_chat() and so have no need for it.

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
        load_chat: bool = True,
    ) -> None:
        self._embed_alias = embed_alias
        self._chat_alias = chat_alias
        self._verbose = verbose

        self._manager = self._init_sdk()
        self._download_eps()
        self._embed_model = self._load_model(embed_alias, kind="embedding")
        self._embed_client = self._embed_model.get_embedding_client()

        self._chat_model = None
        self._chat_client = None
        if load_chat:
            self._chat_model = self._load_model(chat_alias, kind="chat")
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
        """Download and register execution providers (idempotent after first run).

        Deliberately does NOT pass a progress_callback to the SDK. Doing so
        makes the native layer invoke our Python callback from a background
        .NET ThreadPool worker thread via a ctypes/libffi closure — observed
        to segfault (EXC_BAD_ACCESS inside ffi_closure_SYSV_inner, called
        from Microsoft.AI.Foundry.Local.Core.dylib's WebGpuEpBootstrapper
        callback marshaling). Calling with no callback takes the SDK's plain
        synchronous code path instead, which never crosses that boundary.
        """
        self._log("[foundry] Downloading/registering execution providers ...")
        self._manager.download_and_register_eps()
        self._log("[foundry] Execution providers ready.")

    def _load_model(self, alias: str, kind: str):
        """Download (if needed) and load a model by its catalog alias.

        Does NOT pass a progress_callback into model.download() for the
        same reason as _download_eps() above — the SDK routes any
        model.download() call with a callback through the identical
        crash-prone native-to-Python callback path.
        """
        self._log(f"[foundry] Downloading {kind} model: {alias} (skipped if already cached) ...")
        model = self._manager.catalog.get_model(alias)
        model.download()

        self._log(f"[foundry] Loading {kind} model: {alias}")
        model.load()
        self._log(f"[foundry] {alias} ready.")
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def get_embedding_client(self):
        """Return the native embedding client for use by FoundryEmbedder."""
        return self._embed_client

    def chat(self, messages: list[dict]) -> str:
        """Send a messages list and return the complete response string."""
        if self._chat_client is None:
            raise RuntimeError(
                "Chat model was not loaded (FoundryRuntime constructed with load_chat=False)."
            )
        response = self._chat_client.complete_chat(messages)
        return response.choices[0].message.content

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Yield response tokens one at a time for streaming output."""
        if self._chat_client is None:
            raise RuntimeError(
                "Chat model was not loaded (FoundryRuntime constructed with load_chat=False)."
            )
        for chunk in self._chat_client.complete_streaming_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Unload the loaded models and release native resources."""
        self._embed_model.unload()
        if self._chat_model is not None:
            self._chat_model.unload()

    def __enter__(self) -> FoundryRuntime:
        return self

    def __exit__(self, *_) -> None:
        self.close()

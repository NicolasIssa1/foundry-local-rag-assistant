from __future__ import annotations

from typing import Iterator

from foundry_local_sdk import Configuration, FoundryLocalManager

DEFAULT_EMBED_ALIAS = "qwen3-embedding-0.6b"
DEFAULT_CHAT_ALIAS = "qwen3-1.7b"

# Generation safeguard against uncontrolled output (observed once during
# evaluation: extreme repetition and a leaked closing </think> tag). Caps
# the maximum tokens per generation via the SDK's OpenAI-compatible
# ChatClientSettings.max_tokens field.
#
# 1024 (not a tighter value like 512): Qwen3's <think>...</think> reasoning
# block is generated before the visible answer and itself consumes several
# hundred tokens on typical questions. A cap that lands mid-<think> truncates
# the stream before </think> ever appears, and ThinkBlockFilter correctly
# discards everything from an unterminated think block — the whole visible
# answer would silently disappear (verified empirically: 512 reproduced this
# exact empty-answer failure on a real query). 1024 comfortably covers
# reasoning + a concise answer while still bounding true runaway repetition.
#
# frequency_penalty was tried as a second anti-repetition safeguard but is
# NOT used: setting ChatClientSettings.frequency_penalty to any non-None
# value on this backend (qwen3-1.7b via the local ONNX Runtime GenAI
# execution provider) made every generation return zero tokens — verified
# empirically by isolating it from max_tokens in a real query (a run with
# only max_tokens set behaved normally; the same run adding
# frequency_penalty=0.4 produced an empty stream every time).
#
# presence_penalty was tried next and DOES work on this backend (verified
# empirically: normal-length, non-empty, non-repetitive output with
# presence_penalty=0.6 on a real query that previously repeated the same
# sentence three times with fabricated citation numbers). It is used here
# alongside the prompt-level anti-repetition instruction in SYSTEM_PROMPT.
#
# 0.6 was later raised to 1.0 after a second real evaluation run showed
# 0.6 still isn't a hard guarantee: one generation produced a 760-word,
# 15-times-repeated answer even with 0.6 set. A stress test comparing 1.0
# against the same two prompts (2 trials each) showed meaningfully shorter,
# far-less-repetitive output (worst case dropped to ~198 words with a
# couple of repeated fragments, vs. 760 words/15 repeats at 0.6). This
# still isn't a 100% guarantee — a 1.7B local model's occasional
# repetition can't be fully eliminated via the sampling parameters this
# SDK exposes — which is exactly why the evaluation's repeated_blocks
# check (src/evaluation/safety_checks.py) exists as an independent,
# objective backstop rather than relying on generation settings alone.
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PRESENCE_PENALTY = 1.0

# Deterministic decoding (added after repeated evaluation runs showed
# run-to-run variance in length/repetition/factual claims with identical
# prompts). Both fields are real, supported ChatClientSettings parameters
# on the installed foundry_local_sdk (1.2.3) — verified by reading
# foundry_local_sdk/openai/chat_client.py rather than guessed.
#
# temperature=0.0 (pure greedy decoding) was tried FIRST, since it should
# in principle remove all sampling randomness. It was rejected after
# empirical testing: on this backend (qwen3-1.7b via the local ONNX
# Runtime GenAI execution provider), temperature=0 silently disables
# presence_penalty — verified by setting presence_penalty to 1.0, 1.5, and
# 2.0 with temperature=0 on a real query and getting byte-identical output
# every time — and greedy decoding then falls into a degenerate <think>
# repetition loop for at least one real evaluation question ("What is the
# default base URL for Foundry Local?"), consuming the entire max_tokens
# budget without ever closing the think block. Because ThinkBlockFilter
# correctly discards an unterminated think block, this produced a
# byte-identical EMPTY visible answer on every run — deterministic, but
# deterministically broken, which is worse than the variance it was meant
# to fix.
#
# temperature=0.1 was tested next: presence_penalty took effect again, the
# repetition loop did not occur, and the same question/prompt produced
# byte-identical non-empty answers across repeated real calls (verified:
# 3 consecutive calls each for two different questions, all matching
# exactly) once random_seed is also fixed. This is the smallest departure
# from temperature=0 that restores both anti-repetition safeguards and
# real determinism on this backend, so it's used instead of 0.0.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_RANDOM_SEED = 0


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
            self._chat_client.settings.max_tokens = DEFAULT_MAX_TOKENS
            self._chat_client.settings.presence_penalty = DEFAULT_PRESENCE_PENALTY
            self._chat_client.settings.temperature = DEFAULT_TEMPERATURE
            self._chat_client.settings.random_seed = DEFAULT_RANDOM_SEED

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

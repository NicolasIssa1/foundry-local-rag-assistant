# Microsoft Foundry Local — Quick Reference

## What is Foundry Local?

Microsoft Foundry Local is a developer tool that runs AI models directly on your machine.
It exposes an OpenAI-compatible REST API, which means any application written against the
OpenAI SDK can point to Foundry Local instead, with no code changes other than the
`base_url` parameter.

## Supported Model Types

| Type | Used for |
|---|---|
| Chat / completion models | Generating text answers |
| Embedding models | Converting text to numerical vectors |

## Default Endpoints

- **Base URL:** `http://localhost:5272`
- **Chat completions:** `POST /v1/chat/completions`
- **Embeddings:** `POST /v1/embeddings`
- **Available models:** `GET /v1/models`

## Python Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5272/v1",
    api_key="foundry-local",  # value is ignored but required by the SDK
)

# Chat
response = client.chat.completions.create(
    model="phi-3.5-mini",
    messages=[{"role": "user", "content": "Explain RAG in one sentence."}],
)

# Embeddings
embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Retrieval-Augmented Generation is a technique..."],
)
```

## Starting Foundry Local

```bash
# Start the local server (runs in the background)
foundry local start

# List downloaded models
foundry model list

# Download a model
foundry model download phi-3.5-mini
```

## Key Advantages for RAG

1. **No API key required** — the `api_key` field is accepted but ignored.
2. **Same interface as OpenAI** — swap the `base_url`, nothing else changes.
3. **Offline operation** — once models are downloaded, no internet connection is needed.
4. **Privacy** — documents and queries never leave the local machine.
5. **Multiple models** — run a small fast model for embeddings and a larger model for chat.

## Hardware Requirements

| Capability | Minimum |
|---|---|
| RAM | 8 GB (16 GB recommended) |
| Storage | 5–20 GB per model |
| GPU | Optional; CPU inference supported |

## Recommended Models for This Project

- **Chat:** `phi-3.5-mini` (fast, low RAM) or `phi-3-medium` (higher quality)
- **Embeddings:** `all-minilm-l6-v2` (small, fast, good quality)

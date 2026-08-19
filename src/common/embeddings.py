"""
Embedding extraction across the three providers, used by Stage A.5
(embedding-based classification)'s --apply-full-corpus path.

Note on consistency with the published train/test embeddings
(data/labeled/embeddings/*.npy, produced by generate_embeddings.py):
generate_embeddings.py embeds "Purpose: ... \nContract Summary: {that
provider's own summary} \nFolder Name:..." for every row, but the full
21,629-contract corpus (data/ocr_output/summaries/{contract_id}_{model}.json,
written by src/02_summarization/summarize.py) only carries a bare `summary`
field — no Purpose or Folder_Name is produced anywhere upstream for that
corpus. embed_text below is therefore called with the summary text alone,
which is NOT the same feature space the train/test classifier was fit on;
--apply-full-corpus would need a Purpose/Folder_Name source for the full
corpus (none currently exists) before its output is comparable to the
published embeddings, for every provider, not just llama specifically.

The gpt and gemini paths below otherwise use the same model IDs, chunking
(gpt), and task_type (gemini) as generate_embeddings.py. The llama path
additionally does NOT match mechanically — the published *_llama.npy is
mean-pooled hidden states from a local meta-llama/Llama-3.1-70B run (see
generate_embeddings.py), while _embed_llama below calls a chat-server-style
/embeddings endpoint (LLAMA_EMBEDDING_MODEL_ID, e.g. nomic-embed-text on
Ollama) at a different dimensionality. A classifier fit on the 8192-dim
Llama-3.1-70B embeddings will not accept this function's output —
--apply-full-corpus for llama needs to be re-pointed at
generate_embeddings.generate_llama_embeddings before it's usable.
"""

import numpy as np

from . import config
from .llm_clients import _gemini_client, _llama_client, _openai_client


def embed_text(text: str, model_key: str) -> np.ndarray:
    dispatch = {"gpt": _embed_gpt, "gemini": _embed_gemini, "llama": _embed_llama}
    if model_key not in dispatch:
        raise ValueError(f"Unknown model_key: {model_key!r} (expected one of {list(dispatch)})")
    return dispatch[model_key](text)


def chunk_text(text: str, max_tokens: int = 8000, overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks of at most max_tokens (cl100k_base
    tokens), for models with an input token limit. Matches the chunking used
    to produce the published *_gpt.npy embeddings (generate_embeddings.py)."""
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        start += max_tokens - overlap
    return chunks


def _embed_gpt(text: str) -> np.ndarray:
    client = _openai_client()
    chunk_embeddings = [
        client.embeddings.create(model=config.EMBEDDING_MODEL_IDS["gpt"], input=chunk).data[0].embedding
        for chunk in chunk_text(text)
    ]
    return np.mean(chunk_embeddings, axis=0)


def _embed_gemini(text: str) -> np.ndarray:
    from google.genai import types

    client = _gemini_client()
    resp = client.models.embed_content(
        model=config.EMBEDDING_MODEL_IDS["gemini"],
        contents=text,
        config=types.EmbedContentConfig(task_type="CLASSIFICATION"),
    )
    return np.array(resp.embeddings[0].values, dtype=np.float64)


def _embed_llama(text: str) -> np.ndarray:
    # Requires the local server (Ollama/vLLM) to also serve an embedding
    # model — chat checkpoints don't expose an /embeddings endpoint. Point
    # LLAMA_EMBEDDING_MODEL_ID at whatever embedding model you have pulled
    # (e.g. "nomic-embed-text" on Ollama). See the module docstring: this
    # does NOT match the published *_llama.npy's feature space.
    client = _llama_client()
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL_IDS["llama"], input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

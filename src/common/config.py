"""
Central environment/config loading, shared by every LLM-calling stage.

All values are overridable via .env (see .env.example) so the same code runs
against whatever model versions/endpoints you currently have access to —
the paper's exact version strings (GPT-5.2 Pro, Gemini 3 Pro, LLaMA 3.1 70B)
are set as defaults but are not guaranteed to still resolve to the same
underlying weights if you replicate later.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Chat-completion model IDs, one per provider.
CHAT_MODEL_IDS = {
    "gpt": os.getenv("GPT_MODEL_ID", "gpt-5.2-pro"),
    "gemini": os.getenv("GEMINI_MODEL_ID", "gemini-3-pro"),
    "llama": os.getenv("LLAMA_MODEL_ID", "llama3.1:70b"),
}

# Embedding model IDs. LLaMA embeddings need a dedicated embedding model
# (chat checkpoints don't expose an embeddings endpoint) — point this at
# whatever your local server serves, e.g. "nomic-embed-text" on Ollama.
EMBEDDING_MODEL_IDS = {
    "gpt": os.getenv("GPT_EMBEDDING_MODEL_ID", "text-embedding-3-small"),
    "gemini": os.getenv("GEMINI_EMBEDDING_MODEL_ID", "gemini-embedding-001"),
    "llama": os.getenv("LLAMA_EMBEDDING_MODEL_ID", "nomic-embed-text"),
}

# LLaMA embeddings for the published data/labeled/embeddings/*_llama.npy were
# NOT produced through the chat-completion-style endpoint above — they're
# mean-pooled last-hidden-state vectors from this HF model, run locally.
# See generate_embeddings.py. LLAMA_EMBEDDING_MODEL_ID/nomic-embed-text above
# is only used by the live --apply-full-corpus path (src/common/embeddings.py),
# which is dimensionally incompatible with a classifier fit on this model's
# 8192-dim output — see that module's docstring.
LLAMA_HF_EMBEDDING_MODEL_ID = os.getenv("LLAMA_HF_EMBEDDING_MODEL_ID", "meta-llama/Llama-3.1-70B")

# LLaMA is served locally (Ollama or vLLM) behind an OpenAI-compatible API.
LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:11434/v1")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "not-needed")

# Stage A.6 (financial extraction) uses a separate, faster/cheaper GPT
# variant than classification/summarization — confirmed against the actual
# extraction script (gpt-5.4 there vs. gpt-5.2-pro in CHAT_MODEL_IDS).
FINANCIAL_EXTRACTION_MODEL_ID = os.getenv("GPT_EXTRACTION_MODEL_ID", "gpt-5.4")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))

# Contracts are truncated to this many characters before being sent to any
# LLM call, as a hard safety net against context-window overflow. Tune per
# model if you hit truncation issues — the paper doesn't specify an exact
# chunking strategy, so this is a conservative default, not a replication
# of a documented choice.
MAX_INPUT_CHARS = int(os.getenv("LLM_MAX_INPUT_CHARS", "60000"))

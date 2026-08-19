"""
Stage A.5 helper — Embedding Generation

Reproduces data/labeled/embeddings/{split}_{model}.npy from data/labeled/
train.csv and test.csv. The published .npy files are the exact artifacts
used for the results in paper Tables 3-5, so most users won't need to rerun
this — it's here for methodological transparency (paper Appendix), and for
anyone who wants to regenerate the embeddings (e.g. against updated model
versions) or extend to new contracts.

All three providers use the same recipe — each embeds
"Purpose: {Purpose}\\nContract Summary: {model's own summary}\\nFolder
Name:{Folder_Name with the trailing .pdf stripped}" (see _build_texts below),
using that provider's own summary column (gpt_summary / gemini_summary /
llama_summary). Only the embedding call itself differs per provider:

- gpt: EMBEDDING_MODEL_IDS["gpt"] (text-embedding-3-small). Text longer than
  the token limit is split into overlapping chunks via
  src.common.embeddings.chunk_text (8000 tokens, 200 overlap, cl100k_base
  encoding), each chunk embedded separately, and the resulting vectors
  averaged.
- gemini: EMBEDDING_MODEL_IDS["gemini"] (gemini-embedding-001),
  task_type="CLASSIFICATION".
- llama: LLAMA_HF_EMBEDDING_MODEL_ID (meta-llama/Llama-3.1-70B by default),
  loaded locally via transformers, mean-pooled over the final hidden state.
  This needs substantial local compute/VRAM for a 70B model and the
  transformers/torch extras (not installed by default — see
  requirements.txt). max_length here (LLAMA_MAX_LENGTH, default 256 tokens)
  is carried over from the original multi-model comparison script's shared
  config across many candidate encoders; it wasn't tuned specifically for
  this model, so confirm it matches your intended truncation before
  treating output as bit-identical to the published .npy.

Input:  data/labeled/train.csv, data/labeled/test.csv
Output: data/labeled/embeddings/{split}_{model}.npy (overwrites any existing
        file for that split/model)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common import config
from src.common.embeddings import chunk_text
from src.common.llm_clients import _gemini_client, _openai_client

LABELED_DIR = Path("data/labeled")
EMBEDDINGS_DIR = LABELED_DIR / "embeddings"

GEMINI_BATCH_SIZE = 8
LLAMA_BATCH_SIZE = 8
LLAMA_MAX_LENGTH = 256


def _load_split_texts(split_name: str) -> pd.DataFrame:
    path = LABELED_DIR / f"{split_name}.csv"
    df = pd.read_csv(path, dtype={"PDF_ID": str})
    for col in ("Purpose", "gpt_summary", "gemini_summary", "Folder_Name", "llama_summary"):
        df[col] = df[col].fillna("").astype(str)
    return df


def _build_texts(df: pd.DataFrame, summary_col: str) -> list[str]:
    """Builds "Purpose: {Purpose}\\nContract Summary: {summary_col}\\nFolder
    Name:{Folder_Name, trailing .pdf stripped}" — the shared recipe used to
    embed each row, regardless of provider."""
    return (
        "Purpose: " + df["Purpose"] + "\nContract Summary: " + df[summary_col]
        + "\nFolder Name:" + df["Folder_Name"].str[:-4]
    ).tolist()


def generate_gpt_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Embeds each row's recipe text (see _build_texts), chunked/averaged
    (see src.common.embeddings.chunk_text) for anything over the token
    limit."""
    client = _openai_client()
    rows = []
    for text in tqdm(_build_texts(df, "gpt_summary"), desc="Embedding with GPT"):
        chunk_embeddings = [
            client.embeddings.create(model=config.EMBEDDING_MODEL_IDS["gpt"], input=chunk).data[0].embedding
            for chunk in chunk_text(text)
        ]
        rows.append(np.mean(chunk_embeddings, axis=0))
    return np.array(rows, dtype=np.float64)


def generate_gemini_embeddings(df: pd.DataFrame, batch_size: int = GEMINI_BATCH_SIZE) -> np.ndarray:
    """Embeds each row's recipe text (see _build_texts), batched, with
    task_type=CLASSIFICATION."""
    from google.genai import types

    texts = _build_texts(df, "gemini_summary")

    client = _gemini_client()
    rows = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with Gemini"):
        resp = client.models.embed_content(
            model=config.EMBEDDING_MODEL_IDS["gemini"],
            contents=texts[i : i + batch_size],
            config=types.EmbedContentConfig(task_type="CLASSIFICATION"),
        )
        rows.extend(e.values for e in resp.embeddings)
    return np.array(rows, dtype=np.float64)


def _average_pool(last_hidden_states, attention_mask):
    import torch

    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)


def generate_llama_embeddings(
    df: pd.DataFrame,
    model_id: str | None = None,
    batch_size: int = LLAMA_BATCH_SIZE,
    max_length: int = LLAMA_MAX_LENGTH,
) -> np.ndarray:
    """Mean-pools the final hidden state of each row's recipe text (see
    _build_texts) through a local HF model (meta-llama/Llama-3.1-70B by
    default). Requires transformers/torch and enough local VRAM/RAM to hold
    a 70B-parameter model."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = model_id or config.LLAMA_HF_EMBEDDING_MODEL_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    texts = _build_texts(df, "llama_summary")
    rows = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with LLaMA"):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            pooled = _average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            rows.append(pooled.cpu().float().numpy())
    return np.vstack(rows).astype(np.float32)


_GENERATORS = {
    "gpt": lambda df, args: generate_gpt_embeddings(df),
    "gemini": lambda df, args: generate_gemini_embeddings(df, args.gemini_batch_size),
    "llama": lambda df, args: generate_llama_embeddings(
        df, args.llama_model_id, args.llama_batch_size, args.llama_max_length
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate data/labeled/embeddings/{split}_{model}.npy.")
    parser.add_argument("--model", choices=list(_GENERATORS) + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--gemini-batch-size", type=int, default=GEMINI_BATCH_SIZE)
    parser.add_argument("--llama-model-id", default=None, help="Overrides LLAMA_HF_EMBEDDING_MODEL_ID for this run.")
    parser.add_argument("--llama-batch-size", type=int, default=LLAMA_BATCH_SIZE)
    parser.add_argument("--llama-max-length", type=int, default=LLAMA_MAX_LENGTH)
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    models = list(_GENERATORS) if args.model == "all" else [args.model]

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in splits:
        df = _load_split_texts(split_name)
        for model_key in models:
            X = _GENERATORS[model_key](df, args)
            out_path = EMBEDDINGS_DIR / f"{split_name}_{model_key}.npy"
            np.save(out_path, X)
            print(f"Wrote {out_path} {X.shape}")


if __name__ == "__main__":
    main()

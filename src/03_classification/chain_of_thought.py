"""
Stage A.4 — Classification: Chain-of-Thought

Classifies each contract summary into one of four institutional forms
(Service Contract, Resource Sharing, Joint Operations, New Joint Entities)
using zero-shot, 1-shot, and 3-shot chain-of-thought prompting, per
prompts/classification_prompt_template.txt.

Evaluated against the held-out hand-labeled test set (N = 227). Results
reported in paper Table 2.

Expects data/labeled/train.csv (N = 901) and data/labeled/test.csv
(N = 227), each with columns:
    PDF_ID, Folder_Name, surya_ocr, Purpose, gpt_summary, gemini_summary,
    llama_summary, Institutional_Form
Folder_Name and surya_ocr aren't used by this stage.
Institutional_Form is one of CATEGORIES. 

Each classifying model (LLaMA, GPT, Gemini) is prompted with its OWN
summary column (gpt_summary / gemini_summary / llama_summary) — these were
generated per-model to match what each model's embeddings (Stage A.5) are
built from, so a model never sees another model's summary text.

Few-shot examples are chosen deterministically: the first N rows per
category (by contract_id sort order) from train.csv, using the classifying
model's own summary column. 


Input:  data/labeled/train.csv, data/labeled/test.csv (the only inputs this
        script needs, and both are part of the public release)
Output: data/labeled/cot_predictions_{model}_{shots}shot.csv
        data/labeled/cot_report.json (F1 by model/shots/class, paper Table 2)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.common.llm_clients import chat_complete, extract_json_object

PROMPT_TEMPLATE_PATH = Path("prompts/classification_prompt_template.txt")
LABELED_DIR = Path("data/labeled")
TRAIN_PATH = LABELED_DIR / "train.csv"
TEST_PATH = LABELED_DIR / "test.csv"

CATEGORIES = ["Service Contract", "Resource Sharing", "Joint Operations", "New Joint Entities"]
SHOT_SETTINGS = [0, 1, 3]
MODELS = ["llama", "gpt", "gemini"]

_CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}
_COLUMN_RENAME = {"PDF_ID": "contract_id", "Purpose": "purpose", "Institutional_Form": "label"}


def _summary_column(model_key: str) -> str:
    return f"{model_key}_summary"


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"PDF_ID": str}).rename(columns=_COLUMN_RENAME) 
    df["label"] = df["label"].str.strip().str.title()
    df["Purpose"] = df["Purpose"].fillna("")
    return df


def select_few_shot_examples(train_df: pd.DataFrame, n_shots: int, model_key: str) -> dict:
    """Return {category: [example_summary, ...]} with up to n_shots examples
    per category, drawn from the train split, using model_key's own summary
    column (each model is prompted with its own summaries, never another
    model's)."""
    summary_col = _summary_column(model_key)
    examples = {}
    for category in CATEGORIES:
        rows = train_df[train_df["label"] == category].sort_values("contract_id").head(n_shots)
        examples[category] = [s for s in rows[summary_col] if isinstance(s, str) and s.strip()]
    return examples


def build_prompt(purpose: str, summary: str, n_shots: int, examples: dict) -> str:
    """Fill the CoT template with the contract's purpose/summary and the
    appropriate number of few-shot examples per category."""
    template = PROMPT_TEMPLATE_PATH.read_text()
    lines = template.splitlines()
    out_lines = []
    for line in lines:
        if line.strip().startswith("#"):
            continue  # strip the file's leading comment header
        out_lines.append(line)
    template = "\n".join(out_lines)

    prompt = template.replace("[PURPOSE]", purpose or "").replace("[CONTRACT SUMMARY]", summary)

    if n_shots == 0:
        # Zero-shot: drop every line still holding the raw [EXAMPLES]
        # placeholder (the per-category demonstration block) — no
        # numbered "Example N:" lines are emitted at all.
        prompt = "\n".join(l for l in prompt.splitlines() if "[EXAMPLES]" not in l)
        return prompt

    def replacement_for(category: str) -> str:
        shot_examples = examples.get(category, [])[:n_shots]
        if not shot_examples:
            return "  (no example available)"
        return "\n".join(f"  Example {i + 1}: {ex}" for i, ex in enumerate(shot_examples))

    # Walk line-by-line so each [EXAMPLES] placeholder is filled with
    # examples from the category whose bullet it actually belongs to, rather
    # than assuming the bullets appear in CATEGORIES order.
    out_lines = []
    current_category = None
    for line in prompt.splitlines():
        stripped = line.strip()
        for category in CATEGORIES:
            if stripped.startswith(f"- {category}:"):
                current_category = category
                break
        if "[EXAMPLES]" in line and current_category is not None:
            line = line.replace("[EXAMPLES]", replacement_for(current_category))
        out_lines.append(line)
    return "\n".join(out_lines)


def _coerce_category(raw_target: str) -> str | None:
    raw = (raw_target or "").strip().lower()
    if raw in _CATEGORY_LOOKUP:
        return _CATEGORY_LOOKUP[raw]
    for key, canonical in _CATEGORY_LOOKUP.items():
        if key in raw or raw in key:
            return canonical
    return None


def classify_contract(prompt: str, model_key: str) -> str | None:
    """Call the given model's API and parse the returned category. Returns
    None if the response can't be mapped to one of CATEGORIES."""
    response = chat_complete(prompt, model_key, json_mode=True)
    parsed = extract_json_object(response)
    return _coerce_category(parsed.get("Target", ""))


def main():
    parser = argparse.ArgumentParser(description="Run chain-of-thought classification.")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument(
        "--shots", type=int, choices=SHOT_SETTINGS + [-1], default=-1, help="-1 runs all of {0, 1, 3}"
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    args = parser.parse_args()

    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = _load_split(args.train)
    test_df = _load_split(args.test)

    models = MODELS if args.model == "all" else [args.model]
    shot_settings = SHOT_SETTINGS if args.shots == -1 else [args.shots]

    report = {}
    for model_key in models:
        summary_col = _summary_column(model_key)
        for n_shots in shot_settings:
            examples = select_few_shot_examples(train_df, n_shots, model_key) if n_shots else {}
            predictions = []
            for _, row in test_df.iterrows():
                summary = row.get(summary_col, "")
                if not isinstance(summary, str) or not summary.strip():
                    print(f"[WARN] no {summary_col} for {row['contract_id']}, skipping")
                    continue
                prompt = build_prompt(row.get("purpose", ""), summary, n_shots, examples)
                try:
                    predicted = classify_contract(prompt, model_key)
                except Exception as exc:  # noqa: BLE001 — log and continue the batch
                    print(f"[ERROR] {model_key}/{n_shots}-shot failed on {row['contract_id']}: {exc}")
                    predicted = None
                predictions.append(
                    {"contract_id": row["contract_id"], "true_label": row["label"], "predicted_label": predicted}
                )

            pred_df = pd.DataFrame(predictions)
            out_path = LABELED_DIR / f"cot_predictions_{model_key}_{n_shots}shot.csv"
            pred_df.to_csv(out_path, index=False)

            evaluable = pred_df.dropna(subset=["predicted_label"])
            if not evaluable.empty:
                metrics = classification_report(
                    evaluable["true_label"], evaluable["predicted_label"], labels=CATEGORIES,
                    output_dict=True, zero_division=0,
                )
                report[f"{model_key}_{n_shots}shot"] = metrics
                print(f"{model_key} {n_shots}-shot weighted F1: {metrics['weighted avg']['f1-score']:.3f}")

    (LABELED_DIR / "cot_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

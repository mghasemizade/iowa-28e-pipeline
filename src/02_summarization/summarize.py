"""
Stage A.3 — LLM Summarization

Summarizes each OCR'd contract into a 3-5 sentence summary using
prompts/summarization_prompt.txt, run identically across LLaMA 3.1,
GPT-5.2 Pro, and Gemini 3 Pro.

Not runnable from the public release: this depends on data/ocr_output/*.txt
from Stage A.2, which requires a non-public raw PDF corpus (see that
script's docstring and data/raw_pdfs/README.md). The public release
includes one summary per labeled contract per summarizer model, directly in
data/labeled/train.csv and test.csv (gpt_summary / gemini_summary /
llama_summary columns) — see Stage A.4/A.5, which consume those columns
instead of this stage's per-model JSON output.



Input:  data/ocr_output/*.txt
Output: data/ocr_output/summaries/{contract_id}_{model}.json
        each file: {"contract_id": ..., "model": ..., "summary": ...}
"""

import argparse
import json
from pathlib import Path

from src.common.llm_clients import chat_complete
from src.common.text_utils import iter_with_progress, truncate_text

PROMPT_PATH = Path("prompts/summarization_prompt.txt")
OCR_DIR = Path("data/ocr_output")
SUMMARY_DIR = Path("data/ocr_output/summaries")

MODELS = ["llama", "gpt", "gemini"]


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text()


def summarize_contract(contract_text: str, model_key: str) -> str:
    """Call the given model's API with the summarization prompt and return
    the raw summary text."""
    template = load_prompt_template()
    prompt = template.replace("[CONTRACT TEXT]", truncate_text(contract_text))
    return chat_complete(prompt, model_key).strip()


def main():
    parser = argparse.ArgumentParser(description="Summarize OCR'd contracts.")
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--input-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = MODELS if args.model == "all" else [args.model]

    txt_paths = sorted(args.input_dir.glob("*.txt"))
    if not txt_paths:
        print(f"No OCR'd text files found in {args.input_dir}")
        return

    for txt_path in iter_with_progress(txt_paths, desc="Summarizing"):
        contract_id = txt_path.stem
        contract_text = txt_path.read_text(encoding="utf-8")
        for model_key in models:
            out_path = args.output_dir / f"{contract_id}_{model_key}.json"
            if out_path.exists() and not args.overwrite:
                continue
            try:
                summary = summarize_contract(contract_text, model_key)
            except Exception as exc:  # noqa: BLE001 — log and continue the batch
                print(f"[ERROR] {model_key} failed on {contract_id}: {exc}")
                continue
            out_path.write_text(
                json.dumps({"contract_id": contract_id, "model": model_key, "summary": summary}, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()

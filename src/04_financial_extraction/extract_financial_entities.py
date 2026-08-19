"""
Stage A.6 — Financial Entity Extraction

For every contract classified as a Service Contract (weighted F1 = 0.92 per
the embedding classifier, paper Table 3) with prediction_confidence >=
CONFIDENCE_THRESHOLD, extracts the principal, agent, and dollar amount using
gpt-5.4 (config.FINANCIAL_EXTRACTION_MODEL_ID — a faster/cheaper variant
than GPT-5.2 Pro used elsewhere in the pipeline) and
prompts/financial_extraction_prompt.txt. The model also returns
has_transaction; rows where it's false (no monetary exchange — e.g. a
memorandum misclassified as a Service Contract) are dropped rather than
written to the output.

Validation checkpoint: 100 service contracts randomly sampled and manually
verified against the source document by the labeling team; 96% confirmed
correct. That verification step is manual and outside this script's scope —
after running this stage, draw your own random sample from the output and
check it before trusting the network built from it (see prompt file header:
the extraction prompt here is a reconstruction, not the paper's verbatim
prompt).

This script's *output* (data/network/extracted_financial_entities.csv) is
one of the two files in the public release. Its *inputs* below are not
public — full_corpus_classifications_gpt.csv depends on --apply-full-corpus
in Stage A.5, and data/ocr_output/*.txt is the full per-contract OCR text —
both require your own raw PDF corpus (see Stage A.2/A.3 docstrings and
data/raw_pdfs/README.md). This script is kept as method documentation for
how the published CSV was produced; it isn't runnable from the public
release alone. Stage A.7 (entity disambiguation) picks up directly from the
published extracted_financial_entities.csv and *is* runnable as-is.

Input:  data/labeled/full_corpus_classifications_gpt.csv
            (from `python src/03_classification/embedding_classification.py
            --model gpt --apply-full-corpus`; filtered to
            predicted_label == "Service Contract" and
            prediction_confidence >= CONFIDENCE_THRESHOLD)
        data/ocr_output/*.txt (full contract text — the summary can omit
            exact dollar figures/party names that the full text preserves)
Output: data/network/extracted_financial_entities.csv (published)
        columns: contract_id, principal, agent, dollar_amount,
                 service_type
"""

import argparse
from pathlib import Path

import pandas as pd

from src.common import config
from src.common.llm_clients import chat_complete, extract_json_object
from src.common.text_utils import iter_with_progress, truncate_text

PROMPT_PATH = Path("prompts/financial_extraction_prompt.txt")
CLASSIFICATION_PATH = Path("data/labeled/full_corpus_classifications_gpt.csv")
OCR_DIR = Path("data/ocr_output")
OUTPUT_PATH = Path("data/network/extracted_financial_entities.csv")

EXTRACTION_MODEL = "gpt"
CONFIDENCE_THRESHOLD = 0.65


def _load_prompt_template() -> str:
    lines = PROMPT_PATH.read_text().splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#"))


def extract_financial_entities(contract_text: str, purpose: str = "") -> dict:
    """Call gpt-5.2 to extract {principal, agent, dollar_amount,
    payment_structure, has_transaction, service_type} from a single service
    contract."""
    template = _load_prompt_template()
    prompt = template.replace("[PURPOSE]", purpose).replace("[CONTRACT TEXT]", truncate_text(contract_text))
    response = chat_complete(
        prompt, EXTRACTION_MODEL, json_mode=True, model=config.FINANCIAL_EXTRACTION_MODEL_ID
    )
    parsed = extract_json_object(response)

    required_keys = {"principal", "agent", "dollar_amount", "has_transaction", "service_type"}
    missing = required_keys - parsed.keys()
    if missing:
        raise ValueError(f"Extraction response missing keys {missing}: {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Extract financial entities from service contracts.")
    parser.add_argument("--classifications", type=Path, default=CLASSIFICATION_PATH)
    parser.add_argument("--ocr-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    classifications = pd.read_csv(args.classifications, dtype={"contract_id": str})
    is_service_contract = classifications["predicted_label"] == "Service Contract"
    if "prediction_confidence" in classifications.columns:
        is_service_contract &= classifications["prediction_confidence"] >= CONFIDENCE_THRESHOLD
    else:
        print(
            f"[WARN] {args.classifications} has no prediction_confidence column "
            f"(re-run with an updated embedding_classification.py) — skipping confidence filter"
        )
    service_contracts = classifications[is_service_contract]

    already_done = set()
    if args.output.exists() and not args.overwrite:
        already_done = set(pd.read_csv(args.output, dtype={"contract_id": str})["contract_id"])

    records = []
    if args.output.exists() and not args.overwrite:
        records = pd.read_csv(args.output, dtype={"contract_id": str}).to_dict("records")

    for _, row in iter_with_progress(list(service_contracts.iterrows()), desc="Extracting financials"):
        contract_id = row["contract_id"]
        if contract_id in already_done:
            continue
        text_path = args.ocr_dir / f"{contract_id}.txt"
        if not text_path.exists():
            print(f"[WARN] no OCR text for {contract_id}, skipping")
            continue

        try:
            fields = extract_financial_entities(text_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — log and continue the batch
            print(f"[ERROR] extraction failed for {contract_id}: {exc}")
            continue

       

        records.append({"contract_id": contract_id, **fields})
        # Write incrementally so a crash mid-run doesn't lose completed work.
        pd.DataFrame(records).to_csv(args.output, index=False)

    print(f"Wrote {len(records)} extracted financial records -> {args.output}")


if __name__ == "__main__":
    main()

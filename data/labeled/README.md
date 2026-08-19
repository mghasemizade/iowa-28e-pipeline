# Labeled data

`train.csv` (901 rows) and `test.csv` (227 rows) (published — see
repository root README's Data availability section) belong here. Columns:

    PDF_ID, Folder_Name, surya_ocr, Purpose, gpt_summary, gemini_summary,
    llama_summary, Institutional_Form

`Institutional_Form` is one of the four institutional-form categories
(Service Contract, Resource Sharing, Joint Operations, New Joint Entities).
`gpt_summary`, `gemini_summary`, and `llama_summary` are each model's own
contract summary text — generated per-model to match what each model's
embeddings (Stage A.5) are built from, so `src/03_classification/
chain_of_thought.py` and `embedding_classification.py` prompt/embed each
model with its own summary column, never another model's. Both scripts
rename `PDF_ID`/`Purpose`/`Institutional_Form` to `contract_id`/`purpose`/
`label` on load. `Folder_Name` and `surya_ocr` aren't used by either script.
The files are already split — no `split` column is needed.

Everything else this folder's scripts can write (`cot_predictions_*.csv`,
`cot_report.json`, `embedding_predictions_*.csv`,
`embedding_classification_report_*.json`, `embeddings_cache/`) is generated
output, not part of the published release.

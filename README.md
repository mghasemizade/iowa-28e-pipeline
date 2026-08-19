# Iowa 28E Pipeline

Computational pipeline for classifying Iowa 28E intergovernmental agreements by
institutional form and extracting the financial relationships encoded in
service contracts. Companion code for:

> Ghasemizade, M., Gutiérrez-Meave, R., Gramling, C., Chawla, A., Robinette, M.,
> Albrecht, K., & Lovato, J. *Making Local Government Contracts Legible: A
> Computational Pipeline for Classifying and Mapping Intergovernmental Service
> Agreements.* Government Information Quarterly (under review).

Full replication details for every stage (exact model versions, access dates,
prompts, thresholds) are documented in the paper's Supplementary Appendix.
Each folder below corresponds to one pipeline stage and one appendix
subsection.

Only two datasets are published alongside this code (see
[Data availability](#data-availability)) — the raw PDF agreements and the
full 21,629-document OCR corpus are not redistributed. Stages A.2, A.3, the
`--apply-full-corpus` pass of A.5, and A.6's re-extraction path all depend
on that non-public corpus and are included for methodological transparency
only; they are not runnable from the public release by themselves. Stages
A.4 (core train/CV/test path), A.5 (same), A.7, A.8, A.9, and A.10 run
end-to-end from the published files alone.

## Repository structure

```
iowa-28e-pipeline/
├── data/
│   ├── raw_pdfs/         # Input: raw 28E agreement PDFs (not included — see Data Availability)
│   ├── ocr_output/       # Stage A.2 output: OCR-extracted text (not included — see Data Availability)
│   ├── labeled/          # Published: train.csv + test.csv, hand-coded labels + per-model summaries (N = 1,128)
│   └── network/          # Published: extracted_financial_entities.csv; downstream network outputs
├── prompts/              # Verbatim prompts used at every LLM-calling stage
├── src/
│   ├── 01_ocr/                   # A.2 — OCR extraction (SURYA) — requires own PDF corpus
│   ├── 02_summarization/         # A.3 — LLM contract summarization — requires own PDF corpus
│   ├── 03_classification/        # A.4, A.5 — chain-of-thought + embedding classification
│   ├── 04_financial_extraction/  # A.6 — principal/agent/dollar-amount extraction — requires own PDF corpus
│   ├── 05_entity_disambiguation/ # A.7 — fuzzy agency-name matching
│   ├── 06_network_construction/  # A.8 — directed weighted network build
│   ├── 07_null_model/            # A.9 — configuration-model null
│   └── 08_sbm/                   # A.10 — stochastic block model
├── figures/               # Figure-generation scripts
└── notebooks/             # Exploratory / end-to-end run notebooks
```

## Setup

```bash
pip install -r requirements.txt
```

Fill in API keys for GPT, Gemini, and LLaMA access in a `.env` file (see
`.env.example`). Exact model version strings and access dates used in the
paper are recorded in each stage's script header — replace with current
versions if replicating at a later date, and note that results may shift as
providers update model weights. Note that LLaMA, GPT, and Gemini are used
through Stage A.5 (classification); Stage A.6 (financial extraction) uses
GPT only.

## Running the pipeline

Each `src/NN_stage/` folder is runnable independently, provided the previous
stage's output exists in `data/`. See the docstring at the top of each script
for exact inputs/outputs and the corresponding appendix subsection.

```bash
# Requires your own raw PDF corpus — see Data availability
python src/01_ocr/run_ocr.py
python src/02_summarization/summarize.py

# Runnable from the published data/labeled/train.csv + test.csv alone
python src/03_classification/chain_of_thought.py
python src/03_classification/embedding_classification.py

# Optional — only needed to regenerate data/labeled/embeddings/*.npy (e.g.
# against updated model versions); the published .npy files are the actual
# artifacts used in the paper, so embedding_classification.py above doesn't
# need this. Requires live API access (gpt/gemini) and, for llama, local
# GPU compute to run meta-llama/Llama-3.1-70B.
python src/03_classification/generate_embeddings.py

# Requires your own raw PDF corpus — see Data availability
python src/04_financial_extraction/extract_financial_entities.py

# Runnable from the published data/network/extracted_financial_entities.csv alone
python src/05_entity_disambiguation/disambiguate_entities.py
python src/06_network_construction/build_network.py
python src/07_null_model/configuration_model_null.py
python src/08_sbm/fit_sbm.py
python figures/plot_network_map.py   # Figure 1 — network map, $1,000 edge threshold
```

## Data availability

Three files (two datasets) are published alongside this code:

- `data/labeled/train.csv` (N = 901) and `data/labeled/test.csv` (N = 227)
  — the hand-coded classification dataset (N = 1,128 total): `PDF_ID,
  Folder_Name, surya_ocr, Purpose, gpt_summary, gemini_summary,
  llama_summary, Institutional_Form`. gpt_summary/gemini_summary/
  llama_summary are each model's own contract summary text (Stage A.3
  output, one column per summarizer) needed to run Stage A.4/A.5
  classification directly from these files — each classifying model uses
  its own summary column, not a shared one.
- `data/labeled/embeddings/{train,test}_{gpt,llama,gemini}.npy` — the
  embeddings Stage A.5 trains on, one row per contract in the same order as
  the corresponding CSV (gpt: float64, dim 1536 / text-embedding-3-small;
  gemini: float64, dim 3072 / gemini-embedding-001; llama: float32, dim
  8192 / local meta-llama/Llama-3.1-70B). Precomputed and published so
  Stage A.5 is runnable without live API access or local GPU compute. Each
  provider used a different source text, not a shared summary column: gpt
  embeds the full surya_ocr contract text (chunked/averaged over the token
  limit), gemini embeds Purpose + gemini_summary + Folder_Name, and llama
  embeds llama_summary via mean-pooled hidden states. The exact
  regeneration code (for methodological transparency, or to extend to new
  contracts) is `src/03_classification/generate_embeddings.py`.
  `--apply-full-corpus` uses the live-API path in `src/common/embeddings.py`
  instead, since no precomputed embeddings exist for the non-public full
  corpus — see that module's docstring for a caveat on the llama path
  there not matching the published embeddings' feature space.
- `data/network/extracted_financial_entities.csv` — the financial relationships
  extracted from every contract classified as a Service Contract (Stage A.6
  output): `PDF_ID, principal, agent, amount, principal_org_type,
  agent_org_type, org_type, filing_year, service_type, surya_ocr,
  text_summary, Purpose` (`PDF_ID` renamed to `contract_id` on load; see
  `data/network/README.md` for column notes). Stages A.7 onward (entity
  disambiguation, network construction, null model, SBM) run directly from
  this file.

Both are available at [DOI / repository link — TODO]. Raw PDF agreements are
publicly filed with the Iowa Secretary of State and are not redistributed
here; see `data/raw_pdfs/README.md` for acquisition instructions. The full
21,629-document OCR'd corpus and its per-document summaries are likewise not
redistributed — only the labeled subset and its extracted financial data
are.

## License

[TODO — specify license, e.g., MIT for code / CC-BY for data]

## Citation

```bibtex
@article{ghasemizade2026iowa28e,
  title   = {Making Local Government Contracts Legible: A Computational
             Pipeline for Classifying and Mapping Intergovernmental Service
             Agreements},
  author  = {Ghasemizade, Mohsen and Guti\'{e}rrez-Meave, Ra\'{u}l and
             Gramling, Cailin and Chawla, Aviral and Robinette, Michael and
             Albrecht, Kate and Lovato, Juniper},
  journal = {Government Information Quarterly},
  year    = {2026},
  note    = {Under review}
}
```

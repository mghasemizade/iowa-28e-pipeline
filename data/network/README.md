# Network data

`extracted_financial_entities.csv` (published — see repository root README's
Data availability section) belongs here. Columns:

    PDF_ID, principal, agent, amount, principal_org_type, agent_org_type,
    org_type, filing_year, service_type, surya_ocr, text_summary, Purpose

One row per contract classified as a Service Contract, extracted from the
full contract text via GPT (Stage A.6). `PDF_ID` is renamed to
`contract_id` on load by `src/05_entity_disambiguation/disambiguate_entities.py`.
`principal_org_type`/`agent_org_type` are each agency's real org type
(sourced upstream from an agency attribute file, not inferred from the
name) carried on every contract row; `org_type` is a leftover column from
the source file's own build and isn't used downstream. This is the only
input `disambiguate_entities.py` needs — everything from that stage onward
(canonical entity map, edge list, node metadata, null model, SBM) is
generated output, not part of the published release.

"""
Stage A.7 — Entity Disambiguation

Cleans and standardizes agency names extracted in Stage A.6 so that name
variants of the same agency (e.g., "City of Des Moines" / "Des Moines City" /
"Des Moines, IA") collapse to a single canonical node before network
construction.

Method: fuzzy string matching via RapidFuzz (`fuzz.token_sort_ratio`), no
normalization pass — raw names are matched directly. Each agency name is
matched against every OTHER name in the corpus (excluding itself — matching
against a candidate list that includes the query would trivially return a
perfect self-match every time, making the step a no-op) and mapped to that
best match only if it clears SIMILARITY_THRESHOLD *and* passes
`safe_canonical`'s guard that the two names' first tokens agree (catches
false merges like "Grundy County Sheriff's Office" -> "Hardin County
Sheriff's Office", which is what motivated the guard in the first place).
Names with no safe match map to themselves.

The top 100 most frequently appearing agencies are written to
canonical_entity_verification.csv for manual spot-checking (paper: "manually
verified the top 100 most frequently appearing agencies"); confirmed
corrections can be captured via --overrides and are applied last, after the
automated match, so they always win.


Input:  data/network/extracted_financial_entities.csv (published — this
        stage and everything downstream of it run end-to-end from the
        public release alone), columns: PDF_ID, principal, agent, amount,
        principal_org_type, agent_org_type, org_type, filing_year,
        service_type, surya_ocr, text_summary, Purpose. PDF_ID is renamed
        to contract_id on load; every other column passes through
        unchanged except amount, which is coerced to numeric.
Output: data/network/canonical_entity_map.json
        (raw name -> canonical name)
        data/network/canonical_entity_verification.csv
        (top 100 most-frequent raw names, their canonical mapping, and
        whether it changed — for the manual spot-check checkpoint)
        data/network/extracted_financial_entities_canonical.csv
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

INPUT_PATH = Path("data/network/extracted_financial_entities.csv")
CANONICAL_MAP_PATH = Path("data/network/canonical_entity_map.json")
VERIFICATION_PATH = Path("data/network/canonical_entity_verification.csv")
OUTPUT_PATH = Path("data/network/extracted_financial_entities_canonical.csv")

SIMILARITY_THRESHOLD = 87
SCORER = fuzz.token_sort_ratio


def safe_canonical(name_a: str, name_b: str, score: float, threshold: int = SIMILARITY_THRESHOLD) -> bool:
    """Reject a match if the first meaningful token differs (catches
    Grundy/Hardin-County-Sheriff's-Office type false merges) or either name
    is empty; otherwise accept if score clears threshold."""
    tokens_a = name_a.strip().split()
    tokens_b = name_b.strip().split()
    if not tokens_a or not tokens_b:
        return False
    if tokens_a[0].lower() != tokens_b[0].lower():
        return False
    return score >= threshold


def build_canonical_map(agency_names: list[str], threshold: int = SIMILARITY_THRESHOLD) -> dict:
    """Fuzzy-match each agency name against every other name and map it to
    the best safe_canonical-approved match; names with no safe match map to
    themselves."""
    counts = Counter(agency_names)
    all_names = list(counts.keys())

    canonical_map: dict[str, str] = {}
    for name in all_names:
        candidates = [n for n in all_names if n != name]
        match = process.extractOne(name, candidates, scorer=SCORER) if candidates else None
        if match is not None and safe_canonical(name, match[0], match[1], threshold):
            canonical_map[name] = match[0]
        else:
            canonical_map[name] = name  # no safe match — maps to itself
    return canonical_map


def build_verification_table(agency_names: list[str], canonical_map: dict, top_n: int = 100) -> pd.DataFrame:
    """Top-N most frequent raw names, their canonical mapping, and whether
    it changed — for manual spot-checking before trusting the network."""
    counts = Counter(agency_names)
    top_raw = [name for name, _ in counts.most_common(top_n)]
    return pd.DataFrame({
        "raw_name": top_raw,
        "canonical_name": [canonical_map[n] for n in top_raw],
        "freq": [counts[n] for n in top_raw],
    }).assign(changed=lambda d: d.raw_name != d.canonical_name)


def main():
    parser = argparse.ArgumentParser(description="Disambiguate agency entity names.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--threshold", type=int, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--canonical-map", type=Path, default=CANONICAL_MAP_PATH)
    parser.add_argument("--verification", type=Path, default=VERIFICATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--overrides", type=Path, default=None,
        help="Optional JSON file of {raw_name: canonical_name} manual corrections "
             "(e.g. from spot-checking canonical_entity_verification.csv). Applied "
             "after automatic matching, so overrides always win.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype={"PDF_ID": str}).rename(columns={"PDF_ID": "contract_id"})
    all_names = pd.concat([df["principal"], df["agent"]]).dropna().tolist()

    canonical_map = build_canonical_map(all_names, threshold=args.threshold)

    if args.overrides is not None:
        overrides = json.loads(args.overrides.read_text())
        canonical_map.update(overrides)
        print(f"Applied {len(overrides)} manual override(s) from {args.overrides}")

    args.canonical_map.write_text(json.dumps(canonical_map, indent=2))

    verification_df = build_verification_table(all_names, canonical_map)
    verification_df.to_csv(args.verification, index=False)
    n_changed = int(verification_df["changed"].sum())
    print(f"{n_changed}/{len(verification_df)} of the top {len(verification_df)} agencies were remapped "
          f"-> spot-check {args.verification} before trusting the network")

    df["principal"] = df["principal"].map(canonical_map).fillna(df["principal"])
    df["agent"] = df["agent"].map(canonical_map).fillna(df["agent"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df.to_csv(args.output, index=False)

    n_unique_canonical = len(set(canonical_map.values()))
    n_merged = len(canonical_map) - n_unique_canonical
    print(f"{len(canonical_map)} raw names -> {n_unique_canonical} canonical nodes ({n_merged} merged)")
    print(f"Clean rows retained: {len(df):,}")
    print(f"Unique canonical nodes: {pd.concat([df['principal'], df['agent']]).nunique():,}")


if __name__ == "__main__":
    main()

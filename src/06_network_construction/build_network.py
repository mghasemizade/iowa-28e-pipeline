"""
Stage A.8 — Network Construction

Builds the directed, weighted financial network from canonicalized
principal-agent-amount records: edges run principal -> agent, weighted by
summed dollar value across repeated pairs.

Result: 1,642 nodes, 3,234 directed edges (paper Table 7).

Org type (County / City / State Agency / Regional / Special District /
Other — the categories used in paper Table 6 and Figures 1-2) comes from
the input file's own principal_org_type/agent_org_type columns — each
agency's real org_type, sourced upstream from an agency attribute file
(e.g. a "Time Series Attribute File" of agency financials/administrative
data, mode-aggregated per agency; ask the repo owner for it if you need to
reproduce that join from scratch) and carried on every contract row
regardless of which role the agency plays. `infer_org_type`'s keyword
heuristic is only a fallback for agencies missing that real value.

Dollar thresholds ($1,000 for Figure 1, $10,000 for Figure 2) are applied at
plotting time, not here — this script's edge_list.csv is unfiltered so any
downstream threshold can be applied without re-running network construction.

service_type is assigned per PRINCIPAL — the mode of service_type across
*all* of that principal's contracts, not per (principal, agent) pair — so a
principal awarding contracts across several service categories still gets
one representative label on every one of its outgoing edges.

filing_year/year_max per edge are the min/max filing_year across that
(principal, agent) pair's contracts.

Input:  data/network/extracted_financial_entities_canonical.csv, columns
        (post Stage A.7 rename): contract_id, principal, agent, amount,
        principal_org_type, agent_org_type, org_type, filing_year,
        service_type, surya_ocr, text_summary, Purpose. Only principal,
        agent, amount, principal_org_type, agent_org_type, filing_year, and
        service_type are used here — org_type (see module docstring; not to
        be confused with principal_org_type/agent_org_type) is a leftover
        column from the source file's own build and isn't referenced.
Output: data/network/edge_list.csv
        (columns: principal, agent, weight, n_contracts, service_type,
         filing_year, year_max, principal_org_type, agent_org_type)
        data/network/node_metadata.csv
        (columns: node, org_type, out_degree, in_degree, total_out_dollars,
         total_in_dollars, dollar_volume)
"""

import argparse
import re
from pathlib import Path

import networkx as nx
import pandas as pd

INPUT_PATH = Path("data/network/extracted_financial_entities_canonical.csv")
EDGE_LIST_PATH = Path("data/network/edge_list.csv")
NODE_METADATA_PATH = Path("data/network/node_metadata.csv")

_ORG_TYPE_PATTERNS = [
    ("County", re.compile(r"\bcounty\b", re.IGNORECASE)),
    ("State Agency", re.compile(r"\b(state|department of|dept\.? of)\b", re.IGNORECASE)),
    ("Regional", re.compile(r"\bregional\b", re.IGNORECASE)),
    ("Special District", re.compile(r"\b(school district|fire district|special district|district)\b", re.IGNORECASE)),
    ("City", re.compile(r"\b(city|town|village)\b", re.IGNORECASE)),
]


def infer_org_type(agency_name: str) -> str:
    name = str(agency_name)
    for org_type, pattern in _ORG_TYPE_PATTERNS:
        if pattern.search(name):
            return org_type
    return "Other"


def _first_non_null_by(edge_df: pd.DataFrame, name_col: str, value_col: str) -> dict:
    """First non-null value_col per name_col — e.g. an agency's real
    org_type, which shouldn't vary by row even though it's stored
    per-contract as principal_org_type/agent_org_type."""

    def _first_non_null(series: pd.Series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else None

    return edge_df.groupby(name_col)[value_col].agg(_first_non_null).to_dict()


def build_directed_network(
    edge_df: pd.DataFrame, principal_org_lookup: dict, agent_org_lookup: dict
) -> nx.DiGraph:
    """Construct the directed, weighted principal -> agent network,
    aggregating repeated (principal, agent) pairs by summed dollar weight
    and contract count. service_type is assigned per principal (see module
    docstring), so it's computed before the (principal, agent) groupby."""
    edge_df = edge_df.dropna(subset=["principal", "agent"])
    edge_df = edge_df[edge_df["principal"] != edge_df["agent"]]  # drop self-contracts

    def _mode_or_first(series: pd.Series):
        modes = series.dropna().mode()
        return modes.iloc[0] if not modes.empty else None

    principal_service_type = edge_df.groupby("principal")["service_type"].agg(_mode_or_first)

    aggregated = (
        edge_df.groupby(["principal", "agent"])
        .agg(
            weight=("amount", lambda s: s.fillna(0).sum()),
            n_contracts=("amount", "size"),
            filing_year=("filing_year", "min"),
            year_max=("filing_year", "max"),
        )
        .reset_index()
    )
    aggregated["service_type"] = aggregated["principal"].map(principal_service_type)

    graph = nx.DiGraph()
    for _, row in aggregated.iterrows():
        graph.add_edge(
            row["principal"],
            row["agent"],
            weight=row["weight"],
            n_contracts=int(row["n_contracts"]),
            service_type=row["service_type"],
            filing_year=None if pd.isna(row["filing_year"]) else int(row["filing_year"]),
            year_max=None if pd.isna(row["year_max"]) else int(row["year_max"]),
        )
    for node in graph.nodes:
        graph.nodes[node]["org_type"] = (
            principal_org_lookup.get(node) or agent_org_lookup.get(node) or infer_org_type(node)
        )
    return graph, aggregated


def build_node_metadata(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node in graph.nodes:
        out_dollars = sum(d["weight"] for _, _, d in graph.out_edges(node, data=True))
        in_dollars = sum(d["weight"] for _, _, d in graph.in_edges(node, data=True))
        rows.append(
            {
                "node": node,
                "org_type": graph.nodes[node]["org_type"],
                "out_degree": graph.out_degree(node),
                "in_degree": graph.in_degree(node),
                "total_out_dollars": out_dollars,
                "total_in_dollars": in_dollars,
                "dollar_volume": out_dollars + in_dollars,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build the financial contracting network.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-edges", type=Path, default=EDGE_LIST_PATH)
    parser.add_argument("--output-nodes", type=Path, default=NODE_METADATA_PATH)
    args = parser.parse_args()

    args.output_edges.parent.mkdir(parents=True, exist_ok=True)

    edge_df = pd.read_csv(args.input, dtype={"contract_id": str})
    principal_org_lookup = _first_non_null_by(edge_df, "principal", "principal_org_type")
    agent_org_lookup = _first_non_null_by(edge_df, "agent", "agent_org_type")

    graph, aggregated = build_directed_network(edge_df, principal_org_lookup, agent_org_lookup)

    aggregated["principal_org_type"] = aggregated["principal"].map(
        lambda n: principal_org_lookup.get(n) or infer_org_type(n)
    )
    aggregated["agent_org_type"] = aggregated["agent"].map(
        lambda n: agent_org_lookup.get(n) or infer_org_type(n)
    )
    aggregated.to_csv(args.output_edges, index=False)

    node_metadata = build_node_metadata(graph)
    node_metadata.to_csv(args.output_nodes, index=False)

    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} directed edges -> {args.output_edges}")


if __name__ == "__main__":
    main()

"""
Stage A.10 — Stochastic Block Model

Fits a degree-corrected, weighted, nested stochastic block model (SBM) to
the full contracting network to recover latent block structure, using
graph-tool's minimum-description-length (MDL) model selection.

Result: 8 base-level blocks, nesting into 4 broader groups, into 1 trivial
top level (paper Section 5.6, Table 8, Figures 4-5).

The weighted covariate (dollar amount) and the multiplex service-category
covariate (MULTIPLEX_MAP below, collapsing the raw ~33 service_type values
into 4 broad categories) are fit jointly — both were used together in the
fit that produced the paper's 8-block/4-group result, not weight alone.

graph-tool's MCMC-based fitting is stochastic: re-running does not
reproduce the exact same blocks unless you pass --seed. --from-saved
reloads a specific prior fit exactly instead of re-fitting — pass it the
directory of a previously saved run (see `load_saved_sbm`) to reproduce the
paper's 8-block/4-group result bit-for-bit rather than re-running MCMC.


Input:  data/network/edge_list.csv, data/network/node_metadata.csv
        (or, with --from-saved: a directory containing iowa_28e_graph.gt,
        node_df.csv, node_idx.json, sbm_block_assignments.json, and
        optionally sbm_state.pkl — see `load_saved_sbm`)
Output: data/network/sbm_block_assignments.json
        data/network/sbm_block_characteristics.csv  (paper Table 8)
        figures/sbm_hierarchy_1.pdf (skipped under --from-saved if
        sbm_state.pkl isn't present/unpicklable — the hierarchy drawing
        needs the full NestedBlockState, not just flat block assignments)
        figures/sbm_flow_matrix.pdf
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EDGE_LIST_PATH = Path("data/network/edge_list.csv")
NODE_METADATA_PATH = Path("data/network/node_metadata.csv")
BLOCK_ASSIGNMENTS_PATH = Path("data/network/sbm_block_assignments.json")
BLOCK_CHARACTERISTICS_PATH = Path("data/network/sbm_block_characteristics.csv")
HIERARCHY_FIGURE_PATH = Path("figures/sbm_hierarchy.pdf")
FLOW_MATRIX_FIGURE_PATH = Path("figures/sbm_flow_matrix.pdf")
SBM_SAVE_DIR = Path("data/network/sbm_saved")

# Collapses the raw Iowa 28E service_type vocabulary (~33 values, including
# case variants and a couple of filing typos) into 4 broad multiplex
# categories — the discrete covariate used alongside dollar amount in the
# SBM fit.
MULTIPLEX_MAP = {
    # Public Safety
    "Police Protection": "Public Safety",
    "LAW ENFORCEMENT": "Public Safety",
    "Criminal Investigation": "Public Safety",
    "Jail and Corrections": "Public Safety",
    "Fire Response": "Public Safety",
    "Fire Reponse": "Public Safety",
    "FIRE SERVICES": "Public Safety",
    "Hazmat Response": "Public Safety",
    "Emergency Management": "Public Safety",
    # Infrastructure
    "Street and Road Systems": "Infrastructure",
    "Other Public Works": "Infrastructure",
    "Water System": "Infrastructure",
    "Sanitation": "Infrastructure",
    "Electrical and Energy Systems": "Infrastructure",
    "Facilities": "Infrastructure",
    "Airports": "Infrastructure",
    "Other Transportation Including Railroads": "Infrastructure",
    "TRANSPORTATION": "Infrastructure",
    "Public Transit": "Infrastructure",
    "Engineering": "Infrastructure",
    # Human Services
    "Health": "Human Services",
    "Education": "Human Services",
    "EDUCATION": "Human Services",
    "Parks and Recreation": "Human Services",
    "Other Neighborhood Services": "Human Services",
    "COMMUNITY AND NEIGHBORHOOD SERVICES": "Human Services",
    "Risk Management": "Human Services",
    # Administrative
    "Economic Development": "Administrative",
    "Finance and Tax Administration": "Administrative",
    "Information Services": "Administrative",
    "Motor Vehicles": "Administrative",
    "Planning": "Administrative",
}

UNKNOWN_TOKENS = {"unknown service type", "uknown service type"}  # covers the typo too


def map_category(service) -> str:
    """Collapse a raw service_type value into one of MULTIPLEX_MAP's 4
    categories, or "Unknown". Exact match first (fast, handles case
    variants), then a case-insensitive substring scan for near-variants not
    literally in the map."""
    s = str(service).strip()
    if s.lower() in UNKNOWN_TOKENS:
        return "Unknown"
    if s in MULTIPLEX_MAP:
        return MULTIPLEX_MAP[s]
    s_lower = s.lower()
    for key, category in MULTIPLEX_MAP.items():
        if key.lower() in s_lower:
            return category
    return "Unknown"


def load_saved_sbm(save_dir: Path):
    """Reload a previously fit-and-saved SBM instead of re-fitting from
    scratch. graph-tool's MCMC fit is stochastic, so this is the only way to
    exactly reproduce a specific prior run's blocks rather than a re-fit
    that merely tends to land on a similar partition.

    Expects, under save_dir:
        iowa_28e_graph.gt            — the graph, with block assignments
                                        embedded as vertex properties
                                        named block_L0, block_L1, ...
        node_df.csv                  — node metadata, indexed by node name
        node_idx.json                — {node name: vertex index}
        sbm_block_assignments.json   — human-readable block assignments
        sbm_state.pkl                — optional; the full NestedBlockState,
                                        only needed to resume inference or
                                        draw the hierarchy figure

    Returns (graph, node_df, node_idx, assignments, levels, state) where
    `state` is None if sbm_state.pkl is missing or fails to unpickle (e.g.
    graph-tool version mismatch) — block assignments are still usable via
    the .gt file and JSON even without it."""
    graph = gt.load_graph(str(save_dir / "iowa_28e_graph.gt"))
    block_level_keys = sorted(
        (k for k in graph.vp.keys() if k.startswith("block_L")),
        key=lambda k: int(k.removeprefix("block_L")),
    )
    print(f"Graph loaded: {graph.num_vertices()} nodes, {graph.num_edges()} edges")
    print(f"Block levels in graph: {block_level_keys}")

    # The saved graph carries the edge weight as "dollar_amount" (its
    # original property name — see the null-model script it came from);
    # characterize_blocks/plot_flow_matrix expect "weight", matching what a
    # fresh build_graph() produces. Alias rather than requiring callers to
    # know which naming a given .gt file uses.
    if "weight" not in graph.edge_properties and "dollar_amount" in graph.edge_properties:
        graph.edge_properties["weight"] = graph.edge_properties["dollar_amount"]

    node_df = pd.read_csv(save_dir / "node_df.csv", index_col=0)

    with open(save_dir / "node_idx.json") as f:
        node_idx = {n: int(i) for n, i in json.load(f).items()}

    with open(save_dir / "sbm_block_assignments.json") as f:
        assignments = json.load(f)

    state = None
    state_path = save_dir / "sbm_state.pkl"
    if state_path.exists():
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            print("SBM state reloaded (inference can be resumed).")
        except Exception as exc:  # noqa: BLE001 — best-effort, blocks are still usable without it
            print(f"[WARN] could not reload pickled state ({exc})")
            print("  Block assignments are intact via the .gt file and JSON — only live re-inference is unavailable.")

    levels = [graph.vertex_properties[k].a.copy().tolist() for k in block_level_keys]
    return graph, node_df, node_idx, assignments, levels, state


def build_graph(edge_df: pd.DataFrame, node_df: pd.DataFrame):
    n_before = len(edge_df)
    edge_df = edge_df[edge_df["weight"] > 0].copy()
    dropped = n_before - len(edge_df)
    if dropped:
        print(f"[WARN] dropped {dropped} zero/negative-weight edges before SBM fitting")

    edge_df["category"] = edge_df["service_type"].apply(map_category)
    print(edge_df["category"].value_counts())
    admin_mask = edge_df["category"] == "Administrative"
    print("\nRaw service_type values mapped to Administrative:")
    print(edge_df.loc[admin_mask, "service_type"].value_counts())

    graph = gt.Graph(directed=True)
    name_to_vertex = {}
    v_name = graph.new_vertex_property("string")
    for _, row in node_df.iterrows():
        vertex = graph.add_vertex()
        name_to_vertex[row["node"]] = vertex
        v_name[vertex] = row["node"]
    graph.vertex_properties["name"] = v_name

    categories = sorted(edge_df["category"].unique())
    cat2int = {c: i for i, c in enumerate(categories)}

    e_weight = graph.new_edge_property("double")
    e_category = graph.new_edge_property("int")
    for _, row in edge_df.iterrows():
        if row["principal"] not in name_to_vertex or row["agent"] not in name_to_vertex:
            continue
        edge = graph.add_edge(name_to_vertex[row["principal"]], name_to_vertex[row["agent"]])
        e_weight[edge] = float(row["weight"])
        e_category[edge] = cat2int[row["category"]]
    graph.edge_properties["weight"] = e_weight
    graph.edge_properties["category"] = e_category
    return graph, cat2int


def fit_nested_sbm(graph: gt.Graph):
    """Fit the degree-corrected nested weighted SBM on dollar amount
    (real-valued) and multiplex service category (discrete) covariates
    jointly — see module docstring."""
    state = gt.minimize_nested_blockmodel_dl(
        graph,
        state_args=dict(
            deg_corr=True,
            recs=[graph.ep.weight, graph.ep.category],
            rec_types=["real-exponential", "discrete-geometric"],
        ),
    )
    return state


def _vertex_block_hierarchy(state: gt.NestedBlockState, n_vertices: int) -> list[list[int]]:
    """levels[l][v] = block id of vertex v at nesting level l, composed
    bottom-up from graph-tool's per-level block-of-block arrays."""
    bs = [np.asarray(b) for b in state.get_bs()]
    levels = [np.asarray(bs[0])]
    for level_array in bs[1:]:
        candidate = level_array[levels[-1]]
        if len(set(candidate.tolist())) == 1:
            # A single top-level block isn't informative on its own — the
            # hierarchy naturally ends at the last level with >1 block.
            break
        levels.append(candidate)
    return [level.tolist() for level in levels]


def characterize_blocks(graph: gt.Graph, base_blocks: list[int], org_type_by_node: dict) -> pd.DataFrame:
    """Compute per-block agency count, total out/in dollar volume, net flow,
    principal/agent role, and dominant org_type (paper Table 8), using base
    (level-0) blocks."""
    v_name = graph.vertex_properties["name"]
    block_of = {int(v): base_blocks[int(v)] for v in graph.vertices()}
    agencies_per_block = pd.Series(block_of).value_counts()

    out_dollars = {b: 0.0 for b in agencies_per_block.index}
    in_dollars = {b: 0.0 for b in agencies_per_block.index}
    weight = graph.edge_properties["weight"]
    for e in graph.edges():
        src_block, dst_block = block_of[int(e.source())], block_of[int(e.target())]
        out_dollars[src_block] = out_dollars.get(src_block, 0.0) + weight[e]
        in_dollars[dst_block] = in_dollars.get(dst_block, 0.0) + weight[e]

    org_types_per_block = defaultdict(list)
    for v in graph.vertices():
        org_types_per_block[block_of[int(v)]].append(org_type_by_node.get(v_name[v], "Other"))

    rows = []
    for block, n_agencies in agencies_per_block.items():
        out_m = out_dollars.get(block, 0.0) / 1e6
        in_m = in_dollars.get(block, 0.0) / 1e6
        net_m = out_m - in_m
        org_counts = Counter(org_types_per_block[block])
        rows.append(
            {
                "block": block,
                "agencies": int(n_agencies),
                "out_musd": out_m,
                "in_musd": in_m,
                "net_musd": net_m,
                "role": "Principal" if net_m > 0 else "Agent",
                "dominant_org_type": org_counts.most_common(1)[0][0] if org_counts else "Other",
            }
        )
    return pd.DataFrame(rows).sort_values("block").reset_index(drop=True)


def plot_flow_matrix(graph: gt.Graph, base_blocks: list[int], out_path: Path):
    block_of = {int(v): base_blocks[int(v)] for v in graph.vertices()}
    blocks = sorted(set(block_of.values()))
    index = {b: i for i, b in enumerate(blocks)}
    flow = np.zeros((len(blocks), len(blocks)))
    weight = graph.edge_properties["weight"]
    for e in graph.edges():
        i, j = index[block_of[int(e.source())]], index[block_of[int(e.target())]]
        flow[i, j] += weight[e]

    flow_df = pd.DataFrame(np.log10(flow + 1), index=blocks, columns=blocks)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        flow_df, ax=ax, cmap="YlOrBr", linewidths=0.3, linecolor="white",
        cbar_kws={"label": "log₁₀($ flow + 1)"},
    )
    ax.set_xlabel("Agent Block", fontsize=14)
    ax.set_ylabel("Principal Block", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=3000, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Fit stochastic block model.")
    parser.add_argument("--edges", type=Path, default=EDGE_LIST_PATH)
    parser.add_argument("--nodes", type=Path, default=NODE_METADATA_PATH)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--from-saved", type=Path, default=None,
        help="Reload a previously fit SBM from this directory instead of re-fitting "
             "(see load_saved_sbm's docstring for expected contents) — reproduces a "
             "specific prior run exactly, since MCMC fitting is stochastic.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        gt.seed_rng(args.seed)
        np.random.seed(args.seed)

    cat2int = None
    if args.from_saved is not None:
        # node_idx isn't needed here (vertex order already matches the loaded
        # graph); load_saved_sbm still returns it for programmatic callers.
        graph, node_df, _node_idx, assignments, levels, state = load_saved_sbm(args.from_saved)
        # Prefer the graph's own org_type vertex property (set when the
        # graph was originally built) — node_df.csv may only carry $/block
        # columns, not org_type.
        if "org_type" in graph.vertex_properties:
            v_name, v_org = graph.vertex_properties["name"], graph.vertex_properties["org_type"]
            org_type_by_node = {v_name[v]: v_org[v] for v in graph.vertices()}
        elif "org_type" in node_df.columns:
            org_type_by_node = node_df["org_type"].to_dict()
        else:
            org_type_by_node = {}
    else:
        edge_df = pd.read_csv(args.edges)
        node_df = pd.read_csv(args.nodes)
        graph, cat2int = build_graph(edge_df, node_df)

        state = fit_nested_sbm(graph)
        levels = _vertex_block_hierarchy(state, graph.num_vertices())

        v_name = graph.vertex_properties["name"]
        assignments = {
            v_name[v]: {f"level_{i}_block": levels[i][int(v)] for i in range(len(levels))}
            for v in graph.vertices()
        }
        org_type_by_node = dict(zip(node_df["node"], node_df["org_type"]))

    base_blocks = levels[0]

    BLOCK_ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLOCK_ASSIGNMENTS_PATH.write_text(json.dumps(assignments, indent=2))
    for i in range(len(levels)):
        print(f"  Level {i} blocks: {len(set(levels[i]))}")

    if cat2int is not None:
        print("\nCategory encoding:")
        for cat, idx in cat2int.items():
            print(f"  {idx}: {cat}")

    characteristics = characterize_blocks(graph, base_blocks, org_type_by_node)
    characteristics.to_csv(BLOCK_CHARACTERISTICS_PATH, index=False)
    print(characteristics.to_string(index=False))

    if state is not None:
        HIERARCHY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state.draw(
            output=str(HIERARCHY_FIGURE_PATH),
            output_size=(24000, 24000),
            bg_color=(1, 1, 1, 1),  # white
        )
        print(f"Saved hierarchy figure -> {HIERARCHY_FIGURE_PATH}")
    else:
        print(f"[INFO] no SBM state available — skipping {HIERARCHY_FIGURE_PATH} (needs the full NestedBlockState)")

    plot_flow_matrix(graph, base_blocks, FLOW_MATRIX_FIGURE_PATH)
    print(f"Saved flow-matrix figure -> {FLOW_MATRIX_FIGURE_PATH}")


if __name__ == "__main__":
    main()

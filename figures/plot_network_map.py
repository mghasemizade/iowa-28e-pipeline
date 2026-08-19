"""
Figure 1 — Iowa 28E Financial Contracting Network Map

Renders the directed, weighted principal -> agent network built in Stage
A.8 (src/06_network_construction/build_network.py) as a node-link diagram,
filtered to edges >= THRESHOLD dollars (paper Figure 1 uses $1,000; Figure 2
uses $10,000 — see build_network.py's docstring, which applies dollar
thresholds at plotting time rather than at network-construction time).
Node color = org type, node size = weighted in+out degree ($ volume), edge
width = dollar weight. The TOP_N_LABELS highest-degree nodes are labeled via
adjustText to avoid overlap.

TODO (repo owner):
    - org_type here comes from build_network.py's node_metadata.csv, i.e.
      whatever infer_org_type produced (see that script's TODO about
      replacing the keyword heuristic with a real per-agency lookup).

Input:  data/network/edge_list.csv, data/network/node_metadata.csv
        (both from src/06_network_construction/build_network.py)
Output: figures/network.pdf
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from adjustText import adjust_text

EDGE_LIST_PATH = Path("data/network/edge_list.csv")
NODE_METADATA_PATH = Path("data/network/node_metadata.csv")
FIGURE_DIR = Path("figures")

# ── KNOBS — tune these ───────────────────────────────────────────────────────
THRESHOLD = 1_000
NODE_SIZE_MIN = 300
NODE_SIZE_MAX = 12_000
EDGE_WIDTH_MIN = 0.3
EDGE_WIDTH_MAX = 5.0
EDGE_ALPHA = 0.4
TOP_N_LABELS = 20
FIG_SIZE = (26, 22)
LABEL_FONTSIZE = 11.5

# Collapses build_network.py's org_type categories down to the six shown in
# paper Table 6 / network_map.pdf.
ORG_REMAP = {
    "Township": "Other",
    "Other": "Other",
    "Private or Nonprofit": "Other",
    "Fire District": "Special District",
    "Special District": "Special District",
    "School District": "Special District",
    "Regional or District": "Regional",
    "Regional": "Regional",
    "County": "County",
    "City": "City",
    "State Agency": "State Agency",
}

ORG_TYPES = ["County", "City", "State Agency", "Regional", "Special District", "Other"]

ORG_COLOR = {
    "County": "#01696f",
    "City": "#964219",
    "State Agency": "#4a90d9",
    "Regional": "#7b3fa0",
    "Special District": "#e07b00",
    "Other": "#bbbbbb",
}


def remap_type(raw) -> str:
    return ORG_REMAP.get(str(raw).strip(), "Other")


def load_graph(edge_list_path: Path) -> nx.DiGraph:
    edge_df = pd.read_csv(edge_list_path)
    graph = nx.DiGraph()
    for _, row in edge_df.iterrows():
        graph.add_edge(
            row["principal"], row["agent"],
            weight=row["weight"],
            n_contracts=row["n_contracts"],
            service_type=row.get("service_type"),
        )
    return graph


def plot_network_map(graph: nx.DiGraph, otype_lookup: dict, threshold: float, out_path: Path):
    g_fig = nx.DiGraph()
    for u, v, d in graph.edges(data=True):
        if d["weight"] >= threshold:
            g_fig.add_edge(u, v, **d)

    total_volume = sum(d["weight"] for _, _, d in graph.edges(data=True))
    vis_volume = sum(d["weight"] for _, _, d in g_fig.edges(data=True))
    print(
        f"Edges shown (>=${threshold:,}): {g_fig.number_of_edges():,} / {graph.number_of_edges():,} "
        f"({g_fig.number_of_edges() / graph.number_of_edges():.1%} of contracts, "
        f"{vis_volume / total_volume:.1%} of $ volume)"
    )
    print(f"Nodes shown: {g_fig.number_of_nodes():,}")

    pos = nx.kamada_kawai_layout(g_fig, weight="weight")

    degree_score = {
        n: g_fig.in_degree(n, weight="weight") + g_fig.out_degree(n, weight="weight")
        for n in g_fig.nodes
    }

    node_list = list(g_fig.nodes)
    node_colors = [ORG_COLOR.get(otype_lookup.get(n, "Other"), "#bbbbbb") for n in node_list]
    max_deg = max(degree_score.values()) if degree_score else 1
    node_sizes = [
        NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (degree_score[n] / max_deg)
        for n in node_list
    ]

    weights = [g_fig[u][v]["weight"] for u, v in g_fig.edges()]
    max_w = max(weights) if weights else 1
    widths = [EDGE_WIDTH_MIN + (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN) * (w / max_w) for w in weights]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_facecolor("#f7f6f2")
    fig.patch.set_facecolor("#f7f6f2")

    nx.draw_networkx_edges(
        g_fig, pos,
        width=widths,
        alpha=EDGE_ALPHA,
        edge_color="#666666",
        arrows=True,
        arrowsize=40,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.12",
        ax=ax,
        min_source_margin=12,
        min_target_margin=12,
    )

    nx.draw_networkx_nodes(
        g_fig, pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="white",
        ax=ax,
    )

    # Labels: top N nodes only.
    top_n = sorted(degree_score, key=degree_score.get, reverse=True)[:TOP_N_LABELS]
    texts = []
    for n in top_n:
        x, y = pos[n]
        texts.append(ax.text(
            x, y, n,
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
            color="#1a1a1a",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
            zorder=5,
        ))
    adjust_text(
        texts,
        x=[pos[n][0] for n in top_n],
        y=[pos[n][1] for n in top_n],
        ax=ax,
        expand_points=(3.0, 3.0),
        expand_text=(2.0, 2.0),
        force_points=1.0,
        force_text=0.8,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
    )

    # Legend (fixed order, matches Table 6 / network_map.pdf).
    seen_types = {otype_lookup.get(n, "Other") for n in g_fig.nodes}
    leg1 = ax.legend(
        handles=[mpatches.Patch(color=ORG_COLOR[t], label=t) for t in ORG_TYPES if t in seen_types],
        loc="lower right", fontsize=15, framealpha=0.95,
        title="Org Type", title_fontsize=18, edgecolor="#cccccc",
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=[
            ax.scatter([], [], c="#888888", s=size, edgecolors="white", linewidths=0.8, alpha=0.9, label=lbl)
            for lbl, size in [("Low", 80), ("Medium", 250), ("High", 550)]
        ],
        loc="lower left", fontsize=15, framealpha=0.95,
        title="Node Size = $ Volume", title_fontsize=18, edgecolor="#cccccc",
        labelspacing=1.3,
        handleheight=1.6,
        handletextpad=1.2,
        borderpad=0.8,
    )

    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render the Figure 1 network map.")
    parser.add_argument("--edge-list", type=Path, default=EDGE_LIST_PATH)
    parser.add_argument("--node-metadata", type=Path, default=NODE_METADATA_PATH)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--output", type=Path, default=FIGURE_DIR / "network.pdf")
    args = parser.parse_args()

    graph = load_graph(args.edge_list)
    node_df = pd.read_csv(args.node_metadata)
    otype_lookup = {row["node"]: remap_type(row["org_type"]) for _, row in node_df.iterrows()}

    plot_network_map(graph, otype_lookup, args.threshold, args.output)


if __name__ == "__main__":
    main()

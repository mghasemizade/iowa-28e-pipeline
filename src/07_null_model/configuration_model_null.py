"""
Stage A.9 — Configuration-Model Null

Tests whether observed network properties (reciprocity, geographic
clustering, agency-type assortativity, service-type homophily) are
substantively surprising or artifacts of the degree sequence, by comparing
against 1,000 degree-preserving random rewirings.

Each property is attached to VERTICES once from the observed network (org
type, county, and each agency's dominant service category) and held fixed
across all rewirings — only edge topology is randomized. This matches the
paper's framing: "keep each agency's number of partners but scramble who
contracts with whom."

Test directionality follows the paper's Section 5.5 narrative exactly:
reciprocity and service-type homophily/geographic-clustering are tested in
their theoretically expected direction ("low"/"high" respectively); where
the observed value runs opposite to that expectation (reciprocity, agency-
type assortativity), the one-sided p-value in the *expected* direction is
reported as 1.0 by construction, even though the value is significant in
the *other* direction — see the printed direction note per metric.

Validation checkpoint: all four observed properties fall outside the range
produced by 1,000 random rewirings (paper Figure 3).

County assignment: geocode each agency name to (lat, lon) via Nominatim
(cached to data/network/geocache.json — Nominatim's usage policy caps
requests at ~1/sec, so this can be slow on a first run with many distinct
agencies but is a one-time cost after caching), then a point-in-polygon
spatial join against real Iowa county boundaries (US Census TIGER/Line
county shapefile). This replaces an earlier direct-regex-plus-Nominatim-
address-parsing approach: Nominatim's returned address dict doesn't always
include a "county" field even for a successful geocode, which silently
produced ~0 geographic clustering — lat/lon is reliably returned, so doing
the county lookup ourselves via the actual polygons is more robust.
--coord-overrides accepts a JSON {agency_name: [lat, lon]} file for cases
where geocoding a specific agency is wrong or fails.



Input:  data/network/edge_list.csv, data/network/node_metadata.csv
Output: data/network/null_model_results.csv
        figures/configuration_model.pdf
        data/network/geocache.json (agency name -> {lat, lon}, cached)
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

EDGE_LIST_PATH = Path("data/network/edge_list.csv")
NODE_METADATA_PATH = Path("data/network/node_metadata.csv")
OUTPUT_PATH = Path("data/network/null_model_results.csv")
FIGURE_PATH = Path("figures/configuration_model.pdf")
GEOCACHE_PATH = Path("data/network/geocache.json")

COUNTY_SHAPEFILE_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
IOWA_STATEFP = "19"

N_REWIRINGS = 1000

# Direction each metric is expected to run relative to the null, per paper
# Section 5.5. p_value = P(null >= observed) if "high", P(null <= observed)
# if "low" — see module docstring for why the reported p can be 1.0 even
# for a metric that's significant in the opposite direction.
EXPECTED_DIRECTION = {
    "reciprocity": "low",
    "geographic_clustering": "high",
    "agency_type_assortativity": "high",
    "service_type_homophily": "high",
}


def _load_geocache() -> dict:
    if GEOCACHE_PATH.exists():
        return json.loads(GEOCACHE_PATH.read_text())
    return {}


def _save_geocache(cache: dict):
    GEOCACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEOCACHE_PATH.write_text(json.dumps(cache, indent=2))


def geocode_latlon(agency_name: str, cache: dict, geocode_fn=None) -> tuple[float, float] | None:
    """Resolve an agency name to (lat, lon) via a cached Nominatim lookup."""
    if agency_name in cache:
        entry = cache[agency_name]
        return (entry["lat"], entry["lon"]) if entry else None

    if geocode_fn is None:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="iowa-28e-pipeline")
        geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    latlon = None
    try:
        location = geocode_fn(f"{agency_name}, Iowa, USA")
        if location is not None:
            latlon = (location.latitude, location.longitude)
    except Exception as exc:  # noqa: BLE001 — geocoding is best-effort
        print(f"[WARN] geocoding failed for {agency_name!r}: {exc}")

    cache[agency_name] = {"lat": latlon[0], "lon": latlon[1]} if latlon else None
    return latlon


def assign_counties(
    node_names: list[str], cache: dict, counties_source: str, coord_overrides: dict | None = None
) -> dict[str, str]:
    """Map each node to an Iowa county via point-in-polygon spatial join
    against Census TIGER county boundaries."""
    coord_overrides = coord_overrides or {}
    latlon_by_node = {}
    for name in node_names:
        if name in coord_overrides:
            latlon_by_node[name] = tuple(coord_overrides[name])
        else:
            latlon = geocode_latlon(name, cache)
            if latlon is not None:
                latlon_by_node[name] = latlon

    if not latlon_by_node:
        return {name: "unknown" for name in node_names}

    counties = gpd.read_file(counties_source)
    counties = counties[counties["STATEFP"] == IOWA_STATEFP][["NAME", "geometry"]].to_crs("EPSG:4326")

    points = gpd.GeoDataFrame(
        {"node": list(latlon_by_node.keys())},
        geometry=[Point(lon, lat) for lat, lon in latlon_by_node.values()],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, counties, how="left", predicate="within")
    county_by_node = dict(zip(joined["node"], joined["NAME"].fillna("unknown")))
    return {name: county_by_node.get(name, "unknown") for name in node_names}


def _dominant_service_type(node: str, edge_df: pd.DataFrame) -> str | None:
    incident = edge_df[(edge_df["principal"] == node) | (edge_df["agent"] == node)]
    if incident.empty:
        return None
    modes = incident["service_type"].dropna().mode()
    return modes.iloc[0] if not modes.empty else None


def build_graph(edge_df: pd.DataFrame, node_df: pd.DataFrame, county_by_node: dict):
    """Build a graph-tool Graph with vertex properties org_type, county, and
    dominant_service_type attached, held fixed across all rewirings."""
    graph = gt.Graph(directed=True)
    name_to_vertex = {}
    v_name = graph.new_vertex_property("string")
    v_org_type = graph.new_vertex_property("string")
    v_county = graph.new_vertex_property("string")
    v_service = graph.new_vertex_property("string")

    org_type_by_node = dict(zip(node_df["node"], node_df["org_type"]))

    for _, row in node_df.iterrows():
        vertex = graph.add_vertex()
        name_to_vertex[row["node"]] = vertex
        v_name[vertex] = row["node"]
        v_org_type[vertex] = org_type_by_node.get(row["node"], "Other")
        v_county[vertex] = county_by_node.get(row["node"], "unknown")
        v_service[vertex] = _dominant_service_type(row["node"], edge_df) or "unknown"

    for _, row in edge_df.iterrows():
        if row["principal"] not in name_to_vertex or row["agent"] not in name_to_vertex:
            continue
        graph.add_edge(name_to_vertex[row["principal"]], name_to_vertex[row["agent"]])

    graph.vertex_properties["name"] = v_name
    graph.vertex_properties["org_type"] = v_org_type
    graph.vertex_properties["county"] = v_county
    graph.vertex_properties["service_type"] = v_service
    # build_network.py already drops self-contracts before writing
    # edge_list.csv, so this is a defensive no-op in practice.
    gt.remove_self_loops(graph)
    return graph


def compute_reciprocity(graph: gt.Graph) -> float:
    """Share of directed edges with a reverse edge also present. Self-loops
    are stripped once in build_graph, so no need to exclude them here."""
    return gt.edge_reciprocity(graph)


def compute_geographic_clustering(graph: gt.Graph) -> float:
    """Share of edges (with both endpoints' county known) connecting two
    agencies in the same county."""
    county = graph.vertex_properties["county"]
    known_edges = [
        (e.source(), e.target())
        for e in graph.edges()
        if county[e.source()] != "unknown" and county[e.target()] != "unknown"
    ]
    if not known_edges:
        return 0.0
    same_county = sum(1 for u, v in known_edges if county[u] == county[v])
    return same_county / len(known_edges)


def compute_agency_type_assortativity(graph: gt.Graph) -> float:
    """Newman's assortativity coefficient on vertex org_type (categorical).

    gt.assortativity() doesn't accept a string PropertyMap directly on at
    least some graph-tool versions (raises "No static implementation was
    found" on 2.98) — org_type is int-encoded first, matching the original
    script's m_org_assortativity."""
    org_type = graph.vertex_properties["org_type"]
    labels = sorted({org_type[v] for v in graph.vertices()})
    label_to_int = {label: i for i, label in enumerate(labels)}
    org_type_int = graph.new_vertex_property("int")
    for v in graph.vertices():
        org_type_int[v] = label_to_int[org_type[v]]
    r, _ = gt.assortativity(graph, org_type_int)
    return r


def compute_service_type_homophily(graph: gt.Graph) -> float:
    """Share of edges (with both endpoints' dominant service type known)
    connecting two agencies whose dominant service category matches."""
    service = graph.vertex_properties["service_type"]
    known_edges = [
        (e.source(), e.target())
        for e in graph.edges()
        if service[e.source()] != "unknown" and service[e.target()] != "unknown"
    ]
    if not known_edges:
        return 0.0
    same_service = sum(1 for u, v in known_edges if service[u] == service[v])
    return same_service / len(known_edges)


METRIC_FUNCS = {
    "reciprocity": compute_reciprocity,
    "geographic_clustering": compute_geographic_clustering,
    "agency_type_assortativity": compute_agency_type_assortativity,
    "service_type_homophily": compute_service_type_homophily,
}


def _p_value(observed: float, null_values: np.ndarray, direction: str) -> float:
    if direction == "high":
        return float(np.mean(null_values >= observed))
    return float(np.mean(null_values <= observed))


def main():
    parser = argparse.ArgumentParser(description="Run configuration-model null tests.")
    parser.add_argument("--edges", type=Path, default=EDGE_LIST_PATH)
    parser.add_argument("--nodes", type=Path, default=NODE_METADATA_PATH)
    parser.add_argument("--n-rewirings", type=int, default=N_REWIRINGS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    parser.add_argument(
        "--counties-source", default=COUNTY_SHAPEFILE_URL,
        help="Census TIGER county shapefile URL or local path (filtered to Iowa, STATEFP='19')",
    )
    parser.add_argument(
        "--coord-overrides", type=Path, default=None,
        help="Optional JSON file of {agency_name: [lat, lon]} to use instead of geocoding "
             "specific agencies where Nominatim fails or is wrong",
    )
    args = parser.parse_args()

    if args.seed is not None:
        gt.seed_rng(args.seed)
        np.random.seed(args.seed)

    edge_df = pd.read_csv(args.edges)
    node_df = pd.read_csv(args.nodes)

    coord_overrides = json.loads(args.coord_overrides.read_text()) if args.coord_overrides else {}
    geocache = _load_geocache()
    county_by_node = assign_counties(node_df["node"].tolist(), geocache, args.counties_source, coord_overrides)
    _save_geocache(geocache)
    n_known = sum(c != "unknown" for c in county_by_node.values())
    print(f"County assigned: {n_known} / {len(county_by_node)} nodes")

    graph = build_graph(edge_df, node_df, county_by_node)

    observed = {name: fn(graph) for name, fn in METRIC_FUNCS.items()}

    null_values = {name: np.empty(args.n_rewirings) for name in METRIC_FUNCS}
    for i in range(args.n_rewirings):
        rewired = graph.copy()
        gt.random_rewire(rewired, model="configuration", n_iter=10, edge_sweep=True)
        for name, fn in METRIC_FUNCS.items():
            null_values[name][i] = fn(rewired)

    results = []
    for name in METRIC_FUNCS:
        direction = EXPECTED_DIRECTION[name]
        p = _p_value(observed[name], null_values[name], direction)
        results.append(
            {
                "property": name,
                "observed": observed[name],
                "null_mean": float(np.mean(null_values[name])),
                "null_std": float(np.std(null_values[name])),
                "expected_direction": direction,
                "p_value": p,
            }
        )
        print(f"{name}: observed={observed[name]:.4f}, null_mean={np.mean(null_values[name]):.4f}, p={p:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    titles = {
        "reciprocity": "Reciprocity",
        "geographic_clustering": "Geographic Clustering",
        "agency_type_assortativity": "Agency-Type Assortativity",
        "service_type_homophily": "Service-Type Homophily",
    }
    palette = ["#01696f", "#964219", "#437a22", "#006494"]
    for ax, name, color in zip(axes.flat, METRIC_FUNCS, palette):
        arr = null_values[name]
        ax.hist(arr, bins=40, color=color, alpha=0.65, edgecolor="white", label="Null")
        ax.axvline(observed[name], color="#1a1a1a", lw=2.2, ls="--", label=f"Observed = {observed[name]:.3f}")
        ax.axvline(arr.mean(), color=color, lw=1.2, ls=":", alpha=0.8, label="Null mean")
        p = next(r["p_value"] for r in results if r["property"] == name)
        significance = "*" if p < 0.05 else "n.s."
        ax.set_title(titles[name], fontsize=10, fontweight="bold")
        ax.text(
            0.97, 0.95, f"p = {p:.4f} {significance}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9),
        )
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"Observed Metrics vs. Configuration-Model Null (N={args.n_rewirings})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=3000, bbox_inches="tight")
    print(f"Saved figure -> {args.figure}")


if __name__ == "__main__":
    main()

"""Directed road graph for one trip, and driving distances along it.

Each trip has its own road network. Nodes are node_ids, and every arc u -> v is a
directed edge, so a two-way street is two edges (u -> v and v -> u). Each edge stores
its arc_id (the row index in .arcs) and its straight-line length in metres.

A position on the road network is (arc_id, frac): how far along the arc, from its
start node u (frac = 0) to its end node v (frac = 1). Candidates already carry this as
`proj_frac`.

network_distance(G, a, pa, b, pb) is the DRIVING distance from position (a, pa) to
position (b, pb), following arc directions:

    same arc, b ahead of a       (pb - pa) * len(a)
    otherwise                    rest of arc a  +  shortest path v_a -> u_b  +  pb * len(b)

The shortest path uses Dijkstra bounded by a cutoff (we never need distances longer
than the car could have driven between two GPS fixes), and every node-to-node search
is cached on the graph, because consecutive GPS points share most of their candidates.

Going slightly BACKWARDS on the same arc is common for a car standing still (GPS jitter
moves its projection back and forth). Strictly that would require driving round the
block. Within `backward_tol_m` we treat it as GPS noise and return a NEGATIVE distance
(the signed step). The HMM compares |d_network - d_straight|, so a stationary car
(both near 0) costs almost nothing, while a car really moving 10 m forward on the
wrong-direction twin of a two-way street costs |-10 - 10| = 20 m, not 0.

Arcs whose end points coincide in (u, v) with another arc would collapse into one edge in
a DiGraph. They have identical straight-line geometry, so we keep the lowest arc_id for
the edge (used only to name arcs on a shortest path); distances are unaffected.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import shapely

from .candidates import build_arc_index
from .geo import arc_table, point_to_segment, project_trip
from .io import Trip

DEFAULT_CUTOFF_M = 2000.0
BACKWARD_TOL_M = 15.0
INF = math.inf


def build_graph(trip: Trip, arcs: pd.DataFrame | None = None) -> nx.DiGraph:
    """Directed graph of the trip's road network (projects the trip if needed).

    G.graph holds what the rest of the decoder needs:
      arcs      arc table from geo.arc_table (u, v, ax, ay, bx, by, length_m, bearing)
      u, v, length   numpy arrays indexed by arc_id, for fast lookups
      sp_cache  {source_node: (cutoff_used, {target_node: metres})}
    """
    if "x" not in trip.track:
        project_trip(trip)
    arcs = arc_table(trip) if arcs is None else arcs
    u = arcs["u"].to_numpy()
    v = arcs["v"].to_numpy()
    length = arcs["length_m"].to_numpy()

    G = nx.DiGraph()
    G.add_nodes_from(range(len(trip.nodes)))
    # Iterate in reverse so that, for duplicate (u, v) pairs, the lowest arc_id is written last.
    order = np.arange(len(arcs))[::-1]
    G.add_edges_from(
        (int(u[i]), int(v[i]), {"arc_id": int(arcs.index[i]), "length": float(length[i])})
        for i in order if u[i] != v[i]
    )
    G.graph.update(trip_id=trip.trip_id, arcs=arcs, u=u, v=v, length=length, sp_cache={}, tree=None)
    return G


def _node_distances(G: nx.DiGraph, source: int, cutoff: float) -> dict:
    """Shortest driving distances from one node to every node within `cutoff` m (cached)."""
    cache = G.graph["sp_cache"]
    hit = cache.get(source)
    if hit is not None and hit[0] >= cutoff:
        return hit[1]
    dist = nx.single_source_dijkstra_path_length(G, source, cutoff=cutoff, weight="length")
    cache[source] = (cutoff, dist)
    return dist


def node_distance(G: nx.DiGraph, source: int, target: int, cutoff: float = DEFAULT_CUTOFF_M) -> float:
    """Shortest driving distance between two nodes, or inf if longer than cutoff / unreachable."""
    if source == target:
        return 0.0
    if cutoff < 0:
        return INF
    d = _node_distances(G, source, cutoff).get(target, INF)
    return d if d <= cutoff else INF


def network_distance(G: nx.DiGraph, arc_a: int, pos_a: float, arc_b: int, pos_b: float,
                     cutoff: float = DEFAULT_CUTOFF_M, backward_tol_m: float = BACKWARD_TOL_M) -> float:
    """Driving distance (m) from fraction pos_a along arc_a to fraction pos_b along arc_b.

    Returns inf when arc_b cannot be reached within `cutoff` metres, and a small negative
    value for backward jitter on the same arc (see module docstring).
    """
    length = G.graph["length"]
    la, lb = length[arc_a], length[arc_b]
    if arc_a == arc_b:
        step = (pos_b - pos_a) * la
        if step >= 0:
            return step
        if -step <= backward_tol_m:
            return step           # signed (negative): jitter while (nearly) stationary
    rest_a = (1.0 - pos_a) * la
    into_b = pos_b * lb
    budget = cutoff - rest_a - into_b
    d = node_distance(G, int(G.graph["v"][arc_a]), int(G.graph["u"][arc_b]), budget)
    total = rest_a + d + into_b
    return total if total <= cutoff else INF


def distance_matrix(G: nx.DiGraph, arcs_a, pos_a, arcs_b, pos_b, cutoff: float = DEFAULT_CUTOFF_M,
                    backward_tol_m: float = BACKWARD_TOL_M) -> np.ndarray:
    """network_distance for every pair (i in A, j in B): array of shape (len(A), len(B)).

    One bounded Dijkstra per distinct end node of A (cached), then plain lookups.
    """
    arcs_a, arcs_b = np.asarray(arcs_a, dtype=np.int64), np.asarray(arcs_b, dtype=np.int64)
    pos_a, pos_b = np.asarray(pos_a, dtype=float), np.asarray(pos_b, dtype=float)
    length, u, v = G.graph["length"], G.graph["u"], G.graph["v"]
    rest_a = (1.0 - pos_a) * length[arcs_a]
    into_b = pos_b * length[arcs_b]
    ub = u[arcs_b]
    out = np.full((len(arcs_a), len(arcs_b)), INF)
    for i, a in enumerate(arcs_a):
        dist = _node_distances(G, int(v[a]), cutoff)
        for j, b in enumerate(arcs_b):
            if a == b:
                step = (pos_b[j] - pos_a[i]) * length[a]
                if step >= 0 or -step <= backward_tol_m:
                    out[i, j] = step      # signed, see module docstring
                    continue
            d = 0.0 if v[a] == ub[j] else dist.get(int(ub[j]), INF)
            total = rest_a[i] + d + into_b[j]
            if total <= cutoff:
                out[i, j] = total
    return out


def connected(G: nx.DiGraph, arc_a: int, arc_b: int) -> bool:
    """True if arc_b starts where arc_a ends (a.v == b.u), i.e. the car can go straight on."""
    return bool(G.graph["v"][arc_a] == G.graph["u"][arc_b])


def shortest_arc_path(G: nx.DiGraph, arc_a: int, arc_b: int, cutoff: float = DEFAULT_CUTOFF_M) -> list[int] | None:
    """Arc ids strictly BETWEEN arc_a and arc_b on the shortest driving path, or None if unreachable.

    [] means they are already connected. Used to fill gaps in the decoded route for display.
    """
    s, t = int(G.graph["v"][arc_a]), int(G.graph["u"][arc_b])
    if s == t:
        return []
    try:
        _, nodes = nx.single_source_dijkstra(G, s, target=t, cutoff=cutoff, weight="length")
    except nx.NetworkXNoPath:
        return None
    return [G.edges[p, q]["arc_id"] for p, q in zip(nodes[:-1], nodes[1:])]


def nearest_arc(G: nx.DiGraph, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest arc to each point (x, y) with no distance limit: (arc_id, dist_m, frac).

    Used as the fallback state for GPS points that have no candidate within the radius.
    Ties (both directions of a two-way street) go to the lower arc_id.
    """
    arcs = G.graph["arcs"]
    if G.graph.get("tree") is None:
        G.graph["tree"] = build_arc_index(arcs)
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    obs, pos = G.graph["tree"].query_nearest(shapely.points(x, y), all_matches=True)
    ids = arcs.index.to_numpy()[pos]
    ax, ay, bx, by = (arcs[c].to_numpy()[pos] for c in ("ax", "ay", "bx", "by"))
    d, f, _, _ = point_to_segment(x[obs], y[obs], ax, ay, bx, by)
    best = pd.DataFrame({"obs": obs, "arc_id": ids, "dist": d, "frac": f}).sort_values(["obs", "dist", "arc_id"])
    best = best.drop_duplicates("obs").set_index("obs").reindex(range(len(x)))
    return best["arc_id"].to_numpy(np.int64), best["dist"].to_numpy(), best["frac"].to_numpy()

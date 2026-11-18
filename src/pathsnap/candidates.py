"""Candidate generation: which road arcs could each GPS point be on?

For every GPS point we take all arcs within `radius_m`, then keep the `max_k` closest.
A road network can have 300k arcs, so we use a spatial index (shapely STRtree, an
R-tree over the arcs' bounding boxes): each lookup only touches nearby arcs instead
of all of them.

Arcs are directed, so a two-way street appears as TWO arcs (u->v and v->u) at exactly
the same distance. When ranking we give both the same rank, so the cut-off at `max_k`
never drops one direction and keeps the other. Telling them apart is the job of the
heading feature.

If a point's true arc is not in its candidate set, no model can recover it; that is
the "candidate miss rate". radius_m and max_k are chosen on the TRAINING trips in
scripts/radius_analysis.py.
"""
import numpy as np
import pandas as pd
import shapely

from .geo import arc_table, point_to_segment
from .io import Trip

RADIUS_M = 50.0   # chosen in scripts/radius_analysis.py
MAX_K = 16        # chosen in scripts/radius_analysis.py


def build_arc_index(arcs: pd.DataFrame) -> shapely.STRtree:
    """Spatial index over the arcs' straight-line geometries (arcs from geo.arc_table)."""
    coords = np.stack([arcs[["ax", "ay"]].to_numpy(), arcs[["bx", "by"]].to_numpy()], axis=1)  # (M, 2, 2)
    return shapely.STRtree(shapely.linestrings(coords))


def generate_candidates(trip: Trip, radius_m: float = RADIUS_M, max_k: int | None = MAX_K,
                        arcs: pd.DataFrame | None = None, tree: shapely.STRtree | None = None) -> pd.DataFrame:
    """All (GPS point, arc) pairs with the arc within radius_m, capped to the max_k closest per point.

    The trip must be projected. Returns columns: obs_idx, arc_id, dist_m, proj_frac, dist_rank.
    Points with no arc within radius_m get no rows.
    """
    arcs = arc_table(trip) if arcs is None else arcs
    tree = build_arc_index(arcs) if tree is None else tree
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()

    obs, arc_pos = tree.query(shapely.points(px, py), predicate="dwithin", distance=radius_m)
    ax, ay, bx, by = (arcs[c].to_numpy()[arc_pos] for c in ("ax", "ay", "bx", "by"))
    dist, frac, _, _ = point_to_segment(px[obs], py[obs], ax, ay, bx, by)

    cands = pd.DataFrame({
        "obs_idx": obs,
        "arc_id": arcs.index.to_numpy()[arc_pos],
        "dist_m": dist,
        "proj_frac": frac,
    })
    # Rank by distance within each point; twins (same distance to the mm) share a rank.
    cands["dist_rank"] = (cands.assign(d=cands["dist_m"].round(3))
                          .groupby("obs_idx")["d"].rank(method="min").astype(int) - 1)
    if max_k is not None:
        cands = cands[cands["dist_rank"] < max_k]
    return cands.sort_values(["obs_idx", "dist_m", "arc_id"]).reset_index(drop=True)

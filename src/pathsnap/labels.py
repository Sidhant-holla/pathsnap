"""Per-point training labels: which reference-route arc was each GPS point on?

The .route file lists the arcs the car drove, in order, but it does NOT say which GPS
point belongs to which arc. We infer that with a route-constrained alignment, a
Viterbi-style dynamic programme:

  * States are positions j = 0..M-1 in the route list (not raw arc ids, so a road
    driven twice in the trip has two separate positions).
  * Between consecutive GPS points the car can only stay at the same route position or
    move forward (j' >= j). It may skip several positions, because short arcs or GPS
    gaps can be passed without any point landing on them.
  * The cost of assigning point i to position j is the distance (m) from the point to
    that arc. We use the plain distance (not squared) so that a few wild GPS points
    can't drag the alignment around.
  * Starting later than position 0 or ending before the last position costs
    `edge_penalty_m` per skipped arc, so the alignment covers the whole route.

We minimise the total cost over all monotone assignments. The "only move forward"
rule is what fixes the failure cases of nearest-arc labelling: two-way streets (both
directions are at the same distance) and roads the car revisits later.

Points that still end up far from their arc are flagged `low_conf`: either the GPS fix
is bad, or the reference route doesn't match the track there.
"""
import numpy as np
import pandas as pd

from .geo import arc_table, point_to_segment
from .io import Trip

LOW_CONF_DIST_M = 50.0  # see scripts/radius_analysis.py for how this was chosen


def align_route(trip: Trip, edge_penalty_m: float = 5.0, low_conf_dist_m: float = LOW_CONF_DIST_M,
                chunk: int = 500) -> pd.DataFrame:
    """Assign every GPS point to one reference-route arc.

    The trip must be projected (geo.project_trip) and loaded with with_route=True.
    Returns one row per GPS point: obs_idx, route_pos, label_arc, label_dist_m, low_conf.
    """
    if trip.route is None:
        raise ValueError("align_route needs the reference route: load_trip(..., with_route=True)")
    route = np.asarray(trip.route)
    geom = arc_table(trip).loc[route]          # one row per route position (arcs may repeat)
    ax, ay = geom["ax"].to_numpy(), geom["ay"].to_numpy()
    bx, by = geom["bx"].to_numpy(), geom["by"].to_numpy()
    px, py = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()

    n, m = len(px), len(route)
    pos = np.arange(m)
    # backpointer[i, j] = route position of point i-1 on the best path that puts point i at j
    backpointer = np.empty((n, m), dtype=np.int16 if m < 32_000 else np.int32)

    prev_cost = None
    for start in range(0, n, chunk):                      # distances computed in chunks to save memory
        stop = min(n, start + chunk)
        dist, *_ = point_to_segment(px[start:stop, None], py[start:stop, None],
                                    ax[None, :], ay[None, :], bx[None, :], by[None, :])
        for k in range(stop - start):
            i = start + k
            if i == 0:
                cost = dist[k] + edge_penalty_m * pos     # starting at position j skips j arcs
                backpointer[0] = pos
            else:
                # Best predecessor for position j = cheapest of positions 0..j (stay or move forward).
                best = np.minimum.accumulate(prev_cost)
                best_idx = np.maximum.accumulate(np.where(prev_cost == best, pos, 0))
                cost = dist[k] + best
                backpointer[i] = best_idx
            prev_cost = cost

    # Ending at position j skips the remaining m-1-j arcs.
    j = int(np.argmin(prev_cost + edge_penalty_m * (m - 1 - pos)))
    path = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):                        # backtrack
        path[i] = j
        j = backpointer[i, j]

    label_dist, *_ = point_to_segment(px, py, ax[path], ay[path], bx[path], by[path])
    return pd.DataFrame({
        "obs_idx": np.arange(n),
        "route_pos": path,
        "label_arc": route[path],
        "label_dist_m": label_dist,
        "low_conf": label_dist > low_conf_dist_m,
    })

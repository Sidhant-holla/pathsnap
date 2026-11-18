"""Features for each (GPS point, candidate arc) pair.

Leakage rule: nothing in this file may use the reference route or labels. These
features are computed identically at training time and inside the live API.

Feature groups (every column starts with "f_"):

  Geometric       how close the point is to the arc, absolutely and relative to the
                  point's other candidates. Relative features matter because the final
                  decision is "which candidate wins for this point", not "is this one close".
  Directional     the car's heading vs the arc's bearing. This is the only thing that
                  separates the two directions of a two-way street (same distance).
  Neighbourhood   how well the arc also explains the points 1-3 seconds before and
                  after. One noisy fix can sit nearer the wrong road; its neighbours
                  usually don't.
  Topology        nearness to the arc's endpoints (junctions are where mistakes happen)
                  and how many roads meet there.
  GPS quality     time gaps around the point and how jittery the fix is.

The track is split into segments wherever the time gap exceeds GAP_S. Smoothing and
neighbour features never reach across a gap, because a point 40 minutes later says
nothing about the current road.
"""
import numpy as np
import pandas as pd

from .candidates import generate_candidates, build_arc_index
from .geo import angle_diff_deg, arc_table, bearing_deg, point_to_segment, project_trip
from .io import Trip

GAP_S = 10.0             # a time gap above this starts a new segment
HEADING_HALF_WINDOW = 2  # heading/speed from point i-2 to i+2
MIN_MOVE_M = 5.0         # below this displacement the heading is too noisy to trust
CONTEXT_OFFSETS = (-3, -2, -1, 1, 2, 3)
JITTER_HALF_WINDOW = 2
DT_CAP_S = 60.0

FEATURE_COLUMNS = [
    # geometric
    "f_dist", "f_dist_rank", "f_dist_gap", "f_n_cands", "f_proj_frac", "f_arc_len", "f_endpoint_dist",
    # directional
    "f_heading_diff", "f_heading_gap", "f_heading_reliable", "f_speed",
    # neighbourhood
    "f_ctx_dist", "f_ctx_gap",
    # topology
    "f_in_deg", "f_out_deg",
    # GPS quality
    "f_dt_prev", "f_dt_next", "f_jitter",
]


def segment_bounds(t: np.ndarray, gap_s: float = GAP_S):
    """For each point, the first and last index of its gap-free segment."""
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)])
    first = pd.Series(np.arange(len(t))).groupby(seg).transform("min").to_numpy()
    last = pd.Series(np.arange(len(t))).groupby(seg).transform("max").to_numpy()
    return seg, first, last


def point_features(trip: Trip) -> pd.DataFrame:
    """Per-GPS-point features (heading, speed, gaps, jitter). The trip must be projected."""
    x, y, t = (trip.track[c].to_numpy() for c in ("x", "y", "t"))
    n = len(t)
    idx = np.arange(n)
    seg, first, last = segment_bounds(t)

    # Heading and speed from a centred window, clamped to the segment.
    a = np.maximum(idx - HEADING_HALF_WINDOW, first)
    b = np.minimum(idx + HEADING_HALF_WINDOW, last)
    dx, dy, dt = x[b] - x[a], y[b] - y[a], t[b] - t[a]
    move = np.hypot(dx, dy)
    speed = np.divide(move, dt, out=np.zeros(n), where=dt > 0)

    # Jitter: distance from the point to the centred moving average of its neighbours.
    sx = pd.Series(x).groupby(seg).transform(lambda s: s.rolling(2 * JITTER_HALF_WINDOW + 1, center=True, min_periods=1).mean())
    sy = pd.Series(y).groupby(seg).transform(lambda s: s.rolling(2 * JITTER_HALF_WINDOW + 1, center=True, min_periods=1).mean())

    dt_prev = np.concatenate([[DT_CAP_S], np.diff(t)])
    dt_next = np.concatenate([np.diff(t), [DT_CAP_S]])
    return pd.DataFrame({
        "obs_idx": idx,
        "segment": seg,
        "seg_first": first,
        "seg_last": last,
        "heading": bearing_deg(dx, dy),
        "heading_reliable": move >= MIN_MOVE_M,
        "speed": speed,
        "dt_prev": np.minimum(dt_prev, DT_CAP_S),
        "dt_next": np.minimum(dt_next, DT_CAP_S),
        "jitter": np.hypot(x - sx.to_numpy(), y - sy.to_numpy()),
    })


def build_features(trip: Trip, cands: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    """Add f_* feature columns to a candidate table (from candidates.generate_candidates)."""
    pts = point_features(trip)
    df = cands.copy()
    i = df["obs_idx"].to_numpy()
    a = df["arc_id"].to_numpy()          # arc_id == row position in arcs (0..M-1)
    ax, ay, bx, by, length, bearing = (arcs[c].to_numpy()[a] for c in ("ax", "ay", "bx", "by", "length_m", "bearing"))
    grp = df.groupby("obs_idx")

    # Geometric
    df["f_dist"] = df["dist_m"]
    df["f_dist_rank"] = df["dist_rank"]
    df["f_dist_gap"] = df["dist_m"] - grp["dist_m"].transform("min")
    df["f_n_cands"] = grp["arc_id"].transform("size")
    df["f_proj_frac"] = df["proj_frac"]
    df["f_arc_len"] = length
    df["f_endpoint_dist"] = np.minimum(df["proj_frac"], 1 - df["proj_frac"]) * length

    # Directional: unreliable heading (car standing still) gets a neutral 90 degrees.
    reliable = pts["heading_reliable"].to_numpy()[i]
    hd = angle_diff_deg(pts["heading"].to_numpy()[i], bearing)
    df["f_heading_diff"] = np.where(reliable, hd, 90.0)
    df["f_heading_gap"] = df["f_heading_diff"] - df.groupby("obs_idx")["f_heading_diff"].transform("min")
    df["f_heading_reliable"] = reliable.astype(float)
    df["f_speed"] = pts["speed"].to_numpy()[i]

    # Neighbourhood: mean distance from the neighbouring points (same segment) to this arc.
    x, y = trip.track["x"].to_numpy(), trip.track["y"].to_numpy()
    first, last = pts["seg_first"].to_numpy()[i], pts["seg_last"].to_numpy()[i]
    total, count = np.zeros(len(df)), np.zeros(len(df))
    for off in CONTEXT_OFFSETS:
        j = i + off
        ok = (j >= first) & (j <= last)
        jj = np.clip(j, 0, len(x) - 1)
        d, *_ = point_to_segment(x[jj], y[jj], ax, ay, bx, by)
        total += np.where(ok, d, 0.0)
        count += ok
    df["f_ctx_dist"] = np.where(count > 0, total / np.maximum(count, 1), df["dist_m"])
    df["f_ctx_gap"] = df["f_ctx_dist"] - df.groupby("obs_idx")["f_ctx_dist"].transform("min")

    # Topology
    n_nodes = len(trip.nodes)
    in_deg = np.bincount(arcs["v"].to_numpy(), minlength=n_nodes)
    out_deg = np.bincount(arcs["u"].to_numpy(), minlength=n_nodes)
    df["f_in_deg"] = in_deg[arcs["u"].to_numpy()[a]]     # roads arriving where this arc starts
    df["f_out_deg"] = out_deg[arcs["v"].to_numpy()[a]]   # roads leaving where this arc ends

    # GPS quality
    df["f_dt_prev"] = pts["dt_prev"].to_numpy()[i]
    df["f_dt_next"] = pts["dt_next"].to_numpy()[i]
    df["f_jitter"] = pts["jitter"].to_numpy()[i]

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")
    return df


def trip_candidate_features(trip: Trip, radius_m: float | None = None, max_k: int | None = None) -> pd.DataFrame:
    """Full route-free feature pipeline for one trip: project -> candidates -> features.

    Used by both the training-table builder and the live API. Returns one row per
    (obs_idx, arc_id) with the candidate columns plus FEATURE_COLUMNS and t.
    """
    if "x" not in trip.track:
        project_trip(trip)
    arcs = arc_table(trip)
    kwargs = {k: v for k, v in (("radius_m", radius_m), ("max_k", max_k)) if v is not None}
    cands = generate_candidates(trip, arcs=arcs, tree=build_arc_index(arcs), **kwargs)
    feats = build_features(trip, cands, arcs)
    feats.insert(0, "t", trip.track["t"].to_numpy()[feats["obs_idx"].to_numpy()])
    return feats

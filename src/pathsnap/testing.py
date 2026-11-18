"""FAKE candidate and points tables with the real schema (PLAN §4, features.FEATURE_COLUMNS),
for testing the harness and model scripts. Never used for any reported result.

They mimic the real data's awkward cases:
  * 1-16 candidates per point, about 12% positives;
  * two-way streets: twin arcs (u->v and v->u) at exactly the same distance, separated
    only by the heading feature;
  * missed points (true arc not among the candidates, all labels 0);
  * points with no candidates at all (no candidate rows, but present in the points table);
  * low-confidence labels;
  * true arcs that form a connected chain, so the continuity metric can be checked.
"""
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def make_fake_table(n_trips: int = 20, points_per_trip=(150, 400), miss_rate: float = 0.08,
                    no_cand_rate: float = 0.02, seed: int = 0):
    """Returns (candidates, points, arcs_by_trip).

    arcs_by_trip[trip] is a DataFrame(u, v): arc 2k (2k+1 in odd trips) is node k -> k+1
    (the true chain), the other arc of the pair is its reverse twin, and arcs >= 2*N are
    unrelated distractor roads.
    """
    rng = np.random.default_rng(seed)
    rows, pts, arcs_by_trip = [], [], {}
    for t in range(n_trips):
        tid = f"{90000 + t:08d}"
        n = int(rng.integers(*points_per_trip))
        n_chain = n // 5 + 2
        n_distract = 200
        # The true chain is the even arcs in even trips and the odd arcs in odd trips, so the
        # "lower arc_id wins" tie-break does not systematically favour the true arc.
        flip = t % 2
        u = np.empty(2 * n_chain, dtype=np.int64)
        v = np.empty(2 * n_chain, dtype=np.int64)
        u[flip::2], v[flip::2] = np.arange(n_chain), np.arange(1, n_chain + 1)          # chain k -> k+1
        u[1 - flip::2], v[1 - flip::2] = np.arange(1, n_chain + 1), np.arange(n_chain)  # reverse twins
        u = np.r_[u, rng.integers(10_000, 20_000, n_distract)]
        v = np.r_[v, rng.integers(10_000, 20_000, n_distract)]
        arcs_by_trip[tid] = pd.DataFrame({"u": u, "v": v}).rename_axis("arc_id")
        speed = rng.uniform(3, 25)
        for i in range(n):
            true_arc = 2 * min(i // 5, n_chain - 1) + flip
            twin = true_arc + 1 - 2 * flip
            low_conf = bool(rng.random() < 0.03)
            if rng.random() < no_cand_rate:                    # no arc within radius
                pts.append((tid, i, float(i), true_arc, low_conf, 0, False))
                continue
            k = int(rng.integers(1, 12))
            distract = rng.choice(np.arange(2 * n_chain, 2 * n_chain + n_distract), size=k, replace=False)
            dist = rng.gamma(2.0, 9.0, size=k) + 3
            heading = rng.uniform(0, 180, size=k)
            arcs = list(distract)
            if rng.random() >= miss_rate:                      # true arc + its twin are candidates
                d_true = rng.gamma(1.5, 4.0) + (40 if low_conf else 0)
                h_true = abs(rng.normal(0, 15))
                arcs += [true_arc, twin]
                dist = np.r_[dist, d_true, d_true]
                heading = np.r_[heading, h_true, 180 - h_true]
            g = pd.DataFrame({"arc_id": arcs, "f_dist": np.round(dist, 3), "f_heading_diff": heading})
            g = g.nsmallest(16, "f_dist")
            m = len(g)
            g["f_dist_rank"] = g["f_dist"].rank(method="min").astype(int) - 1
            g["f_dist_gap"] = g["f_dist"] - g["f_dist"].min()
            g["f_n_cands"] = m
            g["f_proj_frac"] = rng.uniform(0, 1, m)
            g["f_arc_len"] = rng.gamma(2, 40, m)
            g["f_endpoint_dist"] = rng.gamma(2, 10, m)
            g["f_heading_gap"] = g["f_heading_diff"] - g["f_heading_diff"].min()
            g["f_heading_reliable"] = 1
            g["f_speed"] = speed + rng.normal(0, 1)
            g["f_ctx_dist"] = g["f_dist"] * rng.uniform(0.7, 1.3, m)
            g["f_ctx_gap"] = g["f_ctx_dist"] - g["f_ctx_dist"].min()
            g["f_in_deg"] = rng.integers(1, 4, m)
            g["f_out_deg"] = rng.integers(1, 4, m)
            g["f_dt_prev"] = 1.0
            g["f_dt_next"] = 1.0
            g["f_jitter"] = rng.gamma(1, 2)
            g["label"] = (g["arc_id"] == true_arc).astype("int8")
            g["label_low_conf"] = low_conf
            g["trip_id"], g["obs_idx"], g["t"] = tid, i, float(i)
            rows.append(g)
            pts.append((tid, i, float(i), true_arc, low_conf, m, bool(g["label"].any())))
    df = pd.concat(rows, ignore_index=True)
    df["arc_id"] = df["arc_id"].astype("int64")
    cols = ["trip_id", "obs_idx", "t", "arc_id", "label", "label_low_conf"] + FEATURE_COLUMNS
    points = pd.DataFrame(pts, columns=["trip_id", "obs_idx", "t", "label_arc", "low_conf", "n_cands", "true_in_cands"])
    return df[cols], points, arcs_by_trip

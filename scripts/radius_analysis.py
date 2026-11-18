"""Choose the candidate radius and the max candidates per point, using TRAINING trips only.

Run:  python -m scripts.radius_analysis

For every training GPS point we know its labelled arc (labels.align_route). We
generate candidates with a generous radius and no cap, then measure the
candidate miss rate (true arc not in the candidate set) for each (radius, max_k)
combination, alongside the average number of candidates per point (the cost side).

Outputs:
  experiments/results/candidates/radius_analysis.csv
  experiments/results/candidates/label_quality_per_trip.csv
  reports/figures/candidate_radius_tradeoff.png
  reports/figures/label_distance_hist.png
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap.candidates import generate_candidates
from pathsnap.config import FIGURES_DIR, RESULTS_DIR, TRAIN
from pathsnap.geo import arc_table, project_trip
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import LOW_CONF_DIST_M, align_route

MAX_RADIUS = 150.0
RADII = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150]
KS = [2, 4, 6, 8, 12, 16, 24, 32, None]


def main():
    out_dir = RESULTS_DIR / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    per_point, per_trip = [], []
    for tid in list_trips(TRAIN):
        trip = project_trip(load_trip(tid, TRAIN, with_route=True))
        arcs = arc_table(trip)
        lab = align_route(trip)
        cands = generate_candidates(trip, radius_m=MAX_RADIUS, max_k=None, arcs=arcs)
        hit = cands.merge(lab[["obs_idx", "label_arc"]], on="obs_idx")
        hit = hit[hit["arc_id"] == hit["label_arc"]][["obs_idx", "dist_rank"]]
        lab = lab.merge(hit, on="obs_idx", how="left")       # dist_rank NaN = not within MAX_RADIUS
        lab["n_within"] = lab["obs_idx"].map(cands.groupby("obs_idx").size()).fillna(0)
        # n candidates within each radius (for the cost curve)
        for r in RADII:
            lab[f"n_r{r}"] = lab["obs_idx"].map(cands[cands["dist_m"] <= r].groupby("obs_idx").size()).fillna(0)
        lab["trip_id"] = tid
        per_point.append(lab)
        per_trip.append({"trip_id": tid, "n_points": len(lab),
                         "median_label_dist_m": lab["label_dist_m"].median(),
                         "p95_label_dist_m": lab["label_dist_m"].quantile(0.95),
                         "low_conf_pct": 100 * lab["low_conf"].mean(),
                         "route_arcs_with_points_pct": 100 * lab["route_pos"].nunique() / len(trip.route)})
    df = pd.concat(per_point, ignore_index=True)
    pd.DataFrame(per_trip).to_csv(out_dir / "label_quality_per_trip.csv", index=False)

    conf = df[~df["low_conf"]]
    rows = []
    for r in RADII:
        for k in KS:
            for name, sub in (("all", df), ("confident", conf)):
                found = (sub["label_dist_m"] <= r) & sub["dist_rank"].notna()
                if k is not None:
                    found &= sub["dist_rank"] < k
                rows.append({"subset": name, "radius_m": r, "max_k": k if k is not None else "none",
                             "miss_rate_pct": 100 * (1 - found.mean()),
                             "mean_candidates": np.minimum(sub[f"n_r{r}"], k if k else np.inf).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "radius_analysis.csv", index=False)

    # --- Figures ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    d = df["label_dist_m"].clip(upper=100)
    ax.hist(d, bins=100, color="#3b6ea5")
    ax.axvline(LOW_CONF_DIST_M, color="crimson", ls="--", label=f"low-confidence threshold ({LOW_CONF_DIST_M:.0f} m)")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from GPS point to its labelled route arc (m, clipped at 100)")
    ax.set_ylabel("GPS points (log scale)")
    ax.set_title(f"Label distance, training trips (n={len(df):,})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "label_distance_hist.png", dpi=150)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in [4, 8, 16, 32, None]:
        s = res[(res["subset"] == "confident") & (res["max_k"] == (k if k is not None else "none"))]
        a1.plot(s["radius_m"], s["miss_rate_pct"], marker="o", label=f"max_k={k or 'no cap'}")
        a2.plot(s["radius_m"], s["mean_candidates"], marker="o", label=f"max_k={k or 'no cap'}")
    a1.set_yscale("log")
    a1.set_xlabel("Candidate radius (m)"); a1.set_ylabel("Candidate miss rate (%) - log scale")
    a1.set_title("Miss rate (training, confident points)")
    a2.set_xlabel("Candidate radius (m)"); a2.set_ylabel("Mean candidates per point")
    a2.set_title("Cost: candidates per point")
    for a in (a1, a2):
        a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_radius_tradeoff.png", dpi=150)

    print(f"Training points: {len(df):,}  low-confidence: {100 * df['low_conf'].mean():.2f}%")
    print("Label distance percentiles (confident):",
          {q: round(conf["label_dist_m"].quantile(q / 100), 1) for q in (50, 90, 95, 99, 99.9)})
    print("Rank of true arc among candidates (confident):",
          {q: conf["dist_rank"].quantile(q / 100) for q in (50, 90, 95, 99, 99.9)})
    pivot = res[res["subset"] == "confident"].pivot(index="radius_m", columns="max_k", values="miss_rate_pct")
    print("\nMiss rate % (confident points):\n", pivot.round(2).to_string())
    pivot = res[res["subset"] == "confident"].pivot(index="radius_m", columns="max_k", values="mean_candidates")
    print("\nMean candidates per point (confident points):\n", pivot.round(1).to_string())


if __name__ == "__main__":
    main()

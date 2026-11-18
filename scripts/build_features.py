"""Build the shared feature tables that every model trains and is evaluated on.

Run:  python -m scripts.build_features

Only the train and student_test partitions are processed. The evaluator partition is
deliberately refused here; it is only ever processed live by the API.

Outputs (in data/processed/, not committed, so rebuild them with this script):
  {partition}_candidates.parquet  one row per (trip_id, obs_idx, arc_id):
        trip_id, obs_idx, t, arc_id, label (1 = labelled arc), label_low_conf, f_* features
  {partition}_points.parquet      one row per GPS point (all points, even with no candidates):
        trip_id, obs_idx, t, label_arc, label_dist_m, low_conf, n_cands, true_in_cands
Use the points table as the denominator for Top-1 accuracy and candidate miss rate,
because points with no candidates have no rows in the candidates table.

Summary stats go to experiments/results/candidates/feature_build_summary.csv.
"""
import time

import pandas as pd

from pathsnap.config import DEV_PARTITIONS, PROCESSED_DIR, RESULTS_DIR
from pathsnap.features import FEATURE_COLUMNS, trip_candidate_features
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_route


def build_partition(partition: str):
    if partition not in DEV_PARTITIONS:
        raise ValueError("feature tables are only built for train and student_test")
    cand_parts, point_parts = [], []
    for tid in list_trips(partition):
        # 1) Features: computed WITHOUT the route (same code path as the API).
        trip = load_trip(tid, partition)
        feats = trip_candidate_features(trip)

        # 2) Labels: attached afterwards from the reference route.
        trip.route = load_trip(tid, partition, with_route=True).route
        lab = align_route(trip)
        lab_arc = lab["label_arc"].to_numpy()
        feats["label"] = (feats["arc_id"].to_numpy() == lab_arc[feats["obs_idx"].to_numpy()]).astype("int8")
        feats["label_low_conf"] = lab["low_conf"].to_numpy()[feats["obs_idx"].to_numpy()]
        feats.insert(0, "trip_id", tid)

        pts = lab[["obs_idx", "label_arc", "label_dist_m", "low_conf"]].copy()
        pts.insert(0, "trip_id", tid)
        pts.insert(2, "t", trip.track["t"].to_numpy())
        pts["n_cands"] = pts["obs_idx"].map(feats.groupby("obs_idx").size()).fillna(0).astype(int)
        pts["true_in_cands"] = pts["obs_idx"].isin(feats.loc[feats["label"] == 1, "obs_idx"])

        cand_parts.append(feats[["trip_id", "obs_idx", "t", "arc_id", "label", "label_low_conf"] + FEATURE_COLUMNS])
        point_parts.append(pts)
    return pd.concat(cand_parts, ignore_index=True), pd.concat(point_parts, ignore_index=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for partition in DEV_PARTITIONS:
        t0 = time.time()
        cands, points = build_partition(partition)
        cands.to_parquet(PROCESSED_DIR / f"{partition}_candidates.parquet", index=False)
        points.to_parquet(PROCESSED_DIR / f"{partition}_points.parquet", index=False)
        conf = points[~points["low_conf"]]
        row = {
            "partition": partition,
            "trips": points["trip_id"].nunique(),
            "points": len(points),
            "low_conf_pct": round(100 * points["low_conf"].mean(), 2),
            "candidate_rows": len(cands),
            "mean_cands_per_point": round(points["n_cands"].mean(), 2),
            "positive_rate_pct": round(100 * cands["label"].mean(), 2),
            "miss_rate_all_pct": round(100 * (1 - points["true_in_cands"].mean()), 2),
            "miss_rate_confident_pct": round(100 * (1 - conf["true_in_cands"].mean()), 2),
            "seconds": round(time.time() - t0, 1),
        }
        summary.append(row)
        print(row)
    out = RESULTS_DIR / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "feature_build_summary.csv", index=False)


if __name__ == "__main__":
    main()

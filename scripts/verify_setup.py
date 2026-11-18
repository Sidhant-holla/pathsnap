"""Check that your environment and dataset are set up correctly.

Run from the repo root:  python -m scripts.verify_setup

It loads every train/student-test trip, checks the file structure, and verifies
that the UTM projection agrees with exact WGS84 geodesic distances. It never touches the
evaluator partition.
"""
import time

import numpy as np

from pathsnap.config import DATA_ROOT, DEV_PARTITIONS, STUDENT_TEST, TRAIN
from pathsnap.geo import arc_table, geodesic_m, point_to_segment, project_trip
from pathsnap.io import list_trips, load_trip


def main():
    print(f"Data root: {DATA_ROOT}")
    counts = {p: len(list_trips(p)) for p in DEV_PARTITIONS}
    print(f"Trips: {counts}")
    assert counts[TRAIN] == 80 and counts[STUDENT_TEST] == 10, "expected 80 train / 10 student_test trips"

    t0 = time.time()
    n_obs = 0
    for p in DEV_PARTITIONS:
        for tid in list_trips(p):
            trip = load_trip(tid, p)
            assert trip.route is None, "route must not load by default"
            assert trip.track["t"].is_monotonic_increasing
            assert trip.arcs.to_numpy().max() < len(trip.nodes)
            n_obs += len(trip.track)
    print(f"Loaded all {sum(counts.values())} trips ({n_obs:,} GPS points) in {time.time() - t0:.1f}s")

    # Projection check on one trip: projected distances vs exact WGS84 geodesic distances.
    # We compare points 50 steps apart, because 1 s steps are dominated by the files'
    # 6-decimal coordinate rounding (~0.1 m).
    trip = project_trip(load_trip(list_trips(TRAIN)[0], TRAIN))
    tr = trip.track
    i = np.arange(0, len(tr) - 50, 10)
    j = i + 50
    d_proj = np.hypot(tr["x"].values[j] - tr["x"].values[i], tr["y"].values[j] - tr["y"].values[i])
    d_geo = geodesic_m(tr["lon"].values[i], tr["lat"].values[i], tr["lon"].values[j], tr["lat"].values[j])
    mask = d_geo > 100
    rel_err = np.abs(d_proj[mask] - d_geo[mask]) / d_geo[mask]
    print(f"Trip {trip.trip_id} -> {trip.crs}: max projection error vs WGS84 geodesic = {rel_err.max() * 100:.3f}%")
    assert rel_err.max() < 0.001  # UTM's scale error is at most ~0.04% within a zone

    arcs = arc_table(trip)
    d, frac, *_ = point_to_segment(tr["x"].values[0], tr["y"].values[0],
                                   arcs["ax"].values, arcs["ay"].values, arcs["bx"].values, arcs["by"].values)
    print(f"Nearest arc to first GPS point: arc {arcs.index[d.argmin()]} at {d.min():.1f} m")
    print("Setup OK")


if __name__ == "__main__":
    main()

"""Tests for the data pipeline: geometry, route-label alignment, candidates and features.

Most tests use a small synthetic road network built directly in metres (UTM zone 32N,
near Darmstadt), so the correct answer is known exactly:

        N2 (300,200)
         |
        N1 (300,100)            north road (two-way)
         |
    A0 --- A1 --- A2 --- A3     east-west road (two-way), nodes every 100 m at y = 0
    B0 --- B1 --- B2 --- B3     parallel road 60 m south (two-way), a distractor

Every road is two-way, so each segment exists as TWO arcs (u->v and v->u) at exactly
the same distance from any GPS point. Only the direction of travel can separate them.
"""
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from pathsnap.candidates import generate_candidates
from pathsnap.config import DATA_ROOT, TRAIN
from pathsnap.features import FEATURE_COLUMNS, build_features, segment_bounds, trip_candidate_features
from pathsnap.geo import arc_table, point_to_segment, project_trip, utm_epsg
from pathsnap.io import Trip, list_trips, load_trip
from pathsnap.labels import align_route

ORIGIN = (475_000.0, 5_524_000.0)       # metres in EPSG:32632 (about 8.65 E, 49.87 N)
TO_LONLAT = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

NODES_M = {
    "A0": (0, 0), "A1": (100, 0), "A2": (200, 0), "A3": (300, 0),
    "N1": (300, 100), "N2": (300, 200),
    "B0": (0, -60), "B1": (100, -60), "B2": (200, -60), "B3": (300, -60),
}
ROADS = [("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "N1"), ("N1", "N2"),
         ("B0", "B1"), ("B1", "B2"), ("B2", "B3")]


def _lonlat(x, y):
    return TO_LONLAT.transform(ORIGIN[0] + np.asarray(x, float), ORIGIN[1] + np.asarray(y, float))


def make_network():
    names = list(NODES_M)
    idx = {n: i for i, n in enumerate(names)}
    lon, lat = _lonlat([NODES_M[n][0] for n in names], [NODES_M[n][1] for n in names])
    nodes = pd.DataFrame({"lon": lon, "lat": lat})
    nodes.index.name = "node_id"
    arcs, arc_id = [], {}
    for a, b in ROADS:                       # both directions of every road
        for u, v in ((a, b), (b, a)):
            arc_id[(u, v)] = len(arcs)
            arcs.append((idx[u], idx[v]))
    arcs = pd.DataFrame(arcs, columns=["u", "v"])
    arcs.index.name = "arc_id"
    return nodes, arcs, arc_id


def make_trip(path_m, route_names, noise_m=2.0, seed=0, t=None):
    """GPS track along a polyline (metres), 1 point per 10 m (10 m/s at 1 Hz), plus noise."""
    rng = np.random.default_rng(seed)
    nodes, arcs, arc_id = make_network()
    pts = []
    for (x0, y0), (x1, y1) in zip(path_m[:-1], path_m[1:]):
        n = int(np.hypot(x1 - x0, y1 - y0) // 10)
        for k in range(n):
            pts.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
    pts = np.array(pts) + rng.normal(0, noise_m, size=(len(pts), 2))
    lon, lat = _lonlat(pts[:, 0], pts[:, 1])
    t = np.arange(len(pts), dtype=float) if t is None else t
    track = pd.DataFrame({"lon": lon, "lat": lat, "t": t})
    route = [arc_id[r] for r in route_names]
    return Trip("synthetic", TRAIN, track, nodes, arcs, route), arc_id


EAST_THEN_NORTH = [(5, 0), (300, 0), (300, 195)]
EAST_THEN_NORTH_ROUTE = [("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "N1"), ("N1", "N2")]


# --------------------------------------------------------------------------- geometry

def test_utm_zone_selection():
    assert utm_epsg(8.6, 49.9) == 32632        # Darmstadt
    assert utm_epsg(-58.4, -34.6) == 32721     # southern hemisphere
    assert utm_epsg(106.5, 15.4) == 32648


def test_point_to_segment():
    d, f, qx, qy = point_to_segment(5.0, 5.0, 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx(5.0) and f == pytest.approx(0.5) and (qx, qy) == pytest.approx((5.0, 0.0))
    d, f, *_ = point_to_segment(-3.0, 4.0, 0.0, 0.0, 10.0, 0.0)   # beyond the start: clamped to A
    assert d == pytest.approx(5.0) and f == 0.0
    d, f, *_ = point_to_segment(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)     # zero-length arc behaves like a point
    assert d == pytest.approx(5.0)


def test_projection_recovers_metres():
    trip, _ = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    project_trip(trip)
    assert trip.crs == "EPSG:32632"
    expected = np.array([NODES_M[n] for n in NODES_M]) + np.array(ORIGIN)
    assert np.allclose(trip.nodes[["x", "y"]].to_numpy(), expected, atol=0.01)


# --------------------------------------------------------------------------- labels

def test_alignment_follows_route_in_order():
    trip, arc_id = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    lab = align_route(project_trip(trip))
    assert (np.diff(lab["route_pos"]) >= 0).all()                  # only stays or moves forward
    assert lab["label_dist_m"].max() < 15
    y = trip.track["y"].to_numpy() - ORIGIN[1]
    x = trip.track["x"].to_numpy() - ORIGIN[0]
    on_east = (x > 20) & (x < 280) & (np.abs(y) < 10)
    westbound = {arc_id[("A1", "A0")], arc_id[("A2", "A1")], arc_id[("A3", "A2")]}
    assert not lab.loc[on_east, "label_arc"].isin(westbound).any()  # never the reverse twin
    assert (lab.loc[y > 120, "label_arc"] == arc_id[("N1", "N2")]).all()


def test_alignment_handles_u_turn_on_two_way_street():
    """Drive east to A3, turn round, drive back west on the same road.

    Both directions sit at exactly the same distance, so on the segment where the car
    turns round, distance alone is a tie and the alignment may switch to the westbound
    twin too early. The heading term (heading_weight_m) breaks that tie by making it
    cost extra to be labelled on an arc pointing against the direction of travel."""
    path = [(5, 0), (300, 0), (5, 0)]
    route = [("A0", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "A2"), ("A2", "A1"), ("A1", "A0")]
    trip, arc_id = make_trip(path, route, noise_m=1.0)
    lab = align_route(project_trip(trip), heading_weight_m=10.0)
    n = len(lab)
    first, second = lab.iloc[: n // 2 - 3], lab.iloc[n // 2 + 3:]
    east = {arc_id[r] for r in route[:3]}
    west = {arc_id[r] for r in route[3:]}
    assert first["label_arc"].isin(east).all()
    assert second["label_arc"].isin(west).all()


def test_alignment_needs_route():
    trip, _ = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    trip.route = None
    with pytest.raises(ValueError):
        align_route(project_trip(trip))


# --------------------------------------------------------------------------- candidates

def test_candidates_radius_twins_and_cap():
    trip, arc_id = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    project_trip(trip)
    cands = generate_candidates(trip, radius_m=50, max_k=None)
    assert (cands["dist_m"] <= 50 + 1e-6).all()
    # A point in the middle of A1-A2: both directions are candidates and share rank 0.
    x = trip.track["x"].to_numpy() - ORIGIN[0]
    mid = int(np.argmin(np.abs(x - 150)))
    c = cands[cands["obs_idx"] == mid].set_index("arc_id")
    assert c.loc[arc_id[("A1", "A2")], "dist_rank"] == c.loc[arc_id[("A2", "A1")], "dist_rank"] == 0
    # The parallel road is 60 m away, so it's outside a 50 m radius but inside 80 m.
    assert arc_id[("B1", "B2")] not in c.index
    assert arc_id[("B1", "B2")] in generate_candidates(trip, radius_m=80, max_k=None).query("obs_idx == @mid")["arc_id"].values
    # Cap: ranks below max_k only (twins can make a point exceed max_k rows).
    capped = generate_candidates(trip, radius_m=200, max_k=2)
    assert (capped["dist_rank"] < 2).all()


# --------------------------------------------------------------------------- features

def test_heading_feature_separates_two_way_twins():
    trip, arc_id = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE, noise_m=1.0)
    feats = trip_candidate_features(trip)
    x = trip.track["x"].to_numpy() - ORIGIN[0]
    mid = int(np.argmin(np.abs(x - 150)))
    f = feats[feats["obs_idx"] == mid].set_index("arc_id")
    assert f.loc[arc_id[("A1", "A2")], "f_heading_diff"] < 20       # driving east on the eastbound arc
    assert f.loc[arc_id[("A2", "A1")], "f_heading_diff"] > 160      # its westbound twin
    assert f.loc[arc_id[("A1", "A2")], "f_dist"] == pytest.approx(f.loc[arc_id[("A2", "A1")], "f_dist"])


def test_features_complete_and_finite():
    trip, _ = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    feats = trip_candidate_features(trip)
    assert all(c in feats for c in FEATURE_COLUMNS)
    assert np.isfinite(feats[FEATURE_COLUMNS].to_numpy()).all()


def test_features_do_not_depend_on_the_route():
    """Leakage guard: features must be identical whether or not the route is loaded."""
    with_route, _ = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    without_route, _ = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    without_route.route = None
    pd.testing.assert_frame_equal(trip_candidate_features(with_route), trip_candidate_features(without_route))


def test_segments_split_at_time_gaps():
    t = np.array([0, 1, 2, 3, 40, 41, 42, 200], dtype=float)
    seg, first, last = segment_bounds(t, gap_s=10)
    assert seg.tolist() == [0, 0, 0, 0, 1, 1, 1, 2]
    assert first.tolist() == [0, 0, 0, 0, 4, 4, 4, 7]
    assert last.tolist() == [3, 3, 3, 3, 6, 6, 6, 7]


def test_context_feature_does_not_cross_a_gap():
    """A point right after a long gap must not borrow neighbours from before the gap."""
    path = [(5, 0), (300, 0), (300, 195)]
    trip, _ = make_trip(path, EAST_THEN_NORTH_ROUTE)
    n = len(trip.track)
    t = np.arange(n, dtype=float)
    t[n // 2:] += 3600                           # one-hour gap in the middle
    trip.track["t"] = t
    project_trip(trip)
    arcs = arc_table(trip)
    cands = generate_candidates(trip, arcs=arcs)
    feats = build_features(trip, cands, arcs)
    after_gap = feats[feats["obs_idx"] == n // 2]
    assert (after_gap["f_dt_prev"] == 60).all()  # capped gap length
    assert np.isfinite(after_gap["f_ctx_dist"]).all()


# --------------------------------------------------------------------------- real data (skipped without it)

@pytest.mark.skipif(not (DATA_ROOT / TRAIN).exists(), reason="raw dataset not available")
def test_loader_does_not_load_route_by_default():
    tid = list_trips(TRAIN)[0]
    assert load_trip(tid).route is None
    assert len(load_trip(tid, with_route=True).route) > 0


def test_partition_folder_aliases(tmp_path):
    """The evaluator folder name is only known at the demo, so common variants must work."""
    from pathsnap.io import partition_dir
    for name in ("Evaluator", "train"):
        (tmp_path / name / "00000123").mkdir(parents=True)
    assert list_trips("evaluator_test", data_root=tmp_path) == ["00000123"]
    assert partition_dir("evaluator_test", tmp_path).name == "Evaluator"
    assert list_trips("student_test", data_root=tmp_path) == []

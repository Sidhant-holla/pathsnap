"""Tests for the road graph (graph.py) and the HMM decoder (hmm.py).

They reuse the synthetic network from test_pipeline (every road two-way, so every
segment is two arcs at the same distance):

        N2 (300,200)
         |
        N1 (300,100)
         |
    A0 --- A1 --- A2 --- A3     nodes every 100 m at y = 0
    B0 --- B1 --- B2 --- B3     parallel distractor 60 m south
"""
import math

import numpy as np
import pandas as pd
import pytest

from pathsnap.candidates import generate_candidates
from pathsnap.geo import project_trip
from pathsnap.graph import (build_graph, connected, distance_matrix, nearest_arc, network_distance,
                            shortest_arc_path)
from pathsnap.hmm import HMMParams, argmax_path, transition_logp, viterbi
from tests.test_pipeline import EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE, ORIGIN, make_trip


@pytest.fixture
def east_north():
    trip, arc_id = make_trip(EAST_THEN_NORTH, EAST_THEN_NORTH_ROUTE)
    project_trip(trip)
    return trip, arc_id, build_graph(trip)


# --------------------------------------------------------------------------- graph

def test_graph_is_directed_with_arc_ids(east_north):
    trip, arc_id, G = east_north
    assert G.number_of_edges() == len(trip.arcs)
    e = arc_id[("A0", "A1")]
    u, v = trip.arcs.loc[e, ["u", "v"]]
    assert G.edges[u, v]["arc_id"] == e
    assert G.edges[u, v]["length"] == pytest.approx(100.0, abs=0.01)


def test_network_distance_follows_directions(east_north):
    _, a, G = east_north
    e01, e12, e23 = a[("A0", "A1")], a[("A1", "A2")], a[("A2", "A3")]
    # same arc, forward
    assert network_distance(G, e01, 0.2, e01, 0.7) == pytest.approx(50.0, abs=0.01)
    # next arc: rest of A0A1 (30) + 40 into A1A2
    assert network_distance(G, e01, 0.7, e12, 0.4) == pytest.approx(70.0, abs=0.01)
    # two arcs ahead
    assert network_distance(G, e01, 0.5, e23, 0.5) == pytest.approx(200.0, abs=0.01)
    # reaching the opposite twin needs a U-turn: rest of A1A2 (100) + A2A1 (100) + 70 into A1A0
    assert network_distance(G, e12, 0.0, a[("A1", "A0")], 0.7, cutoff=500) == pytest.approx(270.0, abs=0.01)
    # out of reach within the cutoff
    assert math.isinf(network_distance(G, e01, 0.0, e23, 1.0, cutoff=100))


def test_backward_jitter_is_signed(east_north):
    _, a, G = east_north
    e = a[("A0", "A1")]
    assert network_distance(G, e, 0.50, e, 0.45) == pytest.approx(-5.0, abs=0.01)    # within tolerance
    # beyond the tolerance it has to drive round: rest (50) + A1A0 (100) + 40 back into A0A1
    assert network_distance(G, e, 0.50, e, 0.40, backward_tol_m=5, cutoff=1000) == pytest.approx(190.0, abs=0.01)


def test_distance_matrix_matches_pairwise(east_north):
    _, a, G = east_north
    arcs = [a[("A0", "A1")], a[("A1", "A2")], a[("A1", "A0")], a[("A3", "N1")]]
    pos_a, pos_b = [0.3, 0.9, 0.5, 0.1], [0.6, 0.2, 0.4, 0.8]
    m = distance_matrix(G, arcs, pos_a, arcs, pos_b, cutoff=400)
    for i in range(4):
        for j in range(4):
            d = network_distance(G, arcs[i], pos_a[i], arcs[j], pos_b[j], cutoff=400)
            assert (math.isinf(d) and math.isinf(m[i, j])) or m[i, j] == pytest.approx(d)


def test_connected_and_shortest_path(east_north):
    _, a, G = east_north
    assert connected(G, a[("A0", "A1")], a[("A1", "A2")])
    assert not connected(G, a[("A0", "A1")], a[("A2", "A3")])
    assert shortest_arc_path(G, a[("A0", "A1")], a[("A1", "A2")]) == []
    assert shortest_arc_path(G, a[("A0", "A1")], a[("N1", "N2")]) == [a[("A1", "A2")], a[("A2", "A3")], a[("A3", "N1")]]
    assert shortest_arc_path(G, a[("A0", "A1")], a[("B0", "B1")]) is None      # separate component


def test_nearest_arc_breaks_ties_by_lower_id(east_north):
    trip, a, G = east_north
    arc, dist, frac = nearest_arc(G, [ORIGIN[0] + 150.0], [ORIGIN[1] + 3.0])
    assert arc[0] == min(a[("A1", "A2")], a[("A2", "A1")])
    assert dist[0] == pytest.approx(3.0, abs=0.01)


# --------------------------------------------------------------------------- hmm

def _decode(trip, G, params, cands=None):
    cands = generate_candidates(trip) if cands is None else cands
    return viterbi(cands, G, params, trip.track), cands


def test_transition_soft_break_is_finite():
    p = HMMParams(beta=5, break_penalty=50, v_max=50, speed_slack_m=0)
    lp = transition_logp(np.array([[10.0, np.inf, 1000.0]]), d_straight=10.0, dt=1.0, params=p)
    assert lp[0, 0] == 0.0
    assert lp[0, 1] == -50.0 and lp[0, 2] == -50.0      # unreachable / too fast: finite penalty
    assert np.isfinite(lp).all()


def test_distance_hmm_picks_direction_of_travel(east_north):
    """Distance alone can't separate the two directions; the transitions must."""
    trip, a, G = east_north
    path, _ = _decode(trip, G, HMMParams(emission="distance"))
    assert len(path) == len(trip.track) and path["obs_idx"].tolist() == list(range(len(trip.track)))
    x = trip.track["x"].to_numpy() - ORIGIN[0]
    y = trip.track["y"].to_numpy() - ORIGIN[1]
    on_east = (x > 20) & (x < 280) & (np.abs(y) < 10)
    westbound = {a[("A1", "A0")], a[("A2", "A1")], a[("A3", "A2")]}
    assert not path.loc[on_east, "arc_id"].isin(westbound).any()
    assert (path.loc[y > 120, "arc_id"] == a[("N1", "N2")]).all()
    # the decoded route is drivable: consecutive distinct arcs connect
    arcs = path["arc_id"].to_numpy()
    arcs = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
    assert all(connected(G, p, q) for p, q in zip(arcs[:-1], arcs[1:]))


def test_classifier_emissions_and_lambda(east_north):
    trip, a, G = east_north
    cands = generate_candidates(trip)
    # A "classifier" that strongly prefers the parallel road B (wrong) only at one point.
    cands["proba"] = np.where(cands["dist_m"] < 20, 0.9, 0.1)
    noisy = cands["obs_idx"] == 10
    cands.loc[noisy, "proba"] = np.where(cands.loc[noisy, "dist_m"] > 40, 0.99, 0.01)
    path, _ = _decode(trip, G, HMMParams(emission="classifier", lam=1.0), cands)
    # the HMM overrides the single bad point because jumping to road B is not drivable
    assert path.loc[10, "arc_id"] in {a[("A0", "A1")], a[("A1", "A2")]}


def test_gap_splits_segments_and_no_candidate_fallback(east_north):
    trip, a, G = east_north
    t = trip.track["t"].to_numpy().copy()
    t[15:] += 600                       # a 10-minute gap
    trip.track["t"] = t
    cands = generate_candidates(trip)
    cands = cands[cands["obs_idx"] != 5]          # point 5 loses all its candidates
    path, _ = _decode(trip, G, HMMParams(emission="distance", gap_s=30), cands)
    assert path["segment"].nunique() == 2
    assert path.loc[5, "fallback"] and path["fallback"].sum() == 1
    assert len(path) == len(trip.track)


def test_disconnected_jump_is_a_soft_break_not_a_crash(east_north):
    trip, a, G = east_north
    cands = generate_candidates(trip)
    # Force points 20.. onto road B only (disconnected from road A): decoding must still finish.
    late = cands["obs_idx"] >= 20
    cands = pd.concat([cands[~late], cands[late].assign(arc_id=a[("B0", "B1")], dist_m=60.0, proj_frac=0.5)])
    path, _ = _decode(trip, G, HMMParams(emission="distance"), cands)
    assert path["soft_break"].sum() >= 1
    assert len(path) == len(trip.track)


def test_argmax_path_baseline(east_north):
    trip, a, G = east_north
    cands = generate_candidates(trip)
    cands = cands[cands["obs_idx"] != 3]
    out = argmax_path(cands.assign(score=-cands["dist_m"]), G, trip.track, "score")
    assert len(out) == len(trip.track) and out.loc[3, "fallback"]

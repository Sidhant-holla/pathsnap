"""End-to-end prediction for one trip: raw files -> predicted route.

    load_trip (NO route) -> project to UTM -> candidates + features (features.py, the same
    code that built the training table) -> classifier P(correct) -> HMM/Viterbi -> one arc
    per GPS point -> collapse repeats -> fill gaps with shortest paths -> lat/lon geometry

Leakage rule: this module NEVER loads the reference route. load_trip() is always called
with its default with_route=False, and nothing here imports load_route. Scoring against
the route happens elsewhere (api/scorer.py, scripts/hmm_ablation.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT, MODELS_DIR
from .evaluate import load_model, predict_proba
from .features import trip_candidate_features
from .graph import build_graph
from .hmm import HMMParams, viterbi
from .io import load_trip
from .route_metrics import collapse, fill_route

DEFAULT_MODEL = "lgbm"
HMM_PARAMS_PATH = MODELS_DIR / "hmm_params.json"


def load_hmm_params(path: Path = HMM_PARAMS_PATH) -> HMMParams:
    """Tuned classifier-HMM parameters (scripts/hmm_ablation.py tune), else the defaults."""
    if Path(path).exists():
        return HMMParams.from_dict(json.loads(Path(path).read_text())["classifier_hmm"])
    return HMMParams()


def _polyline(trip, arcs: list[int]) -> list[list[float]]:
    """[lat, lon] vertices of a drivable arc sequence: start node, then every end node."""
    if not arcs:
        return []
    a = np.asarray(arcs, dtype=np.int64)
    nodes = np.r_[trip.arcs["u"].to_numpy()[a[0]], trip.arcs["v"].to_numpy()[a]]
    lat = trip.nodes["lat"].to_numpy()[nodes]
    lon = trip.nodes["lon"].to_numpy()[nodes]
    return np.round(np.c_[lat, lon], 7).tolist()


def predict_trip(trip_id: str, model=DEFAULT_MODEL, params: HMMParams | dict | None = None,
                 partition: str | None = None, data_root: Path = DATA_ROOT,
                 return_context: bool = False):
    """Predicted route for one trip.

    model   a bundle from evaluate.load_model, or a saved model name (default "lgbm")
    params  HMMParams / dict; None loads models/hmm_params.json (tuned on train)

    Returns a JSON-ready dict:
      trip_id, n_points, gps [[lat, lon, t]], point_arcs (decoded arc per GPS point),
      route_arcs (gap-filled route), route_coords (list of [lat, lon] polylines, one per
      drivable piece), flags (fallback points, soft breaks, segments, unfilled gaps),
      timing_s per stage, hmm_params.
    With return_context=True also returns {"trip", "graph", "path"} for server-side use.
    """
    timing = {}
    t_all = t0 = time.perf_counter()
    bundle = load_model(model) if isinstance(model, (str, Path)) else model
    params = load_hmm_params() if params is None else (
        params if isinstance(params, HMMParams) else HMMParams.from_dict(params))

    trip = load_trip(str(trip_id), partition, data_root=data_root)       # route NOT loaded
    timing["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = trip_candidate_features(trip)        # projects the trip, candidates, f_* features
    feats.insert(0, "trip_id", str(trip_id))
    timing["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if len(feats):
        feats["proba"] = predict_proba(bundle["model"], feats[bundle["feature_columns"]],
                                       feats[["trip_id", "obs_idx"]])
    else:
        feats["proba"] = pd.Series(dtype=float)
    timing["classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = build_graph(trip)
    timing["graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    path = viterbi(feats, G, params, trip.track)
    timing["decode"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    point_arcs = path["arc_id"].to_numpy(np.int64)
    pieces, unfilled = fill_route(G, collapse(point_arcs))
    route_coords = [_polyline(trip, p) for p in pieces]
    timing["route"] = time.perf_counter() - t0
    timing["total"] = time.perf_counter() - t_all

    tr = trip.track
    result = {
        "trip_id": str(trip_id),
        "n_points": int(len(tr)),
        "gps": np.round(np.c_[tr["lat"], tr["lon"], tr["t"]], 7).tolist(),
        "point_arcs": point_arcs.tolist(),
        "route_arcs": [int(a) for p in pieces for a in p],
        "route_coords": route_coords,
        "flags": {
            "n_candidates": int(len(feats)),
            "n_fallback_points": int(path["fallback"].sum()),
            "n_soft_breaks": int(path["soft_break"].sum()),
            "n_segments": int(path["segment"].nunique()),
            "n_unfilled_gaps": int(unfilled),
        },
        "timing_s": {k: round(v, 4) for k, v in timing.items()},
        "model": bundle.get("name", str(model)),
        "hmm_params": params.to_dict(),
    }
    if return_context:
        return result, {"trip": trip, "graph": G, "path": path}
    return result

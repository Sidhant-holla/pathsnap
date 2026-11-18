"""Server-side scoring of a prediction against the reference route.

This is the ONLY part of the API that reads the .route file, and it runs AFTER
predict_trip() has finished, so the reference can't influence the prediction. It returns
numbers only: the reference arcs never leave this module, so they never reach the
response or the UI.

Metrics (definitions in pathsnap.route_metrics):
  point top-1 (confident points / all points) vs labels from labels.align_route,
  route continuity of the decode, route precision/recall/F1 (gap-filled route, by arc
  count and by length) and the same for the raw, unfilled route.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pathsnap.config import DATA_ROOT
from pathsnap.geo import project_trip
from pathsnap.io import load_trip, trip_dir
from pathsnap.labels import align_route
from pathsnap.route_metrics import score_decoded


def has_reference(trip_id: str, partition: str, data_root: Path = DATA_ROOT) -> bool:
    return (trip_dir(trip_id, partition, data_root) / f"{trip_id}.route").exists()


def score_prediction(trip_id: str, partition: str, prediction: dict, graph, data_root: Path = DATA_ROOT) -> dict:
    """Metrics for one predicted trip, or {"available": False, ...} if there is no .route file."""
    if not has_reference(trip_id, partition, data_root):
        return {"available": False, "reason": "no reference route on the server for this trip"}
    ref = project_trip(load_trip(trip_id, partition, with_route=True, data_root=data_root))
    labels = align_route(ref)
    m = score_decoded(
        graph,
        point_arcs=np.asarray(prediction["point_arcs"], dtype=np.int64),
        ref_route=ref.route,
        label_arcs=labels["label_arc"].to_numpy(np.int64),
        low_conf=labels["low_conf"].to_numpy(bool),
        filled_route=prediction["route_arcs"],
    )
    m["available"] = True
    m["n_reference_arcs"] = int(len(ref.route))      # a count only, never the arcs themselves
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}

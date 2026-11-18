"""PathSnap API (Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_DATA        folder holding the partitions (default data/raw). On demo day, point
                         it at the folder containing the evaluator trips.
    PATHSNAP_PARTITION   which partition GET /trips lists and POST /predict serves
                         (default evaluator_test; use student_test for dry runs).
    PATHSNAP_MODEL       saved model name in models/ (default lgbm).

Endpoints:
    GET  /health          status, model, partition, number of trips
    GET  /trips           the trip IDs of the served partition (the 10 evaluator trips)
    POST /predict         {"trip_id": "..."} -> GPS points, predicted route geometry, metrics
                          {"trip_id": "...", "score": false} skips scoring entirely: no metrics are
                          computed or logged (used by scripts.smoke_api for crash-only checks)

POST /predict first runs pathsnap.pipeline.predict_trip, which never reads the
reference route. Only afterwards does api.scorer load the route and compute metrics.
The response contains the GPS points, the PREDICTED route and the metrics, never the
reference arcs.

Every request is logged to this console with its trip id, status and timing (the
evaluators check these logs on this laptop).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pathsnap.config import DATA_ROOT, EVALUATOR_TEST, PARTITIONS
from pathsnap.evaluate import load_model
from pathsnap.io import list_trips, partition_dir
from pathsnap.pipeline import load_hmm_params, predict_trip

from .scorer import score_prediction

PARTITION = os.environ.get("PATHSNAP_PARTITION", EVALUATOR_TEST)
MODEL_NAME = os.environ.get("PATHSNAP_MODEL", "lgbm")

log = logging.getLogger("pathsnap.api")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PARTITION not in PARTITIONS:
        raise RuntimeError(f"PATHSNAP_PARTITION={PARTITION!r} must be one of {PARTITIONS}")
    STATE["bundle"] = load_model(MODEL_NAME)
    STATE["params"] = load_hmm_params()
    trips = list_trips(PARTITION)
    log.info(f"PathSnap API ready | model={MODEL_NAME} | partition={PARTITION} "
             f"({partition_dir(PARTITION)}) | {len(trips)} trips | data root={DATA_ROOT}")
    if not trips:
        log.info(f"WARNING: no trips found for partition {PARTITION!r}. Set PATHSNAP_DATA / PATHSNAP_PARTITION.")
    yield
    STATE.clear()


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:                       # pragma: no cover - last-resort guard
        log.info(traceback.format_exc())
        response = JSONResponse({"detail": "internal server error"}, status_code=500)
        status = 500
    ms = 1000 * (time.perf_counter() - t0)
    trip = getattr(request.state, "trip_id", None)
    client = request.client.host if request.client else "?"
    log.info(f"{request.method} {request.url.path} | trip={trip or '-'} | status={status} | "
             f"{ms:.0f} ms | client={client}")
    return response


class PredictRequest(BaseModel):
    trip_id: str
    score: bool = True      # False = crash-only check: the reference route is not even read


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["GET /health", "GET /trips", "POST /predict", "GET /docs"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "partition": PARTITION,
            "n_trips": len(list_trips(PARTITION))}


@app.get("/trips")
def trips():
    return {"partition": PARTITION, "trip_ids": list_trips(PARTITION)}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    trip_id = req.trip_id.strip()
    request.state.trip_id = trip_id
    if trip_id not in list_trips(PARTITION):
        raise HTTPException(status_code=404, detail=f"trip {trip_id!r} is not in partition {PARTITION!r}")
    try:
        pred, ctx = predict_trip(trip_id, STATE["bundle"], STATE["params"], partition=PARTITION,
                                 return_context=True)
    except Exception as e:
        log.info(f"predict failed for trip {trip_id}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"prediction failed: {type(e).__name__}: {e}")

    t0 = time.perf_counter()
    if not req.score:
        metrics = {"available": False, "reason": "scoring skipped on request (crash-only check)"}
    else:
        try:
            metrics = score_prediction(trip_id, PARTITION, pred, ctx["graph"])
        except Exception as e:
            log.info(f"scoring failed for trip {trip_id}:\n{traceback.format_exc()}")
            metrics = {"available": False, "reason": f"scoring failed: {type(e).__name__}: {e}"}
    score_s = time.perf_counter() - t0
    scored = (f"top1={metrics.get('top1_all', 'n/a')} route_f1={metrics.get('route_f1', 'n/a')}"
              if req.score else "scoring skipped (crash-only check)")

    t = pred["timing_s"]
    log.info(f"  trip={trip_id} points={pred['n_points']} candidates={pred['flags']['n_candidates']} | "
             f"load {t['load']:.2f}s features {t['features']:.2f}s classify {t['classify']:.2f}s "
             f"graph {t['graph']:.2f}s decode {t['decode']:.2f}s route {t['route']:.2f}s score {score_s:.2f}s | "
             f"{scored}")

    return {
        "trip_id": trip_id,
        "partition": PARTITION,
        "model": pred["model"],
        "n_points": pred["n_points"],
        "gps": pred["gps"],                         # [lat, lon, t] per GPS fix
        "route_coords": pred["route_coords"],       # predicted route, list of [lat, lon] polylines
        "n_route_arcs": len(pred["route_arcs"]),
        "flags": pred["flags"],
        "timing_s": {**t, "score": round(score_s, 4)},
        "metrics": metrics,
        "hmm_params": pred["hmm_params"],
    }

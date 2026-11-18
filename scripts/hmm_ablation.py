"""HMM decoder: tune its parameters on TRAIN, then run the 4-way ablation on STUDENT-TEST.

Run (from the repo root, after scripts.build_features):

    python -m scripts.hmm_ablation oof        # out-of-fold train probabilities (GroupKFold by trip)
    python -m scripts.hmm_ablation tune       # Optuna TPE for the decoder parameters on train
    python -m scripts.hmm_ablation ablation   # 4 methods on student_test, with the tuned params
    python -m scripts.hmm_ablation all        # the three steps above
    python -m scripts.hmm_ablation ablation --defaults   # same table with untuned decoder defaults

Why out-of-fold probabilities: the saved classifier was fitted on all 80 train trips, so
its probabilities on those trips are over-confident. Tuning lam (how much to trust the
classifier) on them would over-trust it. GroupKFold OOF probabilities behave like the
classifier's output on unseen trips, which is what the decoder sees in the API.

Decoder parameters are chosen on train only and student_test is used once, for the
report. The evaluator partition is never touched here.

Methods (PLAN section 8):
  nearest      each point's nearest candidate arc (ties: lower arc_id)
  dist_hmm     HMM with Gaussian distance emissions (no classifier)
  clf          classifier argmax per point (no HMM)
  clf_hmm      HMM with classifier emissions
Every method uses the same nearest-arc fallback for points with no candidate.

Outputs (experiments/results/hmm/): ablation_student_test.csv (one row per method),
ablation_per_trip.csv, tune_<mode>_trials.csv, <run>.json via evaluate.log_result
(its own summary.csv in this folder, so the shared summary.csv is untouched), and the
decoder config used by the API: models/hmm_params.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

from pathsnap import models as zoo
from pathsnap.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, STUDENT_TEST, TRAIN
from pathsnap.evaluate import KEY, group_cv, load_model, load_points, load_table, log_result, predict_proba
from pathsnap.geo import project_trip
from pathsnap.graph import build_graph
from pathsnap.hmm import HMMParams, argmax_path, viterbi
from pathsnap.io import load_route, load_trip
from pathsnap.route_metrics import score_decoded

OUT = RESULTS_DIR / "hmm"
PARAMS_PATH = MODELS_DIR / "hmm_params.json"
N_TUNE_TRIPS = 24
METHODS = ["nearest", "dist_hmm", "clf", "clf_hmm"]


# --------------------------------------------------------------------------- data

def _cands(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate-table rows in the column names the decoder expects."""
    return df.rename(columns={"f_dist": "dist_m", "f_proj_frac": "proj_frac"})


def oof_path(model_name: str):
    return PROCESSED_DIR / f"train_oof_{model_name}.parquet"


def make_oof(model_name: str) -> pd.DataFrame:
    bundle = load_model(model_name)
    base = bundle["config"].get("model", model_name)
    overrides = {k: v for k, v in bundle["config"].items() if k in zoo.ZOO[base][1]}
    df, pts = load_table(TRAIN), load_points(TRAIN)
    print(f"group CV for OOF probabilities: {base} {overrides}", flush=True)
    cv = group_cv(lambda: zoo.make(base, **overrides), df, n_splits=5, seed=SEED,
                  feature_cols=bundle["feature_columns"], points=pts)
    out = cv.oof[KEY + ["arc_id", "proba", "fold"]]
    out.to_parquet(oof_path(model_name), index=False)
    print(f"saved {oof_path(model_name)}  (pooled OOF top1_conf={cv.summary['pooled']['top1_conf']:.4f})")
    return out


class TripCache:
    """Projected trip, graph, candidates and labels per trip, built once and reused."""

    def __init__(self, partition: str, cand_df: pd.DataFrame, points: pd.DataFrame):
        self.partition, self.items = partition, {}
        self.cand_by_trip = dict(tuple(_cands(cand_df).groupby("trip_id", sort=False)))
        self.pts_by_trip = dict(tuple(points.groupby("trip_id", sort=False)))

    def get(self, tid: str):
        if tid not in self.items:
            trip = project_trip(load_trip(tid, self.partition))
            G = build_graph(trip)
            pts = self.pts_by_trip[tid].sort_values("obs_idx")
            self.items[tid] = dict(trip=trip, G=G, cands=self.cand_by_trip.get(tid, pd.DataFrame()),
                                   label=pts["label_arc"].to_numpy(np.int64),
                                   low_conf=pts["low_conf"].to_numpy(bool),
                                   route=load_route(tid, self.partition))
        return self.items[tid]


def decode(item: dict, method: str, params: HMMParams) -> tuple[np.ndarray, float]:
    trip, G, c = item["trip"], item["G"], item["cands"]
    t0 = time.perf_counter()
    if method == "nearest":
        path = argmax_path(c.assign(_s=-c["dist_m"]), G, trip.track, "_s")
    elif method == "clf":
        path = argmax_path(c, G, trip.track, "proba")
    elif method == "dist_hmm":
        path = viterbi(c, G, HMMParams(**{**params.to_dict(), "emission": "distance"}), trip.track)
    elif method == "clf_hmm":
        path = viterbi(c, G, HMMParams(**{**params.to_dict(), "emission": "classifier"}), trip.track)
    else:
        raise ValueError(method)
    return path["arc_id"].to_numpy(np.int64), time.perf_counter() - t0


def evaluate_method(cache: TripCache, trip_ids, method: str, params: HMMParams) -> tuple[dict, pd.DataFrame]:
    rows = []
    for tid in trip_ids:
        it = cache.get(tid)
        arcs, secs = decode(it, method, params)
        m = score_decoded(it["G"], arcs, it["route"], it["label"], it["low_conf"])
        m.update(trip_id=tid, method=method, decode_s=secs)
        rows.append(m)
    per = pd.DataFrame(rows)
    conf_w = per["n_points_conf"]
    agg = {
        "method": method,
        # point metrics pooled over points (same convention as evaluate.score)
        "top1_conf": float((per["top1_conf"] * conf_w).sum() / conf_w.sum()),
        "top1_all": float((per["top1_all"] * per["n_points"]).sum() / per["n_points"].sum()),
        # continuity pooled over arc pairs
        "continuity": float((per["continuity"].fillna(0) * per["continuity_pairs"]).sum()
                            / max(per["continuity_pairs"].sum(), 1)),
        "decode_s_total": float(per["decode_s"].sum()),
        "decode_ms_per_point": float(1000 * per["decode_s"].sum() / per["n_points"].sum()),
    }
    # route metrics averaged over trips (each trip is one route)
    for k in [c for c in per.columns if c.startswith(("route_", "raw_route_"))]:
        agg[k] = float(per[k].mean())
    return agg, per


# --------------------------------------------------------------------------- tuning

def tune(model_name: str, n_trials_dist: int, n_trials_clf: int) -> dict:
    oof = pd.read_parquet(oof_path(model_name))
    df = load_table(TRAIN).merge(oof[KEY + ["arc_id", "proba"]], on=KEY + ["arc_id"], how="left")
    assert df["proba"].notna().all(), "OOF probabilities missing for some rows"
    pts = load_points(TRAIN)
    rng = np.random.default_rng(SEED)
    trips = sorted(rng.choice(sorted(df["trip_id"].unique()), size=N_TUNE_TRIPS, replace=False).tolist())
    cache = TripCache(TRAIN, df, pts)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"tuning on {len(trips)} train trips: {trips}", flush=True)

    def run(mode: str, space, n_trials: int):
        def objective(trial):
            p = HMMParams(emission=mode, **space(trial))
            agg, _ = evaluate_method(cache, trips, "dist_hmm" if mode == "distance" else "clf_hmm", p)
            for k in ("top1_all", "continuity", "route_f1"):
                trial.set_user_attr(k, agg[k])
            return agg["top1_conf"]

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                    study_name=f"hmm_{mode}")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        study.trials_dataframe().to_csv(OUT / f"tune_{mode}_trials.csv", index=False)
        print(f"{mode}: best top1_conf={study.best_value:.4f} params={study.best_params}", flush=True)
        return study

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    s_dist = run("distance", lambda tr: {
        "sigma": tr.suggest_float("sigma", 2.0, 40.0, log=True),
        "beta": tr.suggest_float("beta", 0.5, 200.0, log=True),
        "gap_s": tr.suggest_float("gap_s", 5.0, 600.0, log=True),
    }, n_trials_dist)
    s_clf = run("classifier", lambda tr: {
        "lam": tr.suggest_float("lam", 0.05, 10.0, log=True),
        "beta": tr.suggest_float("beta", 0.5, 200.0, log=True),
        "gap_s": tr.suggest_float("gap_s", 5.0, 600.0, log=True),
    }, n_trials_clf)

    best = HMMParams(emission="classifier", **s_clf.best_params).to_dict()
    best["sigma"] = s_dist.best_params["sigma"]
    record = {
        "model": model_name,
        "classifier_hmm": best,
        "distance_hmm": HMMParams(emission="distance", **s_dist.best_params).to_dict(),
        "tuned_on": {"partition": TRAIN, "trips": trips, "probabilities": "5-fold GroupKFold OOF",
                     "objective": "pooled top1_conf of the decoded path",
                     "trials": {"distance": n_trials_dist, "classifier": n_trials_clf}},
        "best_train_top1_conf": {"distance": s_dist.best_value, "classifier": s_clf.best_value},
        "tuning_seconds": round(time.time() - t0, 1),
    }
    PARAMS_PATH.write_text(json.dumps(record, indent=2))
    print(f"saved {PARAMS_PATH}")
    return record


# --------------------------------------------------------------------------- ablation

def ablation(model_name: str, defaults: bool = False) -> pd.DataFrame:
    """4-way table on student_test. defaults=True uses the untuned HMMParams defaults instead of
    models/hmm_params.json and writes *_defaults files (for the tuned-vs-untuned comparison)."""
    rec = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() and not defaults else None
    if rec is None:
        print("using untuned HMMParams defaults")
    p_clf = HMMParams.from_dict(rec["classifier_hmm"] if rec else None)
    p_dist = HMMParams.from_dict(rec["distance_hmm"] if rec else {"emission": "distance"})

    bundle = load_model(model_name)
    df = load_table(STUDENT_TEST)
    t0 = time.perf_counter()
    df["proba"] = predict_proba(bundle["model"], df[bundle["feature_columns"]], df[KEY])
    clf_s = time.perf_counter() - t0
    cache = TripCache(STUDENT_TEST, df, load_points(STUDENT_TEST))
    trips = sorted(df["trip_id"].unique())

    rows, per_trip = [], []
    for method in METHODS:
        params = p_dist if method == "dist_hmm" else p_clf
        agg, per = evaluate_method(cache, trips, method, params)
        rows.append(agg)
        per_trip.append(per)
        print(f"{method:9s} top1_conf={agg['top1_conf']:.4f} continuity={agg['continuity']:.4f} "
              f"route_f1={agg['route_f1']:.4f} raw_route_f1={agg['raw_route_f1']:.4f} "
              f"({agg['decode_ms_per_point']:.3f} ms/pt)", flush=True)
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "_defaults" if defaults else ""
    table.to_csv(OUT / f"ablation_student_test{sfx}.csv", index=False)
    pd.concat(per_trip, ignore_index=True).to_csv(OUT / f"ablation_per_trip{sfx}.csv", index=False)
    log_result(f"hmm_ablation_student_test{sfx}",
               {"family": "hmm", "model": model_name, "classifier_hmm": p_clf.to_dict(),
                "distance_hmm": p_dist.to_dict(), "classifier_infer_s": clf_s},
               {m["method"]: m for m in rows}, results_dir=OUT)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["oof", "tune", "ablation", "all"])
    ap.add_argument("--model", default="lgbm", help="saved model name in models/ (default: lgbm)")
    ap.add_argument("--trials-dist", type=int, default=25)
    ap.add_argument("--trials-clf", type=int, default=40)
    ap.add_argument("--defaults", action="store_true", help="ablation with untuned decoder defaults")
    a = ap.parse_args()
    if a.step in ("oof", "all"):
        make_oof(a.model)
    if a.step in ("tune", "all"):
        tune(a.model, a.trials_dist, a.trials_clf)
    if a.step in ("ablation", "all"):
        ablation(a.model, defaults=a.defaults)


if __name__ == "__main__":
    main()

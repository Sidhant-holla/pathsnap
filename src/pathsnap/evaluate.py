"""Shared evaluation harness: cross-validation, metrics and experiment logging.

Every model is evaluated through these functions, so all numbers in the report use
identical definitions and data partitions.

Inputs (built by scripts/build_features.py):
  candidates table  one row per (trip_id, obs_idx, arc_id):
                    trip_id, obs_idx, t, arc_id, label (0/1), label_low_conf, f_*
  points table      one row per GPS point, INCLUDING points with no candidates:
                    trip_id, obs_idx, t, label_arc, label_dist_m, low_conf, n_cands, true_in_cands
Pass the points table to score()/group_cv()/fit_and_score() whenever you have it: it is
the denominator for every point-level metric, so a point with no candidates (no rows in
the candidates table) still counts as wrong. Without it, the denominator falls back to
the points that appear in the candidates table.

"Confident" points are those within 50 m of the reference route (low_conf == False).
Low-confidence points are mostly off-network episodes where the label itself is
unreliable, so confident points are the MAIN evaluation set and all points the secondary
one. Models are trained on confident rows only.

Metric definitions:
  * top1_conf       PRIMARY selection metric. For each confident GPS point, take the
                    candidate with the highest probability and check it is the labelled
                    arc. Points whose labelled arc is not a candidate count as wrong. We
                    select on this rather than a thresholded metric because the route
                    is built from each point's argmax, not from a 0.5 threshold.
  * top1_all        the same over all points.
  * top1_covered    over confident points whose labelled arc IS a candidate: what the
                    classifier itself achieves (top1_conf = top1_covered x (1 - miss_rate_conf)).
  * miss_rate_*     share of points whose labelled arc is not among their candidates.
  * obs_macro_f1_*  observation level: predicted arc vs labelled arc, F1 per arc class,
                    macro-averaged over classes. A class is (trip_id, arc_id), because arc
                    ids repeat across trips. A point with no candidates has no predicted
                    class: it counts as a false negative for its true class.
  * cand_*          candidate level, confident rows: binary "is this the labelled arc" at
                    threshold 0.5 (accuracy, precision, recall, F1, macro F1 over the two
                    classes, confusion matrix [[TN, FP], [FN, TP]]) and log loss.
  * log_loss_all    candidate log loss over all rows.
  * continuity      (needs arcs) classifier-only route over all points: share of
                    consecutive DISTINCT predicted arcs that are connected (a.v == b.u).

Ties: if two candidates of a point have the same probability (e.g. both directions of
a two-way street), the lower arc_id wins, so results never depend on row order.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold

from .config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED

KEY = ["trip_id", "obs_idx"]
EPS = 1e-15
NO_PRED = "<none>"          # predicted class of a point with no candidates


# --------------------------------------------------------------------------- data

def feature_columns(df: pd.DataFrame | None = None) -> list[str]:
    """Model inputs: features.FEATURE_COLUMNS (A), else every f_* column of df."""
    try:
        from .features import FEATURE_COLUMNS
        return list(FEATURE_COLUMNS)
    except ImportError:
        return [c for c in df.columns if c.startswith("f_")]


def load_table(partition: str) -> pd.DataFrame:
    """Candidates table: data/processed/<partition>_candidates.parquet."""
    return pd.read_parquet(PROCESSED_DIR / f"{partition}_candidates.parquet")


def load_points(partition: str) -> pd.DataFrame:
    """Points table: data/processed/<partition>_points.parquet (one row per GPS point)."""
    return pd.read_parquet(PROCESSED_DIR / f"{partition}_points.parquet")


def trip_arcs(partition: str, trip_ids) -> dict[str, pd.DataFrame]:
    """Arc tables (u, v) per trip, for the continuity metric."""
    from .io import load_trip
    return {str(t): load_trip(str(t), partition).arcs for t in trip_ids}


def confident_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of candidate rows belonging to confident points."""
    if "label_low_conf" in df:
        return ~df["label_low_conf"].astype(bool)
    return pd.Series(True, index=df.index)


# --------------------------------------------------------------------------- metrics

def top1_predictions(df: pd.DataFrame, points: pd.DataFrame | None = None,
                     proba_col: str = "proba") -> pd.DataFrame:
    """One row per GPS point: pred_arc (argmax candidate, NaN if no candidates),
    label_arc (NaN if unknown), covered (labelled arc is a candidate), correct, low_conf."""
    best = (df.sort_values(KEY + [proba_col, "arc_id"], ascending=[True, True, False, True])
              .drop_duplicates(KEY)[KEY + ["arc_id"]].rename(columns={"arc_id": "pred_arc"}))
    pos = df.loc[df["label"] == 1, KEY + ["arc_id"]].drop_duplicates(KEY).rename(columns={"arc_id": "pos_arc"})
    if points is not None:
        base = points.loc[points["trip_id"].isin(df["trip_id"].unique()), KEY + ["label_arc", "low_conf"]]
    elif "label_low_conf" in df:
        base = df.groupby(KEY, sort=False)["label_low_conf"].max().rename("low_conf").reset_index()
    else:
        base = df[KEY].drop_duplicates().assign(low_conf=False)
    pts = base.merge(best, on=KEY, how="left").merge(pos, on=KEY, how="left")
    if points is None:
        pts["label_arc"] = pts["pos_arc"]
    elif pts["pos_arc"].notna().any():
        clash = pts["pos_arc"].notna() & (pts["pos_arc"] != pts["label_arc"])
        if clash.any():
            raise ValueError(f"{int(clash.sum())} points: label=1 candidate differs from points.label_arc")
    pts["covered"] = pts["pos_arc"].notna()
    pts["correct"] = pts["pred_arc"].notna() & (pts["pred_arc"] == pts["label_arc"])
    pts["low_conf"] = pts["low_conf"].astype(bool)
    return pts.drop(columns="pos_arc").sort_values(KEY).reset_index(drop=True)


def macro_f1_classes(y_true: pd.Series, y_pred: pd.Series, ignore=(NO_PRED,)) -> float:
    """Macro F1 over the union of true and predicted classes (same as sklearn's
    f1_score(average='macro', zero_division=0)), written with value_counts so it stays
    fast with tens of thousands of arc classes. Classes in `ignore` (the "no prediction"
    marker) still produce false negatives but are not averaged as a class."""
    y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
    tp = y_true[y_true == y_pred].value_counts()
    t = y_true.value_counts()
    p = y_pred.value_counts()
    classes = t.index.union(p.index).difference(list(ignore))
    tp, t, p = (s.reindex(classes, fill_value=0).to_numpy(float) for s in (tp, t, p))
    denom = t + p                                   # 2TP + FP + FN
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.mean()) if len(f1) else float("nan")


def route_continuity(pts: pd.DataFrame, arcs_by_trip: dict[str, pd.DataFrame], arc_col: str = "pred_arc") -> float:
    """Share of consecutive distinct predicted arcs that are connected (a.v == b.u).
    Points without a prediction are skipped."""
    ok = total = 0
    for tid, g in pts.dropna(subset=[arc_col]).sort_values(KEY).groupby("trip_id", sort=False):
        arcs = g[arc_col].to_numpy().astype(np.int64)
        arcs = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]   # collapse repeats
        if len(arcs) < 2:
            continue
        a = arcs_by_trip[str(tid)]
        ok += int((a["v"].to_numpy()[arcs[:-1]] == a["u"].to_numpy()[arcs[1:]]).sum())
        total += len(arcs) - 1
    return ok / total if total else float("nan")


def _obs_f1(pts: pd.DataFrame) -> float:
    known = pts[pts["label_arc"].notna()]
    if known.empty:
        return float("nan")
    tid = known["trip_id"].astype(str) + ":"
    y_true = tid + known["label_arc"].astype("int64").astype(str)
    y_pred = (tid + known["pred_arc"].astype("Int64").astype(str)).where(known["pred_arc"].notna(), NO_PRED)
    return macro_f1_classes(y_true, y_pred)


def _binary(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()

    def f1(tp_, fp_, fn_):
        d = 2 * tp_ + fp_ + fn_
        return 2 * tp_ / d if d else 0.0

    return {
        "cand_accuracy": float((tp + tn) / len(y)) if len(y) else float("nan"),
        "cand_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "cand_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "cand_f1": float(f1(tp, fp, fn)),
        "cand_macro_f1": float((f1(tp, fp, fn) + f1(tn, fn, fp)) / 2),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def score(df: pd.DataFrame, points: pd.DataFrame | None = None, proba_col: str = "proba",
          arcs_by_trip: dict | None = None, threshold: float = 0.5) -> dict:
    """All metrics for one prediction table (see module docstring for definitions)."""
    pts = top1_predictions(df, points, proba_col)
    conf = pts[~pts["low_conf"]]
    cov = conf[conf["covered"]]
    crow = confident_rows(df)
    m = {
        "n_trips": int(pts["trip_id"].nunique()),
        "n_rows": int(len(df)),
        "n_points": int(len(pts)),
        "n_points_conf": int(len(conf)),
        "n_points_no_candidates": int(pts["pred_arc"].isna().sum()),
        "mean_candidates": len(df) / max(int(pts["pred_arc"].notna().sum()), 1),
        "low_conf_rate": float(pts["low_conf"].mean()),
        "top1_conf": float(conf["correct"].mean()) if len(conf) else float("nan"),
        "top1_all": float(pts["correct"].mean()),
        "top1_covered": float(cov["correct"].mean()) if len(cov) else float("nan"),
        "miss_rate_conf": float(1 - conf["covered"].mean()) if len(conf) else float("nan"),
        "miss_rate_all": float(1 - pts["covered"].mean()),
        "obs_macro_f1_conf": _obs_f1(conf),
        "obs_macro_f1_all": _obs_f1(pts),
    }
    m.update(_binary(df.loc[crow, "label"].to_numpy().astype(int), df.loc[crow, proba_col].to_numpy(float), threshold))
    m["log_loss_all"] = float(log_loss(df["label"].to_numpy().astype(int),
                                       np.clip(df[proba_col].to_numpy(float), EPS, 1 - EPS), labels=[0, 1]))
    if arcs_by_trip is not None:
        m["continuity"] = route_continuity(pts, arcs_by_trip)
    return m


# --------------------------------------------------------------------------- training

def _fit(model, X, y, meta):
    """Call model.fit, passing the (trip_id, obs_idx) keys to models that ask for them
    (early stopping on a trip-level split, rankers that need query groups)."""
    if "meta" in inspect.signature(model.fit).parameters:
        return model.fit(X, y, meta=meta)
    return model.fit(X, y)


def predict_proba(model, X, meta: pd.DataFrame | None = None) -> np.ndarray:
    """P(candidate is the true arc). Rankers need the point keys to softmax their scores."""
    if meta is not None and hasattr(model, "set_query_keys"):
        model.set_query_keys(meta)
    return model.predict_proba(X)[:, 1]


@dataclass
class CVResult:
    oof: pd.DataFrame                 # keys + label + proba + fold, one row per candidate
    folds: pd.DataFrame               # one row of metrics per fold
    summary: dict = field(default_factory=dict)   # mean/std over folds + pooled OOF metrics


SUMMARY_METRICS = ["top1_conf", "top1_all", "top1_covered", "obs_macro_f1_conf", "obs_macro_f1_all",
                   "cand_macro_f1", "cand_f1", "cand_precision", "cand_recall", "log_loss", "log_loss_all",
                   "miss_rate_conf", "miss_rate_all"]


def group_cv(make_model, df: pd.DataFrame, n_splits: int = 5, seed: int = SEED,
             feature_cols: list[str] | None = None, points: pd.DataFrame | None = None,
             train_confident_only: bool = True, verbose: bool = True) -> CVResult:
    """GroupKFold by trip_id with out-of-fold probabilities.

    `make_model()` must return a fresh, unfitted estimator, so nothing (scalers,
    early-stopping sets) is ever fitted on the validation trips. Trips are shuffled into
    folds with `seed`. With train_confident_only, low-confidence rows of the TRAINING
    folds are left out of fitting; validation trips are always scored on every row.
    """
    feature_cols = feature_cols or feature_columns(df)
    df = df.reset_index(drop=True)
    X, y, groups = df[feature_cols], df["label"].to_numpy(), df["trip_id"].to_numpy()
    usable = confident_rows(df).to_numpy() if train_confident_only else np.ones(len(df), bool)
    oof = np.full(len(df), np.nan)
    fold_id = np.full(len(df), -1)
    rows = []
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        fit_idx = tr[usable[tr]]
        model = make_model()
        t0 = time.perf_counter()
        _fit(model, X.iloc[fit_idx], y[fit_idx], df.loc[fit_idx, KEY])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        oof[va] = predict_proba(model, X.iloc[va], df.loc[va, KEY])
        pred_s = time.perf_counter() - t0
        fold_id[va] = k
        fm = score(df.loc[va].assign(proba=oof[va]), points=points)
        fm.pop("confusion_matrix")
        fm.update(fold=k, fit_time_s=fit_s, infer_time_s=pred_s, n_fit_rows=len(fit_idx),
                  n_train_trips=len(np.unique(groups[tr])), n_val_trips=len(np.unique(groups[va])))
        rows.append(fm)
        if verbose:
            print(f"  fold {k}: top1_conf={fm['top1_conf']:.4f} top1_all={fm['top1_all']:.4f} "
                  f"logloss={fm['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)
    folds = pd.DataFrame(rows)
    cols = KEY + ["arc_id", "label"] + [c for c in ("label_low_conf", "t") if c in df]
    oof_df = df[cols].assign(proba=oof, fold=fold_id)

    summary = {}
    for mname in SUMMARY_METRICS:
        if mname in folds:
            summary[f"{mname}_mean"] = float(folds[mname].mean())
            summary[f"{mname}_std"] = float(folds[mname].std(ddof=1))
    summary["pooled"] = score(oof_df, points=points)
    summary["fit_time_s_total"] = float(folds["fit_time_s"].sum())
    summary.update(n_splits=n_splits, seed=seed, train_confident_only=train_confident_only)
    return CVResult(oof_df, folds, summary)


def fit_and_score(make_model, train_df: pd.DataFrame, test_df: pd.DataFrame,
                  feature_cols: list[str] | None = None, points: pd.DataFrame | None = None,
                  arcs_by_trip: dict | None = None, train_confident_only: bool = True):
    """Fit on all training trips, score the test partition (student_test).

    Returns (fitted_model, metrics, predictions). metrics include train time and
    inference time on the whole test table.
    """
    feature_cols = feature_cols or feature_columns(train_df)
    train_df = train_df.reset_index(drop=True)
    if train_confident_only:
        train_df = train_df[confident_rows(train_df)].reset_index(drop=True)
    model = make_model()
    t0 = time.perf_counter()
    _fit(model, train_df[feature_cols], train_df["label"].to_numpy(), train_df[KEY])
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = predict_proba(model, test_df[feature_cols], test_df[KEY])
    infer_s = time.perf_counter() - t0
    pred = test_df.assign(proba=proba)
    m = score(pred, points=points, arcs_by_trip=arcs_by_trip)
    m.update(train_time_s=train_s, infer_time_s=infer_s, n_fit_rows=len(train_df),
             infer_ms_per_1k_rows=1e6 * infer_s / max(len(test_df), 1),
             infer_ms_per_trip=1000 * infer_s / max(test_df["trip_id"].nunique(), 1))
    return model, m, pred


# --------------------------------------------------------------------------- logging

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Scalar metrics only, nested dicts become prefix_key (e.g. cv_top1_conf_mean)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}_"))
        elif isinstance(v, (int, float, str, bool, np.integer, np.floating)) or v is None:
            out[f"{prefix}{k}"] = v
    return out


def log_result(name: str, config: dict, metrics: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Write experiments/results/<name>.json and append one flat row to summary.csv.

    `metrics` may be nested (e.g. {"cv": {...}, "student_test": {...}}); the CSV row
    keeps only scalars. Re-running an experiment replaces its row in summary.csv.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    record = {"name": name, "timestamp": stamp, "seed": config.get("seed", SEED),
              "config": config, "metrics": metrics}
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(_jsonable(record), indent=2))

    row = {"name": name, "timestamp": stamp, **_flatten({"family": config.get("family")}),
           **_flatten(metrics)}
    csv = results_dir / "summary.csv"
    new = pd.DataFrame([row])
    if csv.exists():
        old = pd.read_csv(csv, dtype={"name": str})
        new = pd.concat([old[old["name"] != name], new], ignore_index=True)
    new.to_csv(csv, index=False)
    return path


def save_model(name: str, model, feature_cols: list[str], config: dict, models_dir: Path = MODELS_DIR) -> Path:
    """models/<name>.joblib holds the fitted estimator plus the exact feature order."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump({"name": name, "model": model, "feature_columns": list(feature_cols),
                 "config": _jsonable(config)}, path, compress=3)
    return path


def load_model(name_or_path) -> dict:
    """Inverse of save_model: {'model', 'feature_columns', 'config', 'name'}."""
    p = Path(name_or_path)
    if not p.suffix:
        p = MODELS_DIR / f"{name_or_path}.joblib"
    return joblib.load(p)

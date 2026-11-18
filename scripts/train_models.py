"""Train and evaluate Member C's models through the shared harness.

    python -m scripts.train_models                      # logreg, rf, lgbm on the real tables
    python -m scripts.train_models --models lgbm_ranker
    python -m scripts.train_models --fake               # smoke test on a fake table

For each model:
  1. GroupKFold(5) by trip on the 80 train trips -> out-of-fold probabilities + per-fold metrics.
  2. Refit on all 80 train trips -> score the 10 student-test trips (incl. route continuity
     of the classifier-only argmax route).
  3. Log experiments/results/<name>.json + a row in summary.csv, save models/<name>.joblib.

Also written: <name>_cv_folds.csv (fold stability), feature importances (trees),
the real per-iteration train/val loss curves (LightGBM), and the out-of-fold and
student-test probabilities in data/processed/ (D can tune HMM parameters on the OOF
probabilities without the classifier having seen those trips).
--fake writes everything under _fake/ subfolders so it can never mix with real results.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import evaluate as ev  # noqa: E402
from pathsnap import models  # noqa: E402
from pathsnap.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED, STUDENT_TEST, TRAIN  # noqa: E402


def load_data(fake: bool):
    if fake:
        from pathsnap.testing import make_fake_table
        df, points, arcs = make_fake_table(n_trips=30, seed=SEED)
        trips = sorted(df["trip_id"].unique())
        st_trips = set(trips[-6:])
        train, st = df[~df["trip_id"].isin(st_trips)], df[df["trip_id"].isin(st_trips)]
        return train, st, points, {t: arcs[t] for t in st_trips}
    train, st = ev.load_table(TRAIN), ev.load_table(STUDENT_TEST)
    points = pd.concat([ev.load_points(TRAIN), ev.load_points(STUDENT_TEST)], ignore_index=True)
    arcs = ev.trip_arcs(STUDENT_TEST, sorted(st["trip_id"].unique()))
    return train, st, points, arcs


def plot_loss_curve(evals: dict, best_iter: int, name: str, metric: str, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for split, style in (("train", "-"), ("val", "--")):
        ax.plot(evals[split][metric], style, label=f"{split} (inner {'train' if split == 'train' else 'validation'} trips)")
    ax.axvline(best_iter - 1, color="grey", lw=0.8, ls=":", label=f"best iteration = {best_iter}")
    ax.set_xlabel("boosting iteration")
    ax.set_ylabel(metric)
    ax.set_title(f"{name}: training curve (real per-iteration values)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(name: str, train, st, points, arcs, out):
    cfg = models.config(name)
    feats = ev.feature_columns(train)
    cfg.update(features=feats, train_confident_only=True)
    print(f"\n=== {name} ({cfg['family']}) on {len(train):,} train rows, {len(feats)} features", flush=True)

    cv = ev.group_cv(lambda: models.make(name), train, n_splits=5, seed=SEED, feature_cols=feats, points=points)
    cv.folds.to_csv(out["results"] / f"{name}_cv_folds.csv", index=False)
    cv.oof.to_parquet(out["processed"] / f"oof_{name}.parquet", index=False)

    model, st_m, st_pred = ev.fit_and_score(lambda: models.make(name), train, st, feature_cols=feats,
                                            points=points, arcs_by_trip=arcs)
    st_pred[ev.KEY + ["arc_id", "label", "proba"]].to_parquet(out["processed"] / f"st_pred_{name}.parquet", index=False)
    ev.save_model(name, model, feats, cfg, models_dir=out["models"])

    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        pd.Series(est.feature_importances_, index=feats, name="importance").sort_values(ascending=False) \
          .to_csv(out["results"] / f"{name}_feature_importance.csv")
    elif hasattr(est, "coef_"):
        pd.Series(est.coef_[0], index=feats, name="std_coef").sort_values(key=abs, ascending=False) \
          .to_csv(out["results"] / f"{name}_coefficients.csv")
    if hasattr(est, "evals_result_"):
        metric = next(iter(est.evals_result_["val"]))
        pd.DataFrame({f"train_{metric}": est.evals_result_["train"][metric],
                      f"val_{metric}": est.evals_result_["val"][metric]}).rename_axis("iteration") \
          .to_csv(out["results"] / f"{name}_loss_curve.csv")
        plot_loss_curve(est.evals_result_, est.best_iteration_, name, metric, out["figures"] / f"{name}_loss_curve.png")
        cfg["best_iteration"] = est.best_iteration_

    metrics = {"cv": cv.summary, "student_test": st_m,
               "train_time_s": st_m["train_time_s"], "infer_time_s": st_m["infer_time_s"],
               "infer_ms_per_1k_rows": st_m["infer_ms_per_1k_rows"]}
    ev.log_result(name, cfg, metrics, results_dir=out["results"])
    s = cv.summary
    print(f"  CV  top1_conf {s['top1_conf_mean']:.4f} +/- {s['top1_conf_std']:.4f} | top1_all {s['top1_all_mean']:.4f} | "
          f"obsF1 {s['obs_macro_f1_conf_mean']:.4f} | candF1 {s['cand_macro_f1_mean']:.4f} | logloss {s['log_loss_mean']:.4f}")
    print(f"  ST  top1_conf {st_m['top1_conf']:.4f} | top1_all {st_m['top1_all']:.4f} | obsF1 {st_m['obs_macro_f1_conf']:.4f} | "
          f"candF1 {st_m['cand_macro_f1']:.4f} | logloss {st_m['log_loss']:.4f} | continuity {st_m['continuity']:.4f} | "
          f"train {st_m['train_time_s']:.1f}s | infer {st_m['infer_ms_per_1k_rows']:.1f} ms/1k rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "rf", "lgbm"], choices=list(models.ZOO))
    ap.add_argument("--fake", action="store_true", help="smoke test on a fake table (writes to _fake/ folders)")
    args = ap.parse_args()
    sub = "_fake" if args.fake else ""
    out = {k: (d / sub if sub else d) for k, d in
           {"results": RESULTS_DIR, "models": MODELS_DIR, "processed": PROCESSED_DIR, "figures": FIGURES_DIR}.items()}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    train, st, points, arcs = load_data(args.fake)
    for name in args.models:
        run(name, train, st, points, arcs, out)


if __name__ == "__main__":
    main()

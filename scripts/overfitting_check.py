"""Over/underfitting evidence for every model: training-set score vs CV vs student-test.

Run:  python -m scripts.overfitting_check              (scores every model, ~5 min because of KNN)
      python -m scripts.overfitting_check --plot-only  (redraw the figure from the saved CSV)

For each model logged in experiments/results/summary.csv, the saved final model
(models/<name>.joblib, fitted on the confident rows of all 80 train trips) is scored
on its OWN training data (in-sample) with the same harness and metric definitions as
everything else. That is compared with:
  * the cross-validation score (trips the model did not see), and
  * the student-test score (10 held-out trips).

How to read it:
  * large gap train >> CV/test  -> overfitting (memorises training trips)
  * small gap, all scores low   -> underfitting (model too simple for the features)
  * small gap, scores high      -> good fit
Notes:
  * KNN with distance weighting predicts its own stored training points perfectly
    (distance 0 gets all the weight), so its in-sample score is inflated by design.
    It only stores every 5th GPS point, so it is near-perfect on those and ordinary
    on the rest.
  * Top-1 here is over confident points (the main metric).

Outputs:
  experiments/results/overfitting_check.csv
  reports/figures/comparison/overfitting_check.png
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import evaluate as ev  # noqa: E402
from pathsnap.config import FIGURES_DIR, RESULTS_DIR, TRAIN  # noqa: E402

LABELS = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "KNN",
          "rf": "Random Forest", "lgbm": "LightGBM", "lgbm_tuned": "LightGBM (tuned)",
          "lgbm_ranker": "LightGBM Ranker", "mlp": "MLP"}


def score_models() -> pd.DataFrame:
    train, points = ev.load_table(TRAIN), ev.load_points(TRAIN)
    names = pd.read_csv(RESULTS_DIR / "summary.csv")["name"].tolist()
    rows = []
    for name in names:
        try:
            saved = ev.load_model(name)
        except FileNotFoundError:
            print(f"skip {name}: no saved model")
            continue
        logged = json.loads((RESULTS_DIR / f"{name}.json").read_text())["metrics"]
        t0 = time.perf_counter()
        proba = ev.predict_proba(saved["model"], train[saved["feature_columns"]], train[ev.KEY])
        m = ev.score(train.assign(proba=proba), points=points)
        cv, st = logged["cv"], logged["student_test"]
        row = {
            "model": LABELS.get(name, name),
            "train_top1": m["top1_conf"], "cv_top1": cv["top1_conf_mean"], "st_top1": st["top1_conf"],
            "train_log_loss": m["log_loss"], "cv_log_loss": cv["log_loss_mean"], "st_log_loss": st["log_loss"],
            "train_cand_macro_f1": m["cand_macro_f1"], "cv_cand_macro_f1": cv["cand_macro_f1_mean"],
            "st_cand_macro_f1": st["cand_macro_f1"],
        }
        row["gap_top1_train_minus_cv"] = row["train_top1"] - row["cv_top1"]
        row["gap_log_loss_cv_minus_train"] = row["cv_log_loss"] - row["train_log_loss"]
        rows.append(row)
        print(f"{name:12s} train top1 {row['train_top1']:.4f} | CV {row['cv_top1']:.4f} | ST {row['st_top1']:.4f} | "
              f"log loss train {row['train_log_loss']:.4f} CV {row['cv_log_loss']:.4f} | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "overfitting_check.csv", index=False)
    return df


def plot(df: pd.DataFrame):
    out = FIGURES_DIR / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(df) + 2.5))
    y = np.arange(len(df))
    h = 0.27
    for ax, metric, title, xlabel in [
        (axes[0], "top1", "Top-1 accuracy: training data vs unseen trips", "Top-1 accuracy (confident points)"),
        (axes[1], "log_loss", "Log loss: training data vs unseen trips", "binary log loss (lower is better)"),
    ]:
        for k, (split, lab, c) in enumerate([("train", "training data (in-sample)", "#9a9a9a"),
                                             ("cv", "CV (unseen train trips)", "#3b6ea5"),
                                             ("st", "student-test", "#e08214")]):
            vals = df[f"{split}_{metric}"]
            ax.barh(y + (k - 1) * h, vals, height=h, label=lab, color=c)
            for i, v in enumerate(vals):
                ax.text(v, i + (k - 1) * h, f" {v:.3f}", va="center", fontsize=7)
        ax.set_yticks(y, df["model"])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_xlim(0.75, 1.0)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Over/underfitting check: final models scored on their own training data vs held-out trips")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "overfitting_check.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'overfitting_check.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from overfitting_check.csv")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS_DIR / "overfitting_check.csv") if args.plot_only else score_models()
    plot(df)


if __name__ == "__main__":
    main()

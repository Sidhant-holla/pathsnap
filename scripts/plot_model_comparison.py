"""Model comparison table and plots, built only from logged experiment results.

Run:  python -m scripts.plot_model_comparison

Reads experiments/results/<model>.json and <model>_cv_folds.csv for every model in
summary.csv (fake smoke-test runs are never included), and writes:

  experiments/results/model_comparison.csv          report table 3.1
  reports/figures/comparison/cv_macro_f1.png        CV macro F1 per model (mean +/- fold std)
  reports/figures/comparison/student_test_top1_f1.png
  reports/figures/comparison/log_loss.png           CV and student-test log loss
  reports/figures/comparison/fold_stability.png     top-1 of every CV fold, per model
  reports/figures/comparison/accuracy_vs_speed.png  student-test top-1 vs inference time
  reports/figures/comparison/confusion_matrices.png candidate-level, student-test

Re-run it whenever a new model (e.g. the tuned model from HPO) is logged.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402

OUT = FIGURES_DIR / "comparison"
LABELS = {"logreg": "Logistic Regression", "gnb": "Gaussian Naive Bayes", "knn": "KNN (distance-weighted)",
          "rf": "Random Forest", "lgbm": "LightGBM", "lgbm_tuned": "LightGBM (tuned)",
          "lgbm_ranker": "LightGBM Ranker", "mlp": "MLP"}
CONFIG_KEYS = {"logreg": ["C", "class_weight"], "gnb": ["var_smoothing"],
               "knn": ["n_neighbors", "weights", "point_step"],
               "rf": ["n_estimators", "min_samples_leaf", "max_samples"],
               "lgbm": ["learning_rate", "num_leaves", "min_child_samples", "best_iteration"],
               "lgbm_tuned": ["learning_rate", "num_leaves", "min_child_samples", "best_iteration"],
               "lgbm_ranker": ["learning_rate", "num_leaves", "best_iteration"],
               "mlp": ["hidden_layer_sizes", "alpha", "learning_rate_init", "best_epoch_final_model"]}


def load_runs() -> dict:
    names = pd.read_csv(RESULTS_DIR / "summary.csv")["name"].tolist()
    runs = {}
    for n in names:
        path = RESULTS_DIR / f"{n}.json"
        if path.exists():
            runs[n] = json.loads(path.read_text())
    return runs


def key_config(name: str, cfg: dict) -> str:
    keys = CONFIG_KEYS.get(name, CONFIG_KEYS.get(name.replace("_tuned", ""), []))
    parts = []
    for k in keys:
        v = cfg.get(k)
        if v is None:
            continue
        if isinstance(v, float):
            v = f"{v:.3g}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def comparison_table(runs: dict) -> pd.DataFrame:
    rows = []
    for n, r in runs.items():
        cv, st, cfg = r["metrics"]["cv"], r["metrics"]["student_test"], r["config"]
        rows.append({
            "model": LABELS.get(n, n),
            "algorithm_family": cfg.get("family"),
            "key_configuration": key_config(n, cfg),
            "cv_macro_f1_candidate": cv["cand_macro_f1_mean"],
            "cv_macro_f1_candidate_std": cv["cand_macro_f1_std"],
            "cv_macro_f1_observation": cv["obs_macro_f1_conf_mean"],
            "cv_top1": cv["top1_conf_mean"],
            "cv_top1_std": cv["top1_conf_std"],
            "st_top1": st["top1_conf"],
            "st_top1_all_points": st["top1_all"],
            "route_continuity_st": st.get("continuity"),
            "st_macro_f1_candidate": st["cand_macro_f1"],
            "st_macro_f1_observation": st["obs_macro_f1_conf"],
            "st_precision": st["cand_precision"],
            "st_recall": st["cand_recall"],
            "cv_log_loss": cv["log_loss_mean"],
            "st_log_loss": st["log_loss"],
            "train_time_s": r["metrics"]["train_time_s"],
            "inference_ms_per_1k_rows": r["metrics"]["infer_ms_per_1k_rows"],
            "_name": n,
        })
    return pd.DataFrame(rows).sort_values("cv_top1", ascending=False).reset_index(drop=True)


def bar(ax, labels, values, err=None, color="#3b6ea5", fmt="{:.3f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, xerr=err, color=color, capsize=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    offsets = np.zeros(len(values)) if err is None else np.asarray(err, dtype=float)
    for i, (v, e) in enumerate(zip(values, offsets)):
        ax.text(v + e, i, "  " + fmt.format(v), va="center", fontsize=8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    t = comparison_table(runs)
    t.drop(columns="_name").to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    labels = t["model"].tolist()

    # 1) CV macro F1 (both definitions) with fold std
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(t) + 2))
    bar(axes[0], labels, t["cv_macro_f1_candidate"], t["cv_macro_f1_candidate_std"])
    axes[0].set_title("CV macro F1 - candidate level (mean +/- fold std)")
    obs_std = [runs[n]["metrics"]["cv"]["obs_macro_f1_conf_std"] for n in t["_name"]]
    bar(axes[1], labels, t["cv_macro_f1_observation"], obs_std, color="#5aa469")
    axes[1].set_title("CV macro F1 - observation level (mean +/- fold std)")
    for a in axes:
        a.set_xlim(0.6, 1.0)   # zoomed: all models are above 0.6
    fig.suptitle("Cross-validation macro F1, GroupKFold(5) on 80 train trips")
    fig.tight_layout()
    fig.savefig(OUT / "cv_macro_f1.png", dpi=150)
    plt.close(fig)

    # 2) Student-test top-1 and macro F1
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(t) + 2))
    y = np.arange(len(t))
    h = 0.27
    for k, (col, name, c) in enumerate([("st_top1", "Top-1 accuracy (confident points)", "#3b6ea5"),
                                        ("st_macro_f1_candidate", "Macro F1 (candidate)", "#e08214"),
                                        ("st_macro_f1_observation", "Macro F1 (observation)", "#5aa469")]):
        ax.barh(y + (k - 1) * h, t[col], height=h, label=name, color=c)
        for i, v in enumerate(t[col]):
            ax.text(v, i + (k - 1) * h, f" {v:.3f}", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Student-test (10 trips): Top-1 accuracy and macro F1")
    fig.tight_layout()
    fig.savefig(OUT / "student_test_top1_f1.png", dpi=150)
    plt.close(fig)

    # 3) Log loss
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 2))
    ax.barh(y - 0.2, t["cv_log_loss"], height=0.4, label="CV (mean of folds)", color="#3b6ea5")
    ax.barh(y + 0.2, t["st_log_loss"], height=0.4, label="Student-test", color="#e08214")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("binary log loss (lower = better calibrated probabilities)")
    ax.legend()
    ax.set_title("Log loss: the HMM uses these probabilities as emissions")
    fig.tight_layout()
    fig.savefig(OUT / "log_loss.png", dpi=150)
    plt.close(fig)

    # 4) Fold stability
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, lab in zip(t["_name"], labels):
        f = RESULTS_DIR / f"{n}_cv_folds.csv"
        if f.exists():
            folds = pd.read_csv(f)
            ax.plot(folds["fold"], folds["top1_conf"], "-o", label=lab)
    ax.set_xlabel("CV fold (same trip split for every model)")
    ax.set_ylabel("Top-1 accuracy (confident points)")
    ax.set_xticks(range(5))
    ax.legend(fontsize=8)
    ax.set_title("Fold stability: which trips are hard is shared across models")
    fig.tight_layout()
    fig.savefig(OUT / "fold_stability.png", dpi=150)
    plt.close(fig)

    # 5) Accuracy vs inference speed
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(t["inference_ms_per_1k_rows"], t["st_top1"], s=60, color="#3b6ea5")
    for _, r in t.iterrows():
        ax.annotate(r["model"], (r["inference_ms_per_1k_rows"], r["st_top1"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time (ms per 1,000 candidate rows, log scale)")
    ax.set_ylabel("Student-test Top-1 accuracy")
    ax.set_title("Accuracy vs inference cost")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_speed.png", dpi=150)
    plt.close(fig)

    # 6) Candidate-level confusion matrices on student-test (row-normalised)
    n_models = len(t)
    cols = min(4, n_models)
    rows_ = int(np.ceil(n_models / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 3.2 * rows_), squeeze=False)
    for ax, n, lab in zip(axes.flat, t["_name"], labels):
        cm = np.array(runs[n]["metrics"]["student_test"]["confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1], ["pred wrong", "pred true arc"], fontsize=7)
        ax.set_yticks([0, 1], ["wrong arc", "true arc"], fontsize=7)
        ax.set_title(lab, fontsize=9)
    for ax in list(axes.flat)[n_models:]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices, student-test, threshold 0.5 (row-normalised)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    show = t.drop(columns="_name")[["model", "algorithm_family", "cv_macro_f1_candidate", "cv_top1", "st_top1",
                                     "route_continuity_st", "st_macro_f1_candidate", "st_log_loss",
                                     "inference_ms_per_1k_rows"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {RESULTS_DIR / 'model_comparison.csv'} and {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()

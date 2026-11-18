"""Feature-group ablation: how much does each group of features contribute?

Run:  python -m scripts.feature_ablation              (11 CV runs, ~10 min)
      python -m scripts.feature_ablation --plot-only  (redraw the figure from the saved CSV)

For each feature group we retrain LightGBM WITHOUT that group, using exactly the same
GroupKFold(5) folds, seed and confident-rows-only fitting as the model comparison, and
measure the drop in CV Top-1 and the change in log loss. "Only this group" runs show what
each group can do on its own.

Only the 80 TRAIN trips are used (cross-validation). Student-test and evaluator trips are
not touched, so this cannot leak into model or feature choices.

This justifies the feature set with evidence rather than intuition. The feature set
itself is NOT changed by this script; every model keeps all 18 features.

Outputs:
  experiments/results/feature_ablation.csv
  reports/figures/feature_ablation.png
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import evaluate as ev  # noqa: E402
from pathsnap import models  # noqa: E402
from pathsnap.config import FIGURES_DIR, RESULTS_DIR, SEED, TRAIN  # noqa: E402
from pathsnap.features import FEATURE_COLUMNS  # noqa: E402

GROUPS = {
    "Relative distance": ["f_dist_gap", "f_dist_rank", "f_dist", "f_n_cands"],
    "Direction": ["f_heading_diff", "f_heading_gap", "f_heading_reliable"],
    "Neighbourhood": ["f_ctx_gap", "f_ctx_dist"],
    "Topology / geometry": ["f_endpoint_dist", "f_arc_len", "f_proj_frac", "f_in_deg", "f_out_deg"],
    "GPS quality / motion": ["f_speed", "f_jitter", "f_dt_prev", "f_dt_next"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURE_COLUMNS), "groups must cover every feature exactly once"


def run_cv(feats, train, points):
    t0 = time.perf_counter()
    cv = ev.group_cv(lambda: models.make("lgbm"), train, n_splits=5, seed=SEED, feature_cols=feats,
                     points=points, verbose=False)
    s = cv.summary
    return {"cv_top1": s["top1_conf_mean"], "cv_top1_std": s["top1_conf_std"],
            "cv_log_loss": s["log_loss_mean"], "cv_cand_macro_f1": s["cand_macro_f1_mean"],
            "fold_top1": cv.folds["top1_conf"].tolist(), "seconds": time.perf_counter() - t0}


def run_all() -> pd.DataFrame:
    train, points = ev.load_table(TRAIN), ev.load_points(TRAIN)
    runs = [("All 18 features", "baseline", FEATURE_COLUMNS)]
    for g, fs in GROUPS.items():
        runs.append((f"without {g}", "drop", [f for f in FEATURE_COLUMNS if f not in fs]))
    for g, fs in GROUPS.items():
        runs.append((f"only {g}", "only", fs))

    rows = []
    for name, kind, feats in runs:
        r = run_cv(feats, train, points)
        rows.append({"experiment": name, "kind": kind, "n_features": len(feats), **r})
        print(f"{name:34s} {len(feats):2d} features | CV top1 {r['cv_top1']:.4f} +/- {r['cv_top1_std']:.4f} | "
              f"log loss {r['cv_log_loss']:.4f} | {r['seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    base = df.loc[df["kind"] == "baseline"].iloc[0]
    df["delta_top1_vs_all"] = df["cv_top1"] - base["cv_top1"]
    # Paired per-fold comparison: in how many of the 5 folds is the variant worse than all features?
    df["folds_worse_than_all"] = [sum(a < b for a, b in zip(f, base["fold_top1"])) for f in df["fold_top1"]]
    df = df.drop(columns="fold_top1")
    df.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)
    return df


def plot(df: pd.DataFrame):
    base = df.loc[df["kind"] == "baseline"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    d = df[df["kind"] == "drop"]
    labels = [e.replace("without ", "") for e in d["experiment"]]
    vals = (d["delta_top1_vs_all"] * 100).to_numpy()
    ax = axes[0]
    ax.barh(labels, vals, color="#d7301f")
    for i, v in enumerate(vals):
        ax.text(v - 0.4, i, f"{v:.2f} pts", va="center", ha="right", fontsize=8)
    ax.set_xlim(-36, 1)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("change in CV Top-1 (percentage points); worse in all 5 folds for every group")
    ax.set_title("Remove one group (drop vs all 18 features)", fontsize=10)

    d = df[df["kind"] == "only"]
    labels = [e.replace("only ", "") for e in d["experiment"]]
    vals = d["cv_top1"].to_numpy()
    ax = axes[1]
    ax.barh(labels, vals, color="#3b6ea5")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(base["cv_top1"], color="black", lw=0.8, ls="--", label=f"all 18 features ({base['cv_top1']:.3f})")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("CV Top-1 accuracy (confident points)")
    ax.set_title("Use only one group", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Feature-group ablation, LightGBM, GroupKFold(5) on the 80 train trips")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_ablation.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / 'feature_ablation.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from feature_ablation.csv")
    args = ap.parse_args()
    df = pd.read_csv(RESULTS_DIR / "feature_ablation.csv") if args.plot_only else run_all()
    plot(df)


if __name__ == "__main__":
    main()

"""Train and evaluate the MLP (Member A) through the shared harness.

Run:  python -m scripts.train_mlp

Same protocol as scripts/train_models.py, so the numbers are directly comparable:
  1. GroupKFold(5) by trip on the 80 train trips, fitting on confident rows only.
  2. Refit on all 80 train trips, then score the 10 student-test trips.
  3. Log experiments/results/mlp.json + a row in summary.csv, save models/mlp.joblib.

Also written: mlp_cv_folds.csv, the real per-epoch train/validation loss of the final
model (mlp_loss_curve.csv / reports/figures/mlp_loss_curve.png), and the per-fold best
epochs (evidence for or against over/underfitting).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap import evaluate as ev  # noqa: E402
from pathsnap.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, SEED  # noqa: E402
from pathsnap.mlp import EpochMLP, make_mlp  # noqa: E402
from scripts.train_models import load_data  # noqa: E402

NAME = "mlp"


def plot_history(history: pd.DataFrame, best_epoch: int, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], "-o", ms=3, label="train (training trips)")
    ax.plot(history["epoch"], history["val_loss"], "--o", ms=3, label="validation (held-out trips)")
    ax.axvline(best_epoch, color="grey", lw=0.8, ls=":", label=f"best epoch = {best_epoch} (weights kept)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary log loss")
    ax.set_title("MLP: training curve (real per-epoch values)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train, st, points, arcs = load_data(fake=False)
    feats = ev.feature_columns(train)
    params = EpochMLP().get_params()
    cfg = {"model": NAME, "family": "neural network", "seed": SEED, **params,
           "activation": "relu", "solver": "adam", "loss": "binary cross-entropy",
           "scaling": "log1p(skewed) + StandardScaler, fitted on training rows only",
           "features": feats, "train_confident_only": True}
    print(f"=== {NAME} on {len(train):,} train rows, {len(feats)} features", flush=True)

    best_epochs = []

    def make():
        m = make_mlp()
        best_epochs.append(m)          # keep a handle to read best_epoch_ after each fold
        return m

    cv = ev.group_cv(make, train, n_splits=5, seed=SEED, feature_cols=feats, points=points)
    cv.folds["best_epoch"] = [m.best_epoch_ for m in best_epochs]
    cv.folds["epochs_run"] = [len(m.history_) for m in best_epochs]
    cv.folds.to_csv(RESULTS_DIR / f"{NAME}_cv_folds.csv", index=False)
    cv.oof.to_parquet(PROCESSED_DIR / f"oof_{NAME}.parquet", index=False)

    model, st_m, st_pred = ev.fit_and_score(make_mlp, train, st, feature_cols=feats,
                                            points=points, arcs_by_trip=arcs)
    st_pred[ev.KEY + ["arc_id", "label", "proba"]].to_parquet(PROCESSED_DIR / f"st_pred_{NAME}.parquet", index=False)
    ev.save_model(NAME, model, feats, cfg, models_dir=ev.MODELS_DIR)

    history = pd.DataFrame(model.history_)
    history.to_csv(RESULTS_DIR / f"{NAME}_loss_curve.csv", index=False)
    plot_history(history, model.best_epoch_, FIGURES_DIR / f"{NAME}_loss_curve.png")
    cfg["best_epoch_final_model"] = model.best_epoch_
    cfg["best_epoch_per_fold"] = cv.folds["best_epoch"].tolist()

    ev.log_result(NAME, cfg, {"cv": cv.summary, "student_test": st_m,
                              "train_time_s": st_m["train_time_s"], "infer_time_s": st_m["infer_time_s"],
                              "infer_ms_per_1k_rows": st_m["infer_ms_per_1k_rows"]})
    s = cv.summary
    print(f"  CV  top1_conf {s['top1_conf_mean']:.4f} +/- {s['top1_conf_std']:.4f} | obsF1 {s['obs_macro_f1_conf_mean']:.4f} | "
          f"candF1 {s['cand_macro_f1_mean']:.4f} | logloss {s['log_loss_mean']:.4f}")
    print(f"  ST  top1_conf {st_m['top1_conf']:.4f} | top1_all {st_m['top1_all']:.4f} | obsF1 {st_m['obs_macro_f1_conf']:.4f} | "
          f"candF1 {st_m['cand_macro_f1']:.4f} | logloss {st_m['log_loss']:.4f} | continuity {st_m['continuity']:.4f} | "
          f"train {st_m['train_time_s']:.1f}s | infer {st_m['infer_ms_per_1k_rows']:.1f} ms/1k rows")
    print(f"  best epoch per fold {cv.folds['best_epoch'].tolist()}, final model best epoch {model.best_epoch_}")


if __name__ == "__main__":
    main()

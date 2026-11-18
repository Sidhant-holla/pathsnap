"""Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.

    python -m scripts.hpo --model lgbm --trials 50
    python -m scripts.hpo --fake --trials 6           # smoke test (writes to _fake/ folders)

Objective: mean observation Top-1 on confident points over GroupKFold(5) by trip on the 80
TRAIN trips only, fitting on confident rows like the model comparison
(same folds and seed as the model comparison). Student-test is only used AFTER the search,
to compare the tuned model with the untuned one; the evaluator partition is never touched.

Each trial also records mean log loss, macro F1s and fit time as user attributes, and a
MedianPruner stops clearly bad trials after a couple of folds. The study is stored in
experiments/hpo/optuna.db, so an interrupted run resumes where it stopped.

Outputs (experiments/hpo/): optuna.db, trials.csv, best_params.json, tuned_vs_untuned.csv;
reports/figures/: hpo_history.png, hpo_param_importance.png. The refitted tuned model is
logged as <model>_tuned like any other experiment and saved to models/<model>_tuned.joblib.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from pathsnap import evaluate as ev  # noqa: E402
from pathsnap import models  # noqa: E402
from pathsnap.config import FIGURES_DIR, MODELS_DIR, REPO_ROOT, RESULTS_DIR, SEED  # noqa: E402
from scripts.train_models import load_data  # noqa: E402


# Search spaces. Ranges are chosen around the defaults, on log scales where the effect is
# multiplicative (learning rate, leaves, regularisation strengths).
def space_lgbm(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def space_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_features": trial.suggest_float("max_features", 0.2, 0.9),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.05, 0.5),
    }


SPACES = {"lgbm": space_lgbm, "rf": space_rf, "lgbm_ranker": space_lgbm}


def objective_factory(model_name, train, points, n_splits, seed):
    """Same folds, same confident-rows-only fitting and same metric as evaluate.group_cv."""
    feats = ev.feature_columns(train)
    train = train.reset_index(drop=True)
    X, y, groups = train[feats], train["label"].to_numpy(), train["trip_id"].to_numpy()
    usable = ev.confident_rows(train).to_numpy()
    splits = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups))

    def objective(trial):
        params = SPACES[model_name](trial)
        top1s, lls, cf1, of1, fit_s = [], [], [], [], 0.0
        for k, (tr, va) in enumerate(splits):
            fit_idx = tr[usable[tr]]
            model = models.make(model_name, **params)
            t0 = time.perf_counter()
            ev._fit(model, X.iloc[fit_idx], y[fit_idx], train.loc[fit_idx, ev.KEY])
            fit_s += time.perf_counter() - t0
            p = ev.predict_proba(model, X.iloc[va], train.loc[va, ev.KEY])
            m = ev.score(train.loc[va].assign(proba=p), points=points)
            top1s.append(m["top1_conf"]); lls.append(m["log_loss"])
            cf1.append(m["cand_macro_f1"]); of1.append(m["obs_macro_f1_conf"])
            trial.report(float(np.mean(top1s)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("top1_conf_std", float(np.std(top1s, ddof=1)))
        trial.set_user_attr("log_loss_mean", float(np.mean(lls)))
        trial.set_user_attr("cand_macro_f1_mean", float(np.mean(cf1)))
        trial.set_user_attr("obs_macro_f1_conf_mean", float(np.mean(of1)))
        trial.set_user_attr("fit_time_s", fit_s)
        if hasattr(model, "best_iteration_"):
            trial.set_user_attr("best_iteration_last_fold", int(model.best_iteration_))
        return float(np.mean(top1s))

    return objective


def _set_title(ax, text):
    # Optuna writes its own title (left-aligned on some plots); clear every slot first so
    # the titles don't overprint.
    for loc in ("left", "center", "right"):
        ax.set_title("", loc=loc)
    ax.set_title(text, loc="left")


def save_plots(study, fig_dir, tag):
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    ax = plot_optimization_history(study)
    _set_title(ax, f"{tag}: optimisation history (objective = mean CV top-1, confident points)")
    ax.figure.set_size_inches(7, 4); ax.figure.tight_layout()
    ax.figure.savefig(fig_dir / "hpo_history.png", dpi=150); plt.close(ax.figure)
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 3:
        ax = plot_param_importances(study)
        _set_title(ax, f"{tag}: hyperparameter importance (fANOVA)")
        ax.figure.set_size_inches(7, 4); ax.figure.tight_layout()
        ax.figure.savefig(fig_dir / "hpo_param_importance.png", dpi=150); plt.close(ax.figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm", choices=list(SPACES))
    ap.add_argument("--trials", type=int, default=50, help="target number of finished trials in the study")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    sub = "_fake" if args.fake else ""
    hpo_dir = REPO_ROOT / "experiments" / "hpo" / sub
    res_dir, mod_dir, fig_dir = (d / sub if sub else d for d in (RESULTS_DIR, MODELS_DIR, FIGURES_DIR))
    for d in (hpo_dir, res_dir, mod_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    train, st, points, arcs = load_data(args.fake)
    study = optuna.create_study(
        study_name=f"{args.model}_top1", direction="maximize",
        storage=f"sqlite:///{(hpo_dir / 'optuna.db').as_posix()}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    remaining = max(0, args.trials - done)
    print(f"study {study.study_name}: {done} trials already done, running {remaining} more")
    t0 = time.perf_counter()
    if remaining:
        study.optimize(objective_factory(args.model, train, points, 5, SEED), n_trials=remaining)
    search_s = time.perf_counter() - t0

    study.trials_dataframe().to_csv(hpo_dir / "trials.csv", index=False)
    save_plots(study, fig_dir, args.model)
    best = study.best_trial
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    total_trial_s = float(sum((t.datetime_complete - t.datetime_start).total_seconds()
                              for t in study.trials if t.datetime_complete))
    info = {"model": args.model, "objective": "mean GroupKFold(5) observation top-1 on confident points, train trips",
            "n_trials": len(study.trials), "trial_states": states, "best_trial": best.number,
            "best_value": best.value, "best_params": best.params, "best_user_attrs": best.user_attrs,
            "search_time_s_this_run": search_s, "search_time_s_all_trials": total_trial_s}
    (hpo_dir / "best_params.json").write_text(json.dumps(info, indent=2))
    print(f"best trial #{best.number}: top1_conf={best.value:.4f} params={best.params}")

    # Refit the best config with the SAME harness as the untuned model: full CV metrics, then
    # fit on all train trips and score student-test.
    name = f"{args.model}_tuned"
    cfg = models.config(args.model, **best.params)
    feats = ev.feature_columns(train)
    cfg.update(features=feats, hpo_best_trial=best.number, hpo_n_trials=len(study.trials))
    print(f"\n=== refit {name}")
    cv = ev.group_cv(lambda: models.make(args.model, **best.params), train, seed=SEED,
                     feature_cols=feats, points=points)
    cv.folds.to_csv(res_dir / f"{name}_cv_folds.csv", index=False)
    model, st_m, _ = ev.fit_and_score(lambda: models.make(args.model, **best.params), train, st,
                                      feature_cols=feats, points=points, arcs_by_trip=arcs)
    ev.save_model(name, model, feats, cfg, models_dir=mod_dir)
    if hasattr(model, "evals_result_"):
        from scripts.train_models import plot_loss_curve
        metric = next(iter(model.evals_result_["val"]))
        plot_loss_curve(model.evals_result_, model.best_iteration_, name, metric, fig_dir / f"{name}_loss_curve.png")
        cfg["best_iteration"] = model.best_iteration_
    ev.log_result(name, cfg, {"cv": cv.summary, "student_test": st_m, "train_time_s": st_m["train_time_s"],
                              "infer_time_s": st_m["infer_time_s"], "infer_ms_per_1k_rows": st_m["infer_ms_per_1k_rows"],
                              "hpo_search_time_s": total_trial_s}, results_dir=res_dir)

    # Tuned vs untuned (the untuned run comes from scripts.train_models).
    rows = {}
    untuned_path = res_dir / f"{args.model}.json"
    if untuned_path.exists():
        u = json.loads(untuned_path.read_text())["metrics"]
        rows["untuned"] = u
    else:
        print(f"(no {untuned_path.name}; run scripts.train_models --models {args.model} first for the comparison)")
    rows["tuned"] = {"cv": cv.summary, "student_test": st_m}
    table = pd.DataFrame({k: {
        "CV top-1 conf (mean)": v["cv"]["top1_conf_mean"],
        "CV top-1 conf (std)": v["cv"]["top1_conf_std"],
        "CV candidate macro F1": v["cv"]["cand_macro_f1_mean"],
        "CV observation macro F1": v["cv"]["obs_macro_f1_conf_mean"],
        "CV log loss": v["cv"]["log_loss_mean"],
        "ST top-1 conf": v["student_test"]["top1_conf"],
        "ST top-1 all points": v["student_test"]["top1_all"],
        "ST candidate macro F1": v["student_test"]["cand_macro_f1"],
        "ST observation macro F1": v["student_test"]["obs_macro_f1_conf"],
        "ST log loss": v["student_test"]["log_loss"],
        "ST continuity (argmax)": v["student_test"].get("continuity"),
        "Inference ms / 1k rows": v["student_test"]["infer_ms_per_1k_rows"],
        "Train time s (80 trips)": v["student_test"]["train_time_s"],
    } for k, v in rows.items()})
    table.to_csv(hpo_dir / "tuned_vs_untuned.csv")
    print("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

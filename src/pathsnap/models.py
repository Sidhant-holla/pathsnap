"""Model definitions (Member C): Logistic Regression, Random Forest, LightGBM, LightGBM ranker.

Each entry in ZOO is (family, default config). `make(name, **overrides)` builds a fresh
unfitted estimator from that config, so the config that gets logged is exactly the one
that was trained. All estimators expose predict_proba(X)[:, 1] = P(candidate is the true arc).

Scaling lives inside a Pipeline wherever the algorithm needs it (Logistic Regression uses
the team's shared preprocessing.make_scaler: log1p on skewed features, then
StandardScaler), so in cross-validation the scaler is fitted on the training folds only.
Trees don't need it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .config import SEED

ZOO = {
    "logreg": ("linear", {
        "C": 1.0, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs",
    }),
    "rf": ("ensemble (bagging)", {
        # max_samples < 1 keeps each tree to a 20% bootstrap: faster, and the saved model
        # stays small enough to commit. min_samples_leaf limits depth/overfitting.
        "n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 25,
        "max_samples": 0.2, "class_weight": None, "n_jobs": -1,
    }),
    "lgbm": ("boosting", {
        "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "reg_alpha": 0.0, "n_estimators": 3000,
        "early_stopping_rounds": 100, "val_frac": 0.15,
    }),
    "lgbm_ranker": ("learning to rank", {
        "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "n_estimators": 3000,
        "early_stopping_rounds": 100, "val_frac": 0.15, "eval_at": 1,
    }),
}


def config(name: str, **overrides) -> dict:
    family, cfg = ZOO[name]
    return {"model": name, "family": family, "seed": SEED, **cfg, **overrides}


def make(name: str, **overrides):
    """Fresh, unfitted estimator for ZOO[name] with optional parameter overrides."""
    cfg = config(name, **overrides)
    params = {k: v for k, v in cfg.items() if k not in ("model", "family", "seed")}
    if name == "logreg":
        from .preprocessing import make_scaler          # shared log1p + StandardScaler (A)
        return Pipeline([("scale", make_scaler()),
                         ("clf", LogisticRegression(random_state=SEED, **params))])
    if name == "rf":
        return RandomForestClassifier(random_state=SEED, **params)
    if name == "lgbm":
        return LGBMEarlyStop(params=params, seed=SEED)
    if name == "lgbm_ranker":
        return LGBMRankerProba(params=params, seed=SEED)
    raise KeyError(name)


def _inner_split(meta: pd.DataFrame, val_frac: float, seed: int):
    """Hold out whole TRIPS (never single rows) of the training data for early stopping.
    The outer CV validation fold is never touched, so CV scores are not optimistic."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    return next(gss.split(meta, groups=meta["trip_id"].to_numpy()))


class LGBMEarlyStop(ClassifierMixin, BaseEstimator):
    """LGBMClassifier with early stopping on a trip-level split of the training data.

    After fit: `evals_result_` holds the real per-iteration binary log loss on the inner
    train and validation trips (the loss curves), and `best_iteration_` is the number of
    trees kept.
    """

    def __init__(self, params: dict | None = None, seed: int = SEED):
        self.params = params
        self.seed = seed

    def fit(self, X, y, meta: pd.DataFrame | None = None):
        import lightgbm as lgb
        p = dict(self.params or {})
        stop = p.pop("early_stopping_rounds", 100)
        val_frac = p.pop("val_frac", 0.15)
        y = np.asarray(y)
        if meta is None:
            raise ValueError("LGBMEarlyStop needs meta (trip_id) for its trip-level early-stopping split")
        tr, va = _inner_split(meta.reset_index(drop=True), val_frac, self.seed)
        self.model_ = lgb.LGBMClassifier(objective="binary", random_state=self.seed, verbose=-1, **p)
        # Passing the SAME objects as the first eval set lets LightGBM recognise it as the
        # training set, so early stopping is decided by the validation trips only.
        Xtr, ytr = X.iloc[tr], y[tr]
        self.model_.fit(Xtr, ytr, eval_X=(Xtr, X.iloc[va]), eval_y=(ytr, y[va]),
                        eval_names=["train", "val"], eval_metric="binary_logloss",
                        callbacks=[lgb.early_stopping(stop, verbose=False)])
        self.evals_result_ = self.model_.evals_result_
        self.best_iteration_ = int(self.model_.best_iteration_ or self.model_.n_estimators)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X, num_iteration=self.best_iteration_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.model_.booster_.feature_importance("gain")


class LGBMRankerProba(ClassifierMixin, BaseEstimator):
    """LightGBM LambdaRank where each GPS point's candidates form one query group, so it
    directly optimises getting the true arc to rank 1 (NDCG@1 = top-1 per point).

    A ranker outputs scores, not probabilities. predict_proba converts them with a
    softmax over each point's candidates, which needs the point keys: call
    `set_query_keys(meta)` before predicting, or pass a DataFrame that still has
    trip_id/obs_idx (they are not used as features). Without keys it falls back to a
    sigmoid of the score, which is not a calibrated probability.
    """

    def __init__(self, params: dict | None = None, seed: int = SEED):
        self.params = params
        self.seed = seed

    @staticmethod
    def _sorted(meta):
        m = meta.reset_index(drop=True)
        order = np.lexsort((m["obs_idx"].to_numpy(), m["trip_id"].astype(str).to_numpy()))
        sizes = m.iloc[order].groupby(["trip_id", "obs_idx"], sort=False).size().to_numpy()
        return order, sizes

    def fit(self, X, y, meta: pd.DataFrame | None = None):
        import lightgbm as lgb
        if meta is None:
            raise ValueError("LGBMRankerProba needs meta (trip_id, obs_idx) to form query groups")
        p = dict(self.params or {})
        stop = p.pop("early_stopping_rounds", 100)
        val_frac = p.pop("val_frac", 0.15)
        eval_at = p.pop("eval_at", 1)
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        meta = meta.reset_index(drop=True)
        tr, va = _inner_split(meta, val_frac, self.seed)
        o_tr, g_tr = self._sorted(meta.iloc[tr])
        o_va, g_va = self._sorted(meta.iloc[va])
        tr, va = tr[o_tr], va[o_va]
        self.model_ = lgb.LGBMRanker(objective="lambdarank", random_state=self.seed, verbose=-1, **p)
        Xtr, ytr = X.iloc[tr], y[tr]
        self.model_.fit(Xtr, ytr, group=g_tr,
                        eval_X=(Xtr, X.iloc[va]), eval_y=(ytr, y[va]), eval_group=[g_tr, g_va],
                        eval_names=["train", "val"], eval_at=[eval_at],
                        callbacks=[lgb.early_stopping(stop, verbose=False)])
        self.evals_result_ = self.model_.evals_result_
        self.best_iteration_ = int(self.model_.best_iteration_ or self.model_.n_estimators)
        self.classes_ = np.array([0, 1])
        self._keys = None
        return self

    def set_query_keys(self, meta: pd.DataFrame):
        self._keys = meta.reset_index(drop=True)[["trip_id", "obs_idx"]]
        return self

    def decision_function(self, X):
        return self.model_.predict(X, num_iteration=self.best_iteration_)

    def predict_proba(self, X):
        s = self.decision_function(X)
        keys = getattr(self, "_keys", None)
        if keys is not None and len(keys) == len(s):
            g = pd.Series(s).groupby([keys["trip_id"].to_numpy(), keys["obs_idx"].to_numpy()])
            e = np.exp(s - g.transform("max").to_numpy())
            p1 = e / pd.Series(e).groupby([keys["trip_id"].to_numpy(), keys["obs_idx"].to_numpy()]).transform("sum").to_numpy()
        else:
            p1 = 1 / (1 + np.exp(-s))
        return np.column_stack([1 - p1, p1])

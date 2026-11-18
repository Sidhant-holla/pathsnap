"""Member B's models: Gaussian Naive Bayes and distance-weighted K-Nearest Neighbours.

Both are registered in models.ZOO ("gnb", "knn"), so they are trained and evaluated
by exactly the same harness and protocol as every other model:
    python -m scripts.train_models --models gnb knn

Both use the shared preprocessing.make_scaler (log1p on heavy-tailed features, then
StandardScaler), fitted on the training folds only.

Gaussian Naive Bayes (probabilistic family)
    Models each feature as a Gaussian per class and assumes the features are
    independent given the class. Our features are strongly correlated (f_dist,
    f_dist_rank, f_dist_gap and f_ctx_dist all measure closeness), so the independence
    assumption is badly violated. NB "counts the same evidence several times" and its
    probabilities become over-confident. It is a fast, interpretable baseline that shows
    what that assumption costs.

K-Nearest Neighbours (instance-based family)
    A candidate gets the (distance-weighted) share of positives among its k most
    similar training candidates in the scaled feature space. No training phase, but
    prediction must search the stored training set, so it is the slowest model at
    inference and its saved model contains the training data.
    To keep that practical we fit on every `point_step`-th GPS point of each trip, with
    ALL candidates of the kept points. GPS points are 1 s apart, so neighbouring points
    are near-duplicates and dropping them loses little information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import make_scaler


def make_gnb(var_smoothing: float = 1e-9) -> Pipeline:
    return Pipeline([("scale", make_scaler()), ("clf", GaussianNB(var_smoothing=var_smoothing))])


class SubsampledKNN(ClassifierMixin, BaseEstimator):
    """Scaled, distance-weighted KNN fitted on every `point_step`-th GPS point.

    fit() receives the (trip_id, obs_idx) keys from the harness (`meta`) so the
    subsample keeps whole points, i.e. all candidates of a kept point.
    """

    def __init__(self, n_neighbors: int = 50, weights: str = "distance", point_step: int = 5,
                 n_jobs: int = -1):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.point_step = point_step
        self.n_jobs = n_jobs

    def fit(self, X, y, meta: pd.DataFrame | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if meta is None:
            keep = np.ones(len(y), dtype=bool)
        else:
            keep = meta["obs_idx"].to_numpy() % self.point_step == 0
        self.model_ = Pipeline([
            ("scale", make_scaler()),
            ("knn", KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=self.n_jobs)),
        ])
        self.model_.fit(X[keep], y[keep])
        self.classes_ = self.model_.classes_
        self.n_fit_rows_ = int(keep.sum())
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

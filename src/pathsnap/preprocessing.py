"""Shared input scaling for scale-sensitive models (Logistic Regression, KNN, SVM, MLP).

Tree models (Random Forest, LightGBM) don't need this, since splits ignore scale.

Two steps, both fitted on training data only (put them INSIDE the sklearn Pipeline so
cross-validation refits them on each training fold):
  1. log1p on heavy-tailed, non-negative features (distances, lengths, time gaps, speed,
     jitter). A 300 m arc and a 3 km arc should not differ 10x in the model's eyes.
  2. StandardScaler: zero mean, unit variance per feature.

Input X must have the columns of features.FEATURE_COLUMNS in that order (a DataFrame
or a numpy array).
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import FEATURE_COLUMNS

SKEWED_FEATURES = [
    "f_dist", "f_dist_gap", "f_arc_len", "f_endpoint_dist", "f_speed",
    "f_ctx_dist", "f_ctx_gap", "f_dt_prev", "f_dt_next", "f_jitter",
]
_SKEWED_IDX = [FEATURE_COLUMNS.index(c) for c in SKEWED_FEATURES]


def log_skewed(X):
    X = np.array(X, dtype=np.float64, copy=True)
    X[:, _SKEWED_IDX] = np.log1p(np.clip(X[:, _SKEWED_IDX], 0, None))
    return X


def make_scaler() -> Pipeline:
    """log1p(skewed features) -> StandardScaler. Use as the first step of a model Pipeline."""
    return Pipeline([
        ("log_skewed", FunctionTransformer(log_skewed)),
        ("standardise", StandardScaler()),
    ])

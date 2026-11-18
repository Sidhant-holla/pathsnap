"""Multi-layer perceptron with a real per-epoch training/validation loss history.

sklearn's MLPClassifier only records the training loss. We need validation loss per
epoch too (for the loss curves in the report and for early stopping), so we train
one epoch at a time with partial_fit and measure both losses after each epoch.

Validation split: the candidate tables are sorted by trip_id, so taking the LAST
`val_fraction` of the rows holds out whole trips (apart from at most one partial
trip at the boundary). A random row split would put neighbouring 1-second points of
the same trip on both sides and make the validation loss look too good.

Objective: binary cross-entropy (log loss) of "is this candidate the true arc",
optimised with Adam. No class weighting: we want calibrated probabilities, because
the HMM uses them directly as emission probabilities.

After training, the weights from the epoch with the lowest validation loss are
restored (early stopping with patience).
"""
import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from .config import SEED
from .preprocessing import make_scaler


class EpochMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size=1024, max_epochs=60, patience=6, val_fraction=0.1,
                 loss_sample=200_000, random_state=SEED):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.loss_sample = loss_sample
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)

        n_val = int(len(y) * self.val_fraction)
        tr, va = np.arange(len(y) - n_val), np.arange(len(y) - n_val, len(y))

        # The scaler is fitted on the training part only.
        self.scaler_ = make_scaler().fit(X[tr])
        Xtr, Xva = self.scaler_.transform(X[tr]), self.scaler_.transform(X[va])
        ytr, yva = y[tr], y[va]
        # Measure the training loss on a fixed random sample (the full set is slow).
        sample = rng.choice(len(ytr), size=min(self.loss_sample, len(ytr)), replace=False)

        self.net_ = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation="relu",
                                  solver="adam", alpha=self.alpha, batch_size=self.batch_size,
                                  learning_rate_init=self.learning_rate_init, random_state=self.random_state)
        self.classes_ = np.array([0, 1])
        self.history_ = []
        best_loss, best_net, bad_epochs = np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(ytr))          # new shuffle each epoch
            self.net_.partial_fit(Xtr[order], ytr[order], classes=self.classes_)
            train_loss = log_loss(ytr[sample], self.net_.predict_proba(Xtr[sample])[:, 1], labels=[0, 1])
            val_loss = log_loss(yva, self.net_.predict_proba(Xva)[:, 1], labels=[0, 1])
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_loss - 1e-5:
                best_loss, best_net, bad_epochs = val_loss, copy.deepcopy(self.net_), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net_ = best_net
        self.best_epoch_ = int(np.argmin([h["val_loss"] for h in self.history_]) + 1)
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(self.scaler_.transform(np.asarray(X, dtype=np.float64)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_mlp(**params) -> EpochMLP:
    """Factory used by training/CV code. The scaler lives inside EpochMLP so it can split train/val first."""
    return EpochMLP(**params)

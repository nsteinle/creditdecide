"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             proba: np.ndarray | None = None) -> dict[str, float]:
    """Label metrics for any classifier, plus ranking and calibration metrics when class
    probabilities are available. Binary problems use class 1 as the positive class."""
    y_true = np.asarray(y_true)
    binary = len(np.unique(y_true)) == 2
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary" if binary else "macro",
                       zero_division=0),
    }
    if proba is not None and binary:
        p1 = proba[:, 1]
        out["roc_auc"] = roc_auc_score(y_true, p1)
        out["average_precision"] = average_precision_score(y_true, p1)
        out["brier"] = brier_score_loss(y_true, p1)
    elif proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr")
    return {k: float(v) for k, v in out.items()}

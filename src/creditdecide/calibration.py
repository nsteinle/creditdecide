"""Probability calibration.

A model can rank accounts well and still give probabilities that are too high or too low.
Ranking is enough to decide *which* accounts to act on; calibrated probabilities are needed to
answer *how many* defaults to expect (staffing, provisioning: expected loss = PD x LGD x
exposure) and to make per-account expected-cost decisions (creditdecide.decision).

Method
------
The scoring model is fixed. A one-dimensional map from its raw score to a probability is
fitted on the VALIDATION split, which the model never trained on:

* ``none``      identity;
* ``sigmoid``   logistic regression on the log-odds of the raw score (Platt scaling);
* ``isotonic``  monotone step function (pool-adjacent-violators).

The method is chosen by 5-fold stratified cross-validation *within* the validation split on log
loss (a proper scoring rule), with the one-standard-error rule: the simplest method (none, then
sigmoid, then isotonic) whose cross-validated log loss is within one standard error of the best
is used, so a map is only added when it clearly helps. It is then refitted on the whole
validation split. The test split is only used to report the result. Every map is monotone
non-decreasing, so the ranking of accounts, and therefore every threshold decision, is
unchanged by calibration.

The fitted map is stored as plain numbers (two coefficients, or the isotonic breakpoints) in the
model manifest and applied with ``np.interp``, so no extra code needs to be deserialized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

METHODS = ("none", "sigmoid", "isotonic")
EPS = 1e-6
N_BINS = 10


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_map(method: str, score, y) -> dict:
    """Fit one calibration map; returns its JSON-serialisable parameters."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    if method == "none":
        return {"method": "none"}
    if method == "sigmoid":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score)[:, None], y)
        return {"method": "sigmoid", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score, y)
        return {"method": "isotonic", "x": iso.X_thresholds_.tolist(),
                "y": iso.y_thresholds_.tolist()}
    raise ValueError(f"unknown calibration method {method!r}; choose from {METHODS}")


def apply_map(params: dict, score) -> np.ndarray:
    score = np.asarray(score, float)
    m = params["method"]
    if m == "none":
        return np.clip(score, 0.0, 1.0)
    if m == "sigmoid":
        return 1.0 / (1.0 + np.exp(-(params["a"] * _logit(score) + params["b"])))
    if m == "isotonic":
        return np.interp(score, params["x"], params["y"])  # constant beyond the end points
    raise ValueError(f"unknown calibration method {m!r}")


def choose_and_fit(score_val, y_val, method: str = "auto", seed: int = 0,
                   n_folds: int = 5) -> tuple[dict, pd.DataFrame]:
    """Pick a method by cross-validated log loss on the validation split (or use ``method``),
    then fit it on the whole validation split. Returns (params, cross-validation table)."""
    s, y = np.asarray(score_val, float), np.asarray(y_val, int)
    rows = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for m in METHODS:
        ll, br = [], []
        for fit_idx, ev_idx in folds.split(s, y):
            p = apply_map(fit_map(m, s[fit_idx], y[fit_idx]), s[ev_idx])
            ll.append(log_loss(y[ev_idx], np.clip(p, EPS, 1 - EPS), labels=[0, 1]))
            br.append(brier_score_loss(y[ev_idx], p))
        rows.append({"method": m, "cv_log_loss": float(np.mean(ll)),
                     "cv_log_loss_se": float(np.std(ll, ddof=1) / np.sqrt(n_folds)),
                     "cv_brier": float(np.mean(br))})
    table = pd.DataFrame(rows)
    if method == "auto":  # one-standard-error rule; METHODS is ordered simplest first
        best = table.loc[table.cv_log_loss.idxmin()]
        ok = table.cv_log_loss <= best.cv_log_loss + best.cv_log_loss_se
        method = str(table.loc[ok, "method"].iloc[0])
    elif method not in METHODS:
        raise ValueError(f"calibration must be 'auto' or one of {METHODS}")
    table["chosen"] = table["method"] == method
    return fit_map(method, s, y), table


def reliability_table(y, p, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean prediction vs observed rate."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, n_bins), start=1):
        if len(idx) == 0:
            continue
        rows.append({"bin": i, "n": len(idx), "mean_predicted": float(p[idx].mean()),
                     "observed_rate": float(y[idx].mean())})
    return pd.DataFrame(rows)


def calibration_metrics(y, p, n_bins: int = N_BINS) -> dict:
    """Brier score, log loss, expected calibration error (equal-count bins), and the
    portfolio check: expected number of positives (sum of probabilities) vs actual."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    t = reliability_table(y, p, n_bins)
    ece = float(np.sum(t["n"] * (t["mean_predicted"] - t["observed_rate"]).abs()) / len(y))
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "ece": ece,
        "max_bin_gap": float((t["mean_predicted"] - t["observed_rate"]).abs().max()),
        "expected_positives": float(p.sum()),
        "actual_positives": int(y.sum()),
    }

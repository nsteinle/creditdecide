"""Outcome audit by protected group.

Excluding protected attributes from the features does not guarantee equal treatment: other
features can act as proxies, and groups can differ in base rates. Lenders therefore audit the
decision's outcomes by group. This module reports, for each protected attribute present in the
data (whether or not it was used as a feature):

* per-group volume, actual positive (default) rate, flag rate, capture rate (true-positive
  rate), false-alarm rate (non-defaulters flagged) and precision;
* the adverse impact ratio (AIR): each group's favorable-outcome rate (not flagged) divided by
  the highest group's. Values below 0.8 are flagged against the "four-fifths" reference from the
  US Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4(D)). The guidelines say
  a lower rate will "generally" be regarded as evidence of adverse impact, and that smaller
  differences can still count, so it is a reference point rather than a definitive test. It is
  also used informally in fair-lending analysis;
* a proxy check: how well the model's own inputs predict group membership (ROC AUC of a
  logistic regression trained on the training split, scored on the evaluation split). 0.5 means
  the inputs carry no information about the attribute.

These are descriptive statistics, not a legal determination. Differences in base rates can
produce different flag rates without any unfair treatment, and small groups are noisy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FOUR_FIFTHS = 0.8
MIN_GROUP = 50  # groups smaller than this are marked as too small to interpret


def group_labels(values: pd.Series, labels: dict | None = None,
                 bins: tuple | None = None) -> pd.Series:
    """Readable group names: explicit labels for codes, bands for numeric attributes."""
    if bins:
        edges = [-np.inf, *bins, np.inf]
        names = [f"< {bins[0]}"] + [f"{lo}-{hi - 1}" for lo, hi in zip(bins[:-1], bins[1:],
                                                                      strict=True)]
        names.append(f"{bins[-1]}+")
        return pd.cut(values, edges, right=False, labels=names).astype(str)
    if labels:
        return values.map(lambda v: labels.get(v, labels.get(str(v), f"code {v}"))).astype(str)
    return values.astype(str)


def outcome_audit(groups: pd.Series, y: np.ndarray, flag: np.ndarray) -> pd.DataFrame:
    y, flag = np.asarray(y), np.asarray(flag, bool)
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        n, yy, ff = int(m.sum()), y[m], flag[m]
        pos, neg = int(yy.sum()), int((1 - yy).sum())
        rows.append({
            "group": g, "n": n,
            "positive_rate": yy.mean(),
            "flag_rate": ff.mean(),
            "capture_rate": (ff & (yy == 1)).sum() / pos if pos else np.nan,
            "false_alarm_rate": (ff & (yy == 0)).sum() / neg if neg else np.nan,
            "precision": yy[ff].mean() if ff.any() else np.nan,
        })
    t = pd.DataFrame(rows)
    favorable = 1 - t["flag_rate"]
    t["adverse_impact_ratio"] = favorable / favorable.max()
    t["below_four_fifths"] = t["adverse_impact_ratio"] < FOUR_FIFTHS
    t["too_small"] = t["n"] < MIN_GROUP
    return t


def proxy_auc(X_train: np.ndarray, g_train: pd.Series, X_eval: np.ndarray,
              g_eval: pd.Series) -> dict[str, float]:
    """For each group (vs everyone else), ROC AUC of predicting membership from the model's
    processed inputs. Skips groups too rare to fit or score."""
    out = {}
    for g in sorted(g_train.unique(), key=str):
        t, e = (g_train == g).to_numpy(), (g_eval == g).to_numpy()
        if min(t.sum(), (~t).sum()) < MIN_GROUP or min(e.sum(), (~e).sum()) < 10:
            continue
        clf = LogisticRegression(max_iter=2000).fit(X_train, t)
        out[str(g)] = float(roc_auc_score(e, clf.predict_proba(X_eval)[:, 1]))
    return out


def fairness_audit(df: pd.DataFrame, spec, X: np.ndarray, y: np.ndarray, flag: np.ndarray,
                   train_idx: np.ndarray, eval_idx: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Outcome audit on ``eval_idx`` and proxy check, for every protected attribute present."""
    tables, proxies = [], {}
    for col in [c for c in spec.protected if c in df.columns]:
        g = group_labels(df[col], spec.group_labels.get(col), spec.group_bins.get(col))
        t = outcome_audit(g.iloc[eval_idx].reset_index(drop=True), y[eval_idx], flag[eval_idx])
        t.insert(0, "attribute", col)
        tables.append(t)
        proxies[col] = proxy_auc(X[train_idx], g.iloc[train_idx].reset_index(drop=True),
                                 X[eval_idx], g.iloc[eval_idx].reset_index(drop=True))
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return table, proxies

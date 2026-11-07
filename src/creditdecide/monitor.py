"""Drift monitoring with the population stability index (PSI).

A model is fit to one population; it is used on later ones. PSI measures how far a feature's
(or the score's) distribution has moved from the reference sample:

    PSI = sum_bins (actual% - expected%) * ln(actual% / expected%)

Common credit-scorecard reading (e.g. the R `scorecard` package documentation):
< 0.1 stable, 0.1-0.25 some change (check other monitoring metrics), > 0.25 major shift
(investigate before trusting the model). These are conventions, not statistical tests.

The reference is the validation split saved with the model: data the model did not train on,
scored by the final model. Continuous columns use the reference deciles as bin edges;
columns with at most ``MAX_DISCRETE`` distinct values, and categorical columns, use their
values, with one extra bin for values never seen in the reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 10
MAX_DISCRETE = 12
EPS = 1e-4  # floor for empty bins, so ln() stays finite
STABLE, SIGNIFICANT = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(value: float) -> str:
    if value < STABLE:
        return "stable"
    return "moderate shift" if value <= SIGNIFICANT else "significant shift"


def _profile(values: pd.Series, categorical: bool) -> dict:
    v = values.to_numpy()
    if categorical or values.nunique() <= MAX_DISCRETE:
        cats = sorted(map(str, pd.unique(v)))
        counts = pd.Series(v.astype(str)).value_counts()
        return {"kind": "discrete", "values": cats,
                "expected": [float(counts.get(c, 0)) / len(v) for c in cats] + [0.0]}
    edges = np.unique(np.quantile(v.astype(float), np.linspace(0, 1, N_BINS + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, v.astype(float), side="right"),
                         minlength=len(edges) + 1)
    return {"kind": "binned", "edges": edges.tolist(),
            "expected": (counts / len(v)).tolist()}


def build_reference(df: pd.DataFrame, columns: list[str], categorical: tuple[str, ...],
                    scores: np.ndarray, sample: str) -> dict:
    return {
        "sample": sample,
        "n": int(len(df)),
        "features": {c: _profile(df[c], c in categorical) for c in columns},
        "score": _profile(pd.Series(np.asarray(scores, float)), categorical=False),
        "thresholds": {"stable_below": STABLE, "significant_above": SIGNIFICANT},
    }


def drift_report(reference: dict, df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """PSI of every model input and of the score, new data vs the reference sample. The
    binning, the shares and the PSI sum are DuckDB SQL (creditdecide.sql, psi.sql)."""
    from creditdecide import sql

    return sql.drift(reference, df, scores, eps=EPS, stable=STABLE, significant=SIGNIFICANT)

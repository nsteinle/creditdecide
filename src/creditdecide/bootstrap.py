"""Bootstrap confidence intervals for results measured on the test split.

The test split is one sample of customers (6,002 in the bundled data). A different month, or a
different random split, would give somewhat different numbers, so every headline figure gets
an interval.

How it works: test accounts are drawn with replacement, as many as there are, and every result
is recomputed on the new sample; this is repeated (1,000 times by default) and the 2.5th and
97.5th percentiles of the repeats form the 95% interval (the percentile bootstrap, Efron &
Tibshirani 1993, ch. 13). Rows with identical inputs are drawn together, matching the
group-aware split. Comparisons (model minus rule, one model minus another) are computed on the
same draw, so the interval of a difference accounts for the two results moving together.

What is held fixed: the trained models, the threshold, the calibration map and the chosen
rule. The intervals therefore measure how much the test figures depend on which customers
happened to be in the test sample. They do not include the variation from retraining on
different data or with another seed, which would widen them.

Instead of copying rows, each draw is a vector of weights: how many times each row was
drawn (0, 1, 2, ...). Weight 1 for every row reproduces the original figures exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

DEFAULT_REPLICATES = 1000
LEVEL = 0.95


def resample_weights(groups, n_boot: int, seed: int = 0):
    """Yield ``n_boot`` weight vectors, one per bootstrap sample.

    ``groups`` gives each row's group id; groups are drawn with replacement, as many as there
    are, and every row of a group gets the group's draw count.
    """
    g = pd.factorize(np.asarray(groups))[0]
    n_groups = int(g.max()) + 1 if len(g) else 0
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        yield counts[g].astype(float)


def capture_rate(w, y, flag) -> float:
    """Weighted share of positives that are flagged."""
    pos = w * (np.asarray(y) == 1)
    total = pos.sum()
    return float((pos * flag).sum() / total) if total else float("nan")


def precision(w, y, flag) -> float:
    """Weighted share of flagged rows that are positive."""
    f = w * np.asarray(flag, bool)
    total = f.sum()
    return float((f * (np.asarray(y) == 1)).sum() / total) if total else float("nan")


def flagged_fraction(w, flag) -> float:
    return float((w * np.asarray(flag, bool)).sum() / w.sum())


class RankedScores:
    """A score vector prepared once for fast weighted ROC AUC on many bootstrap samples."""

    def __init__(self, y, score):
        self.pos = np.asarray(y) == 1
        _, self.inverse = np.unique(np.asarray(score, float), return_inverse=True)
        self.n_unique = int(self.inverse.max()) + 1 if len(self.inverse) else 0

    def auc(self, w) -> float:
        """Weighted ROC AUC: the probability that a random positive outscores a random
        negative, ties counting one half (the Mann-Whitney form)."""
        pos = np.bincount(self.inverse, weights=w * self.pos, minlength=self.n_unique)
        neg = np.bincount(self.inverse, weights=w * ~self.pos, minlength=self.n_unique)
        below = np.cumsum(neg) - neg
        denom = pos.sum() * neg.sum()
        return float((pos * (below + 0.5 * neg)).sum() / denom) if denom else float("nan")


def weighted_roc_auc(y, score, w=None) -> float:
    """ROC AUC with row weights (equal to scikit-learn's roc_auc_score with sample_weight)."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return RankedScores(y, score).auc(w)


def cost(w, y, flag, cost_fp: float, missed) -> float:
    """Weighted cost of a rule: cost_fp per flagged negative plus the missed cost of every
    positive that is not flagged."""
    y, flag = np.asarray(y) == 1, np.asarray(flag, bool)
    return float(cost_fp * (w * (flag & ~y)).sum() + (w * missed * (~flag & y)).sum())


def saving_vs_best_trivial(w, y, flag, cost_fp: float, missed) -> float:
    """Cost saving against the better of flagging nobody and flagging everybody, as a
    fraction of that strategy's cost (same definition as decision.summarize)."""
    none = cost(w, y, np.zeros(len(y), bool), cost_fp, missed)
    everyone = cost(w, y, np.ones(len(y), bool), cost_fp, missed)
    best = min(none, everyone)
    return float((best - cost(w, y, flag, cost_fp, missed)) / best) if best else 0.0


def bootstrap(statistic: Callable[[np.ndarray], dict], groups, n_boot: int = DEFAULT_REPLICATES,
              seed: int = 0, level: float = LEVEL, differences=(),
              units: dict | None = None) -> pd.DataFrame:
    """Percentile bootstrap intervals for every number ``statistic`` returns.

    ``statistic(w)`` takes a vector of row weights and returns ``{name: value}``. Names listed
    in ``differences`` are comparisons; for those the table says whether the interval excludes
    zero, i.e. whether the sign of the difference is clear at this confidence level. ``units``
    maps names to "fraction", "relative", "auc" or "ratio" (used only for printing; default
    "fraction").
    """
    units = units or {}
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    n = len(np.asarray(groups))
    estimate = statistic(np.ones(n))
    draws = {k: np.empty(n_boot) for k in estimate}
    for b, w in enumerate(resample_weights(groups, n_boot, seed)):
        for k, v in statistic(w).items():
            draws[k][b] = v
    tail = (1 - level) / 2
    rows = []
    for k, v in estimate.items():
        d = draws[k]
        lo, hi = np.nanquantile(d, [tail, 1 - tail])
        is_diff = k in differences
        rows.append({"quantity": k, "type": "difference" if is_diff else "estimate",
                     "unit": units.get(k, "fraction"),
                     "estimate": float(v), "ci_low": float(lo), "ci_high": float(hi),
                     "std_error": float(np.nanstd(d, ddof=1)) if n_boot > 1 else float("nan"),
                     "excludes_zero": bool(lo > 0 or hi < 0) if is_diff else None})
    out = pd.DataFrame(rows)
    out.attrs.update({"replicates": n_boot, "level": level, "seed": seed})
    return out


def format_interval(row) -> str:
    """One line per quantity: estimate and interval in readable units. A difference of two
    fractions is in percentage points; a relative error (unit "relative") in percent."""
    unit, diff = row["unit"], row["type"] == "difference"
    sign = "+" if diff else ""
    if unit in ("fraction", "relative"):
        suffix = " points" if diff and unit == "fraction" else "%"
        est, lo, hi = (f"{100 * row[c]:{sign}.1f}" for c in ("estimate", "ci_low", "ci_high"))
        text = (f"{est}{suffix} [{lo}, {hi}]" if suffix == " points"
                else f"{est}% [{lo}%, {hi}%]")
    else:
        digits = 3 if unit == "auc" else 2
        est, lo, hi = (f"{row[c]:{sign}.{digits}f}" for c in ("estimate", "ci_low", "ci_high"))
        text = f"{est} [{lo}, {hi}]"
    if diff:
        text += "; sign clear" if row["excludes_zero"] else "; includes zero"
    return f"{row['quantity']}: {text}"

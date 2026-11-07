"""Reason codes: why was this account flagged?

Method (the first method named in the Regulation B official interpretation, comment
9(b)(2)-5: identify "the factors for which the applicant's score fell furthest below the average
score" of applicants at or slightly above the minimum passing score):

1. Reference profile: validation-split accounts that the rule did NOT flag but that were closest
   to being flagged (the top ``REFERENCE_FRACTION`` of not-flagged accounts by decision margin),
   i.e. the marginal "pass" population. For each input: the median, or the most common value
   for categorical and low-cardinality columns.
2. Inputs are grouped into business-meaningful factors (e.g. all six bill amounts form one
   factor), because correlated inputs neutralized one at a time understate their joint effect.
3. For a flagged account, each factor in turn is set to the reference values and the account is
   re-scored. The factor's impact is the fall in the decision margin (score minus threshold, or
   the expected-cost margin for the expected-loss rule). Factors are ranked by impact; only
   factors with a positive impact are reasons. Up to four are reported (comment 9(b)(2)-1:
   "disclosure of more than four reasons is not likely to be helpful").

This is an approximation of the model, so its accuracy is measured (CFPB Circular 2022-03:
creditors must be able to verify that explanations of complex models are accurate):

* sufficiency: share of flagged accounts that would no longer be flagged if their stated
  reasons were set to the reference values (and, more strictly, if only the top reason was);
* ranking check: average margin drop from neutralizing the top reason vs a randomly chosen
  factor.

Reasons describe the model, not causes in the world, and are computed only for flagged rows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

MAX_REASONS = 4
REFERENCE_FRACTION = 0.2
MAX_DISCRETE = 12

# decide(frame) -> (decision margin, flag, calibrated probability)
Decider = Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray, np.ndarray]]


def factor_groups(features: list[str], declared: dict[str, tuple[str, ...]] | None) -> dict:
    """Declared factors restricted to model inputs; every other input becomes its own factor."""
    groups, used = {}, set()
    for label, cols in (declared or {}).items():
        cols = tuple(c for c in cols if c in features)
        overlap = used & set(cols)
        if overlap:
            raise ValueError(f"columns {sorted(overlap)} belong to more than one reason factor")
        if cols:
            groups[label] = cols
            used |= set(cols)
    for c in features:
        if c not in used:
            groups[c] = (c,)
    return groups


def reference_profile(df_val: pd.DataFrame, margin_val: np.ndarray, flag_val: np.ndarray,
                      features: list[str], categorical: tuple[str, ...]) -> dict:
    """Typical inputs of the marginally not-flagged validation accounts."""
    margin_val, flag_val = np.asarray(margin_val, float), np.asarray(flag_val, bool)
    ok = np.flatnonzero(~flag_val)
    if len(ok) == 0:
        raise ValueError("every validation account is flagged; no reference population")
    n = max(1, int(np.ceil(REFERENCE_FRACTION * len(ok))))
    near = ok[np.argsort(-margin_val[ok], kind="stable")[:n]]
    ref = df_val.iloc[near]
    values = {}
    for c in features:
        col = ref[c]
        if c in categorical or col.nunique() <= MAX_DISCRETE:
            values[c] = col.mode(dropna=True).sort_values().iloc[0]
        else:
            values[c] = float(col.median())
    values = {c: (v.item() if hasattr(v, "item") else v) for c, v in values.items()}
    return {"population": "validation accounts not flagged, closest to the decision boundary",
            "fraction_of_not_flagged": REFERENCE_FRACTION, "n": int(n), "values": values}


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)):
        return str(v)
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _neutralise(rows: pd.DataFrame, cols, ref: dict) -> pd.DataFrame:
    out = rows.copy()
    for c in cols:
        out[c] = ref[c]
    return out


def _assign(frame: pd.DataFrame, mask: np.ndarray, cols, ref: dict) -> None:
    """Set ``cols`` to the reference values on the masked rows, upcasting integer columns
    when a reference value is fractional (pandas refuses a lossy in-place assignment)."""
    for c in cols:
        v = ref[c]
        if (pd.api.types.is_integer_dtype(frame[c]) and isinstance(v, float)
                and not float(v).is_integer()):
            frame[c] = frame[c].astype(float)
        frame.loc[mask, c] = v


def explain(rows: pd.DataFrame, decide: Decider, groups: dict, reference: dict,
            k: int = MAX_REASONS, seed: int = 0, id_column: str | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Reason codes for ``rows`` (normally the flagged ones).

    Returns (wide table: one row per account with reason_1..k, long table: every factor's
    impact per account, accuracy summary)."""
    if not 1 <= k <= len(groups):
        raise ValueError(f"k must be between 1 and the number of factors ({len(groups)})")
    rows = rows.reset_index(drop=True)
    ref = reference["values"]
    labels = list(groups)
    base, _, p_base = decide(rows)
    impact = np.empty((len(rows), len(labels)))
    p_impact = np.empty_like(impact)
    for j, label in enumerate(labels):
        m, _, p = decide(_neutralise(rows, groups[label], ref))
        impact[:, j] = base - m
        p_impact[:, j] = p_base - p

    order = np.argsort(-impact, axis=1, kind="stable")
    ids = rows[id_column].to_numpy() if id_column else np.arange(len(rows))
    id_key = id_column or "row"
    wide = {id_key: ids, "margin": base, "probability": p_base}
    long = []
    stated = []
    for i in range(len(rows)):
        picks = [j for j in order[i] if impact[i, j] > 0][:k]
        stated.append(picks)
        for r, j in enumerate(order[i], start=1):
            cols = groups[labels[j]]
            long.append({
                id_key: ids[i],
                "rank": r, "factor": labels[j], "impact": float(impact[i, j]),
                "probability_impact": float(p_impact[i, j]),
                "is_reason": j in picks,
                "account_values": ", ".join(_fmt(rows.at[i, c]) for c in cols),
                "reference_values": ", ".join(_fmt(ref[c]) for c in cols),
                "columns": ", ".join(cols),
            })
    for r in range(k):
        wide[f"reason_{r + 1}"] = [labels[p[r]] if len(p) > r else "" for p in stated]
        wide[f"impact_{r + 1}"] = [float(impact[i, p[r]]) if len(p) > r else np.nan
                                   for i, p in enumerate(stated)]

    # accuracy checks
    # one re-scoring with every account's own stated reasons set to the reference values
    joint = rows.copy()
    for j, label in enumerate(labels):
        mask = np.array([j in p for p in stated], dtype=bool)
        if mask.any():
            _assign(joint, mask, groups[label], ref)
    _, still_flagged, _ = decide(joint)
    has_reason = np.array([bool(p) for p in stated])
    flipped = has_reason & ~np.asarray(still_flagged, bool)
    top = rows.copy()
    for j, label in enumerate(labels):
        mask = np.array([bool(p) and p[0] == j for p in stated], dtype=bool)
        if mask.any():
            _assign(top, mask, groups[label], ref)
    _, still_top, _ = decide(top)
    flipped_top = has_reason & ~np.asarray(still_top, bool)
    rng = np.random.default_rng(seed)
    rand = rng.integers(0, len(labels), len(rows))
    top_drop = impact[np.arange(len(rows)), order[:, 0]]
    rand_drop = impact[np.arange(len(rows)), rand]
    counts = pd.Series([labels[p[0]] for p in stated if p]).value_counts()
    summary = {
        "accounts_explained": int(len(rows)),
        "reasons_per_account": k,
        "accounts_without_a_positive_reason": int(sum(not p for p in stated)),
        "sufficiency_rate": float(flipped.mean()) if len(rows) else float("nan"),
        "top_reason_sufficiency_rate": float(flipped_top.mean()) if len(rows) else float("nan"),
        "mean_margin_drop_top_reason": float(top_drop.mean()) if len(rows) else float("nan"),
        "mean_margin_drop_random_factor": float(rand_drop.mean()) if len(rows) else float("nan"),
        "primary_reason_share": {str(a): float(b / len(rows)) for a, b in counts.items()},
        "reference": {kk: v for kk, v in reference.items() if kk != "values"},
    }
    return pd.DataFrame(wide), pd.DataFrame(long), summary

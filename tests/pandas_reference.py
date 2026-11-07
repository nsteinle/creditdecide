"""The pandas implementations that creditdecide.sql replaced, kept only as the reference the
SQL tests compare against (tests/test_sql.py). Copied unchanged from the pandas versions,
apart from these comments and the imports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditdecide.datasets import FeatureSpec
from creditdecide.monitor import EPS, SIGNIFICANT, STABLE


def _status(value: float) -> str:
    if value < STABLE:
        return "stable"
    return "moderate shift" if value <= SIGNIFICANT else "significant shift"


def quality_report(df: pd.DataFrame, spec: FeatureSpec,
                   documented_codes: dict[str, set] | None = None) -> dict:
    """Summarise size, duplicates, missing values, constant columns, class balance and, when
    ``documented_codes`` is given, values outside the documented code sets."""
    id_cols = [c for c in spec.drop if c in df.columns]
    content = df.drop(columns=id_cols)
    features = [c for c in content.columns if c != spec.target]
    dup = df[df.duplicated(subset=features, keep=False)]
    conflicting = (int((dup.groupby(features)[spec.target].nunique() > 1).sum())
                   if len(dup) else 0)
    # The model never sees the protected columns (by default), so rows that differ only in
    # those look identical to it. The group-aware split uses these model-input groups.
    inputs = spec.feature_columns(df)
    dup_in = df[df.duplicated(subset=inputs, keep=False)]
    conflicting_in = (int((dup_in.groupby(inputs, dropna=False)[spec.target].nunique() > 1)
                          .sum()) if len(dup_in) else 0)
    counts = df[spec.target].value_counts().sort_index()
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows_ignoring_ids": int(content.duplicated().sum()),
        "rows_sharing_features_with_another_row": int(len(dup)),
        "feature_groups_with_conflicting_labels": conflicting,
        "rows_sharing_model_inputs_with_another_row": int(len(dup_in)),
        "model_input_groups_with_conflicting_labels": conflicting_in,
        "duplicate_ids": {c: int(df[c].duplicated().sum()) for c in id_cols},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_fractions": {str(k): round(float(v) / len(df), 4) for k, v in counts.items()},
        "majority_class_accuracy": round(float(counts.max()) / len(df), 4),
    }
    if documented_codes:
        undocumented = {}
        for col, codes in documented_codes.items():
            if col in df.columns:
                bad = df.loc[~df[col].isin(codes), col].value_counts().sort_index()
                if len(bad):
                    undocumented[col] = {str(k): int(v) for k, v in bad.items()}
        report["undocumented_codes"] = undocumented
    return report


def gains_table(y, score, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gains and lift by score decile (decile 1 = highest scores)."""
    y, s = np.asarray(y), np.asarray(score, float)
    order = np.argsort(-s, kind="stable")
    bins = np.array_split(order, n_bins)
    pos, n = y.sum(), len(y)
    rows, cum_pos, cum_n = [], 0, 0
    for i, idx in enumerate(bins, start=1):
        p = int(y[idx].sum())
        cum_pos += p
        cum_n += len(idx)
        rows.append({"decile": i, "accounts": len(idx), "min_score": float(s[idx].min()),
                     "positives": p, "positive_rate": p / len(idx),
                     "cumulative_capture": cum_pos / pos if pos else np.nan,
                     "cumulative_lift": (cum_pos / cum_n) / (pos / n) if pos else np.nan})
    return pd.DataFrame(rows)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _actual(values, prof: dict) -> np.ndarray:
    v = np.asarray(values)
    if prof["kind"] == "discrete":
        s = pd.Series(v.astype(str))
        known = s.isin(prof["values"])
        counts = s[known].value_counts()
        a = [float(counts.get(c, 0)) for c in prof["values"]] + [float((~known).sum())]
        return np.asarray(a) / len(v)
    edges = np.asarray(prof["edges"])
    counts = np.bincount(np.searchsorted(edges, v.astype(float), side="right"),
                         minlength=len(edges) + 1)
    return counts / len(v)


def drift_report(reference: dict, df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """PSI of every model input and of the score, new data vs the reference sample."""
    rows = []
    items = [*reference["features"].items(), ("[model score]", reference["score"])]
    for name, prof in items:
        values = scores if name == "[model score]" else df[name]
        value = psi(prof["expected"], _actual(values, prof))
        rows.append({"variable": name, "psi": value, "status": _status(value)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    out.attrs["n_new"] = int(len(df))
    return out

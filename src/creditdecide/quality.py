"""Data-quality report: the checks to run before any model sees the data."""

from __future__ import annotations

import pandas as pd

from creditdecide import sql
from creditdecide.datasets import FeatureSpec


def quality_report(df: pd.DataFrame, spec: FeatureSpec,
                   documented_codes: dict[str, set] | None = None) -> dict:
    """Summarise size, duplicates, missing values, constant columns, class balance and, when
    ``documented_codes`` is given, values outside the documented code sets. The counts are
    DuckDB SQL (creditdecide.sql, queries duplicates.sql, class_counts.sql and
    undocumented_codes.sql)."""
    id_cols = [c for c in spec.drop if c in df.columns]
    features = [c for c in df.columns if c not in id_cols and c != spec.target]
    # The model never sees the protected columns (by default), so rows that differ only in
    # those look identical to it. The group-aware split uses these model-input groups.
    inputs = spec.feature_columns(df)
    c = sql.quality_counts(df, spec.target, id_cols, features, inputs, documented_codes)
    n = c["n_rows"]
    top = max(c["class_counts"].values())
    report = {
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        **{k: c[k] for k in ("duplicate_rows_ignoring_ids",
                             "rows_sharing_features_with_another_row",
                             "feature_groups_with_conflicting_labels",
                             "rows_sharing_model_inputs_with_another_row",
                             "model_input_groups_with_conflicting_labels",
                             "duplicate_ids", "missing_values", "constant_columns",
                             "class_counts")},
        "class_fractions": {k: round(v / n, 4) for k, v in c["class_counts"].items()},
        "majority_class_accuracy": round(top / n, 4),
    }
    if documented_codes:
        report["undocumented_codes"] = c["undocumented_codes"]
    return report


def format_report(report: dict) -> str:
    lines = [
        f"rows: {report['n_rows']:,}   columns: {report['n_columns']}",
        f"duplicate rows (ignoring IDs): {report['duplicate_rows_ignoring_ids']}",
        f"rows sharing all features with another row: "
        f"{report['rows_sharing_features_with_another_row']} "
        f"({report['feature_groups_with_conflicting_labels']} groups with conflicting labels)",
        f"rows sharing all model inputs (protected columns excluded) with another row: "
        f"{report['rows_sharing_model_inputs_with_another_row']} "
        f"({report['model_input_groups_with_conflicting_labels']} groups with conflicting "
        f"labels; the split keeps each group together)",
        f"missing values: {report['missing_values'] or 'none'}",
        f"constant columns: {report['constant_columns'] or 'none'}",
        f"class counts: {report['class_counts']}  "
        f"(majority-class accuracy {report['majority_class_accuracy']:.3f})",
    ]
    for col, bad in report.get("undocumented_codes", {}).items():
        lines.append(f"undocumented codes in {col}: {bad}")
    return "\n".join(lines)

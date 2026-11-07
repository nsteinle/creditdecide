"""Feature preprocessing, always fit on the training split only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from creditdecide.datasets import FeatureSpec


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(df: pd.DataFrame, spec: FeatureSpec,
                      include_protected: bool = False) -> ColumnTransformer:
    """signed-log + z-score for heavy-tailed columns, z-score for other numeric columns,
    one-hot for categoricals (unseen categories at prediction time are all-zero)."""
    features = spec.feature_columns(df, include_protected=include_protected)
    cat = [c for c in features if c in spec.categorical]
    log = [c for c in features if c in spec.signed_log]
    num = [c for c in features if c not in cat and c not in log]
    parts = []
    if log:
        parts.append(("signed_log", make_pipeline(
            FunctionTransformer(signed_log1p, feature_names_out="one-to-one"),
            StandardScaler()), log))
    if num:
        parts.append(("numeric", StandardScaler(), num))
    if cat:
        parts.append(("categorical", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False), cat))
    if not parts:
        raise ValueError("no feature columns left after exclusions")
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)

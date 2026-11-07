"""Dataset specifications and loaders.

Bundled data
------------
``credit-default``: "Default of Credit Card Clients", I.-C. Yeh, UCI Machine Learning
Repository, doi:10.24432/C55S3H, CC BY 4.0. 30,000 Taiwanese card holders (2005), 23
features, target = default on the next month's payment. The package ships a CSV converted
from the original .xls by ``scripts/build_credit_default.py``; no values are altered (see
DATA_LICENSE.md). The file's SHA-256 is checked every time it is loaded.

User data
---------
``load_csv`` reads any numeric/categorical table with a label column, with strict validation.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PER_CLASS = 20
MAX_CLASSES = 20


@dataclass(frozen=True)
class FeatureSpec:
    """How to turn a table into features.

    target: label column. categorical: one-hot encoded (categories learned on training data).
    signed_log: transformed with sign(x) * log1p(|x|) before scaling (heavy-tailed amounts).
    protected: excluded from features unless explicitly included; still used to audit
        outcomes by group (creditdecide.fairness).
    drop: always excluded (e.g. identifiers).
    positive_label: the class a decision acts on (e.g. "default"); binary targets only.
    """

    target: str
    categorical: tuple[str, ...] = ()
    signed_log: tuple[str, ...] = ()
    protected: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    positive_label: object = 1
    label_names: dict = field(default_factory=dict)
    group_labels: dict = field(default_factory=dict)  # readable names for protected codes
    group_bins: dict = field(default_factory=dict)    # band edges for numeric protected cols
    reason_groups: dict = field(default_factory=dict)  # factor label -> columns (reason codes)

    def feature_columns(self, df: pd.DataFrame, include_protected: bool = False) -> list[str]:
        excluded = {self.target, *self.drop}
        if not include_protected:
            excluded |= set(self.protected)
        return [c for c in df.columns if c not in excluded]


PAY_STATUS = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
BILL_AMT = tuple(f"BILL_AMT{i}" for i in range(1, 7))
PAY_AMT = tuple(f"PAY_AMT{i}" for i in range(1, 7))
CREDIT_DEFAULT_COLUMNS = (
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_STATUS, *BILL_AMT, *PAY_AMT, "DEFAULT",
)

# SEX, MARRIAGE and AGE are excluded by default: sex and marital status are prohibited bases
# under the US Equal Credit Opportunity Act (Regulation B), and age is restricted.
CREDIT_DEFAULT_SPEC = FeatureSpec(
    target="DEFAULT",
    categorical=("EDUCATION",),
    signed_log=("LIMIT_BAL", *BILL_AMT, *PAY_AMT),
    protected=("SEX", "MARRIAGE", "AGE"),
    drop=("ID",),
    positive_label=1,
    label_names={0: "no default", 1: "default"},
    group_labels={
        "SEX": {1: "male", 2: "female"},
        "MARRIAGE": {1: "married", 2: "single", 3: "others", 0: "undocumented (0)"},
    },
    # Regulation B treats applicants aged 62 or older as "elderly"; other bands are for reading.
    group_bins={"AGE": (25, 35, 45, 62)},
    reason_groups={
        "Repayment status last month": ("PAY_0",),
        "Repayment status in the five months before": PAY_STATUS[1:],
        "Credit limit": ("LIMIT_BAL",),
        "Bill amounts over the last six months": BILL_AMT,
        "Payment amounts over the last six months": PAY_AMT,
        "Education level": ("EDUCATION",),
    },
)

# Codes documented by the data donor (Yeh & Lien 2009 / UCI page); others are flagged.
CREDIT_DEFAULT_DOCUMENTED_CODES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    **{c: {-1, *range(1, 10)} for c in PAY_STATUS},
}

DATASETS = {"credit-default": CREDIT_DEFAULT_SPEC}


def read_csv(path, **kwargs) -> pd.DataFrame:
    """``pandas.read_csv`` with exact float parsing on every pandas version.

    pandas 2's default parser can read a float back one unit in the last place off, so a score
    written by ``predict`` and read again would differ in the 16th digit. There the
    ``round_trip`` parser is used; pandas 3 parses exactly by default and deprecates the
    option, so it is left out."""
    if int(pd.__version__.split(".")[0]) < 3:
        kwargs.setdefault("float_precision", "round_trip")
    return pd.read_csv(path, **kwargs)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_credit_default() -> pd.DataFrame:
    """Load the bundled credit-default table after verifying its checksum and schema."""
    pkg = resources.files("creditdecide.data")
    meta_file = pkg / "credit_default.json"
    data_file = pkg / "credit_default.csv.gz"
    if not data_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError(
            "bundled credit-default data not found; this installation is incomplete"
        )
    meta = json.loads(meta_file.read_text())
    raw = data_file.read_bytes()
    if _sha256(raw) != meta["sha256_csv_gz"]:
        raise ValueError("credit-default data failed its SHA-256 integrity check")
    df = read_csv(io.BytesIO(gzip.decompress(raw)))
    validate_credit_default(df)
    return df


def validate_credit_default(df: pd.DataFrame) -> None:
    """Schema checks for the credit-default table (columns, size, types, label values)."""
    if tuple(df.columns) != CREDIT_DEFAULT_COLUMNS:
        raise ValueError(f"unexpected columns: {list(df.columns)}")
    if len(df) != 30_000:
        raise ValueError(f"expected 30,000 rows, found {len(df)}")
    if not all(pd.api.types.is_integer_dtype(df[c]) for c in df.columns):
        raise ValueError("all credit-default columns must be integers")
    if set(df["DEFAULT"].unique()) != {0, 1}:
        raise ValueError("DEFAULT must contain exactly the labels 0 and 1")
    if df["ID"].duplicated().any():
        raise ValueError("duplicate IDs")


def load_csv(path: str | Path, spec: FeatureSpec, dropna: bool = False) -> pd.DataFrame:
    """Read a user table and validate it against ``spec``.

    Feature columns must be numeric unless listed as categorical. Missing values are an error
    unless ``dropna=True``, in which case incomplete rows are removed and counted.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = read_csv(path)
    validate_table(df, spec, dropna=dropna)
    if dropna:
        n = len(df)
        df = df.dropna().reset_index(drop=True)
        df.attrs["rows_dropped_na"] = n - len(df)
    return df


def validate_table(df: pd.DataFrame, spec: FeatureSpec, dropna: bool = False) -> None:
    named = [spec.target, *spec.categorical, *spec.signed_log, *spec.protected, *spec.drop]
    missing = sorted(set(named) - set(df.columns))
    if missing:
        raise ValueError(f"columns not found in table: {missing}")
    overlap = set(spec.categorical) & set(spec.signed_log)
    if overlap:
        raise ValueError(f"columns cannot be both categorical and signed-log: {sorted(overlap)}")
    na = df.columns[df.isna().any()].tolist()
    if na and not dropna:
        raise ValueError(f"missing values in {na}; clean the data or pass dropna=True")
    features = spec.feature_columns(df, include_protected=True)
    non_numeric = [c for c in features if c not in spec.categorical
                   and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"non-numeric feature columns (declare them categorical): {non_numeric}")
    numeric = [c for c in features if c not in spec.categorical]
    if numeric and not np.isfinite(df[numeric].dropna().to_numpy(float)).all():
        raise ValueError("feature columns contain infinite values")
    counts = df[spec.target].dropna().value_counts()
    if len(counts) < 2 or len(counts) > MAX_CLASSES:
        raise ValueError(f"target must have between 2 and {MAX_CLASSES} classes, has {len(counts)}")
    if counts.min() < MIN_PER_CLASS:
        raise ValueError(f"each class needs at least {MIN_PER_CLASS} rows; smallest has "
                         f"{counts.min()}")

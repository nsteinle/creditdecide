"""Convert the official UCI archive into the package's bundled data file.

Maintainer-only script (not shipped in the wheel). Usage:

    pip install xlrd
    python scripts/build_credit_default.py "default+of+credit+card+clients.zip"

Download the archive yourself from
https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
The script refuses any file whose SHA-256 differs from the archive verified on 2026-10-15.

Changes made to the data (required to be stated by CC BY 4.0): the first header row (X1..X23,
Y) is dropped, the target column "default payment next month" is renamed DEFAULT, and the table
is stored as gzip-compressed CSV. No values are changed, added or removed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
MEMBER = "default of credit card clients.xls"
OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "creditdecide" / "data"

sys.path.insert(0, str(OUT_DIR.parents[1]))
from creditdecide.datasets import validate_credit_default  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip_path", type=Path)
    a = ap.parse_args()

    blob = a.zip_path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != ZIP_SHA256:
        sys.exit(f"SHA-256 mismatch: got {digest}, expected {ZIP_SHA256}")

    with zipfile.ZipFile(io.BytesIO(blob)) as z:  # read one named member; never extract
        xls = z.read(MEMBER)
    df = pd.read_excel(io.BytesIO(xls), header=1, engine="xlrd")
    df = df.rename(columns={"default payment next month": "DEFAULT"})
    validate_credit_default(df)

    csv = df.to_csv(index=False, lineterminator="\n").encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=7) as gz:
        gz.write(csv)
    gz_bytes = buf.getvalue()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "credit_default.csv.gz").write_bytes(gz_bytes)
    meta = {
        "name": "Default of Credit Card Clients",
        "creator": "I.-C. Yeh",
        "source": "UCI Machine Learning Repository",
        "doi": "10.24432/C55S3H",
        "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        "license": "CC BY 4.0",
        "citation": ("Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining "
                     "techniques for the predictive accuracy of probability of default of "
                     "credit card clients. Expert Systems with Applications, 36(2), 2473-2480."),
        "changes": ("first header row dropped; 'default payment next month' renamed DEFAULT; "
                    "stored as gzip CSV; no values altered"),
        "sha256_source_zip": ZIP_SHA256,
        "sha256_csv_gz": hashlib.sha256(gz_bytes).hexdigest(),
        "n_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df["DEFAULT"].value_counts()
                         .sort_index().items()},
    }
    (OUT_DIR / "credit_default.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

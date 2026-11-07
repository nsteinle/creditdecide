"""Save a trained model as a self-describing artifact, load it safely, and score new data.

Artifact layout (one directory):

    manifest.json        format version, input schema, decision rule, threshold and policy,
                         calibration map, reason-code factors and reference profile,
                         validation/test results, SHA-256 of every file, library versions
    pipeline.skops       preprocessing + scoring model (skops format, not pickle)
    reference.json       training-time distributions for drift monitoring (creditdecide.monitor)
    MODEL_CARD.md        human-readable summary

Loading verifies every checksum, refuses any file it does not know and a skops file containing
any type outside a short allow-list. Nothing is ever unpickled. A minimal manifest
(format 1: no calibration map, decision rule or reason-code profile) also loads: it scores with
an uncalibrated probability and a threshold rule, and cannot produce reason codes.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
from sklearn.exceptions import InconsistentVersionWarning

from creditdecide._version import __version__
from creditdecide.calibration import apply_map
from creditdecide.decision import expected_loss_margin
from creditdecide.forest import predict_on_one_thread

FORMAT_VERSION = 2
SUPPORTED_FORMATS = (1, 2)
ARTIFACT_FILES = frozenset({"pipeline.skops", "reference.json"})
TRUSTED_TYPES = frozenset({
    "creditdecide.preprocess.signed_log1p",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_schema(df: pd.DataFrame, features: list[str], categorical: tuple[str, ...]) -> dict:
    """Columns the model needs at scoring time, their kinds and the training categories."""
    return {
        "columns": features,
        "numeric": [c for c in features if c not in categorical],
        "categorical": {c: sorted(map(str, df[c].unique())) for c in features
                        if c in categorical},
    }


def write_artifact(outdir: Path, manifest: dict, preprocessor, model,
                   reference: dict | None = None, model_card: str | None = None) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    sio.dump({"preprocessor": preprocessor, "model": model}, out / "pipeline.skops")
    files["pipeline.skops"] = sha256_file(out / "pipeline.skops")
    if reference is not None:
        (out / "reference.json").write_text(json.dumps(reference, indent=2))
        files["reference.json"] = sha256_file(out / "reference.json")
    manifest = {"format_version": FORMAT_VERSION, **manifest, "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if model_card is not None:
        (out / "MODEL_CARD.md").write_text(model_card)
    return out


@dataclass
class ScoringModel:
    manifest: dict
    preprocessor: object
    model: object
    reference: dict | None

    @property
    def rule(self) -> str:
        return self.manifest["decision"].get("rule", "threshold")

    @property
    def threshold(self) -> float | None:
        t = self.manifest["decision"]["threshold"]
        return None if t is None else float(t)

    @property
    def calibration(self) -> dict:
        return self.manifest.get("calibration", {"method": "none"})

    @property
    def schema(self) -> dict:
        return self.manifest["input_schema"]

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Raise on anything that would make scores wrong; return warnings for anything that
        makes them less trustworthy (e.g. unseen categories)."""
        cols = self.schema["columns"] + self.manifest.get("decision_inputs", [])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        na = [c for c in cols if df[c].isna().any()]
        if na:
            raise ValueError(f"missing values in {na}; impute or remove those rows first")
        bad = [c for c in self.schema["numeric"] if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"non-numeric values in numeric columns {bad}")
        num = self.schema["numeric"]
        if num and not np.isfinite(df[num].to_numpy(float)).all():
            raise ValueError("numeric columns contain infinite values")
        warns = []
        for c, cats in self.schema["categorical"].items():
            unseen = ~df[c].astype(str).isin(cats)
            if unseen.any():
                warns.append(f"{int(unseen.sum())} rows have categories in {c!r} not seen in "
                             f"training: {sorted(df.loc[unseen, c].astype(str).unique())[:10]}")
        return warns

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Raw model score (ranking; used by the threshold rule and drift monitoring)."""
        X = np.asarray(self.preprocessor.transform(df[self.schema["columns"]]), dtype=float)
        return self.model.predict_proba(X)[:, 1]

    def probability(self, df: pd.DataFrame, score: np.ndarray | None = None) -> np.ndarray:
        """Calibrated probability of the positive class."""
        return apply_map(self.calibration, self.score(df) if score is None else score)

    def margin(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(score, probability, decision margin, flag) under the saved decision rule."""
        s = self.score(df)
        p = self.probability(df, s)
        if self.rule == "threshold":
            m = s - self.threshold
            return s, p, m, m >= 0
        pol = self.manifest["decision"]["policy"]
        if "cost_fn" in pol:
            missed = np.full(len(df), float(pol["cost_fn"]))
        else:
            missed = pol["lgd"] * np.clip(df[pol["exposure_column"]].to_numpy(float), 0, None)
        m = expected_loss_margin(p, missed, float(pol["cost_fp"]))
        return s, p, m, m > 0

    def decide(self, df: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
        s, p, _, f = self.margin(df)
        out = pd.DataFrame({"score": s, "probability": p, "flag": f.astype(int)})
        if id_column:
            out.insert(0, id_column, df[id_column].to_numpy())
        return out

    def reasons(self, df: pd.DataFrame, k: int | None = None,
                id_column: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Reason codes for the rows of ``df`` that the saved rule flags."""
        from creditdecide.reasons import explain

        info = self.manifest.get("reasons")
        if info is None:
            raise ValueError("this model artifact has no reason-code profile (format 1)")
        k = k or info["max_reasons"]
        groups = {label: tuple(cols) for label, cols in info["groups"].items()}
        flagged = df[self.margin(df)[3]].copy()

        def decide(frame):
            _, p, m, f = self.margin(frame)
            return m, f, p

        if not len(flagged):
            return pd.DataFrame(), pd.DataFrame(), {"accounts_explained": 0}
        return explain(flagged, decide, groups, info["reference"], k=min(k, len(groups)),
                       id_column=id_column)


def load_model(model_dir: str | Path) -> ScoringModel:
    d = Path(model_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest.get("format_version") not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported artifact format {manifest.get('format_version')!r}")
    unknown = sorted(set(manifest["files"]) - ARTIFACT_FILES)
    if unknown or "pipeline.skops" not in manifest["files"]:
        raise ValueError(f"not a creditdecide {__version__} model directory: unexpected files "
                         f"{unknown}" if unknown else "manifest lists no pipeline.skops")
    for name, digest in manifest["files"].items():
        if sha256_file(d / name) != digest:
            raise ValueError(f"{name} failed its SHA-256 check; the artifact was modified")
    untrusted = set(sio.get_untrusted_types(file=d / "pipeline.skops"))
    if not untrusted <= TRUSTED_TYPES:
        raise ValueError(f"refusing to load untrusted types: {sorted(untrusted - TRUSTED_TYPES)}")
    trained_with = manifest["environment"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        warnings.warn(f"model saved with scikit-learn {trained_with}, running "
                      f"{sklearn.__version__}; check scores against the manifest", stacklevel=2)
    with warnings.catch_warnings():  # one warning above instead of one per estimator
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        obj = sio.load(d / "pipeline.skops", trusted=sorted(untrusted))
    reference = (json.loads((d / "reference.json").read_text())
                 if "reference.json" in manifest["files"] else None)
    # score forests on one thread even if the saved model says otherwise (creditdecide.forest)
    model = predict_on_one_thread(obj["model"])
    return ScoringModel(manifest, obj["preprocessor"], model, reference)

"""The data and reporting layer in DuckDB SQL; the models stay in Python.

Each query is a ``.sql`` file next to this module. DuckDB runs in memory inside the Python
process and reads pandas DataFrames in place, so there is no server, no copy of the data and
nothing to install beyond the ``duckdb`` package.

Every connection has external access (files, network) and extension auto-loading switched off,
so a query can only see the tables registered here. The only text put into a query is column
names, quoted as identifiers; values are passed as query parameters.

``tests/test_sql.py`` checks every query against the pandas implementation it replaced, on
the bundled data.
"""

from __future__ import annotations

from importlib import resources

import duckdb
import numpy as np
import pandas as pd

__all__ = ["connect", "query", "quality_counts", "gains", "outcomes", "drift"]

_SETTINGS = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
}


def connect(**tables: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with external access off and ``tables`` registered
    under their keyword names."""
    con = duckdb.connect(config=dict(_SETTINGS))
    for name, frame in tables.items():
        con.register(name, frame)
    return con


def ident(name: str) -> str:
    """A column or table name quoted as a SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def load(name: str, **identifiers: str | list[str]) -> str:
    """The text of ``<name>.sql`` with ``{placeholders}`` replaced by quoted identifiers
    (a list becomes a comma-separated column list)."""
    text = (resources.files(__package__) / f"{name}.sql").read_text()
    quoted = {k: ", ".join(map(ident, v)) if isinstance(v, (list, tuple)) else ident(v)
              for k, v in identifiers.items()}
    return text.format(**quoted)


def query(sql: str, **tables: pd.DataFrame) -> pd.DataFrame:
    """Run any read-only query over the given DataFrames (``creditdecide sql``)."""
    with connect(**tables) as con:
        return con.execute(sql).df()


# ------------------------------------------------------------------ data quality
def quality_counts(df: pd.DataFrame, target: str, id_columns: list[str],
                   feature_columns: list[str], model_inputs: list[str],
                   documented_codes: dict[str, set] | None = None) -> dict:
    """The counts behind creditdecide.quality.quality_report."""
    content = [c for c in df.columns if c not in id_columns]
    with connect(data=df) as con:
        def dup(cols):
            return con.execute(load("duplicates", columns=cols, target=target)).fetchone()

        _, _, duplicate_rows = dup(content)
        rows_shared, conflicting, _ = dup(feature_columns)
        rows_shared_in, conflicting_in, _ = dup(model_inputs)
        per_column = ", ".join(
            f"COUNT(*) - COUNT({ident(c)}), "
            f"COUNT(DISTINCT {ident(c)}) + (COUNT(*) > COUNT({ident(c)}))::INTEGER"
            for c in df.columns)
        row = con.execute(f"SELECT COUNT(*), {per_column} FROM data").fetchone()
        n, values = int(row[0]), row[1:]
        missing = {c: int(values[2 * i]) for i, c in enumerate(df.columns)}
        distinct = {c: int(values[2 * i + 1]) for i, c in enumerate(df.columns)}
        classes = con.execute(load("class_counts", target=target)).fetchall()
        undocumented = None
        if documented_codes:
            undocumented = {}
            for col, codes in documented_codes.items():
                if col in df.columns:
                    bad = con.execute(load("undocumented_codes", column=col),
                                      {"codes": sorted(codes)}).fetchall()
                    if bad:
                        undocumented[col] = {str(k): int(v) for k, v in bad}
    return {
        "n_rows": n,
        "duplicate_rows_ignoring_ids": int(duplicate_rows),
        "rows_sharing_features_with_another_row": int(rows_shared),
        "feature_groups_with_conflicting_labels": int(conflicting),
        "rows_sharing_model_inputs_with_another_row": int(rows_shared_in),
        "model_input_groups_with_conflicting_labels": int(conflicting_in),
        "duplicate_ids": {c: n - distinct[c] for c in id_columns},
        "missing_values": {c: v for c, v in missing.items() if v},
        "constant_columns": [c for c in df.columns if distinct[c] <= 1],
        "class_counts": {str(k): int(v) for k, v in classes},
        "undocumented_codes": undocumented,
    }


# ------------------------------------------------------------------ gains table
def gains(y, score, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gains and lift by score decile (decile 1 = highest scores)."""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(score, float)
    n, pos = len(y), int(y.sum())
    frame = pd.DataFrame({"row_id": np.arange(n), "y": y, "score": s})
    with connect(scored=frame) as con:
        out = con.execute(load("gains"), {"n": n, "bins": n_bins, "positives": pos}).df()
    return out.astype({"decile": "int64", "accounts": "int64", "positives": "int64"})


# ------------------------------------------------------------------ outcomes join
def outcomes(scores: pd.DataFrame, labels: pd.DataFrame, id_column: str, target: str,
             positive_label: str, threshold: float | None = None) -> tuple[dict, pd.DataFrame]:
    """Join saved decisions (``id_column``, score, probability, flag) to labels that arrived
    later (``id_column``, ``target``). Returns the decision outcome (the same fields as
    creditdecide.decision.summarize, plus unmatched-row counts) and the matched rows."""
    for frame, name in ((scores, "scores"), (labels, "labels")):
        if id_column not in frame:
            raise ValueError(f"{name} has no {id_column!r} column")
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} has duplicate values in {id_column!r}")
    if target not in labels:
        raise ValueError(f"labels have no {target!r} column")
    s = scores[[id_column, "score", "probability", "flag"]].assign(row_id=np.arange(len(scores)))
    lab = labels[[id_column, target]]
    params = {"positive_label": str(positive_label)}
    with connect(scores=s, labels=lab) as con:
        c = con.execute(load("outcomes", id=id_column, target=target), params).df().iloc[0]
        rows = con.execute(load("outcomes_rows", id=id_column, target=target), params).df()
    n, pos, flagged = int(c["n"]), int(c["positives"]), int(c["flagged"])
    tp, fp, fn, tn = (int(c[k]) for k in ("tp", "fp", "fn", "tn"))
    nan = float("nan")
    summary = {
        "n": n,
        "threshold": None if threshold is None else float(threshold),
        "flagged": flagged,
        "flagged_fraction": flagged / n if n else nan,
        "positives": pos,
        "capture_rate": tp / pos if pos else nan,
        "precision": tp / flagged if flagged else nan,
        "false_alarm_rate": fp / (n - pos) if n > pos else nan,
        "lift": (tp / flagged) / (pos / n) if flagged and pos else nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "scored_without_label": int(c["scored_without_label"]),
        "labelled_without_score": int(c["labelled_without_score"]),
    }
    return summary, rows


# ------------------------------------------------------------------ drift (PSI)
def drift(reference: dict, df: pd.DataFrame, scores: np.ndarray, eps: float,
          stable: float, significant: float) -> pd.DataFrame:
    """PSI of every model input and of the score against a saved reference profile
    (creditdecide.monitor.build_reference)."""
    items = [*reference["features"].items(), ("[model score]", reference["score"])]
    numeric, text, edges, known, unseen, expected = [], [], [], [], [], []
    n = len(df)
    row_id = np.arange(n)
    for position, (name, prof) in enumerate(items):
        values = np.asarray(scores) if name == "[model score]" else df[name].to_numpy()
        expected += [(name, position, b, float(e)) for b, e in enumerate(prof["expected"])]
        if prof["kind"] == "binned":
            numeric.append(pd.DataFrame({"variable": name, "row_id": row_id,
                                         "value": values.astype(float)}))
            edges += [(name, float(e)) for e in prof["edges"]]
        else:  # compare as text, exactly as the reference profile stored the values
            text.append(pd.DataFrame({"variable": name, "row_id": row_id,
                                      "value": values.astype(str)}))
            known += [(name, v, b) for b, v in enumerate(prof["values"])]
            unseen.append((name, len(prof["values"])))

    def frame(rows, cols, dtypes):
        return pd.DataFrame(rows, columns=cols).astype(dtypes)

    tables = {
        "values_numeric": (pd.concat(numeric, ignore_index=True) if numeric else
                           frame([], ["variable", "row_id", "value"],
                                 {"variable": str, "row_id": "int64", "value": float})),
        "values_text": (pd.concat(text, ignore_index=True) if text else
                        frame([], ["variable", "row_id", "value"],
                              {"variable": str, "row_id": "int64", "value": str})),
        "edges": frame(edges, ["variable", "edge"], {"variable": str, "edge": float}),
        "known_values": frame(known, ["variable", "value", "bin"],
                              {"variable": str, "value": str, "bin": "int64"}),
        "unseen": frame(unseen, ["variable", "unseen_bin"],
                        {"variable": str, "unseen_bin": "int64"}),
        "expected": frame(expected, ["variable", "position", "bin", "expected"],
                          {"variable": str, "position": "int64", "bin": "int64",
                           "expected": float}),
    }
    params = {"n": n, "eps": eps, "stable": stable, "significant": significant}
    with connect(**tables) as con:
        out = con.execute(load("psi"), params).df()
    out.attrs["n_new"] = int(n)
    return out

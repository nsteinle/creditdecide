"""Tableau Public extract: a run folder (and optionally a scored folder) as flat CSV tables.

Tableau Public opens text files, Excel, JSON, PDF, spatial and statistical files, and not
Parquet, so the extract is CSV. Every table is built by a DuckDB query in creditdecide/sql
(tableau_*.sql) or copied from the run unchanged. docs/TABLEAU.md describes the dashboard
built from them.

    accounts.csv      one row per test-split account: inputs, readable group names, score,
                      calibrated probability, decision, score decile, top reason code
    decision.csv      one row: the decision model, policy and test-split outcome (KPI tiles)
    gains.csv         gains and lift by score decile, with the share of accounts reviewed
    calibration.csv   reliability table, raw and calibrated scores
    models.csv        test-split metrics, one row per model and metric
    intervals.csv     95% bootstrap intervals
    reasons.csv       how often each factor is a stated reason, by rank
    fairness.csv      outcomes by protected group
    drift.csv         PSI per input and for the score (only with a scored folder)
    tables.json       what each file holds, its columns and row count, and its source
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from creditdecide import sql
from creditdecide._version import __version__
from creditdecide.datasets import read_csv
from creditdecide.fairness import group_labels

REQUIRED = ("samples.csv", "decision.json", "run.json", "gains_test.csv",
            "metrics_test.csv")

DESCRIPTIONS = {
    "accounts.csv": "one row per test-split account: inputs, readable group names, score, "
                    "calibrated probability, decision, score decile, top reason code",
    "decision.csv": "the decision model, policy and test-split outcome",
    "gains.csv": "gains and lift by score decile, with the share of accounts reviewed",
    "calibration.csv": "reliability table (10 equal-count bins), raw and calibrated scores",
    "models.csv": "test-split metrics, one row per model and metric",
    "intervals.csv": "95% bootstrap intervals for the test-split results",
    "reasons.csv": "how often each factor is a stated reason for flagged accounts, by rank",
    "fairness.csv": "outcomes by protected group (adverse impact ratio, four-fifths flag)",
    "drift.csv": "population stability index per input and for the score, new file vs the "
                 "model's reference",
}


def _accounts(run_dir: Path, meta: dict) -> pd.DataFrame:
    spec = meta["spec"]
    target = spec["target"]
    samples = read_csv(run_dir / "samples.csv", low_memory=False)
    samples = samples.drop(columns=[c for c in samples.columns if c.startswith("pred_")])
    samples.insert(0, "row_id", range(len(samples)))
    if "label_name" in samples:
        samples = samples.rename(columns={"label_name": "actual_label"})
    for col in spec["protected"]:
        if col in samples:
            labels = {int(k) if str(k).lstrip("-").isdigit() else k: v
                      for k, v in spec["group_labels"].get(col, {}).items()}
            bins = spec["group_bins"].get(col)
            samples[f"{col}_group"] = group_labels(samples[col], labels or None,
                                                   tuple(bins) if bins else None).to_numpy()
    reasons_file = run_dir / "reasons_test.csv"
    reasons = (read_csv(reasons_file) if reasons_file.exists() else
               pd.DataFrame({"row": pd.Series(dtype="int64"), "reason_1": pd.Series(dtype=str),
                             "impact_1": pd.Series(dtype=float)}))
    q = sql.load("tableau_accounts", target=target)
    with sql.connect(samples=samples, reasons=reasons[["row", "reason_1", "impact_1"]]) as con:
        return con.execute(q).df()


def _decision(d: dict) -> pd.DataFrame:
    t, pol = d["test"], d["policy"]
    row = {"model": d["model"], "selected_by": d["selected_by"],
           "rule": d.get("rule", "threshold"), "policy": pol["kind"],
           "budget": pol.get("budget"), "threshold": d.get("threshold"),
           **{k: t[k] for k in ("n", "flagged", "flagged_fraction", "positives",
                                "capture_rate", "precision", "lift", "false_alarm_rate")}}
    for k in ("saving_fraction_vs_flag_nobody", "saving_fraction_vs_best_trivial"):
        if k in t:
            row[k] = t[k]
    bench = d.get("simple_rule_benchmark") or {}
    if bench:
        row["benchmark_rule"] = bench.get("rule")
    return pd.DataFrame([row])


def export(run_dir: str | Path, out_dir: str | Path, scored_dir: str | Path | None = None,
           overwrite: bool = False) -> dict:
    """Write the Tableau tables for one run; returns the contents of tables.json."""
    run_dir, out = Path(run_dir), Path(out_dir)
    missing = [f for f in REQUIRED if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is not a creditdecide run folder with a decision "
                                f"(missing {', '.join(missing)})")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; use --overwrite to replace its files")
    meta = json.loads((run_dir / "run.json").read_text())
    decision = json.loads((run_dir / "decision.json").read_text())

    tables = {"accounts.csv": _accounts(run_dir, meta), "decision.csv": _decision(decision)}
    with sql.connect(gains=read_csv(run_dir / "gains_test.csv"),
                     metrics=read_csv(run_dir / "metrics_test.csv")) as con:
        tables["gains.csv"] = con.execute(sql.load("tableau_gains")).df()
        tables["models.csv"] = con.execute(sql.load("tableau_models")).df()
    if (run_dir / "reasons_test_detail.csv").exists():
        detail = read_csv(run_dir / "reasons_test_detail.csv")
        with sql.connect(reasons_detail=detail) as con:
            tables["reasons.csv"] = con.execute(sql.load("tableau_reasons")).df()
    for src, name in (("calibration_test.csv", "calibration.csv"),
                      ("bootstrap_test.csv", "intervals.csv"),
                      ("fairness_test.csv", "fairness.csv")):
        if (run_dir / src).exists():
            tables[name] = read_csv(run_dir / src)
    if scored_dir is not None:
        drift_file = Path(scored_dir) / "drift.csv"
        if not drift_file.exists():
            raise FileNotFoundError(f"{drift_file} not found; run creditdecide predict first")
        tables["drift.csv"] = read_csv(drift_file)

    out.mkdir(parents=True, exist_ok=True)
    index = {"creditdecide_version": __version__, "run_dir": str(run_dir),
             "scored_dir": None if scored_dir is None else str(scored_dir),
             "decision_model": decision["model"], "tables": {}}
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)
        index["tables"][name] = {"description": DESCRIPTIONS[name], "rows": int(len(frame)),
                                 "columns": list(frame.columns)}
    (out / "tables.json").write_text(json.dumps(index, indent=2))
    return index

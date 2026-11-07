"""creditdecide command line.

  creditdecide datasets
  creditdecide quality --dataset credit-default
  creditdecide run --dataset credit-default --out results/credit --plot          # budget 10%
  creditdecide run --dataset credit-default --out results/cost \\
      --cost-fp 8000 --lgd 0.6 --exposure-column BILL_AMT1                        # cost policy
  creditdecide run --dataset credit-default --out results/el --cost-fp 8000 --lgd 0.6 \\
      --exposure-column BILL_AMT1 --decision-rule expected_loss             # per-account rule
  creditdecide run --dataset credit-default --out results/tuned --tune         # ~4 min on 2 cores
  creditdecide predict --model-dir results/credit/model --csv new.csv --out scored --id-column ID \\
      --reasons 4
  creditdecide drift --model-dir results/credit/model --csv new.csv
  creditdecide outcomes --model-dir results/credit/model --scores scored/scores.csv \\
      --labels labels.csv --id-column ID --out outcomes.json        # labels that arrive later
  creditdecide tableau --run-dir results/credit --scored scored --out results/tableau
  creditdecide sql --dataset credit-default --query \\
      'SELECT EDUCATION, AVG("DEFAULT") AS default_rate FROM data GROUP BY ALL ORDER BY ALL'
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from creditdecide import sql
from creditdecide.bootstrap import format_interval
from creditdecide.calibration import calibration_metrics
from creditdecide.datasets import (
    CREDIT_DEFAULT_DOCUMENTED_CODES,
    CREDIT_DEFAULT_SPEC,
    DATASETS,
    FeatureSpec,
    load_credit_default,
    load_csv,
    read_csv,
)
from creditdecide.decision import DecisionPolicy
from creditdecide.metrics import evaluate
from creditdecide.modelio import load_model
from creditdecide.monitor import drift_report
from creditdecide.pipeline import DECISION_RULES, SCORING_MODELS, TUNED_ONLY, Config, run
from creditdecide.quality import format_report, quality_report
from creditdecide.reasons import MAX_REASONS
from creditdecide.tableau import export as tableau_export


def _data_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", choices=sorted(DATASETS))
    src.add_argument("--csv", type=Path, help="your own table (one row per example)")
    p.add_argument("--target", help="label column (required with --csv)")
    p.add_argument("--positive-label", default="1",
                   help="target value a decision acts on (default 1)")
    p.add_argument("--categorical", nargs="*", default=[], help="columns to one-hot encode")
    p.add_argument("--signed-log", nargs="*", default=[],
                   help="heavy-tailed columns to transform with sign(x)*log1p(|x|)")
    p.add_argument("--drop", nargs="*", default=[], help="columns to exclude (e.g. IDs)")
    p.add_argument("--protected", nargs="*", default=[],
                   help="excluded from features unless --include-protected; always audited")
    p.add_argument("--dropna", action="store_true", help="drop rows with missing values")


def _model_args(p: argparse.ArgumentParser) -> None:
    d = Config()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--include-protected", action="store_true",
                   help="use protected attributes as features (off by default)")


def _decision_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("decision policy (binary targets)")
    g.add_argument("--budget", type=float, default=None,
                   help="act on this fraction of accounts (default 0.10 unless costs given)")
    g.add_argument("--cost-fp", type=float, help="cost of acting on a would-be negative")
    g.add_argument("--cost-fn", type=float, help="constant cost of a missed positive")
    g.add_argument("--lgd", type=float, help="loss given default, 0-1 (with --exposure-column)")
    g.add_argument("--exposure-column", help="balance at risk per row (with --lgd)")
    g.add_argument("--decision-model", default="auto",
                   choices=["auto", *SCORING_MODELS, TUNED_ONLY])
    g.add_argument("--decision-rule", default="threshold", choices=DECISION_RULES,
                   help="expected_loss: per-account rule from calibrated probabilities "
                        "(cost policies only)")
    g.add_argument("--calibration", default="auto",
                   choices=["auto", "none", "sigmoid", "isotonic"])
    g.add_argument("--reasons", type=int, default=MAX_REASONS,
                   help=f"reason codes per flagged test account (1-{MAX_REASONS})")
    g.add_argument("--bootstrap", type=int, default=Config().bootstrap,
                   help="bootstrap samples for 95%% intervals on test results (0 = off)")
    g = p.add_argument_group("hyperparameter tuning")
    g.add_argument("--tune", action="store_true",
                   help="tune models by cross-validation on the training split; adds "
                        "gradient boosting")
    g.add_argument("--tune-iter", type=int, default=Config().tune_iter,
                   help="candidates per model family")


def _policy(a) -> DecisionPolicy:
    if a.cost_fp is None and a.cost_fn is None and a.lgd is None:
        return DecisionPolicy(budget=a.budget if a.budget is not None else 0.10)
    if a.budget is not None:
        raise SystemExit("give either --budget or cost options, not both")
    return DecisionPolicy(kind="cost", budget=None, cost_fp=a.cost_fp, cost_fn=a.cost_fn,
                          lgd=a.lgd, exposure_column=a.exposure_column)


def _load(a) -> tuple[pd.DataFrame, FeatureSpec, dict | None, str]:
    if a.dataset == "credit-default":
        return (load_credit_default(), CREDIT_DEFAULT_SPEC, CREDIT_DEFAULT_DOCUMENTED_CODES,
                "credit-card default (UCI, Yeh 2009)")
    if not a.target:
        raise SystemExit("--target is required with --csv")
    df = read_csv(a.csv, nrows=1)
    pos = a.positive_label
    if a.target in df and pd.api.types.is_numeric_dtype(df[a.target]):
        try:
            pos = int(pos)
        except ValueError:
            pass
    spec = FeatureSpec(target=a.target, categorical=tuple(a.categorical),
                       signed_log=tuple(a.signed_log), protected=tuple(a.protected),
                       drop=tuple(a.drop), positive_label=pos)
    return load_csv(a.csv, spec, dropna=a.dropna), spec, None, a.csv.name


def _config(a, **kw) -> Config:
    return Config(seed=a.seed, include_protected=a.include_protected, **kw)


def _check_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise SystemExit(f"{out} is not empty; use --overwrite to replace its files")


def _print_decision(d: dict) -> None:
    t = d["test"]
    rule = (f"threshold {d['threshold']:.4f}" if d.get("rule", "threshold") == "threshold"
            else "expected-loss rule")
    print(f"\ndecision model: {d['model']} ({d['selected_by']}); policy {d['policy']}; {rule}")
    print(f"test: flagged {t['flagged']:,}/{t['n']:,} ({100 * t['flagged_fraction']:.1f}%), "
          f"captured {100 * t['capture_rate']:.1f}% of positives at "
          f"{100 * t['precision']:.1f}% precision (lift {t['lift']:.2f})")
    if "expected_cost" in t:
        print(f"      expected-cost saving vs flag nobody "
              f"{100 * t['saving_fraction_vs_flag_nobody']:.1f}%, vs best trivial strategy "
              f"{100 * t['saving_fraction_vs_best_trivial']:.1f}%")


def _print_drift(rep: pd.DataFrame) -> None:
    flagged = rep[rep.status != "stable"]
    print(f"drift vs reference: {len(flagged)} of {len(rep)} variables shifted "
          f"(PSI >= 0.10)")
    if len(flagged):
        print(flagged.round(3).to_string(index=False))


def _outcomes(model, scores: pd.DataFrame, labels: pd.DataFrame, id_column: str) -> dict:
    """Decision outcome (DuckDB join of decisions to labels), ranking and calibration."""
    target = model.manifest["target"]
    decision, rows = sql.outcomes(scores, labels, id_column, target,
                                  model.manifest["positive_label"], model.threshold)
    if not len(rows):
        raise SystemExit(f"no rows match on {id_column!r} between the scores and the labels")
    y, s = rows["y"].to_numpy(), rows["score"].to_numpy()
    both = len(np.unique(y)) == 2
    outcome = {"decision": decision,
               "ranking": (evaluate(y, rows["flag"].to_numpy(), np.column_stack([1 - s, s]))
                           if both else None),
               "calibration": calibration_metrics(y, rows["probability"]) if both else None}
    ref = model.manifest["decision"]["test"]
    unmatched = decision["scored_without_label"] + decision["labelled_without_score"]
    auc = f", ROC AUC {outcome['ranking']['roc_auc']:.3f}" if both else ""
    print(f"outcomes: {decision['n']:,} accounts with labels; captured "
          f"{100 * decision['capture_rate']:.1f}% at {100 * decision['precision']:.1f}% "
          f"precision{auc} (test split at training: {100 * ref['capture_rate']:.1f}% / "
          f"{100 * ref['precision']:.1f}%)")
    if unmatched:
        print(f"  unmatched: {decision['scored_without_label']:,} scored rows without a label, "
              f"{decision['labelled_without_score']:,} labels without a score")
    return outcome


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="creditdecide", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="list bundled datasets with source and license")

    p = sub.add_parser("quality", help="data-quality report")
    _data_args(p)

    p = sub.add_parser("run", help="train, evaluate, choose a decision rule, export the model")
    _data_args(p)
    _model_args(p)
    _decision_args(p)
    p.add_argument("--plot", action="store_true",
                   help="write ROC/PR and calibration figures")

    p = sub.add_parser("predict", help="score a new file with a saved model")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--id-column", help="column copied to the output to identify rows")
    p.add_argument("--reasons", type=int, default=0,
                   help=f"also write reason codes (up to {MAX_REASONS}) for flagged rows")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("drift", help="population stability of a new file vs the model's reference")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", type=Path, help="optional CSV for the full report")

    p = sub.add_parser("outcomes",
                       help="join saved decisions to labels that arrived later (DuckDB SQL)")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--scores", required=True, type=Path, help="scores.csv from predict")
    p.add_argument("--labels", required=True, type=Path,
                   help="CSV with the ID column and the model's target column")
    p.add_argument("--id-column", required=True)
    p.add_argument("--out", type=Path, help="optional JSON for the full result")

    p = sub.add_parser("tableau", help="write a run's results as Tableau Public tables (CSV)")
    p.add_argument("--run-dir", required=True, type=Path, help="output folder of run")
    p.add_argument("--scored", type=Path, help="output folder of predict, adds drift.csv")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("sql", help="run a read-only DuckDB SQL query on the data (table: data)")
    _data_args(p)
    p.add_argument("--query", required=True, help="SQL over the table named data")
    p.add_argument("--out", type=Path, help="optional CSV for the full result")

    a = ap.parse_args(argv)

    if a.cmd == "datasets":
        from importlib import resources

        meta = json.loads((resources.files("creditdecide.data") / "credit_default.json")
                          .read_text())
        print(f"credit-default: {meta['name']} ({meta['creator']}), {meta['n_rows']:,} rows\n"
              f"  {meta['source']}, doi:{meta['doi']}, {meta['license']}\n"
              f"  changes: {meta['changes']}\n  cite: {meta['citation']}")
        return

    if a.cmd == "outcomes":
        model = load_model(a.model_dir)
        scores, labels = read_csv(a.scores), read_csv(a.labels)
        result = _outcomes(model, scores, labels, a.id_column)
        if a.out:
            a.out.write_text(json.dumps(result, indent=2))
        return

    if a.cmd == "tableau":
        try:
            index = tableau_export(a.run_dir, a.out, a.scored, overwrite=a.overwrite)
        except (FileNotFoundError, FileExistsError) as e:
            raise SystemExit(str(e)) from None
        for name, info in index["tables"].items():
            print(f"  {name:<16} {info['rows']:>6,} rows  {info['description']}")
        print(f"wrote {a.out} (see tables.json and docs/TABLEAU.md)")
        return

    if a.cmd in ("predict", "drift"):
        model = load_model(a.model_dir)
        new = read_csv(a.csv)
        for w in model.validate(new):
            warnings.warn(w, stacklevel=1)
        if a.cmd == "drift":
            rep = drift_report(model.reference, new, model.score(new))
            _print_drift(rep)
            if a.out:
                rep.to_csv(a.out, index=False)
            return
        _check_out(a.out, a.overwrite)
        a.out.mkdir(parents=True, exist_ok=True)
        if a.id_column and a.id_column not in new:
            raise SystemExit(f"--id-column {a.id_column!r} not in {a.csv}")
        scored = model.decide(new, id_column=a.id_column)
        scored.to_csv(a.out / "scores.csv", index=False)
        rep = drift_report(model.reference, new, scored["score"].to_numpy())
        rep.to_csv(a.out / "drift.csv", index=False)
        how = (f"at threshold {model.threshold:.4f}" if model.rule == "threshold"
               else "by the expected-loss rule")
        print(f"scored {len(scored):,} rows; flagged {int(scored.flag.sum()):,} "
              f"({100 * scored.flag.mean():.1f}%) {how}")
        _print_drift(rep)
        if a.reasons:
            if not 1 <= a.reasons <= MAX_REASONS:
                raise SystemExit(f"--reasons must be between 1 and {MAX_REASONS}")
            wide, long, rsum = model.reasons(new, k=a.reasons, id_column=a.id_column)
            wide.to_csv(a.out / "reasons.csv", index=False)
            long.to_csv(a.out / "reasons_detail.csv", index=False)
            (a.out / "reasons_summary.json").write_text(json.dumps(rsum, indent=2))
            if rsum.get("accounts_explained"):
                print(f"reason codes for {rsum['accounts_explained']:,} flagged rows; stated "
                      f"reasons alone change the decision for "
                      f"{100 * rsum['sufficiency_rate']:.1f}%")
        target = model.manifest["target"]
        if target in new:  # outcomes analysis once labels are known, joined by row
            rows = pd.DataFrame({"_row": range(len(new))})
            outcome = _outcomes(model, scored.assign(_row=rows["_row"]),
                                new[[target]].assign(_row=rows["_row"]), "_row")
            (a.out / "outcomes.json").write_text(json.dumps(outcome, indent=2))
        print(f"wrote {a.out}")
        return

    df, spec, codes, name = _load(a)

    if a.cmd == "quality":
        print(format_report(quality_report(df, spec, codes)))

    elif a.cmd == "sql":
        try:
            result = sql.query(a.query, data=df)
        except Exception as e:  # duckdb errors: report the message, not a traceback
            raise SystemExit(f"query failed: {e}") from None
        if a.out:
            result.to_csv(a.out, index=False)
        with pd.option_context("display.max_rows", 50, "display.width", 120):
            print(result.to_string(index=False, max_rows=50))
        print(f"({len(result):,} rows" + (f"; wrote {a.out})" if a.out else ")"))

    elif a.cmd == "run":
        _check_out(a.out, a.overwrite)
        res = run(df, spec, _config(a, decision_model=a.decision_model,
                                    decision_rule=a.decision_rule, calibration=a.calibration,
                                    n_reasons=a.reasons, tune=a.tune, tune_iter=a.tune_iter,
                                    bootstrap=a.bootstrap),
                  _policy(a))
        res.extra["quality"] = quality_report(df, spec, codes)
        res.extra["dataset_name"] = name
        out = res.save(a.out, overwrite=a.overwrite)
        if a.plot:
            from creditdecide.plotting import plot_reliability, plot_roc_pr

            if res.decision is not None:
                plot_roc_pr(res.samples, spec.target, out / "roc_pr.png")
                plot_reliability(res.calibration["reliability_test"], out / "calibration.png")
        print(res.summary("test").round(3).to_string(index=False))
        if res.decision is not None:
            _print_decision(res.decision)
            c = res.calibration
            print(f"calibration: {c['method']}; test ECE {c['test_calibrated']['ece']:.3f}; "
                  f"expected {c['test_calibrated']['expected_positives']:,.0f} positives, "
                  f"actual {c['test_calibrated']['actual_positives']:,}")
            rs = res.decision.get("reasons_test")
            if rs:
                print(f"reason codes: {rs['accounts_explained']:,} flagged test accounts; "
                      f"stated reasons change the decision for "
                      f"{100 * rs['sufficiency_rate']:.1f}%, top reason alone for "
                      f"{100 * rs['top_reason_sufficiency_rate']:.1f}%")
            if res.tuning is not None:
                print("tuning (5-fold CV AUC on training split, default -> tuned):")
                for _, r in res.tuning.iterrows():
                    print(f"  {r['model']}: {r['default_cv_auc']:.4f} -> {r['tuned_cv_auc']:.4f}"
                          f" (sd {r['tuned_cv_auc_sd']:.4f})")
            if res.bootstrap is not None:
                print(f"95% bootstrap intervals ({res.bootstrap.attrs['replicates']:,} "
                      "resamples of the test split; see bootstrap_test.csv):")
                for _, r in res.bootstrap.iterrows():
                    print(f"  {format_interval(r)}")
            if res.fairness is not None and len(res.fairness):
                low = res.fairness[res.fairness.below_four_fifths & ~res.fairness.too_small]
                print(f"fairness audit: {len(low)} group(s) below the four-fifths reference "
                      f"(see fairness_test.csv)")
        print(f"\nwrote {out}" + (f" (model in {out / 'model'})" if res.decision else ""))


if __name__ == "__main__":
    main()

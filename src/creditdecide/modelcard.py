"""Model card (Mitchell et al. 2019 style) generated from a run, saved with the model artifact.

Written for a reader who did not build the model: what it is for, how it was built and
checked, what it gets wrong, and what to monitor. Section 'Validation coverage' maps the
contents to the three validation elements of the Federal Reserve / OCC model-risk guidance
SR 11-7 (conceptual soundness, ongoing monitoring, outcomes analysis).
"""

from __future__ import annotations

import pandas as pd

from creditdecide.bootstrap import format_interval


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and pd.isna(v):
                cells.append("–")
            elif c in fmt:
                cells.append(fmt[c](v))
            else:
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def model_card(res, dataset_name: str = "user-supplied table") -> str:
    d = res.decision
    t, v = d["test"], d["validation"]
    pol = d["policy"]
    missed_txt = (f"{pol['cost_fn']:g}" if "cost_fn" in pol
                  else f"{pol.get('lgd', 0):g} x {pol.get('exposure_column')}")
    if pol["kind"] == "budget":
        policy_txt = (f"act on the {_pct(pol['budget'])} highest-scoring accounts "
                      f"(threshold {d['threshold']:.4f}, chosen on the validation split)")
    elif d.get("rule") == "expected_loss":
        policy_txt = (f"act on an account when its expected loss avoided, p x {missed_txt}, "
                      f"exceeds the expected cost of acting, {pol['cost_fp']:g} x (1 - p), "
                      "with p the calibrated probability")
    elif "cost_fn" in pol:
        policy_txt = (f"minimize expected cost with a false alarm costing {pol['cost_fp']:g} and "
                      f"a missed positive costing {pol['cost_fn']:g} (threshold "
                      f"{d['threshold']:.4f}, chosen on the validation split)")
    else:
        policy_txt = (f"minimize expected cost with a false alarm costing {pol['cost_fp']:g} and "
                      f"a missed positive costing {pol['lgd']:g} x {pol['exposure_column']} "
                      f"(threshold {d['threshold']:.4f}, chosen on the validation split)")
    summ = res.summary("test")
    pos_name = res.spec.label_names.get(res.classes[1])
    pos_txt = f"`{res.classes[1]}`" + (f" ({pos_name})" if pos_name else "")
    excluded = [c for c in res.data.columns if c not in res.features and c != res.spec.target]
    parts = [
        f"# Model card: {d['model']} on {dataset_name}",
        "",
        "## Purpose",
        f"Ranks rows by the probability that `{res.spec.target}` = {pos_txt} and turns "
        f"the ranking into a decision: {policy_txt}. Intended as decision support with human "
        "review; not for fully automated adverse decisions without reason codes.",
        "",
        "## Data and features",
        f"- Training / validation / test rows: {len(res.splits['train']):,} / "
        f"{len(res.splits['val']):,} / {len(res.splits['test']):,} (group-aware stratified "
        "split; rows with identical model inputs never cross splits).",
        f"- Model inputs ({len(res.features)}): {', '.join(res.features)}.",
        f"- Excluded: {', '.join(excluded) or 'none'}.",
        f"- Positive rate in the test split: {_pct(t['positives'] / t['n'])}.",
        "",
        "## Model selection",
        f"Scoring model chosen by validation ROC AUC among "
        f"{', '.join(f'{k} ({x:.3f})' for k, x in d['validation_roc_auc'].items())}.",
        *_tuning_lines(res),
        "",
        "## Test-split performance (all models)",
        _table(summ, ["model", "accuracy", "balanced_accuracy", "roc_auc", "average_precision"]),
        "",
        "## Decision outcome (test split)",
        f"- Flagged {t['flagged']:,} of {t['n']:,} ({_pct(t['flagged_fraction'])}); caught "
        f"{_pct(t['capture_rate'])} of positives at {_pct(t['precision'])} precision "
        f"(lift {t['lift']:.2f} over random selection).",
        f"- Validation estimate was capture {_pct(v['capture_rate'])}, precision "
        f"{_pct(v['precision'])}: the operating point held on unseen data within "
        f"{abs(t['capture_rate'] - v['capture_rate']) * 100:.1f} percentage points.",
    ]
    if "expected_cost" in t:
        parts.append(f"- Expected cost saving vs flagging nobody: "
                     f"{_pct(t['saving_fraction_vs_flag_nobody'])}; vs the better of flag-"
                     f"nobody / flag-everybody: {_pct(t['saving_fraction_vs_best_trivial'])}.")
    parts += _calibration_lines(res) + _reason_lines(res) + _interval_lines(res)
    if res.fairness is not None and len(res.fairness):
        f = res.fairness.assign(note=lambda t: t["too_small"].map(
            {True: "too small to interpret", False: ""}))
        parts += ["", "## Outcomes by protected group (test split)",
                  "Protected attributes are audited whether or not they are model inputs. "
                  "Adverse impact ratio = favorable-outcome rate / highest group's; < 0.8 is "
                  "marked against the four-fifths reference (an indicator, not a legal test).",
                  "",
                  _table(f, ["attribute", "group", "n", "positive_rate", "flag_rate",
                             "false_alarm_rate", "adverse_impact_ratio", "note"]),
                  "",
                  "Proxy check (ROC AUC of predicting group membership from model inputs; "
                  "0.5 = no information):"]
        for col, aucs in res.proxies.items():
            parts.append(f"- {col}: " + (", ".join(f"{g} {a:.3f}" for g, a in aucs.items())
                                          or "groups too small"))
    parts += [
        "",
        "## Monitoring",
        "`creditdecide predict` and `creditdecide drift` compare new data with the validation "
        "sample saved in `reference.json` using the population stability index: < 0.10 stable, "
        "0.10-0.25 check other metrics, > 0.25 investigate before relying on the scores. When "
        "outcomes become available, rerun `predict` on a file that includes the target column "
        "to compare realized capture and precision with the figures above.",
        "",
        "## Validation coverage (SR 11-7 elements)",
        "- Conceptual soundness: documented data checks, leakage-safe splits, baselines, "
        "selection on validation data only.",
        "- Ongoing monitoring: PSI on every input and on the score; schema and category checks "
        "at scoring time.",
        "- Outcomes analysis: test-split results above, with bootstrap intervals when computed; "
        "realized-outcome comparison when "
        "labels arrive.",
        "- Not covered: independent review, champion-challenger testing, back-testing over "
        "many periods.",
        "",
        "## Limitations",
        "- Reason codes approximate the model by setting one factor at a time to typical values "
        "from a reference population; their accuracy is measured above, not guaranteed.",
        "- Calibration is fitted on one validation sample; recheck it with `predict` on "
        "labeled data, since calibration drifts before ranking does.",
        "- Results describe one historical sample; populations change. Bootstrap intervals cover "
        "the choice of test customers, not retraining.",
    ]
    return "\n".join(parts) + "\n"


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3f}"


def _margin_unit(res) -> str:
    return ("expected-cost units" if res.decision.get("rule") == "expected_loss"
            else "score points")


def _tuning_lines(res) -> list[str]:
    if res.tuning is None:
        return ["Hyperparameters: library defaults (random forest: 200 trees, depth 10); tuning "
                "was not run."]
    lines = ["", "Hyperparameters tuned by 5-fold group cross-validation on the training split "
             "(ROC AUC, default -> tuned):"]
    for _, r in res.tuning.iterrows():
        lines.append(f"- {r['model']}: {r['default_cv_auc']:.4f} -> {r['tuned_cv_auc']:.4f} "
                     f"(sd {r['tuned_cv_auc_sd']:.4f}); {r['best_params']}")
    return lines


def _calibration_lines(res) -> list[str]:
    c = res.calibration
    raw, cal, fl = c["test_raw"], c["test_calibrated"], c["flagged_test"]
    lines = [
        "", "## Probabilities (calibration, test split)",
        f"- Map: {c['method']} (chosen by {c['selected_by']}). Monotone, so it does not "
        "change which accounts a threshold rule flags.",
        f"- Expected calibration error {raw['ece']:.3f} raw, {cal['ece']:.3f} calibrated; "
        f"Brier score {raw['brier']:.4f} raw, {cal['brier']:.4f} calibrated.",
        f"- Portfolio forecast: {cal['expected_positives']:,.0f} positives expected, "
        f"{cal['actual_positives']:,} occurred. Among flagged accounts: "
        f"{fl['expected_positives']:,.0f} expected, {fl['actual_positives']:,} occurred.",
    ]
    if "loss_test" in c:
        lines.append(f"- Expected loss {c['loss_test']['expected']:,.0f} vs realized "
                     f"{c['loss_test']['realised']:,.0f} (same cost assumptions as the policy).")
    return lines


def _reason_lines(res) -> list[str]:
    r = res.decision.get("reasons_test")
    if not r:
        return []
    share = ", ".join(f"{k} ({_pct(v)})" for k, v in r["primary_reason_share"].items())
    return [
        "", "## Reason codes (flagged test accounts)",
        f"- Up to {r['reasons_per_account']} reasons per flagged account: the factors whose "
        "values, set to those of marginally not-flagged validation accounts, lower the decision "
        "margin most (Regulation B comment 9(b)(2)-5, first method).",
        f"- Accuracy: setting the stated reasons to the reference values changes the decision for "
        f"{_pct(r['sufficiency_rate'])} of {r['accounts_explained']:,} flagged accounts; the top "
        f"reason alone for {_pct(r['top_reason_sufficiency_rate'])}. Setting the top reason to "
        f"the reference values lowers the decision margin by "
        f"{_num(r['mean_margin_drop_top_reason'])} on average vs "
        f"{_num(r['mean_margin_drop_random_factor'])} for a random factor ({_margin_unit(res)}).",
        f"- Primary reason: {share}.",
    ]


def _interval_lines(res) -> list[str]:
    b = getattr(res, "bootstrap", None)
    if b is None:
        return []
    return [
        "", "## Uncertainty (test split)",
        f"95% intervals from {b.attrs['replicates']:,} bootstrap resamples of the test accounts "
        "(rows with identical inputs drawn together). The models, threshold, calibration map "
        "and rule are held fixed, so the intervals show how much the figures depend on which "
        "customers were in the test sample, not the extra variation from retraining. For a "
        "difference, \"sign clear\" means the interval excludes zero.",
        "",
        *(f"- {format_interval(r)}" for _, r in b.iterrows()),
    ]

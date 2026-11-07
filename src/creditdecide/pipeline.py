"""End-to-end experiment.

1. Group-aware stratified split, 60/20/20: rows with identical feature vectors always share a
   split, so the test set never contains an exact copy of a training input.
2. Preprocessing (signed-log, z-score, one-hot) fit on the training split only.
3. Models: a majority-class baseline, logistic regression and a random forest on the processed
   features.
4. Every model scored on every split. Labels are used only by the models, never by a
   transform.
5. Binary targets: the scoring model with the best validation ROC AUC (or the one requested)
   is turned into a decision rule whose threshold is chosen on the validation split
   (creditdecide.decision) and evaluated once on the test split.
6. Its scores are calibrated on the validation split (creditdecide.calibration); calibration is
   monotone, so it never changes which accounts a threshold rule flags.
7. Flagged test accounts get reason codes (creditdecide.reasons), with accuracy checks.
8. Test-split results get 95% bootstrap intervals (creditdecide.bootstrap): the decision
   outcome, every model's ROC AUC, model-vs-model and model-vs-rule differences, and the
   accuracy of the default (and, for cost policies, loss) forecast.
Optional: hyperparameter tuning of the supervised models by cross-validation inside the
training split (creditdecide.tuning), which also adds a gradient-boosting model.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from creditdecide import bootstrap as bs
from creditdecide._version import __version__
from creditdecide.calibration import (
    apply_map,
    calibration_metrics,
    choose_and_fit,
    reliability_table,
)
from creditdecide.datasets import FeatureSpec, validate_table
from creditdecide.decision import (
    DecisionPolicy,
    choose_threshold,
    cost_sensitivity,
    expected_loss_margin,
    gains_table,
    simple_rule_benchmark,
    summarize,
)
from creditdecide.fairness import fairness_audit
from creditdecide.forest import predict_on_one_thread
from creditdecide.metrics import evaluate
from creditdecide.modelcard import model_card
from creditdecide.modelio import input_schema, write_artifact
from creditdecide.monitor import build_reference
from creditdecide.preprocess import make_preprocessor
from creditdecide.reasons import MAX_REASONS, explain, factor_groups, reference_profile
from creditdecide.tuning import tune

MODELS = ("majority_baseline", "logistic_regression", "random_forest")
SCORING_MODELS = ("logistic_regression", "random_forest")
TUNED_ONLY = "gradient_boosting"  # trained only when tuning is requested
DECISION_RULES = ("threshold", "expected_loss")


@dataclass
class Config:
    seed: int = 0
    test_size: float = 0.2
    val_size: float = 0.25  # of the non-test rows, giving 60/20/20
    include_protected: bool = False
    rf_n_estimators: int = 200
    rf_max_depth: int | None = 10
    decision_model: str = "auto"  # "auto" = best validation ROC AUC among scoring models
    decision_rule: str = "threshold"  # or "expected_loss" (cost policy, calibrated p)
    calibration: str = "auto"  # "auto" (CV on validation), "none", "sigmoid", "isotonic"
    n_reasons: int = MAX_REASONS
    tune: bool = False
    tune_iter: int = 12
    bootstrap: int = bs.DEFAULT_REPLICATES  # 0 = no confidence intervals

    def __post_init__(self):
        allowed = ("auto", *SCORING_MODELS, TUNED_ONLY)
        if self.decision_model not in allowed:
            raise ValueError(f"decision_model must be one of {allowed}")
        if self.decision_model == TUNED_ONLY and not self.tune:
            raise ValueError("gradient_boosting is only trained with tune=True")
        if self.decision_rule not in DECISION_RULES:
            raise ValueError(f"decision_rule must be one of {DECISION_RULES}")
        if self.calibration not in ("auto", "none", "sigmoid", "isotonic"):
            raise ValueError("calibration must be auto, none, sigmoid or isotonic")
        if not 1 <= self.n_reasons <= MAX_REASONS:
            raise ValueError(f"n_reasons must be between 1 and {MAX_REASONS}")
        if self.tune_iter < 1:
            raise ValueError("tune_iter must be positive")
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be 0 (off) or a number of replicates")


@dataclass
class Result:
    config: Config
    spec: FeatureSpec
    policy: DecisionPolicy
    classes: list
    features: list[str]
    feature_names: list[str]
    splits: dict[str, np.ndarray]
    metrics: pd.DataFrame
    samples: pd.DataFrame
    preprocessor: object
    models: dict
    decision: dict | None
    gains: pd.DataFrame | None
    sensitivity: pd.DataFrame | None
    fairness: pd.DataFrame | None
    proxies: dict | None
    data: pd.DataFrame
    model_names: tuple[str, ...] = MODELS
    calibration: dict | None = None
    tuning: pd.DataFrame | None = None
    tuning_details: pd.DataFrame | None = None
    reasons: pd.DataFrame | None = None
    reasons_long: pd.DataFrame | None = None
    reason_groups: dict | None = None
    reason_reference: dict | None = None
    bootstrap: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    def summary(self, split: str = "test") -> pd.DataFrame:
        m = self.metrics[self.metrics.split == split]
        wide = m.pivot_table(index="model", columns="metric", values="value")
        return wide.reindex(list(self.model_names)).reset_index().rename_axis(columns=None)

    def probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of the positive class from the decision model."""
        s = self.predict_proba(df, model=self.decision["model"])[:, 1]
        return apply_map(self.calibration["params"], s)

    def predict_proba(self, df: pd.DataFrame, model: str = "random_forest") -> np.ndarray:
        """Class probabilities for new rows (same columns as the training table)."""
        X = np.asarray(self.preprocessor.transform(df), dtype=float)
        return self.models[model].predict_proba(X)

    def save(self, outdir: str | Path, overwrite: bool = False) -> Path:
        """Write evaluation outputs; the deployable model goes to <outdir>/model."""
        out = Path(outdir)
        if out.exists() and any(out.iterdir()) and not overwrite:
            raise FileExistsError(f"{out} is not empty; pass overwrite=True to replace files")
        out.mkdir(parents=True, exist_ok=True)
        self.metrics.to_csv(out / "metrics_long.csv", index=False)
        for split in ("train", "val", "test"):
            self.summary(split).to_csv(out / f"metrics_{split}.csv", index=False)
        self.samples.to_csv(out / "samples.csv", index=False)
        self.data.iloc[self.splits["test"]].to_csv(out / "holdout_test.csv", index=False)
        if self.decision is not None:
            (out / "decision.json").write_text(json.dumps(self.decision, indent=2))
            self.gains.to_csv(out / "gains_test.csv", index=False)
        if self.sensitivity is not None:
            self.sensitivity.to_csv(out / "cost_sensitivity.csv", index=False)
        if self.fairness is not None and len(self.fairness):
            self.fairness.to_csv(out / "fairness_test.csv", index=False)
        if self.calibration is not None:
            pd.DataFrame(self.calibration["cv"]).to_csv(out / "calibration_cv.csv", index=False)
            self.calibration["reliability_test"].to_csv(out / "calibration_test.csv",
                                                        index=False)
        if self.tuning is not None:
            self.tuning.to_csv(out / "tuning.csv", index=False)
            self.tuning_details.to_csv(out / "tuning_details.csv", index=False)
        if self.reasons is not None:
            self.reasons.to_csv(out / "reasons_test.csv", index=False)
            self.reasons_long.to_csv(out / "reasons_test_detail.csv", index=False)
        if self.bootstrap is not None:
            self.bootstrap.to_csv(out / "bootstrap_test.csv", index=False)
        meta = {
            "config": asdict(self.config),
            "spec": spec_to_dict(self.spec),
            "classes": [str(c) for c in self.classes],
            "feature_names": self.feature_names,
            "split_sizes": {s: int(len(i)) for s, i in self.splits.items()},
            "proxy_auc": self.proxies,
            "bootstrap": (None if self.bootstrap is None else
                          {**self.bootstrap.attrs, "resampled": "test accounts, rows with "
                           "identical inputs drawn together", "held fixed": "models, "
                           "threshold, calibration map, decision rule"}),
            "environment": environment_info(),
            **self.extra,
        }
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))
        if self.decision is not None:
            self.export_model(out / "model")
        return out

    def export_model(self, outdir: str | Path) -> Path:
        """Write the deployable artifact for the decision model (binary targets only):
        pipeline, weights, drift reference (validation split) and model card."""
        if self.decision is None:
            raise ValueError("model export needs a binary target with a decision rule")
        name = self.decision["model"]
        va = self.splits["val"]
        reference = build_reference(
            self.data.iloc[va], self.features, self.spec.categorical,
            self.samples["decision_score"].to_numpy()[va], sample="validation split")
        manifest = {
            "creditdecide_version": __version__,
            "model": name,
            "target": self.spec.target,
            "positive_label": str(self.classes[1]),
            "negative_label": str(self.classes[0]),
            "input_schema": input_schema(self.data, self.features, self.spec.categorical),
            "excluded_columns": [c for c in self.data.columns
                                 if c not in self.features and c != self.spec.target],
            "decision": self.decision,
            "decision_inputs": decision_inputs(self.decision, self.features),
            "calibration": self.calibration["params"],
            "reasons": {"groups": {k: list(v) for k, v in self.reason_groups.items()},
                        "reference": self.reason_reference,
                        "max_reasons": self.config.n_reasons},
            "test_metrics": self.summary("test").set_index("model").loc[name].dropna()
            .to_dict(),
            "training_rows": int(len(self.splits["train"])),
            "config": asdict(self.config),
            "environment": environment_info(),
        }
        card = model_card(self, self.extra.get("dataset_name", "user-supplied table"))
        return write_artifact(Path(outdir), manifest, self.preprocessor, self.models[name],
                              reference=reference, model_card=card)


def spec_to_dict(spec: FeatureSpec) -> dict:
    d = asdict(spec)
    d["label_names"] = {str(k): v for k, v in spec.label_names.items()}
    d["group_labels"] = {c: {str(k): v for k, v in m.items()}
                         for c, m in spec.group_labels.items()}
    d["reason_groups"] = {k: list(v) for k, v in spec.reason_groups.items()}
    return d


def decision_inputs(decision: dict, features: list[str]) -> list[str]:
    """Columns a scoring file needs beyond the model inputs (the exposure column of an
    expected-loss rule, when it is not itself a model input)."""
    col = decision["policy"].get("exposure_column")
    if decision.get("rule") == "expected_loss" and col and col not in features:
        return [col]
    return []


def environment_info() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "duckdb": duckdb.__version__, "creditdecide": __version__,
            "platform": platform.platform()}


def duplicate_groups(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Group id per row; rows with identical feature vectors share an id."""
    return df.groupby(features, sort=False, dropna=False).ngroup().to_numpy()


def group_split(df: pd.DataFrame, features: list[str], y: np.ndarray, cfg: Config):
    """Stratified train/val/test indices with identical feature vectors kept together."""
    groups = duplicate_groups(df, features)
    first = pd.Series(np.arange(len(df))).groupby(groups).first().to_numpy()
    g_ids = np.arange(len(first))
    g_lab = y[first]  # stratify groups by the label of their first row
    g_tmp, g_test = train_test_split(g_ids, test_size=cfg.test_size, stratify=g_lab,
                                     random_state=cfg.seed)
    g_train, g_val = train_test_split(g_tmp, test_size=cfg.val_size, stratify=g_lab[g_tmp],
                                      random_state=cfg.seed)
    return {name: np.flatnonzero(np.isin(groups, g))
            for name, g in (("train", g_train), ("val", g_val), ("test", g_test))}


def _ordered_classes(values, positive_label) -> list:
    """Sorted classes; for binary targets the positive label is placed last (index 1)."""
    classes = sorted(values, key=str)
    if len(classes) == 2:
        pos = [c for c in classes if str(c) == str(positive_label)]
        if not pos:
            raise ValueError(f"positive label {positive_label!r} not among target values "
                             f"{[str(c) for c in classes]}; set FeatureSpec.positive_label")
        classes = [c for c in classes if c is not pos[0]] + pos
    return classes


def run(df: pd.DataFrame, spec: FeatureSpec, config: Config | None = None,
        policy: DecisionPolicy | None = None) -> Result:
    cfg = config or Config()
    policy = policy or DecisionPolicy()
    validate_table(df, spec)
    if cfg.decision_rule == "expected_loss" and policy.kind != "cost":
        raise ValueError("the expected-loss rule needs a cost policy (cost_fp and a missed cost)")
    df = df.reset_index(drop=True)
    classes = _ordered_classes(df[spec.target].unique(), spec.positive_label)
    y = df[spec.target].map({c: i for i, c in enumerate(classes)}).to_numpy(int)
    k = len(classes)
    if cfg.tune and k != 2:
        raise ValueError("tuning is scored by ROC AUC and needs a binary target")
    features = spec.feature_columns(df, include_protected=cfg.include_protected)
    if policy.kind == "cost" and policy.exposure_column and policy.exposure_column not in df:
        raise ValueError(f"exposure column {policy.exposure_column!r} not in the table")
    splits = group_split(df, features, y, cfg)
    tr = splits["train"]

    pre = make_preprocessor(df, spec, include_protected=cfg.include_protected)
    pre.fit(df.iloc[tr])
    X = np.asarray(pre.transform(df), dtype=float)
    names = list(pre.get_feature_names_out())

    rf_kw = dict(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                 random_state=cfg.seed, n_jobs=-1)
    full_feature = ["majority_baseline", "logistic_regression", "random_forest"]
    models = {"majority_baseline": DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])}
    tuning_table = tuning_details = None
    if cfg.tune:
        best, tuning_table, tuning_details = tune(
            df.iloc[tr], y[tr], duplicate_groups(df, features)[tr],
            make_preprocessor(df, spec, include_protected=cfg.include_protected),
            seed=cfg.seed, n_iter=cfg.tune_iter, rf_n_estimators=cfg.rf_n_estimators)
        for name, est in best.items():
            models[name] = predict_on_one_thread(est.fit(X[tr], y[tr]))
        full_feature.append(TUNED_ONLY)
    else:
        models["logistic_regression"] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        models["random_forest"] = predict_on_one_thread(
            RandomForestClassifier(**rf_kw).fit(X[tr], y[tr]))
    model_names = tuple(full_feature)
    scoring_models = tuple(m for m in model_names if m in (*SCORING_MODELS, TUNED_ONLY))

    preds, probas = {}, {}
    for name in model_names:
        probas[name] = models[name].predict_proba(X)
        preds[name] = probas[name].argmax(axis=1)

    rows = []
    for split, ii in splits.items():
        for name in model_names:
            p = probas.get(name)
            for metric, value in evaluate(y[ii], preds[name][ii],
                                          None if p is None else p[ii]).items():
                rows.append({"model": name, "split": split, "metric": metric, "value": value})
    metrics = pd.DataFrame(rows)

    split_col = np.empty(len(df), dtype=object)
    for split, ii in splits.items():
        split_col[ii] = split
    samples = df.copy()
    samples.insert(0, "split", split_col)
    if spec.label_names:
        samples["label_name"] = df[spec.target].map(spec.label_names)
    cls = np.asarray(classes, dtype=object)
    for name in model_names:
        p = preds[name]
        samples[f"pred_{name}"] = np.where(p >= 0, cls[np.clip(p, 0, None)], None)

    result = Result(
        config=cfg, spec=spec, policy=policy, classes=classes, features=features,
        feature_names=names, splits=splits, metrics=metrics, samples=samples,
        preprocessor=pre, models=models, decision=None, gains=None, sensitivity=None,
        fairness=None, proxies=None, data=df, model_names=model_names,
        tuning=tuning_table, tuning_details=tuning_details,
        extra={"classes": [str(c) for c in classes]},
    )
    if k == 2:
        _decide(result, X, y, scoring_models, probas)
    return result


def _decide(res: Result, X: np.ndarray, y: np.ndarray, scoring_models: tuple[str, ...],
            probas: dict) -> None:
    """Binary targets: choose the decision model, calibrate it, apply the decision rule,
    audit outcomes and explain flagged accounts. Fills the corresponding Result fields."""
    cfg, policy, df, spec = res.config, res.policy, res.data, res.spec
    tr, va, te = res.splits["train"], res.splits["val"], res.splits["test"]
    samples, metrics = res.samples, res.metrics
    for name in scoring_models:
        samples[f"p_{name}"] = probas[name][:, 1]
    val_auc = metrics[(metrics.split == "val") & (metrics.metric == "roc_auc")
                      & metrics.model.isin(scoring_models)].set_index("model")["value"]
    chosen = val_auc.idxmax() if cfg.decision_model == "auto" else cfg.decision_model
    s = probas[chosen][:, 1]

    cal_params, cal_cv = choose_and_fit(s[va], y[va], cfg.calibration, seed=cfg.seed)
    prob = apply_map(cal_params, s)

    miss_all = policy.missed_cost(df, len(df))
    miss_v, miss_t = miss_all[va], miss_all[te]
    if cfg.decision_rule == "threshold":
        thr = choose_threshold(y[va], s[va], policy, miss_v)
        margin = s - thr
        flag = margin >= 0
    else:
        thr = None
        margin = expected_loss_margin(prob, miss_all, policy.cost_fp)
        flag = margin > 0
    res.decision = {
        "model": chosen,
        "selected_by": "validation ROC AUC" if cfg.decision_model == "auto" else "user",
        "validation_roc_auc": {m: float(v) for m, v in val_auc.items()},
        "policy": policy.to_dict(),
        "rule": cfg.decision_rule,
        "threshold": thr,
        "validation": summarize(y[va], s[va], thr, policy, miss_v, flag=flag[va]),
        "test": summarize(y[te], s[te], thr, policy, miss_t, flag=flag[te]),
    }
    numeric = [c for c in res.features if c not in spec.categorical]
    res.decision["simple_rule_benchmark"] = simple_rule_benchmark(
        df, numeric, y, s, va, te, target_fraction=float(flag[va].mean()))
    if cfg.decision_rule == "expected_loss":
        res.decision["note"] = ("validation figures use the split on which probabilities were "
                                "calibrated; the test figures are the independent estimate")
    res.gains = gains_table(y[te], s[te])
    if policy.kind == "cost" and policy.exposure_column:
        exp = df[policy.exposure_column].to_numpy(float)
        res.sensitivity = cost_sensitivity(
            y[va], s[va], y[te], s[te], exp[va], exp[te],
            cost_fps=[policy.cost_fp / 4, policy.cost_fp, policy.cost_fp * 4],
            lgds=sorted({0.3, 0.6, 0.9, policy.lgd}), p_test=prob[te])

    flagged_te = flag[te]
    rel = pd.concat([reliability_table(y[te], s[te]).assign(scores="raw"),
                     reliability_table(y[te], prob[te]).assign(scores="calibrated")])
    res.calibration = {
        "params": cal_params,
        "method": cal_params["method"],
        "selected_by": ("5-fold CV log loss within the validation split, one-standard-error "
                        "rule" if cfg.calibration == "auto" else "user"),
        "cv": cal_cv.to_dict(orient="records"),
        "test_raw": calibration_metrics(y[te], s[te]),
        "test_calibrated": calibration_metrics(y[te], prob[te]),
        "flagged_test": {"flagged": int(flagged_te.sum()),
                         "expected_positives": float(prob[te][flagged_te].sum()),
                         "actual_positives": int(y[te][flagged_te].sum())},
        "reliability_test": rel,
    }
    if policy.kind == "cost":
        res.calibration["loss_test"] = {
            "expected": float(np.sum(prob[te] * miss_t)),
            "realised": float(np.sum(miss_t[y[te] == 1])),
        }

    samples["decision_score"] = s
    samples["probability"] = prob
    samples["decision_margin"] = margin
    samples["flag"] = flag.astype(int)
    res.fairness, res.proxies = fairness_audit(df, spec, X, y, flag.astype(int), tr, te)

    def decide_frame(frame: pd.DataFrame):
        sc = res.predict_proba(frame, model=chosen)[:, 1]
        p = apply_map(cal_params, sc)
        if cfg.decision_rule == "threshold":
            m = sc - thr
            return m, m >= 0, p
        m = expected_loss_margin(p, policy.missed_cost(frame, len(frame)), policy.cost_fp)
        return m, m > 0, p

    res.reason_groups = factor_groups(res.features, spec.reason_groups)
    res.reason_reference = reference_profile(df.iloc[va], margin[va], flag[va], res.features,
                                             spec.categorical)
    idx = te[flagged_te]
    if len(idx):
        rows = df.iloc[idx].assign(_row=idx)
        n = min(cfg.n_reasons, len(res.reason_groups))
        wide, long, rsum = explain(rows, decide_frame, res.reason_groups, res.reason_reference,
                                   k=n, seed=cfg.seed, id_column="_row")
        ids = [c for c in spec.drop if c in df.columns]
        wide = wide.rename(columns={"_row": "row"})
        long = long.rename(columns={"_row": "row"})
        for c in reversed(ids):
            wide.insert(1, c, df[c].to_numpy()[wide["row"].to_numpy()])
            long.insert(1, c, df[c].to_numpy()[long["row"].to_numpy()])
        res.reasons, res.reasons_long = wide, long
        res.decision["reasons_test"] = rsum
    if cfg.bootstrap:
        res.bootstrap = _test_intervals(res, y, s, prob, flag, miss_all, scoring_models, probas)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _test_intervals(res: Result, y, s, prob, flag, miss_all, scoring_models, probas):
    """95% bootstrap intervals for the test-split results (creditdecide.bootstrap)."""
    cfg, policy, df, d = res.config, res.policy, res.data, res.decision
    te = res.splits["test"]
    yt, st, pt, ft, mt = y[te], s[te], prob[te], flag[te], miss_all[te]
    groups = duplicate_groups(df, res.features)[te]
    chosen = d["model"]
    ranked = {m: bs.RankedScores(yt, probas[m][te, 1]) for m in scoring_models}
    diffs = [(chosen, m) for m in scoring_models if m != chosen]
    bench = d.get("simple_rule_benchmark") or {}
    if bench:
        x = df[bench["column"]].to_numpy(float)[te]
        rule = x >= bench["cutoff"] if bench["direction"] == ">=" else x <= bench["cutoff"]
        top = np.zeros(len(te), bool)
        top[np.argsort(-st, kind="stable")[: int(rule.sum())]] = True
    both_rules = policy.kind == "cost"
    if both_rules:  # the other cost rule, so the two can be compared on the same draws
        if d["rule"] == "threshold":
            cut, per_account = ft, expected_loss_margin(pt, mt, policy.cost_fp) > 0
        else:
            va = res.splits["val"]
            thr = choose_threshold(y[va], s[va], policy, miss_all[va])
            cut, per_account = st >= thr, ft

    def stat(w):
        out = {"capture rate": bs.capture_rate(w, yt, ft),
               "precision": bs.precision(w, yt, ft),
               "flagged fraction": bs.flagged_fraction(w, ft)}
        out["lift"] = _ratio(out["precision"] * np.sum(w), np.sum(w * yt))
        auc = {m: r.auc(w) for m, r in ranked.items()}
        for m in scoring_models:
            out[f"ROC AUC {m}"] = auc[m]
        for a, b in diffs:
            out[f"ROC AUC {a} minus {b}"] = auc[a] - auc[b]
        if bench:
            out[f"capture rate, model minus rule {bench['rule']} at the rule's volume"] = (
                bs.capture_rate(w, yt, top) - bs.capture_rate(w, yt, rule))
        out["defaults forecast error, all accounts (expected / actual - 1)"] = (
            _ratio(np.sum(w * pt), np.sum(w * yt)) - 1)
        out["defaults forecast error, flagged accounts (expected / actual - 1)"] = (
            _ratio(np.sum(w * pt * ft), np.sum(w * yt * ft)) - 1)
        if both_rules:
            sc = bs.saving_vs_best_trivial(w, yt, cut, policy.cost_fp, mt)
            sp = bs.saving_vs_best_trivial(w, yt, per_account, policy.cost_fp, mt)
            out["saving vs best trivial, one cut-off"] = sc
            out["saving vs best trivial, per-account rule"] = sp
            out["saving, per-account rule minus one cut-off"] = sp - sc
            out["loss forecast error (expected / realized - 1)"] = (
                _ratio(np.sum(w * mt * pt), np.sum(w * mt * yt)) - 1)
        return out

    diff_names = {f"ROC AUC {a} minus {b}" for a, b in diffs}
    if bench:
        diff_names.add(f"capture rate, model minus rule {bench['rule']} at the rule's volume")
    diff_names |= {"defaults forecast error, all accounts (expected / actual - 1)",
                   "defaults forecast error, flagged accounts (expected / actual - 1)",
                   "saving, per-account rule minus one cut-off",
                   "loss forecast error (expected / realized - 1)"}
    auc_names = [f"ROC AUC {m}" for m in scoring_models] + [f"ROC AUC {a} minus {b}"
                                                            for a, b in diffs]
    units = {"lift": "ratio", **dict.fromkeys(auc_names, "auc"),
             **{k: "relative" for k in diff_names if "forecast error" in k}}
    return bs.bootstrap(stat, groups, n_boot=cfg.bootstrap, seed=cfg.seed,
                        differences=diff_names, units=units)

"""Calibration, expected-loss rule, reason codes, tuning, and loading format-1 artifacts."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from creditdecide import Config, DecisionPolicy, load_model, run
from creditdecide.calibration import (
    apply_map,
    calibration_metrics,
    choose_and_fit,
    fit_map,
    reliability_table,
)
from creditdecide.decision import expected_loss_margin, summarize
from creditdecide.reasons import explain, factor_groups, reference_profile


# ---------------------------------------------------------------- calibration
def test_isotonic_map_matches_sklearn():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = (rng.random(500) < s**2).astype(int)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
    grid = np.linspace(-0.1, 1.1, 300)
    assert np.allclose(apply_map(fit_map("isotonic", s, y), grid), iso.predict(grid))


def test_maps_are_monotone_and_preserve_ranking():
    rng = np.random.default_rng(1)
    s = rng.random(2000)
    y = (rng.random(2000) < 0.3 + 0.4 * s).astype(int)
    for m in ("none", "sigmoid", "isotonic"):
        p = apply_map(fit_map(m, s, y), np.sort(s))
        assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0 and p.max() <= 1


def test_sigmoid_fixes_overconfident_scores():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.05, 0.6, 20000)
    y = (rng.random(20000) < true_p).astype(int)
    raw = true_p ** 0.5  # systematically too high
    params, table = choose_and_fit(raw[:10000], y[:10000])
    assert params["method"] != "none"
    before = calibration_metrics(y[10000:], raw[10000:])
    after = calibration_metrics(y[10000:], apply_map(params, raw[10000:]))
    assert after["ece"] < before["ece"] / 3
    assert abs(after["expected_positives"] - after["actual_positives"]) < 0.05 * y[10000:].sum()
    assert table["chosen"].sum() == 1


def test_one_standard_error_rule_keeps_identity_for_calibrated_scores():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.9, 6000)
    y = (rng.random(6000) < p).astype(int)
    params, _ = choose_and_fit(p, y)
    assert params["method"] == "none"


def test_reliability_and_metrics_hand_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.6, 0.8])
    t = reliability_table(y, p, n_bins=2)
    assert t["mean_predicted"].tolist() == pytest.approx([0.2, 0.7])
    m = calibration_metrics(y, p, n_bins=2)
    assert m["ece"] == pytest.approx(0.25)  # |0.2-0| and |0.7-1|, weighted equally
    assert m["expected_positives"] == pytest.approx(1.8) and m["actual_positives"] == 2


# ---------------------------------------------------------------- expected-loss rule
def test_expected_loss_margin_and_summary():
    p = np.array([0.1, 0.5, 0.9])
    missed = np.array([100.0, 10.0, 10.0])
    m = expected_loss_margin(p, missed, cost_fp=5.0)
    assert m == pytest.approx([10 - 4.5, 5 - 2.5, 9 - 0.5])
    flag = m > 0
    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=5.0, lgd=1.0, exposure_column="e")
    r = summarize(np.array([0, 1, 1]), p, None, pol, missed, flag=flag)
    assert r["threshold"] is None and r["flagged"] == 3


# ---------------------------------------------------------------- reason codes
def _linear_decider(weights, threshold):
    """Toy model: score = sum of weights * columns; flagged when score >= threshold."""
    def decide(frame):
        score = sum(w * frame[c].to_numpy(float) for c, w in weights.items())
        m = score - threshold
        return m, m >= 0, 1 / (1 + np.exp(-score))
    return decide


def test_reasons_recover_known_drivers():
    rows = pd.DataFrame({"a": [5.0, 0.0], "b": [0.0, 4.0], "c": [1.0, 1.0]})
    ref = {"population": "test", "values": {"a": 0.0, "b": 0.0, "c": 0.0}}
    groups = factor_groups(["a", "b", "c"], {"A and C": ("a", "c")})
    assert groups == {"A and C": ("a", "c"), "b": ("b",)}
    wide, long, summary = explain(rows, _linear_decider({"a": 1, "b": 2, "c": 0.5}, 3.0),
                                  groups, ref, k=2)
    assert wide.loc[0, "reason_1"] == "A and C" and wide.loc[0, "impact_1"] == pytest.approx(5.5)
    assert wide.loc[1, "reason_1"] == "b" and wide.loc[1, "impact_1"] == pytest.approx(8.0)
    assert wide.loc[0, "reason_2"] == ""  # b contributes nothing for row 0: not a reason
    assert summary["sufficiency_rate"] == 1.0 and summary["top_reason_sufficiency_rate"] == 1.0
    assert set(long.columns) >= {"factor", "impact", "probability_impact", "account_values",
                                 "reference_values", "is_reason"}


def test_reference_is_marginal_not_flagged_population():
    df = pd.DataFrame({"x": np.arange(100.0), "cat": ["u"] * 50 + ["v"] * 50})
    margin = df["x"].to_numpy() - 89.5  # rows 90..99 flagged
    ref = reference_profile(df, margin, margin >= 0, ["x", "cat"], ("cat",))
    assert ref["n"] == 18  # 20% of 90 not-flagged rows
    assert ref["values"]["x"] == pytest.approx(80.5)  # median of rows 72..89
    assert ref["values"]["cat"] == "v"


def test_factor_groups_rejects_overlap():
    with pytest.raises(ValueError, match="more than one"):
        factor_groups(["a", "b"], {"one": ("a",), "two": ("a", "b")})


# ---------------------------------------------------------------- pipeline integration
def test_calibration_never_changes_threshold_flags(toy, toy_spec, fast):
    base = run(toy, toy_spec, fast)
    for method in ("none", "sigmoid", "isotonic"):
        res = run(toy, toy_spec, Config(**{**fast.__dict__, "calibration": method}))
        assert res.calibration["method"] == method
        assert np.array_equal(res.samples["flag"], base.samples["flag"])
        assert res.decision["threshold"] == base.decision["threshold"]


def test_reasons_in_run_and_reloaded_model_agree(toy, toy_spec, fast, tmp_path):
    res = run(toy, toy_spec, fast)
    res.save(tmp_path / "r")
    m = load_model(tmp_path / "r" / "model")
    holdout = toy.iloc[res.splits["test"]]
    wide, _, summary = m.reasons(holdout, id_column="id")
    assert summary["accounts_explained"] == int(res.decision["test"]["flagged"])
    mine = res.reasons.set_index("id").sort_index()
    theirs = wide.set_index("id").sort_index()
    assert (mine["reason_1"] == theirs["reason_1"]).all()
    assert np.allclose(mine["impact_1"], theirs["impact_1"])
    assert np.allclose(m.probability(holdout), res.samples["probability"].to_numpy()[
        res.splits["test"]])


def test_tuning_adds_gradient_boosting_and_reports_cv(toy, toy_spec, fast, tmp_path):
    res = run(toy, toy_spec, Config(**{**fast.__dict__, "tune": True, "tune_iter": 2,
                                        "decision_model": "gradient_boosting"}))
    assert "gradient_boosting" in res.model_names
    assert set(res.tuning["model"]) == {"logistic_regression", "random_forest",
                                        "gradient_boosting"}
    assert res.tuning["tuned_cv_auc"].between(0, 1).all()
    assert res.decision["model"] == "gradient_boosting"
    res.export_model(tmp_path / "m")  # gradient boosting passes the skops allow-list
    m = load_model(tmp_path / "m")
    holdout = toy.iloc[res.splits["test"]]
    assert np.allclose(m.score(holdout), res.predict_proba(holdout, "gradient_boosting")[:, 1])


def test_tuning_requires_flag_for_gradient_boosting():
    with pytest.raises(ValueError, match="tune=True"):
        Config(decision_model="gradient_boosting")


def _fold_vector(y, g, seed=0):
    from creditdecide.tuning import group_folds

    fold = np.full(len(y), -1)
    for k, (train, test) in enumerate(group_folds(y, g, 5, seed)):
        assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == len(y)
        fold[test] = k
    return fold


def test_group_folds_keep_groups_together_and_stratify():
    rng = np.random.RandomState(1)
    y = (rng.random_sample(4000) < 0.22).astype(int)
    g = rng.randint(0, 3000, 4000)
    fold = _fold_vector(y, g)
    assert (fold >= 0).all()
    assert pd.Series(fold).groupby(g).nunique().max() == 1  # no group spans two folds
    sizes = np.bincount(fold)
    assert sizes.max() - sizes.min() <= 0.05 * len(y) / 5
    rates = pd.Series(y).groupby(fold).mean()
    assert (rates - y.mean()).abs().max() < 0.02
    assert np.array_equal(fold, _fold_vector(y, g))  # deterministic
    assert not np.array_equal(fold, _fold_vector(y, g, 1))  # and seeded


def test_tuning_folds_and_candidates_do_not_depend_on_library_versions():
    """Pinned outputs: the same folds and candidates on every NumPy and scikit-learn version
    (checked under scikit-learn 1.7.2 and 1.9.1, NumPy 2.2.6 and 2.5.4). A change here changes
    the tuned run, so it must be deliberate and the documented tuned figures rerun."""
    import hashlib

    from creditdecide.tuning import sample_candidates, search_spaces

    rng = np.random.RandomState(0)
    y, g = rng.randint(0, 2, 3000), rng.randint(0, 2400, 3000)
    fold = _fold_vector(y, g)
    assert hashlib.sha256(fold.astype(np.int64).tobytes()).hexdigest()[:16] == "0fe5703e643505ae"
    c = [sample_candidates(space, 12, 0) for _, space in search_spaces(0, 200).values()]
    assert hashlib.sha256(repr(c).encode()).hexdigest()[:16] == "80dc186a4f0715e8"
    assert [len(x) for x in c] == [6, 12, 12]  # logistic regression: the whole grid of 6


def test_expected_loss_rule_in_run(toy, toy_spec, fast):
    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=1.0, lgd=0.5, exposure_column="x2")
    res = run(toy, toy_spec, Config(**{**fast.__dict__, "decision_rule": "expected_loss"}), pol)
    assert res.decision["rule"] == "expected_loss" and res.decision["threshold"] is None
    p = res.samples["probability"].to_numpy()
    missed = 0.5 * np.clip(toy["x2"].to_numpy(), 0, None)
    assert np.array_equal(res.samples["flag"].to_numpy(),
                          (p * missed > 1.0 * (1 - p)).astype(int))
    assert set(res.sensitivity["rule"]) == {"threshold", "expected_loss"}


def test_loads_format_1_artifact(toy, toy_spec, fast, tmp_path):
    """A format-1 manifest (no calibration, rule or reasons) still scores."""
    res = run(toy, toy_spec, fast)
    res.export_model(tmp_path / "m")
    old = tmp_path / "old"
    shutil.copytree(tmp_path / "m", old)
    manifest = json.loads((old / "manifest.json").read_text())
    for key in ("calibration", "reasons", "decision_inputs"):
        manifest.pop(key)
    manifest["decision"].pop("rule")
    manifest["format_version"] = 1
    (old / "manifest.json").write_text(json.dumps(manifest))
    m = load_model(old)
    holdout = toy.iloc[res.splits["test"]]
    out = m.decide(holdout)
    assert np.array_equal(out["flag"], (out["score"] >= res.decision["threshold"]).astype(int))
    assert np.allclose(out["probability"], out["score"])  # uncalibrated
    with pytest.raises(ValueError, match="no reason-code profile"):
        m.reasons(holdout)


# ---------------------------------------------------------------- simple-rule benchmark
def test_simple_rule_benchmark_finds_the_informative_feature():
    from creditdecide.decision import simple_rule_benchmark

    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({
        "late": np.where(y == 1, rng.integers(1, 4, n), rng.integers(-1, 2, n)),  # informative
        "noise": rng.normal(size=n),
        "neg": -np.where(y == 1, rng.normal(1, 1, n), rng.normal(0, 1, n)),  # lower = riskier
    })
    val, test = np.arange(0, 2000), np.arange(2000, 4000)
    score = df["late"].to_numpy() + rng.normal(0, 0.01, n)  # model = the rule, plus jitter
    b = simple_rule_benchmark(df, ["late", "noise", "neg"], y, score, val, test, 0.15)
    assert b["column"] == "late" and b["direction"] == ">="
    assert abs(b["validation_flagged_fraction"] - 0.15) <= 0.075
    assert abs(b["capture_gain_of_model"]) < 0.05  # a model that copies the rule gains ~0
    assert b["test_rule"]["flagged_fraction"] == b["test_model_same_volume"]["flagged_fraction"]


def test_other_random_draws_are_pinned():
    """The main split and the calibration folds come from scikit-learn, the bootstrap and the
    reason-code check from NumPy's Generator. They agree today on scikit-learn 1.7.2 and 1.9.1
    and NumPy 2.2.6 and 2.5.4; pinning them here makes CI, which runs on three Python versions,
    fail if a library release ever changes them, instead of the results moving silently."""
    import hashlib

    from sklearn.model_selection import StratifiedKFold

    from creditdecide.bootstrap import resample_weights
    from creditdecide.pipeline import group_split

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]

    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.randint(0, 40, 3000), "b": rng.randint(0, 40, 3000)})
    y = (rng.random_sample(3000) < 0.25).astype(int)
    parts = group_split(df, ["a", "b"], y, Config())
    split = np.zeros(3000, int)
    split[parts["val"]], split[parts["test"]] = 1, 2
    assert h(split) == "7ff8050cb1d65f8c"
    fold = np.zeros(1000, int)  # as creditdecide.calibration.choose_and_fit draws its folds
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (_, test) in enumerate(cv.split(np.zeros(1000), y[:1000])):
        fold[test] = k
    assert h(fold) == "71ddd458c4e68495"
    weights = np.vstack(list(resample_weights(rng.randint(0, 800, 1000), 3, seed=0)))
    assert h(weights) == "321213b8a96bd02f"
    assert h(np.random.default_rng(0).integers(0, 7, 1000)) == "ec6e9edc7059cb0b"  # reasons

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from creditdecide import DecisionPolicy, run
from creditdecide import bootstrap as bs
from creditdecide.decision import summarize
from creditdecide.pipeline import duplicate_groups


def test_weighted_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = np.round(rng.normal(y * 0.7, 1.0), 1)  # rounding creates many ties
    w = rng.integers(0, 4, 400).astype(float)
    assert bs.weighted_roc_auc(y, s, w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert bs.weighted_roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_weights_resample_whole_groups_and_are_reproducible():
    groups = np.array([0, 0, 1, 2, 2, 2, 3])
    a = list(bs.resample_weights(groups, 50, seed=3))
    b = list(bs.resample_weights(groups, 50, seed=3))
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    for w in a:
        assert w[0] == w[1] and w[3] == w[4] == w[5]  # a group is drawn as a unit
        # four groups drawn four times in total
        assert w[0] + w[2] + w[3] + w[6] == 4


def test_interval_width_matches_binomial_theory():
    """Known answer: the bootstrap standard error of a capture rate is close to
    sqrt(c (1 - c) / positives)."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.random(n) < 0.25
    flag = np.where(y, rng.random(n) < 0.4, rng.random(n) < 0.05)
    t = bs.bootstrap(lambda w: {"c": bs.capture_rate(w, y, flag)}, np.arange(n), n_boot=2000)
    c = flag[y].mean()
    theory = np.sqrt(c * (1 - c) / y.sum())
    row = t.iloc[0]
    assert row["estimate"] == pytest.approx(c)
    assert row["std_error"] == pytest.approx(theory, rel=0.1)
    assert row["ci_low"] < c < row["ci_high"]


def test_saving_matches_decision_summary():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    flag = rng.random(300) < 0.3
    missed = rng.uniform(0, 50, 300)
    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=5.0, lgd=1.0, exposure_column="e")
    ref = summarize(y, np.zeros(300), None, pol, missed, flag=flag)
    got = bs.saving_vs_best_trivial(np.ones(300), y, flag, 5.0, missed)
    assert got == pytest.approx(ref["saving_fraction_vs_best_trivial"])


def test_run_intervals_reproduce_the_point_figures(toy, toy_spec, fast):
    res = run(toy, toy_spec, fast)
    b = res.bootstrap.set_index("quantity")
    t, bench = res.decision["test"], res.decision["simple_rule_benchmark"]
    assert b.loc["capture rate", "estimate"] == pytest.approx(t["capture_rate"])
    assert b.loc["precision", "estimate"] == pytest.approx(t["precision"])
    assert b.loc["lift", "estimate"] == pytest.approx(t["lift"])
    s = res.summary("test").set_index("model")
    assert b.loc["ROC AUC random_forest", "estimate"] == pytest.approx(
        s.loc["random_forest", "roc_auc"])
    gain = [q for q in b.index if q.startswith("capture rate, model minus rule")]
    assert b.loc[gain[0], "estimate"] == pytest.approx(bench["capture_gain_of_model"])
    c = res.calibration["test_calibrated"]
    assert b.loc["defaults forecast error, all accounts (expected / actual - 1)",
                 "estimate"] == pytest.approx(c["expected_positives"] / c["actual_positives"] - 1)
    assert (b.ci_low <= b.estimate + 1e-12).all() and (b.estimate <= b.ci_high + 1e-12).all()
    assert set(b.loc[b["type"] == "difference", "excludes_zero"]) <= {True, False}
    again = run(toy, toy_spec, fast).bootstrap
    pd.testing.assert_frame_equal(res.bootstrap, again)  # same seed, same intervals


def test_cost_run_compares_both_rules(toy, toy_spec, fast):
    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=1.0, lgd=0.5, exposure_column="x1")
    a = run(toy, toy_spec, fast, pol).bootstrap.set_index("quantity")
    el = run(toy, toy_spec, replace(fast, decision_rule="expected_loss"), pol)
    e = el.bootstrap.set_index("quantity")
    for q in ("saving vs best trivial, one cut-off", "saving vs best trivial, per-account rule",
              "saving, per-account rule minus one cut-off"):
        assert a.loc[q, "estimate"] == pytest.approx(e.loc[q, "estimate"])
        assert a.loc[q, "ci_low"] == pytest.approx(e.loc[q, "ci_low"])
    assert e.loc["saving vs best trivial, per-account rule", "estimate"] == pytest.approx(
        el.decision["test"]["saving_fraction_vs_best_trivial"])


def test_groups_and_switch_off(toy, toy_spec, fast, tmp_path):
    res = run(toy, toy_spec, fast)
    te = res.splits["test"]
    g = duplicate_groups(res.data, res.features)[te]
    assert len(np.unique(g)) <= len(te)
    res.save(tmp_path / "on")
    assert (tmp_path / "on" / "bootstrap_test.csv").exists()
    meta = json.loads((tmp_path / "on" / "run.json").read_text())
    assert meta["bootstrap"]["replicates"] == fast.bootstrap
    assert "Uncertainty" in (tmp_path / "on" / "model" / "MODEL_CARD.md").read_text()
    off = run(toy, toy_spec, replace(fast, bootstrap=0))
    assert off.bootstrap is None
    off.save(tmp_path / "off")
    assert not (tmp_path / "off" / "bootstrap_test.csv").exists()
    pd.testing.assert_frame_equal(off.metrics, res.metrics)  # intervals change nothing else
    assert off.decision == res.decision

import numpy as np
import pytest

from creditdecide.decision import (
    DecisionPolicy,
    choose_threshold,
    cost_sensitivity,
    expected_cost,
    gains_table,
    summarize,
)


def test_policy_validation():
    with pytest.raises(ValueError):
        DecisionPolicy(budget=1.2)
    with pytest.raises(ValueError):
        DecisionPolicy(kind="cost", budget=None)                      # no cost_fp
    with pytest.raises(ValueError):
        DecisionPolicy(kind="cost", budget=None, cost_fp=1)           # no missed cost
    with pytest.raises(ValueError):
        DecisionPolicy(kind="cost", budget=None, cost_fp=1, cost_fn=5, lgd=0.5,
                       exposure_column="e")                           # both kinds given
    with pytest.raises(ValueError):
        DecisionPolicy(kind="cost", budget=None, cost_fp=1, lgd=1.5, exposure_column="e")
    with pytest.raises(ValueError):
        DecisionPolicy(kind="other")


def test_cost_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 40))
        y = rng.integers(0, 2, n)
        s = np.round(rng.random(n), 1)  # many ties
        miss = rng.random(n) * 10
        cfp = float(rng.random() * 3)
        pol = DecisionPolicy(kind="cost", budget=None, cost_fp=cfp, lgd=0.5,
                             exposure_column="e")
        t = choose_threshold(y, s, pol, miss)
        cands = np.concatenate([np.unique(s), [np.inf]])  # brute force: every possible cut
        brute = min(expected_cost(y, s >= c, cfp, miss) for c in cands)
        assert np.isclose(expected_cost(y, s >= t, cfp, miss), brute)


def test_cost_threshold_extremes():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    free_alarms = DecisionPolicy(kind="cost", budget=None, cost_fp=0.0, cost_fn=1.0)
    assert (s >= choose_threshold(y, s, free_alarms)).sum() >= 2
    costly_alarms = DecisionPolicy(kind="cost", budget=None, cost_fp=1e9, cost_fn=1.0)
    # the two positives outrank every negative, so flagging exactly them costs nothing
    assert choose_threshold(y, s, costly_alarms) == 0.8
    # when a negative outranks the positive, expensive alarms mean flag nobody
    t = choose_threshold(np.array([1, 0]), np.array([0.1, 0.9]), costly_alarms)
    assert 1.0 < t < 1.0 + 1e-9 and np.isfinite(t)


def test_budget_threshold():
    s = np.linspace(0, 1, 1000)
    t = choose_threshold(np.zeros(1000, int), s, DecisionPolicy(budget=0.1))
    assert (s >= t).mean() == pytest.approx(0.1, abs=0.002)


def test_summarize_hand_example():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=1.0, cost_fn=10.0)
    r = summarize(y, s, 0.5, pol, np.full(5, 10.0))
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 2)
    assert r["capture_rate"] == 0.5 and r["precision"] == 0.5
    assert r["lift"] == pytest.approx(0.5 / 0.4)
    assert r["expected_cost"] == 11.0
    assert r["cost_flag_nobody"] == 20.0 and r["cost_flag_everybody"] == 3.0
    assert r["saving_fraction_vs_best_trivial"] == pytest.approx((3 - 11) / 3)


def test_missed_cost_from_exposure():
    import pandas as pd

    pol = DecisionPolicy(kind="cost", budget=None, cost_fp=1, lgd=0.5, exposure_column="bal")
    m = pol.missed_cost(pd.DataFrame({"bal": [100.0, -50.0]}), 2)
    assert m.tolist() == [50.0, 0.0]  # negative balances (credits) carry no loss


def test_gains_table():
    y = np.array([1] * 10 + [0] * 90)
    s = np.concatenate([np.linspace(1, 0.9, 10), np.linspace(0.5, 0, 90)])
    g = gains_table(y, s)
    assert g.loc[0, "cumulative_capture"] == 1.0 and g.loc[0, "cumulative_lift"] == 10.0
    assert g["cumulative_capture"].iloc[-1] == 1.0


def test_cost_sensitivity_shape():
    rng = np.random.default_rng(1)
    y, s, e = rng.integers(0, 2, 200), rng.random(200), rng.random(200) * 100
    t = cost_sensitivity(y, s, y, s, e, e, cost_fps=[1, 10], lgds=[0.3, 0.6])
    assert len(t) == 4 and t["flagged_fraction"].between(0, 1).all()

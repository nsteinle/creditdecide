import numpy as np
import pandas as pd
import pytest

from creditdecide import Config, DecisionPolicy, load_model, run
from creditdecide.fairness import group_labels, outcome_audit, proxy_auc
from creditdecide.monitor import build_reference, drift_report, psi, status


# ---------------------------------------------------------------- PSI
def test_psi_hand_example():
    e, a = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert psi(e, a) == pytest.approx(expected)
    assert psi(e, e) == 0.0


def test_status_bands():
    assert status(0.05) == "stable"
    assert status(0.1) == "moderate shift" and status(0.25) == "moderate shift"
    assert status(0.26) == "significant shift"


def test_drift_detects_shift_and_ignores_resample():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "c": rng.choice(["a", "b"], 5000),
                        "d": rng.integers(0, 4, 5000)})
    profile = build_reference(ref, ["x", "c", "d"], ("c",), rng.random(5000), "ref")
    same = pd.DataFrame({"x": rng.normal(size=5000), "c": rng.choice(["a", "b"], 5000),
                         "d": rng.integers(0, 4, 5000)})
    rep = drift_report(profile, same, rng.random(5000)).set_index("variable")
    assert (rep["psi"] < 0.02).all()
    moved = same.assign(x=same.x + 1.0, c="z", d=9)
    rep = drift_report(profile, moved, rng.random(5000)).set_index("variable")
    assert rep.loc["x", "status"] == "significant shift"
    assert rep.loc["c", "status"] == "significant shift"  # unseen category bin
    assert rep.loc["d", "status"] == "significant shift"  # unseen discrete value
    assert profile["features"]["d"]["kind"] == "discrete"
    assert profile["features"]["x"]["kind"] == "binned"


# ---------------------------------------------------------------- fairness
def test_group_labels():
    bands = group_labels(pd.Series([20, 25, 61, 62]), bins=(25, 62))
    assert bands.tolist() == ["< 25", "25-61", "25-61", "62+"]
    named = group_labels(pd.Series([1, 2, 7]), labels={1: "a", 2: "b"})
    assert named.tolist() == ["a", "b", "code 7"]


def test_outcome_audit_hand_example():
    groups = pd.Series(["a"] * 4 + ["b"] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    flag = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    t = outcome_audit(groups, y, flag).set_index("group")
    assert t.loc["a", "flag_rate"] == 0.5 and t.loc["b", "flag_rate"] == 0.25
    assert t.loc["a", "adverse_impact_ratio"] == pytest.approx(0.5 / 0.75)
    assert t.loc["b", "adverse_impact_ratio"] == 1.0
    assert bool(t.loc["a", "below_four_fifths"]) and not bool(t.loc["b", "below_four_fifths"])
    assert t.loc["a", "false_alarm_rate"] == pytest.approx(1 / 3)
    assert t.loc["b", "capture_rate"] == 0.5
    assert t["too_small"].all()


def test_proxy_auc_detects_leaky_feature():
    rng = np.random.default_rng(0)
    g = pd.Series(rng.choice(["x", "y"], 2000))
    X_leaky = np.column_stack([(g == "x").to_numpy() + rng.normal(0, 0.3, 2000),
                               rng.normal(size=2000)])
    X_clean = rng.normal(size=(2000, 2))
    leaky = proxy_auc(X_leaky[:1000], g[:1000], X_leaky[1000:], g[1000:].reset_index(drop=True))
    clean = proxy_auc(X_clean[:1000], g[:1000], X_clean[1000:], g[1000:].reset_index(drop=True))
    assert leaky["x"] > 0.9 and abs(clean["x"] - 0.5) < 0.07


# ---------------------------------------------------------------- integration
def test_run_produces_audit_reference_and_card(toy, toy_spec, fast, tmp_path):
    from dataclasses import replace

    spec = replace(toy_spec, group_bins={"age": (30, 50)})
    res = run(toy, spec, fast, DecisionPolicy(budget=0.2))
    assert set(res.fairness["attribute"]) == {"age"}
    assert set(res.fairness["group"]) == {"< 30", "30-49", "50+"}
    assert res.fairness["n"].sum() == len(res.splits["test"])
    res.save(tmp_path / "r")
    m = load_model(tmp_path / "r" / "model")
    assert m.reference["sample"] == "validation split"
    assert set(m.reference["features"]) == set(res.features)
    card = (tmp_path / "r" / "model" / "MODEL_CARD.md").read_text()
    for heading in ("## Purpose", "## Decision outcome", "## Outcomes by protected group",
                    "## Monitoring", "SR 11-7"):
        assert heading in card
    rep = drift_report(m.reference, toy, m.score(toy))
    assert len(rep) == len(res.features) + 1


def test_include_protected_still_audits(toy, toy_spec, fast):
    res = run(toy, toy_spec, Config(**{**fast.__dict__, "include_protected": True}))
    assert "age" in res.features and set(res.fairness["attribute"]) == {"age"}

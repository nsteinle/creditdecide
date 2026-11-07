import numpy as np
import pandas as pd
import pytest

from creditdecide import CREDIT_DEFAULT_SPEC, Config, load_credit_default, run
from creditdecide.cli import main


def test_datasets(capsys):
    main(["datasets"])
    out = capsys.readouterr().out
    assert "CC BY 4.0" in out and "10.24432/C55S3H" in out


def test_quality_credit(capsys):
    main(["quality", "--dataset", "credit-default"])
    assert "conflicting labels" in capsys.readouterr().out


def _csv(toy, tmp_path):
    p = tmp_path / "toy.csv"
    toy.to_csv(p, index=False)
    return str(p)


CSV_ARGS = ["--target", "label", "--categorical", "region", "--signed-log", "amount",
            "--drop", "id", "--protected", "age"]
FAST = ["--bootstrap", "200"]


def test_run_predict_drift_workflow(toy, tmp_path, capsys):
    out = tmp_path / "r"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST, "--plot"])
    for f in ("roc_pr.png", "calibration.png", "decision.json",
              "gains_test.csv", "fairness_test.csv", "holdout_test.csv", "calibration_cv.csv",
              "calibration_test.csv", "reasons_test.csv", "reasons_test_detail.csv",
              "model/manifest.json", "model/MODEL_CARD.md", "model/reference.json"):
        assert (out / f).exists(), f
    with pytest.raises(SystemExit):  # refuses to overwrite
        main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST])

    holdout = out / "holdout_test.csv"
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(holdout),
          "--out", str(tmp_path / "p"), "--id-column", "id", "--reasons", "2"])
    scores = pd.read_csv(tmp_path / "p" / "scores.csv")
    assert list(scores.columns) == ["id", "score", "probability", "flag"]
    reasons = pd.read_csv(tmp_path / "p" / "reasons.csv")
    assert len(reasons) == scores["flag"].sum()
    assert set(reasons["id"]) == set(scores.loc[scores.flag == 1, "id"])
    assert {"reason_1", "reason_2"} <= set(reasons.columns)
    assert (tmp_path / "p" / "drift.csv").exists()
    assert (tmp_path / "p" / "outcomes.json").exists()  # labels present -> outcomes analysis
    main(["drift", "--model-dir", str(out / "model"), "--csv", str(holdout),
          "--out", str(tmp_path / "d.csv")])
    assert "drift vs reference" in capsys.readouterr().out


def test_cost_policy_cli(toy, tmp_path):
    out = tmp_path / "c"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST,
          "--cost-fp", "1", "--lgd", "0.5", "--exposure-column", "x1"])
    assert (out / "cost_sensitivity.csv").exists()
    with pytest.raises(SystemExit):
        main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(tmp_path / "x"),
              *FAST, "--budget", "0.1", "--cost-fp", "1", "--cost-fn", "2"])


def test_expected_loss_and_tuning_cli(toy, tmp_path):
    out = tmp_path / "e"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST,
          "--cost-fp", "1", "--cost-fn", "4", "--decision-rule", "expected_loss",
          "--tune", "--tune-iter", "2"])
    assert (out / "tuning.csv").exists() and (out / "tuning_details.csv").exists()
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(out / "holdout_test.csv"),
          "--out", str(tmp_path / "ep")])
    scored = pd.read_csv(tmp_path / "ep" / "scores.csv")
    expected = (scored.probability * 4 > 1 * (1 - scored.probability)).astype(int)
    assert (scored.flag == expected).all()
    with pytest.raises(ValueError, match="cost policy"):
        main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(tmp_path / "x"),
              *FAST, "--decision-rule", "expected_loss"])


def test_csv_requires_target(toy, tmp_path):
    with pytest.raises(SystemExit):
        main(["quality", "--csv", _csv(toy, tmp_path)])


def test_outcomes_join_labels_that_arrive_later(toy, tmp_path, capsys):
    """Scores written without labels, labels delivered later in another file: the joined
    outcome must equal the outcome computed when the labels were in the scoring file."""
    out = tmp_path / "r"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST])
    holdout = pd.read_csv(out / "holdout_test.csv")
    holdout.drop(columns=["label"]).to_csv(tmp_path / "new.csv", index=False)
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(tmp_path / "new.csv"),
          "--out", str(tmp_path / "p"), "--id-column", "id"])
    assert not (tmp_path / "p" / "outcomes.json").exists()
    # labels arrive in a different order, one is missing and one belongs to another file
    labels = holdout[["id", "label"]].sample(frac=1, random_state=0).iloc[1:]
    extra = pd.DataFrame({"id": [10**6], "label": [1]})
    pd.concat([labels, extra]).to_csv(tmp_path / "labels.csv", index=False)
    main(["outcomes", "--model-dir", str(out / "model"), "--scores",
          str(tmp_path / "p" / "scores.csv"), "--labels", str(tmp_path / "labels.csv"),
          "--id-column", "id", "--out", str(tmp_path / "o.json")])
    text = capsys.readouterr().out
    assert "1 scored rows without a label, 1 labels without a score" in text
    import json

    joined = json.loads((tmp_path / "o.json").read_text())["decision"]
    # the same join done in pandas, scored with the summary used at training time
    from creditdecide import load_model
    from creditdecide.decision import summarize

    scores = pd.read_csv(tmp_path / "p" / "scores.csv")
    merged = scores.merge(labels, on="id", how="inner")
    want = summarize(merged.label.to_numpy(), merged.score.to_numpy(),
                     load_model(out / "model").threshold, flag=merged.flag.to_numpy())
    for k, v in want.items():
        assert joined[k] == pytest.approx(v), k
    assert joined["n"] == len(holdout) - 1
    # labels inside the scoring file give the same numbers as the full join
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(out / "holdout_test.csv"),
          "--out", str(tmp_path / "q"), "--id-column", "id"])
    inline = json.loads((tmp_path / "q" / "outcomes.json").read_text())["decision"]
    full = scores.merge(holdout[["id", "label"]], on="id")
    want = summarize(full.label.to_numpy(), full.score.to_numpy(),
                     load_model(out / "model").threshold, flag=full.flag.to_numpy())
    for k, v in want.items():
        assert inline[k] == pytest.approx(v), k
    with pytest.raises(SystemExit):  # nothing matches
        pd.DataFrame({"id": [-1], "label": [0]}).to_csv(tmp_path / "none.csv", index=False)
        main(["outcomes", "--model-dir", str(out / "model"), "--scores",
              str(tmp_path / "p" / "scores.csv"), "--labels", str(tmp_path / "none.csv"),
              "--id-column", "id"])


def test_outcomes_from_a_labels_file_equal_inline_outcomes_exactly(toy, tmp_path):
    """Scores written to scores.csv and read back must give the same outcomes, to the last
    digit, as scoring with the labels in the file (exact CSV parsing on pandas 2 and 3)."""
    import json

    from creditdecide.datasets import read_csv

    out = tmp_path / "r"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST])
    holdout = read_csv(out / "holdout_test.csv")  # exact, so new.csv has the same inputs
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(out / "holdout_test.csv"),
          "--out", str(tmp_path / "inline"), "--id-column", "id"])
    holdout.drop(columns=["label"]).to_csv(tmp_path / "new.csv", index=False)
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(tmp_path / "new.csv"),
          "--out", str(tmp_path / "later"), "--id-column", "id"])
    holdout[["id", "label"]].sample(frac=1, random_state=1).to_csv(tmp_path / "labels.csv",
                                                                  index=False)
    main(["outcomes", "--model-dir", str(out / "model"), "--scores",
          str(tmp_path / "later" / "scores.csv"), "--labels", str(tmp_path / "labels.csv"),
          "--id-column", "id", "--out", str(tmp_path / "o.json")])
    inline = json.loads((tmp_path / "inline" / "outcomes.json").read_text())
    assert json.loads((tmp_path / "o.json").read_text()) == inline


def test_read_csv_is_exact(tmp_path):
    from creditdecide.datasets import read_csv

    x = np.random.default_rng(0).random(20_000)
    pd.DataFrame({"x": x}).to_csv(tmp_path / "x.csv", index=False)
    assert np.array_equal(read_csv(tmp_path / "x.csv").x.to_numpy(), x)


def test_tableau_export(toy, tmp_path, capsys):
    import json

    out = tmp_path / "r"
    main(["run", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--out", str(out), *FAST])
    main(["predict", "--model-dir", str(out / "model"), "--csv", str(out / "holdout_test.csv"),
          "--out", str(tmp_path / "p"), "--id-column", "id"])
    tab = tmp_path / "t"
    main(["tableau", "--run-dir", str(out), "--scored", str(tmp_path / "p"), "--out", str(tab)])
    index = json.loads((tab / "tables.json").read_text())
    assert set(index["tables"]) == {"accounts.csv", "decision.csv", "gains.csv",
                                    "calibration.csv", "models.csv", "intervals.csv",
                                    "reasons.csv", "fairness.csv", "drift.csv"}
    for name, info in index["tables"].items():
        frame = pd.read_csv(tab / name)
        assert list(frame.columns) == info["columns"] and len(frame) == info["rows"], name
    acc = pd.read_csv(tab / "accounts.csv")
    holdout = pd.read_csv(out / "holdout_test.csv")
    assert sorted(acc["id"]) == sorted(holdout["id"])  # the test split, each account once
    assert {"score", "probability", "flag", "decision", "score_decile", "top_reason",
            "age_group"} <= set(acc.columns)
    gains = pd.read_csv(out / "gains_test.csv")
    by_decile = acc.groupby("score_decile").agg(accounts=("id", "size"),
                                                positives=("actual", "sum"))
    assert by_decile["accounts"].tolist() == gains["accounts"].tolist()
    assert by_decile["positives"].tolist() == gains["positives"].tolist()
    reasons = pd.read_csv(out / "reasons_test.csv")
    flagged = acc[acc.flag == 1].set_index("id")
    assert flagged["top_reason"].notna().all() and acc[acc.flag == 0].top_reason.isna().all()
    assert (flagged.loc[reasons["id"], "top_reason"].to_numpy() == reasons.reason_1).all()
    assert len(acc[acc.flag == 1]) == len(reasons)
    assert pd.read_csv(tab / "gains.csv")["share_reviewed"].iloc[-1] == pytest.approx(1.0)
    models = pd.read_csv(tab / "models.csv")
    m = pd.read_csv(out / "metrics_test.csv").set_index("model")
    for _, r in models.iterrows():
        assert r["value"] == pytest.approx(m.loc[r["model"], r["metric"]])
    with pytest.raises(SystemExit):  # refuses to overwrite
        main(["tableau", "--run-dir", str(out), "--out", str(tab)])
    with pytest.raises(SystemExit):  # not a run folder
        main(["tableau", "--run-dir", str(tmp_path / "p"), "--out", str(tmp_path / "t2")])


def test_sql_command(toy, tmp_path, capsys):
    q = "SELECT region, COUNT(*) AS n, AVG(label) AS rate FROM data GROUP BY ALL ORDER BY ALL"
    main(["sql", "--csv", _csv(toy, tmp_path), *CSV_ARGS, "--query", q,
          "--out", str(tmp_path / "q.csv")])
    got = pd.read_csv(tmp_path / "q.csv")
    want = toy.groupby("region").agg(n=("label", "size"), rate=("label", "mean")).reset_index()
    pd.testing.assert_frame_equal(got, want)
    assert "(3 rows" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="query failed"):  # files are out of reach
        main(["sql", "--dataset", "credit-default", "--query",
              f"SELECT * FROM read_csv('{tmp_path / 'q.csv'}')"])


@pytest.mark.slow
def test_credit_default_full_run():
    """Full dataset, default settings (~30 s CPU): the figures quoted in the README."""
    res = run(load_credit_default(), CREDIT_DEFAULT_SPEC, Config(seed=0))
    s = res.summary("test").set_index("model")
    assert s.loc["majority_baseline", "accuracy"] == pytest.approx(0.778, abs=0.002)
    assert 0.76 <= s.loc["random_forest", "roc_auc"] <= 0.80
    assert s.loc["logistic_regression", "roc_auc"] < s.loc["random_forest", "roc_auc"]
    t = res.decision["test"]
    assert res.decision["model"] == "random_forest"
    assert t["flagged_fraction"] == pytest.approx(0.10, abs=0.01)
    assert 0.28 <= t["capture_rate"] <= 0.35 and t["lift"] > 2.5
    assert res.proxies["AGE"]["< 25"] > 0.75  # age information survives its exclusion
    c = res.calibration  # the random forest is already close to calibrated on this data
    assert c["method"] == "none" and c["test_calibrated"]["ece"] < 0.03
    assert abs(c["test_calibrated"]["expected_positives"] - 1332) < 60
    r = res.decision["reasons_test"]
    assert r["accounts_explained"] == t["flagged"]
    assert r["sufficiency_rate"] > 0.95 and r["top_reason_sufficiency_rate"] > 0.5
    assert max(r["primary_reason_share"], key=r["primary_reason_share"].get) == (
        "Repayment status last month")
    b = res.decision["simple_rule_benchmark"]  # a one-line rule matches the model at 10%
    assert b["rule"] == "PAY_0 >= 2" and abs(b["capture_gain_of_model"]) < 0.02
    ci = res.bootstrap.set_index("quantity")  # 95% bootstrap intervals on the test split
    gain = ci.loc["capture rate, model minus rule PAY_0 >= 2 at the rule's volume"]
    assert gain.ci_low < 0 < gain.ci_high  # the model and the rule cannot be told apart
    assert ci.loc["ROC AUC random_forest minus logistic_regression", "ci_low"] > 0
    assert 0.27 < ci.loc["capture rate", "ci_low"] < t["capture_rate"] < ci.loc[
        "capture rate", "ci_high"] < 0.36

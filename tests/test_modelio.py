import json

import numpy as np
import pytest

from creditdecide import Config, DecisionPolicy, load_model, run


@pytest.fixture(scope="module")
def trained(request, tmp_path_factory):
    toy = request.getfixturevalue("toy_module")
    spec = request.getfixturevalue("toy_spec_module")
    cfg = Config(rf_n_estimators=20)
    res = run(toy, spec, cfg, DecisionPolicy(budget=0.2))
    out = res.save(tmp_path_factory.mktemp("run"))
    return res, out / "model", toy


def test_artifact_contents(trained):
    _, model_dir, _ = trained
    manifest = json.loads((model_dir / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert set(manifest["files"]) >= {"pipeline.skops"}
    assert "age" in manifest["excluded_columns"]  # protected column never a feature
    assert "age" not in manifest["input_schema"]["columns"]
    assert manifest["input_schema"]["categorical"]["region"] == ["a", "b", "c"]


def test_reload_reproduces_scores_and_decisions(trained):
    res, model_dir, toy = trained
    m = load_model(model_dir)
    out = m.decide(toy, id_column="id")
    direct = res.predict_proba(toy, model=res.decision["model"])[:, 1]
    assert np.allclose(out["score"], direct)
    assert np.array_equal(out["flag"], (direct >= res.decision["threshold"]).astype(int))
    assert list(out.columns) == ["id", "score", "probability", "flag"]


def test_validation_errors_and_warnings(trained):
    _, model_dir, toy = trained
    m = load_model(model_dir)
    with pytest.raises(ValueError, match="missing required"):
        m.validate(toy.drop(columns="x1"))
    bad = toy.copy()
    bad.loc[0, "x2"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        m.validate(bad)
    bad = toy.copy()
    bad["x1"] = bad["x1"].astype(str)
    with pytest.raises(ValueError, match="non-numeric"):
        m.validate(bad)
    new = toy.copy()
    new.loc[:2, "region"] = "zzz"
    assert any("not seen in training" in w for w in m.validate(new))
    assert m.validate(toy.drop(columns=["age", "label", "id"])) == []  # extras not needed


def test_tampering_is_detected(trained, tmp_path):
    import shutil

    _, model_dir, _ = trained
    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    p = copy / "pipeline.skops"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 1
    p.write_bytes(b)
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(copy)


def test_logistic_regression_export(toy, toy_spec, tmp_path):
    res = run(toy, toy_spec, Config(rf_n_estimators=20, decision_model="logistic_regression"))
    res.export_model(tmp_path / "m")
    m = load_model(tmp_path / "m")
    assert np.allclose(m.score(toy), res.predict_proba(toy, "logistic_regression")[:, 1])


def test_unexpected_artifact_file_is_refused(trained, tmp_path):
    """A model directory listing a file this version does not know (for example weights
    for a model type it cannot rebuild) must not load and silently score without it."""
    import hashlib
    import shutil

    _, model_dir, _ = trained
    copy = tmp_path / "m"
    shutil.copytree(model_dir, copy)
    (copy / "weights.bin").write_bytes(b"0" * 16)
    manifest = json.loads((copy / "manifest.json").read_text())
    manifest["files"]["weights.bin"] = hashlib.sha256(b"0" * 16).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unexpected files"):
        load_model(copy)


def test_positive_label_must_exist(toy, toy_spec):
    from dataclasses import replace

    named = toy.assign(label=np.where(toy.label == 1, "bad", "good"))
    with pytest.raises(ValueError, match="positive label"):
        run(named, toy_spec, Config(rf_n_estimators=20))
    res = run(named, replace(toy_spec, positive_label="bad"),
              Config(rf_n_estimators=20))
    assert res.classes == ["good", "bad"]

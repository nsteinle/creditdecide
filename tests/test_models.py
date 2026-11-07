import numpy as np
import pandas as pd
import pytest

from creditdecide import run
from creditdecide.metrics import evaluate
from creditdecide.pipeline import group_split
from creditdecide.preprocess import make_preprocessor, signed_log1p


# ---------------------------------------------------------------- metrics
def test_evaluate_binary_with_probabilities():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
    m = evaluate(y, proba.argmax(1), proba)
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["brier"] == pytest.approx(0.075)
    assert "roc_auc" not in evaluate(y, np.array([0, 1, 1, 1]))


# ---------------------------------------------------------------- preprocessing
def test_signed_log():
    assert np.allclose(signed_log1p(np.array([-np.e + 1, 0, np.e - 1])), [-1, 0, 1])


def test_preprocessor_unseen_category_is_all_zero(toy, toy_spec):
    pre = make_preprocessor(toy, toy_spec).fit(toy[toy.region != "c"])
    X = pre.transform(toy[toy.region == "c"].head(3))
    names = list(pre.get_feature_names_out())
    cat_cols = [i for i, n in enumerate(names) if n.startswith("region_")]
    assert np.all(X[:, cat_cols] == 0)
    assert "age" not in names and "id" not in names and "label" not in names


# ---------------------------------------------------------------- pipeline
def test_group_split_keeps_identical_rows_together(toy, toy_spec, fast):
    feats = toy_spec.feature_columns(toy)
    y = toy.label.to_numpy()
    s = group_split(toy, feats, y, fast)
    where = {i: name for name, idx in s.items() for i in idx}
    dups = toy.index[toy.duplicated(subset=feats, keep=False)]
    assert len({where[i] for i in dups}) == 1
    assert sorted(np.concatenate(list(s.values()))) == list(range(len(toy)))
    assert len(s["test"]) == pytest.approx(0.2 * len(toy), abs=3)


def test_run_end_to_end(toy, toy_spec, fast):
    res = run(toy, toy_spec, fast)
    assert set(res.metrics.model) == {"majority_baseline", "logistic_regression",
                                      "random_forest"}
    assert set(res.metrics.metric) == {"accuracy", "balanced_accuracy", "f1", "roc_auc",
                                       "average_precision", "brier"}
    test = res.summary("test").set_index("model")
    assert test.loc["majority_baseline", "balanced_accuracy"] == 0.5
    assert test.loc["random_forest", "roc_auc"] > 0.6
    assert "age" not in res.feature_names
    p = res.predict_proba(toy.head(4), model="logistic_regression")
    assert p.shape == (4, 2) and np.allclose(p.sum(1), 1)


def test_preprocessing_ignores_label_values(toy, toy_spec):
    """The fitted transform must be identical whatever the labels are: only the models may
    use labels."""
    shuffled = toy.assign(label=np.random.default_rng(1).permutation(toy.label))
    a = make_preprocessor(toy, toy_spec).fit(toy)
    b = make_preprocessor(shuffled, toy_spec).fit(shuffled)
    assert np.array_equal(a.transform(toy), b.transform(toy))
    assert "label" not in a.get_feature_names_out()


def test_deterministic(toy, toy_spec, fast):
    a, b = run(toy, toy_spec, fast), run(toy, toy_spec, fast)
    pd.testing.assert_frame_equal(a.metrics, b.metrics, check_exact=True)


def test_forests_predict_on_one_thread(toy, toy_spec, fast, tmp_path):
    """Forests are fitted on all cores but predict on one thread, in the run and after
    loading, so that reruns match bit for bit (creditdecide.forest)."""
    from sklearn.ensemble import RandomForestClassifier

    from creditdecide import load_model

    res = run(toy, toy_spec, fast)

    def forests(m):
        steps = [s for _, s in m.steps] if hasattr(m, "steps") else [m]
        return [s for s in steps if isinstance(s, RandomForestClassifier)]

    found = [f for m in res.models.values() for f in forests(m)]
    assert len(found) == 1  # the random forest on all inputs
    assert {f.n_jobs for f in found} == {1}
    # a model saved with n_jobs=-1 (for example by another script) is scored on one thread
    from dataclasses import replace

    old = run(toy, toy_spec, replace(fast, decision_model="random_forest"))
    old.models["random_forest"].set_params(n_jobs=-1)
    old.save(tmp_path / "r")
    loaded = load_model(tmp_path / "r" / "model")
    assert isinstance(loaded.model, RandomForestClassifier) and loaded.model.n_jobs == 1


def test_save_refuses_non_empty_directory(toy, toy_spec, fast, tmp_path):
    res = run(toy, toy_spec, fast)
    out = res.save(tmp_path / "r")
    for f in ("metrics_long.csv", "metrics_test.csv", "samples.csv", "run.json"):
        assert (out / f).exists()
    with pytest.raises(FileExistsError):
        res.save(tmp_path / "r")
    res.save(tmp_path / "r", overwrite=True)


def test_string_labels(toy, toy_spec, fast):
    from dataclasses import replace

    named = toy.assign(label=np.where(toy.label == 1, "yes", "no"))
    res = run(named, replace(toy_spec, positive_label="yes"), fast)
    assert set(res.samples["pred_random_forest"]) <= {"yes", "no"}

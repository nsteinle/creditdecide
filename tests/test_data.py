import json
from importlib import resources

import numpy as np
import pytest

from creditdecide.datasets import (
    CREDIT_DEFAULT_DOCUMENTED_CODES,
    CREDIT_DEFAULT_SPEC,
    load_credit_default,
    load_csv,
    validate_credit_default,
)
from creditdecide.quality import quality_report


@pytest.fixture(scope="module")
def credit():
    return load_credit_default()


def test_bundled_data_matches_metadata(credit):
    meta = json.loads((resources.files("creditdecide.data") / "credit_default.json").read_text())
    assert meta["license"] == "CC BY 4.0" and meta["doi"] == "10.24432/C55S3H"
    assert credit.shape == (30_000, 25)
    assert credit["DEFAULT"].value_counts().to_dict() == {0: 23_364, 1: 6_636}


def test_schema_validation_rejects_tampering(credit):
    with pytest.raises(ValueError):
        validate_credit_default(credit.drop(columns="AGE"))
    with pytest.raises(ValueError):
        validate_credit_default(credit.iloc[:-1])
    bad = credit.copy()
    bad.loc[0, "DEFAULT"] = 2
    with pytest.raises(ValueError):
        validate_credit_default(bad)


def test_quality_report_on_credit(credit):
    r = quality_report(credit, CREDIT_DEFAULT_SPEC, CREDIT_DEFAULT_DOCUMENTED_CODES)
    assert r["missing_values"] == {}
    assert r["duplicate_rows_ignoring_ids"] == 35
    assert r["rows_sharing_features_with_another_row"] == 108
    assert r["feature_groups_with_conflicting_labels"] == 21
    # without sex, marriage and age many more rows look identical to the model
    assert r["rows_sharing_model_inputs_with_another_row"] == 833
    assert r["model_input_groups_with_conflicting_labels"] == 109
    assert r["majority_class_accuracy"] == pytest.approx(0.7788, abs=1e-4)
    assert r["undocumented_codes"]["EDUCATION"] == {"0": 14, "5": 280, "6": 51}
    assert r["undocumented_codes"]["MARRIAGE"] == {"0": 54}


def test_protected_attributes_excluded_by_default(credit):
    cols = CREDIT_DEFAULT_SPEC.feature_columns(credit)
    assert not {"SEX", "MARRIAGE", "AGE", "ID", "DEFAULT"} & set(cols)
    assert {"SEX", "MARRIAGE", "AGE"} <= set(
        CREDIT_DEFAULT_SPEC.feature_columns(credit, include_protected=True))


# ---------------------------------------------------------------- user CSV validation
def _write(df, tmp_path):
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    return p


def test_load_csv_ok(toy, toy_spec, tmp_path):
    assert len(load_csv(_write(toy, tmp_path), toy_spec)) == len(toy)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns="amount"), "not found"),
    (lambda d: d.assign(x1=d.x1.astype(str) + "z"), "non-numeric"),
    (lambda d: d.assign(x2=np.where(d.index == 3, np.inf, d.x2)), "infinite"),
    (lambda d: d.assign(label=np.where(d.index < 5, 1, 0)), "at least"),
    (lambda d: d.assign(label=d.index % 25), "between 2 and"),
])
def test_load_csv_rejects_bad_tables(toy, toy_spec, tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        load_csv(_write(mutate(toy.copy()), tmp_path), toy_spec)


def test_load_csv_missing_values(toy, toy_spec, tmp_path):
    bad = toy.copy()
    bad.loc[[1, 2], "x1"] = np.nan
    path = _write(bad, tmp_path)
    with pytest.raises(ValueError, match="missing"):
        load_csv(path, toy_spec)
    out = load_csv(path, toy_spec, dropna=True)
    assert len(out) == len(toy) - 2 and out.attrs["rows_dropped_na"] == 2


def test_load_csv_missing_file(tmp_path, toy_spec):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "none.csv", toy_spec)


def test_spec_conflict(toy, tmp_path):
    from creditdecide import FeatureSpec

    spec = FeatureSpec(target="label", categorical=("x1",), signed_log=("x1",))
    with pytest.raises(ValueError, match="both"):
        load_csv(_write(toy, tmp_path), spec)


def test_target_column_types(toy, toy_spec, tmp_path):
    named = toy.assign(label=np.where(toy.label == 1, "yes", "no"))
    df = load_csv(_write(named, tmp_path), toy_spec)
    assert set(df["label"]) == {"yes", "no"}

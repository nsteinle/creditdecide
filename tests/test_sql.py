"""Each DuckDB query against the pandas implementation it replaced (tests/pandas_reference.py),
on the bundled credit-card default data. These tests are the evidence that the SQL is correct:
integer results must match exactly, and floating-point results exactly too, except the PSI,
whose sum may be added up in a different order (equal to 1e-12)."""

import duckdb
import numpy as np
import pandas as pd
import pandas_reference as ref
import pytest
from sklearn.linear_model import LogisticRegression

from creditdecide import CREDIT_DEFAULT_SPEC, load_credit_default, sql
from creditdecide.datasets import CREDIT_DEFAULT_DOCUMENTED_CODES
from creditdecide.decision import gains_table, summarize
from creditdecide.monitor import build_reference, drift_report
from creditdecide.preprocess import make_preprocessor
from creditdecide.quality import quality_report


@pytest.fixture(scope="module")
def credit():
    return load_credit_default()


@pytest.fixture(scope="module")
def scored(credit):
    """A real score on the real data: logistic regression fitted on accounts with odd IDs,
    scored on the accounts with even IDs."""
    spec = CREDIT_DEFAULT_SPEC
    fit, new = credit[credit.ID % 2 == 1], credit[credit.ID % 2 == 0].reset_index(drop=True)
    pre = make_preprocessor(credit, spec).fit(fit)
    lr = LogisticRegression(max_iter=2000).fit(pre.transform(fit), fit.DEFAULT)
    return new, lr.predict_proba(pre.transform(new))[:, 1]


# ---------------------------------------------------------------- quality report
def test_quality_report_matches_pandas(credit):
    codes = CREDIT_DEFAULT_DOCUMENTED_CODES
    got = quality_report(credit, CREDIT_DEFAULT_SPEC, codes)
    assert got == ref.quality_report(credit, CREDIT_DEFAULT_SPEC, codes)
    # the documented numbers (README, docs/FINDINGS.md)
    assert got["rows_sharing_features_with_another_row"] == 108
    assert got["feature_groups_with_conflicting_labels"] == 21
    assert got["rows_sharing_model_inputs_with_another_row"] == 833
    assert got["model_input_groups_with_conflicting_labels"] == 109
    assert got["undocumented_codes"]["EDUCATION"] == {"0": 14, "5": 280, "6": 51}


def test_quality_report_missing_values_and_constants(credit):
    df = credit.head(3000).copy()
    df.loc[df.index[:7], "BILL_AMT1"] = np.nan
    df.loc[df.index[:3], "EDUCATION"] = np.nan
    df["constant"] = 1
    df.loc[df.index[5], "ID"] = df.loc[df.index[4], "ID"]  # one duplicate ID
    got = quality_report(df, CREDIT_DEFAULT_SPEC)
    want = ref.quality_report(df, CREDIT_DEFAULT_SPEC)
    for key in ("n_rows", "missing_values", "constant_columns", "duplicate_ids",
                "class_counts", "duplicate_rows_ignoring_ids"):
        assert got[key] == want[key], key
    assert got["missing_values"] == {"EDUCATION": 3, "BILL_AMT1": 7}


# ---------------------------------------------------------------- gains table
def test_gains_matches_pandas(scored):
    new, s = scored
    y = new.DEFAULT.to_numpy()
    pd.testing.assert_frame_equal(gains_table(y, s), ref.gains_table(y, s), check_exact=True)


def test_gains_ties_and_uneven_deciles(scored):
    new, s = scored
    y = new.DEFAULT.to_numpy()[:14_997]  # 14,997 rows: the first seven deciles get one extra
    s = np.round(s[:14_997], 2)  # many tied scores: ties keep the input order
    got = gains_table(y, s)
    pd.testing.assert_frame_equal(got, ref.gains_table(y, s), check_exact=True)
    assert got.accounts.tolist() == [1500] * 7 + [1499] * 3


# ---------------------------------------------------------------- outcomes join
def test_outcomes_join_matches_pandas(scored):
    new, s = scored
    thr = float(np.quantile(s, 0.9))
    scores = pd.DataFrame({"ID": new.ID, "score": s, "probability": s, "flag": (s >= thr)})
    labels = new[["ID", "DEFAULT"]].sample(frac=1, random_state=0)  # arrive in another order
    got, rows = sql.outcomes(scores, labels, "ID", "DEFAULT", "1", thr)
    want = summarize(new.DEFAULT.to_numpy(), s, thr)
    assert {k: got[k] for k in want} == want
    assert got["scored_without_label"] == got["labelled_without_score"] == 0
    assert rows.id.tolist() == new.ID.tolist() and rows.y.tolist() == new.DEFAULT.tolist()


def test_outcomes_join_counts_unmatched_rows(scored):
    new, s = scored
    scores = pd.DataFrame({"ID": new.ID, "score": s, "probability": s, "flag": s >= 0.5})
    labels = new[["ID", "DEFAULT"]].iloc[100:]
    labels = pd.concat([labels, pd.DataFrame({"ID": [-1, -2], "DEFAULT": [1, 0]})])
    got, rows = sql.outcomes(scores, labels, "ID", "DEFAULT", "1")
    assert (got["n"], got["scored_without_label"], got["labelled_without_score"]) == (
        len(new) - 100, 100, 2)
    kept = slice(100, None)
    want = summarize(new.DEFAULT.to_numpy()[kept], s[kept], 0.5)
    assert {k: got[k] for k in want if k != "threshold"} == {
        k: v for k, v in want.items() if k != "threshold"}
    with pytest.raises(ValueError, match="duplicate"):
        sql.outcomes(pd.concat([scores, scores.head(1)]), labels, "ID", "DEFAULT", "1")


# ---------------------------------------------------------------- PSI drift bins
def _assert_same_drift(got: pd.DataFrame, want: pd.DataFrame):
    g, w = got.set_index("variable"), want.set_index("variable")
    assert set(g.index) == set(w.index)
    assert np.allclose(g.loc[w.index, "psi"], w["psi"], rtol=1e-12, atol=1e-15)
    assert (g.loc[w.index, "status"] == w["status"]).all()
    assert got.psi.is_monotonic_decreasing


@pytest.mark.parametrize("subset", ["second half", "under 30", "unseen codes"])
def test_drift_matches_pandas(credit, scored, subset):
    new, s = scored
    spec = CREDIT_DEFAULT_SPEC
    cols = spec.feature_columns(credit)
    half = len(new) // 2
    reference = build_reference(new.iloc[:half], cols, spec.categorical, s[:half],
                                sample="first half")
    other, scores = new.iloc[half:].reset_index(drop=True), s[half:]
    if subset == "under 30":
        keep = (other.AGE < 30).to_numpy()
        other, scores = other[keep].reset_index(drop=True), scores[keep]
    elif subset == "unseen codes":  # a category and a discrete code never seen in the reference
        other = other.copy()
        other.loc[other.index[:40], "EDUCATION"] = 9
        other.loc[other.index[:25], "PAY_0"] = 12
    got = drift_report(reference, other, scores)
    want = ref.drift_report(reference, other, scores)
    _assert_same_drift(got, want)
    if subset == "under 30":
        assert got.set_index("variable").loc["LIMIT_BAL", "psi"] > 0.10  # younger accounts
    if subset == "unseen codes":
        assert got.set_index("variable").loc["EDUCATION", "psi"] > 0.01


def test_drift_reruns_bit_for_bit(credit, scored):
    new, s = scored
    spec = CREDIT_DEFAULT_SPEC
    reference = build_reference(new, spec.feature_columns(credit), spec.categorical, s,
                                sample="all")
    first = drift_report(reference, credit, np.r_[s, s])
    for _ in range(5):
        pd.testing.assert_frame_equal(drift_report(reference, credit, np.r_[s, s]), first,
                                      check_exact=True)


# ---------------------------------------------------------------- safety
def test_connections_cannot_reach_files(tmp_path):
    p = tmp_path / "secret.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(duckdb.Error):
        sql.query(f"SELECT * FROM read_csv('{p}')")
    with pytest.raises(duckdb.Error):
        sql.query(f"COPY (SELECT 1) TO '{tmp_path / 'out.csv'}'")
    assert not (tmp_path / "out.csv").exists()


def test_column_names_are_quoted():
    """A column name cannot change a query: names with quotes and SQL keywords work."""
    df = pd.DataFrame({'odd"name': [1, 1, 2], "select": [0, 0, 1], "label": [0, 1, 1]})
    q = sql.load("duplicates", columns=['odd"name', "select"], target="label")
    with sql.connect(data=df) as con:
        assert con.execute(q).fetchone() == (2, 1, 1)

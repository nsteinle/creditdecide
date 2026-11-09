# A Tableau Public dashboard from a run

`creditdecide tableau` writes a run's results as flat CSV tables, built by the package's DuckDB
SQL layer. This page lists the tables and describes a five-panel dashboard built from them in
Tableau Public, the free desktop app (now called Tableau Desktop Public Edition). Tableau
Public opens text files but not Parquet, and allows up to 15 million rows, so 6,002 test
accounts are well within its limits.

The dashboard described here is published at
[Tableau Public](https://public.tableau.com/app/profile/nathan.steinle/viz/creditdecidewhichcreditcardaccountstoreview/creditdecide).
I built it in the Tableau Public web editor; the steps below are written for the desktop app.

The tables are tested (`tests/test_cli.py::test_tableau_export`); the steps in Tableau are not,
because they are done by hand in the app. Menu names can differ slightly between versions.

## 1. Write the tables

```bash
creditdecide run --dataset credit-default --out results/credit
creditdecide predict --model-dir results/credit/model --csv results/credit/holdout_test.csv \
    --out results/scored --id-column ID
creditdecide tableau --run-dir results/credit --scored results/scored --out results/tableau
```

`--scored` is optional and adds `drift.csv`. The command refuses to write into a folder that
already has files, unless you add `--overwrite`.

| file | rows (bundled data) | grain and use |
|---|---|---|
| `accounts.csv` | 6,002 | one row per test account: all inputs, `SEX_group`, `MARRIAGE_group` and `AGE_group` (the same groups as the fairness audit), `actual` and `actual_label`, `score`, `probability`, `flag`, `decision`, `score_decile` (the deciles of `gains.csv`) and, for flagged accounts, `top_reason` and `top_reason_impact` |
| `decision.csv` | 1 | the decision model, policy, threshold and test outcome: `capture_rate`, `precision`, `lift`, `flagged`, `flagged_fraction` |
| `gains.csv` | 10 | per score decile: `accounts`, `positives`, `positive_rate`, `cumulative_capture`, `cumulative_lift`, and `share_reviewed` (the share of accounts reviewed down to that decile) |
| `calibration.csv` | 20 | reliability table, 10 equal-count bins for the raw and for the calibrated scores (`scores`) |
| `models.csv` | 18 | one row per model and metric on the test split |
| `intervals.csv` | 10 (budget run) | 95% bootstrap intervals: `quantity`, `estimate`, `ci_low`, `ci_high` |
| `reasons.csv` | 23 | per factor and rank: the number of flagged accounts with that stated reason and its mean impact |
| `fairness.csv` | 11 | per protected group: `flag_rate`, `capture_rate`, `adverse_impact_ratio`, `below_four_fifths`, `too_small` |
| `drift.csv` | 21 | PSI and status per input and for the score (with `--scored`) |
| `tables.json` | | every file's description, columns and row count, and the run it came from |

The files have different grains, so connect each one as its own data source rather than
joining them.

## 2. The dashboard

In Tableau Public: **Connect → To a File → Text file**, choose `accounts.csv`, then add the
other files with **Data → New Data Source**.

1. **Outcome tiles** (`decision.csv`). One text sheet each for `capture_rate`, `precision`
   and `lift`, formatted as percentages (lift as a number). On the bundled data: 31.5%, 69.9%
   and 3.15. Add the 95% interval for the capture rate from `intervals.csv` (29.0% to 33.8%)
   in the tile's caption.
2. **Gains chart** (`gains.csv`). `share_reviewed` on Columns; `Measure Values` on Rows,
   filtered to `cumulative_capture` and `share_reviewed`; `Measure Names` on Color; `decile`
   on Detail; mark type Line. The second line is the diagonal, which is what random selection
   would reach. The first decile reaches 31.5% of the defaulters with 10% of the accounts.
3. **Reliability diagram** (`calibration.csv`). `mean_predicted` on Columns, `observed_rate`
   on Rows, `scores` on Color, `bin` on Detail, mark type Line with markers. Add a reference
   line where predicted equals observed (a calculated field equal to `mean_predicted`, on a
   synchronized dual axis). Points on that line mean the probabilities can be used for
   planning.
4. **Why accounts are flagged** (`reasons.csv`). Filter `rank` to 1, `factor` on Rows,
   `accounts` on Columns, sorted descending. On the bundled data, "Repayment status last month"
   is the top reason for 495 of the 599 flagged accounts.
5. **Flag rate by group** (`fairness.csv`). `attribute` and `group` on Rows,
   `adverse_impact_ratio` on Columns, `too_small` on Color, and a constant reference line at
   0.8 (the four-fifths reference). Say in the caption that the four-fifths reference is an
   indicator, not a legal test, and that groups under 50 accounts are too small to interpret.

`accounts.csv` also supports drill-down views: flag rate or default rate by `AGE_group`,
`EDUCATION` or `score_decile`, or a table of flagged accounts with their `top_reason`. Its
`score_decile` counts match `gains.csv` exactly.

Put the five sheets on one dashboard, and add a text box with the data credit, which the
CC BY 4.0 license requires: Yeh, I.-C. (2009), *Default of Credit Card Clients*, UCI Machine
Learning Repository, doi:10.24432/C55S3H. Say also that all figures are from the test split
of one historical sample (Taiwan, 2005, amounts in NT$).

## 3. Publish and update

**File → Save to Tableau Public As** saves the workbook to your Tableau Public profile. It
needs a free account, and everything saved there, including the data, is public. The bundled
data may be shared under its CC BY 4.0 license; do not publish your own data this way unless it
can be public.

After a new run, write the tables again with `--overwrite` into the same folder, then refresh
the data sources in Tableau and save to Tableau Public again.

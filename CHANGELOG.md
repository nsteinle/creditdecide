# Changelog

## 0.1.0 (2026-11-07)

First public release.

- Data-quality report (`creditdecide quality`): duplicate rows and duplicates with conflicting
  labels, counted on all columns and on the model inputs; undocumented category codes; missing
  values; constant columns; class balance.
- Group-aware stratified 60/20/20 split, so rows with identical model inputs never cross
  splits, with preprocessing fit on the training split only.
- Models: a majority-class baseline, logistic regression and a random forest. Optional tuning
  (`--tune`) by group cross-validation inside the training split, which adds histogram
  gradient boosting; its folds and candidates are drawn by the package, so they do not depend
  on the scikit-learn version.
- Decision layer: capacity (budget) and cost policies with thresholds chosen on validation, a
  per-account expected-loss rule (`--decision-rule expected_loss`), a cost sensitivity table,
  and the best one-feature rule as a benchmark at the model's own volume.
- Probability calibration (identity, sigmoid or isotonic) chosen by cross-validated log loss
  with the one-standard-error rule, with reliability tables and expected vs actual defaults
  and losses.
- Reason codes for flagged accounts, grouped into business factors, with measured accuracy.
- 95% bootstrap intervals for the test results (`bootstrap_test.csv`, `--bootstrap N`), with
  comparisons computed on the same redraws.
- Fairness audit by protected group, with adverse impact ratios and a proxy check.
- A saved model directory with SHA-256 checksums, a skops type allow-list and a generated model
  card; `predict`, `drift`, and outcomes analysis when labels arrive.
- A DuckDB SQL data layer (`creditdecide.sql`): the data-quality counts, the gains table, the
  outcomes join and the drift bins, each tested against the pandas version it replaced.
  `creditdecide outcomes` joins saved decisions to labels that arrive later, and
  `creditdecide sql` runs a read-only query on the data.
- `creditdecide tableau` writes a run's results as CSV tables for a Tableau Public dashboard
  (docs/TABLEAU.md).
- Random forests are fitted on all cores but predict on one thread (`creditdecide.forest`), so
  reruns match bit for bit.
- Bundled UCI credit-card default data (CC BY 4.0), CI on Python 3.10, 3.12 and 3.13, a Docker
  image, and a PyPI Trusted Publishing workflow.

# Findings, interpretation and verification log

Everything below was produced on the bundled credit-card default data with seed 0 unless
stated otherwise, and rerun under two sets of library versions: Python 3.10 with
scikit-learn 1.7.2, NumPy 2.2.6, pandas 2.3.3 and DuckDB 1.5.6, and Python 3.13 with
scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1 and DuckDB 2.0.0. Every figure is the same under
both, except the tuned gradient boosting model (section 5). The runs are:

```bash
creditdecide run --dataset credit-default --out budget --plot               # 10% budget
creditdecide run --dataset credit-default --out cost \
    --cost-fp 8000 --lgd 0.6 --exposure-column BILL_AMT1                   # one cut-off
creditdecide run --dataset credit-default --out el \
    --cost-fp 8000 --lgd 0.6 --exposure-column BILL_AMT1 --decision-rule expected_loss
creditdecide run --dataset credit-default --out tuned --tune                # ~4 min
```

Ranges in brackets are 95% bootstrap intervals over the test accounts (section 4).

## 1. What the data show

| Finding | Value | Consequence |
|---|---|---|
| Class balance | 22.1% default (6,636 of 30,000) | "Always no default" is 77.9% accurate, so accuracy is not used to judge models |
| Exact duplicates (ignoring ID) | 35 rows | kept, with each set of copies in a single split |
| Rows sharing all columns except ID and the label with another row | 108 rows in 52 groups, 21 groups with conflicting labels | label noise that no model can remove |
| Rows sharing all **model inputs** with another row (sex, marriage and age are not inputs) | 833 rows in 170 groups, 109 groups with conflicting labels | the split keeps each group together, so no test input also appears in training; 88% of these rows are accounts with six months of zero bills and zero payments |
| Undocumented category codes | EDUCATION 0, 5 and 6 (345 rows), MARRIAGE 0 (54), and repayment status 0 and −2 (the majority of every PAY column) | kept as they are (they carry signal), reported, and listed as a question for the data owner |
| Missing values | none | no imputation needed |

The second count is the one that matters for the model. Compared on all columns only 108 rows
look duplicated, which understates how many rows look identical to the model, so the quality
report gives both counts and the split groups on the model inputs.

## 2. Model results (test split, 6,002 clients)

| model | accuracy | balanced accuracy | ROC AUC [95% interval] | average precision |
|---|---|---|---|---|
| majority class | 0.778 | 0.500 | 0.500 | 0.222 |
| logistic regression | 0.804 | 0.604 | 0.738 [0.720, 0.755] | 0.485 |
| random forest | 0.817 | 0.655 | 0.782 [0.768, 0.797] | 0.550 |

Decision, fairness and monitoring results are in [BUSINESS_CASE.md](BUSINESS_CASE.md).

## 3. The data layer in SQL

The data-quality counts, the gains table, the outcomes join and the drift bins are DuckDB
queries (`src/creditdecide/sql/*.sql`). The models, calibration, reason codes and bootstrap stay
in Python. DuckDB runs in memory inside the Python process and reads the pandas DataFrames in
place.

**How the SQL is checked.** Each query replaced a pandas function. The pandas versions are kept
in `tests/pandas_reference.py`, and `tests/test_sql.py` runs both on the bundled data:

| query | what is compared | result |
|---|---|---|
| data-quality counts | the whole quality report on the 30,000 rows, and a copy with missing values, a constant column and a duplicate ID added | identical |
| gains table | a logistic regression's scores for 15,000 accounts; the same scores rounded to two decimals (many ties) on 14,997 rows (deciles of unequal size) | identical, bit for bit |
| outcomes join | labels delivered in a different order; labels with 100 accounts missing and 2 unknown IDs | identical to `summarize` on the matched rows; the unmatched rows counted on both sides |
| drift bins and PSI | 21 inputs and the score, on half the accounts, on the clients under 30, and with unseen codes added | every PSI equal to within 1e-12 (the bins are summed in a different order), every status identical |

On the full runs the SQL also reproduces the pandas outputs of the previous version: the gains
tables and the outcomes are identical, and the drift values differ by at most 9e-19.

**Determinism.** The counts and the gains table are integer sums or ordered window sums, so a
rerun cannot change them. The PSI sum adds the bins in bin order (`SUM(... ORDER BY bin)`) so
that the last bit does not depend on how DuckDB splits the work between threads. A test reruns
the drift report six times and requires identical output.

**Safety.** Every connection has external access and extension loading switched off, so a query
cannot read or write files or reach the network, including through `creditdecide sql`. Column
names are quoted as identifiers and values are passed as parameters; a test uses column names
containing a quote and a SQL keyword.

**What it does not show.** 30,000 rows fit in memory many times over, so these queries show
correct SQL, not out-of-core processing or scale.

## 4. Uncertainty: bootstrap intervals

The test split is 6,002 customers from one period. A different sample of customers would give
somewhat different figures, and several conclusions below rest on differences of less than one
percentage point, so the size of the noise matters. Every run redraws the test accounts with replacement 1,000 times (accounts with identical inputs are drawn
together, like the split), recomputes each figure, and reports the 2.5th and 97.5th
percentiles (the percentile bootstrap; Efron and Tibshirani 1993, chapter 13). Comparisons are
computed on the same redraws, so a difference gets its own interval, which is narrower than
what the two separate intervals would suggest when the two figures move together. The results
are in `bootstrap_test.csv`, in the printed run summary and in the model card.

**10% budget run:**

| quantity | estimate [95% interval] | conclusion |
|---|---|---|
| defaulters reached (capture rate) | 31.5% [29.0%, 33.8%] | plan on 29% to 34% |
| precision | 69.9% [66.3%, 73.5%] | |
| model minus `PAY_0 >= 2`, same volume (10.9%) | −0.4 points [−1.7, +0.9] | can't be told apart; at best about 12 more defaulters per 6,000 accounts |
| model minus `PAY_0 >= 1`, same volume (22.6%) | +3.5 points [+1.8, +5.2] | the model is ahead, by 24 to 69 defaulters per 6,000 accounts |
| random forest minus logistic regression, ROC AUC | +0.044 [+0.033, +0.055] | clear |
| default forecast error, all accounts | +0.4% [−3.5%, +4.6%] | no sign of bias |
| default forecast error, flagged accounts | −1.1% [−6.0%, +4.6%] | no sign of bias |

**Cost runs (NT$8,000 per false alarm, 60% of the current bill lost per missed default):**

| quantity | estimate [95% interval] |
|---|---|
| saving against the best trivial strategy, one cut-off | 43.1% [35.8%, 48.2%] |
| saving against the best trivial strategy, per-account rule | 63.9% [60.0%, 66.2%] |
| per-account rule minus one cut-off | +20.8 points [+15.2, +27.7] |
| the same difference in money, 6,002 test accounts | NT$7.8 million [5.4, 10.3] |
| loss forecast error (expected / realized − 1) | −2.7% [−9.6%, +4.6%] |

What this adds to the business conclusions:

- At a 10% budget the rule `PAY_0 >= 2` does as well as the model, and the interval rules out
  any large advantage for the model. The point estimate alone (−0.4 points) could not say this.
- At 22.6% of accounts the model's advantage over `PAY_0 >= 1` is clear, and so is the
  per-account rule's advantage in money (NT$7.8 million on the test month, at least NT$5.4
  million). The case for the model is in larger volumes and in cost-based decisions.
- The default forecast is good to within about 5%. The loss forecast is only good to within
  about 10%, because a few large balances carry much of the loss. A finance team should get the
  wider range.
- The random forest's lead over logistic regression is not an accident of the sample. The
  cross-validated tuning gains (0.001 to 0.007) are smaller than these interval widths, which
  agrees with the conclusion that tuning buys nothing here.

**What is held fixed.** The trained models, threshold, calibration map and rules do not change
between redraws, so the intervals measure how much the figures depend on which customers were
in the test sample. They do not include retraining. A different seed, which changes the split
and the training, moves the random forest's test AUC by an amount of the same order (0.775 vs
0.782 for seeds 1 and 0), so the full uncertainty is somewhat larger than these intervals.

**A caveat about the benchmark in the cost runs.** The package compares the model with the best
one-feature rule at the model's volume. In the cost runs it picks `PAY_5 >= 2` (one cut-off run)
and `PAY_2 >= 1` (per-account run), which are weaker rules than `PAY_0 >= 2` and
`PAY_0 >= 1`. This happens because the search takes, for each input, the cut-off whose volume
is closest to the model's, and then picks the highest precision among those. The resulting
model advantages of +8.3 [+5.8, +10.8] and +5.6 [+3.2, +7.9] points therefore overstate the gap
to the best simple rule. Use the `PAY_0 >= 1` comparison above (+3.5 points) instead.

The `PAY_0 >= 1` rows are computed from the budget run's `samples.csv` with the package's own
functions:

```python
import numpy as np, pandas as pd
from creditdecide import CREDIT_DEFAULT_SPEC, load_credit_default
from creditdecide import bootstrap as bs
from sklearn.metrics import roc_auc_score
s = pd.read_csv("budget/samples.csv"); t = s[s.split == "test"]
y = t.DEFAULT.to_numpy()
inputs = CREDIT_DEFAULT_SPEC.feature_columns(load_credit_default())
groups = t.groupby(inputs, sort=False, dropna=False).ngroup().to_numpy()
rule = (t.PAY_0 >= 1).to_numpy()                                # 1,358 accounts (22.6%)
top = np.zeros(len(t), bool)
top[np.argsort(-t.decision_score.to_numpy(), kind="stable")[: rule.sum()]] = True
print(bs.bootstrap(lambda w: {"rule": bs.capture_rate(w, y, rule),           # 0.514
                              "model": bs.capture_rate(w, y, top),           # 0.548
                              "model minus rule": bs.capture_rate(w, y, top)
                              - bs.capture_rate(w, y, rule)},                # 0.035
                   groups, differences={"model minus rule"}).round(3).to_string())
for band in (t.PAY_0 >= 2, t.PAY_0 < 2):                        # 0.589, 0.709
    print(roc_auc_score(t[band].DEFAULT, t[band].decision_score))
```

## 5. Challenger, calibration, cost rule, reason codes and tuning

**Simple-rule challenger.** At the 10% budget the benchmark selects `PAY_0 >= 2` ("two or
more months late last month") on validation. On test it reaches 34.1% of defaulters at 69.4%
precision while flagging 10.9%, against 33.7% and 68.7% for the model at the same volume. The
comparison at 22.6% and the ranking within each delinquency band come from the snippet above.

**Calibration.** The random forest is already calibrated (test ECE 0.016; 1,337 defaults
expected and 1,332 occurred), so the one-standard-error rule keeps the identity map. A
class-weighted random forest on the same splits forecasts 2,498 defaults. Calibration (a
sigmoid map, chosen automatically) brings that to 1,296 and the error from 0.194 to 0.015
(`scripts/check_calibration.py`).

**Per-account expected-loss rule.** It beats the single cut-off in all nine cost scenarios on
test, for example 63.9% vs 43.1% saving against the best trivial strategy at (8,000, 0.6). It
does this by targeting large balances while reaching fewer defaulters (45.7% vs 49.5%). It acts
on 646 customers who were not late last month, against 141 for the cut-off and 3 at the 10%
budget.

**Reason codes.** Across the three rules, the stated reasons change the decision for 95.8% to
100% of flagged accounts, and the top reason alone for 64.9% to 82.1%. Setting the top reason
to typical values lowers the score 3.5 times as much as a random factor does (0.275 vs 0.078,
10% budget).

**Tuning.** The cross-validated gains are +0.0013 (logistic regression), +0.0038 (random
forest) and +0.0068 (gradient boosting) in AUC, each within one standard deviation (0.009 to
0.014). The tuned gradient boosting model, selected on validation, scores test ROC AUC 0.782,
the same as the default random forest, and within the tuned run it differs from the tuned
random forest by −0.001 [−0.005, +0.002]. The logistic regression's best regularization
(C = 0.001) is at the edge of the searched grid, but with a gain of 0.0013 I did not extend the
grid. The folds and the candidates are drawn by the package (creditdecide.tuning), so they are
the same on every library version; the gradient boosting figures still need scikit-learn 1.8 or
later, because 1.8 changed its gradient boosting (BUSINESS_CASE.md gives the figures for 1.7).

## 6. Why the negative results are reported

Class imbalance, duplicates with conflicting labels, undocumented codes, a random forest that
ties with a one-line rule at the default budget, and tuning that doesn't help are all normal
for real data. Finding these, measuring how
much they matter and designing around them is most of the work in data science, data quality
and ML engineering, so I report them rather than tune them away.

The risk is that a reader skims and concludes that "the model is useless". So the documents
ask explicit questions and answer them with numbers. At 10% capacity, use the simple rule. Use the model when
capacity grows or when the goal is money, where a per-account rule driven by calibrated
probabilities saves much more than any single cut-off.

## 7. What the project shows

| Skill | Where to find it |
|---|---|
| Translating a business problem into a technical one | BUSINESS_CASE.md sections 1 and 2: the manager's questions mapped to the unit of decision, outcome, constraints, costs, planning numbers, explanations and metrics |
| Building the whole workflow | `run`, `predict`, `drift` and `outcomes`; a deployable model directory; command line and Python API |
| SQL | the data-quality counts, gains table, outcomes join and drift bins as DuckDB queries, each tested against the pandas version (section 3) |
| Reporting | `creditdecide tableau` writes Tableau Public tables built by the same SQL layer (TABLEAU.md) |
| Decisions under constraints | capacity policy, two cost rules with sensitivity analysis, comparison with trivial strategies and with the best simple rule |
| Recommending against complexity when it doesn't pay | the simple rule at 10% capacity, defaults over tuning, no calibration map when none is needed |
| Data quality | quality report before modeling, group-aware split, schema validation at scoring time |
| Evaluation discipline | baselines, tuning inside the training split only, selection on validation only, the test split used once, metrics suited to imbalance, the one-standard-error rule |
| Uncertainty in the reported figures | bootstrap intervals on every test figure, and paired intervals on every comparison that drives a decision |
| Probabilistic forecasting | calibrated probabilities checked against realized defaults and losses |
| Responsible ML in a regulated domain | protected attributes excluded by default but audited, a proxy check, reason codes with measured accuracy, and the regulatory context (ECOA and Regulation B adverse action and its delinquency exclusion, SR 11-7, CFPB Circular 2022-03) |
| MLOps for batch scoring | a versioned, checksum-verified model directory that stays loadable across versions, safe loading, drift monitoring, outcomes and calibration analysis, a model card, CI on three Python versions, a Docker image, Trusted Publishing to PyPI |

## 8. Verification log

| Check | How | Result |
|---|---|---|
| Bundled data integrity | SHA-256 of the source archive and of the bundled file; rebuild twice from the UCI archive | rebuild byte-identical; one appended byte makes loading fail |
| Data schema | 30,000 rows × 25 integer columns; labels {0, 1}; class counts | matches the published description (6,636 defaults) |
| Duplicate counts | quality report (DuckDB SQL) vs a direct pandas `groupby` on all columns and on the 20 model inputs | 108 / 52 / 21 and 833 / 170 / 109, as reported |
| SQL against pandas | every query vs the pandas function it replaced, on the bundled data (section 3) | counts, gains and outcomes identical; PSI within 1e-12 |
| Outcomes join | labels in another file, shuffled, with missing and unknown IDs, from the command line | same outcome as labels inside the scoring file; unmatched rows counted |
| SQL cannot reach files | `read_csv` and `COPY ... TO` through the package's connections | refused; no file written |
| Tableau tables | `tableau` on a run and a scored folder | each account of the test split once; deciles match `gains_test.csv`; top reasons match `reasons_test.csv`; metrics match `metrics_test.csv`; `tables.json` lists the columns and rows of every file |
| Cost-optimal threshold | compared with brute force over every threshold, 200 random problems with tied scores | identical expected cost in all 200 |
| Budget threshold | flagged fraction on the data it was chosen on | 10.0% for a 10% budget |
| Model artifact | save, reload in a fresh object, rescore the test split | scores and flags identical to the in-memory model |
| Artifact tampering | flip one byte of the pipeline file; list a file the format does not know | load refused, with a checksum error and an unexpected-file error |
| PSI | hand computation for LIMIT_BAL from the saved deciles | identical to 6 decimals (0.003479) |
| Drift detection | reference vs itself; test split; clients under 30; balances × 1.5 | about 0; all 21 stable (max 0.006); credit limit 0.146 (moderate); credit limit 0.80 (significant) |
| Fairness table | flag rates and adverse impact ratios by sex recomputed directly with pandas | identical |
| Full workflow | `run`, `predict` on unlabeled and labeled files, and `drift` from the command line | as reported in the README |
| Installation | built wheel installed in a fresh environment and run outside the source tree | identical results; refuses to overwrite existing output |
| Cross-version scoring | model saved with scikit-learn 1.9.1, loaded and scored with 1.7.2 | scores and flags identical, bit for bit; one warning naming both versions |
| Behavior kept as features were added and removed | default run before and after adding calibration and reason codes, the bootstrap and single-thread forest prediction, and before and after removing the low-dimensional summary models and moving the data layer to SQL | same model, threshold (0.5649), capture (31.5%) and precision (69.9%) every time; after the last change, every decision, metric, interval, reason code and fairness figure identical bit for bit in the budget, both cost and the tuned runs |
| Minimal model directories load | format-1 manifest without calibration, rule or reasons | scores and flags with a threshold rule on the uncalibrated score; reason codes refused with a clear error |
| Isotonic map | stored breakpoints with `np.interp` vs scikit-learn `IsotonicRegression.predict` on 300 points | identical |
| Calibration is monotone | all three maps on sorted scores | non-decreasing, within [0, 1]; threshold flags identical for every method |
| Calibration repairs a miscalibrated model | class-weighted random forest, same splits | expected defaults 2,498 → 1,296 (actual 1,332); ECE 0.194 → 0.015; ranking unchanged |
| Expected-loss rule | flags recomputed as p × missed > cost × (1 − p) in Python and after `predict` | identical |
| Reason codes, known answer | linear toy model with planted drivers and a grouped factor | correct factors, impacts and blanks for factors that don't contribute |
| Reason codes, saved model | reasons from the run vs from the reloaded model on the same accounts | identical reasons and impacts |
| Reason codes, data types | reference medians that are fractional in integer columns (pandas 3 rejects lossy assignment) | explicit upcast; covered by a unit test |
| Simple-rule benchmark | planted informative feature; real data | finds the planted feature; on real data `PAY_0 >= 2`, matching a manual computation |
| Tuned gradient boosting | export, skops allow-list, reload, rescore | scores identical; outcomes on the holdout match the run |
| Weighted ROC AUC used by the bootstrap | against scikit-learn `roc_auc_score` with `sample_weight`, scores with many ties | identical |
| Bootstrap, known answer | standard error of a capture rate vs the binomial formula √(c(1 − c)/n) | within 10% |
| Bootstrap point estimates | weights of 1 for every account vs the reported test figures, benchmark gain and forecast | identical |
| Bootstrap reproducibility | same run twice from the command line; same run under both library sets | identical intervals, bit for bit, in both cases |
| Across library versions | budget, both cost and the tuned runs under Python 3.10 (scikit-learn 1.7.2, pandas 2.3.3) and Python 3.13 (1.9.1, 3.1.1) | without `--tune`: every decision, interval, reason, fairness and gains figure identical, metrics within 5e-16. With `--tune`: the same folds, candidates, logistic regression and random forest; gradient boosting differs, because scikit-learn 1.8 changed it (it gives different predictions on identical data and settings) |
| Folds, candidates and draws pinned | fingerprints of the tuning folds and candidates, the main split, the calibration folds and the bootstrap and reason-code draws, on scikit-learn 1.7.2 and 1.9.1 and NumPy 2.2.6 and 2.5.4 | identical; tests pin them, so CI on three Python versions fails if a library release changes them |
| Exact CSV reading | `outcomes` on scores read back from `scores.csv` vs labels inside the scoring file, under pandas 2.3.3 and 3.1.1 | identical `outcomes.json`. pandas 2's default parser read 3,910 of 6,002 scores back about 1e-16 off; the package now reads exactly on both |
| Reruns bit for bit | toy run repeated 150 times; budget, both cost and tuned runs on the bundled data run twice | every output file identical, because forests predict on one thread. With forests predicting on several threads, reruns differed in the last bit and `test_deterministic` failed about once in 30 to 60 runs |
| Reason codes only for factors with an effect | stated impacts in the budget, cut-off and per-account runs | no stated reason has an impact below 1e-12; the smallest at the 10% budget is 1.5e-5. With multi-threaded prediction, noise of about 1e-16 let factors with no effect be listed: 13 to 20 of about 2,250 stated reasons at the 10% budget, 19 of 4,198 for the cut-off and 386 of 4,152 for the per-account rule |
| Saving intervals across rules | cost run with the cut-off vs with the per-account rule | the same two savings and difference in both |
| `PAY_0 >= 1` snippet | section 4 snippet on the budget run's `samples.csv` | reproduces the 22.6% rows |
| Library versions | full test suite under Python 3.10 / scikit-learn 1.7.2 / pandas 2.3.3 / DuckDB 1.5.6, Python 3.13 with the same and DuckDB 1.1.3, and Python 3.13 / 1.9.1 / 3.1.1 / 2.0.0 | all tests pass on all three |

Not verified here: the GitHub Actions workflows and the Docker image (there was no Docker daemon
in the build environment). Both run on the first push to GitHub.

## 9. Known limitations

- PSI with decile bins understated a 1.5× increase in the bill amounts (PSI about 0.07) while
  catching it in the credit limit (0.80), so heavy-tailed variables need other checks as well.
- Reason codes set one factor at a time to reference values, so interactions between factors
  only show up through the sufficiency check.
- The cost model assumes that acting prevents the whole loss and costs the same for every
  account.
- The simple-rule benchmark searches one-feature rules only and, at volumes well above 10%, can
  pick a weaker rule than the best one (section 4). Two-feature rules (for example repayment
  status and balance) would be a stronger challenger.
- The bootstrap intervals cover the choice of test customers, not retraining or changes in the
  population over time.
- One historical sample from one market.
- The tuned gradient boosting model depends on the scikit-learn version (1.7 vs 1.8 and
  later), so a tuned result should be reported with the version it was run on.

## 10. Next steps (not pursued)

Development stops here. These are the questions I would answer next, roughly in order of
expected value.

1. **A stronger challenger.** Search every cut-off within the volume band rather than the one
   nearest the model's volume for each input, and add two-feature rules (for example repayment
   status combined with balance), the kind of rule a credit team could still apply by hand.
2. **Monotonic constraints** (gradient boosting supports them), so that, for example, a longer
   payment delay can never lower the score, which makes reason codes and model review simpler.
3. **Time-based validation.** These data are one snapshot. With several months of data, train
   on earlier months and test on later ones.
4. **Uncertainty from retraining.** Repeat the whole pipeline over several seeds and splits,
   or bootstrap the training data, to add the model's own variability to the intervals in
   section 4.

# Model card: random_forest on credit-card default (UCI, Yeh 2009)

## Purpose
Ranks rows by the probability that `DEFAULT` = `1` (default) and turns the ranking into a decision: act on the 10.0% highest-scoring accounts (threshold 0.5649, chosen on the validation split). Intended as decision support with human review; not for fully automated adverse decisions without reason codes.

## Data and features
- Training / validation / test rows: 18,005 / 5,993 / 6,002 (group-aware stratified split; rows with identical model inputs never cross splits).
- Model inputs (20): LIMIT_BAL, EDUCATION, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6.
- Excluded: ID, SEX, MARRIAGE, AGE.
- Positive rate in the test split: 22.2%.

## Model selection
Scoring model chosen by validation ROC AUC among logistic_regression (0.746), random_forest (0.782).
Hyperparameters: library defaults (random forest: 200 trees, depth 10); tuning was not run.

## Test-split performance (all models)
| model | accuracy | balanced_accuracy | roc_auc | average_precision |
|---|---|---|---|---|
| majority_baseline | 0.778 | 0.500 | 0.500 | 0.222 |
| logistic_regression | 0.804 | 0.604 | 0.738 | 0.485 |
| random_forest | 0.817 | 0.655 | 0.782 | 0.550 |

## Decision outcome (test split)
- Flagged 599 of 6,002 (10.0%); caught 31.5% of positives at 69.9% precision (lift 3.15 over random selection).
- Validation estimate was capture 31.3%, precision 68.7%: the operating point held on unseen data within 0.2 percentage points.

## Probabilities (calibration, test split)
- Map: none (chosen by 5-fold CV log loss within the validation split, one-standard-error rule). Monotone, so it does not change which accounts a threshold rule flags.
- Expected calibration error 0.016 raw, 0.016 calibrated; Brier score 0.1352 raw, 0.1352 calibrated.
- Portfolio forecast: 1,337 positives expected, 1,332 occurred. Among flagged accounts: 414 expected, 419 occurred.

## Reason codes (flagged test accounts)
- Up to 4 reasons per flagged account: the factors whose values, set to those of marginally not-flagged validation accounts, lower the decision margin most (Regulation B comment 9(b)(2)-5, first method).
- Accuracy: setting the stated reasons to the reference values changes the decision for 100.0% of 599 flagged accounts; the top reason alone for 82.1%. Setting the top reason to the reference values lowers the decision margin by 0.275 on average vs 0.078 for a random factor (score points).
- Primary reason: Repayment status last month (82.6%), Repayment status in the five months before (12.5%), Bill amounts over the last six months (2.8%), Payment amounts over the last six months (1.8%), Credit limit (0.2%).

## Uncertainty (test split)
95% intervals from 1,000 bootstrap resamples of the test accounts (rows with identical inputs drawn together). The models, threshold, calibration map and rule are held fixed, so the intervals show how much the figures depend on which customers were in the test sample, not the extra variation from retraining. For a difference, "sign clear" means the interval excludes zero.

- capture rate: 31.5% [29.0%, 33.8%]
- precision: 69.9% [66.3%, 73.5%]
- flagged fraction: 10.0% [9.2%, 10.8%]
- lift: 3.15 [2.97, 3.33]
- ROC AUC logistic_regression: 0.738 [0.720, 0.755]
- ROC AUC random_forest: 0.782 [0.768, 0.797]
- ROC AUC random_forest minus logistic_regression: +0.044 [+0.033, +0.055]; sign clear
- capture rate, model minus rule PAY_0 >= 2 at the rule's volume: -0.4 points [-1.7, +0.9]; includes zero
- defaults forecast error, all accounts (expected / actual - 1): +0.4% [-3.5%, +4.6%]; includes zero
- defaults forecast error, flagged accounts (expected / actual - 1): -1.1% [-6.0%, +4.6%]; includes zero

## Outcomes by protected group (test split)
Protected attributes are audited whether or not they are model inputs. Adverse impact ratio = favorable-outcome rate / highest group's; < 0.8 is marked against the four-fifths reference (an indicator, not a legal test).

| attribute | group | n | positive_rate | flag_rate | false_alarm_rate | adverse_impact_ratio | note |
|---|---|---|---|---|---|---|---|
| SEX | female | 3623 | 0.212 | 0.096 | 0.034 | 1.000 |  |
| SEX | male | 2379 | 0.237 | 0.106 | 0.045 | 0.990 |  |
| MARRIAGE | married | 2674 | 0.240 | 0.101 | 0.036 | 0.939 |  |
| MARRIAGE | others | 72 | 0.181 | 0.042 | 0.034 | 1.000 |  |
| MARRIAGE | single | 3241 | 0.209 | 0.100 | 0.039 | 0.939 |  |
| MARRIAGE | undocumented (0) | 15 | 0.067 | 0.267 | 0.214 | 0.765 | too small to interpret |
| AGE | 25-34 | 2610 | 0.214 | 0.096 | 0.041 | 0.993 |  |
| AGE | 35-44 | 1796 | 0.213 | 0.089 | 0.031 | 1.000 |  |
| AGE | 45-61 | 978 | 0.226 | 0.100 | 0.034 | 0.988 |  |
| AGE | 62+ | 43 | 0.256 | 0.093 | 0.062 | 0.996 | too small to interpret |
| AGE | < 25 | 575 | 0.277 | 0.151 | 0.058 | 0.932 |  |

Proxy check (ROC AUC of predicting group membership from model inputs; 0.5 = no information):
- SEX: female 0.552, male 0.552
- MARRIAGE: married 0.628, others 0.669, single 0.632
- AGE: 25-34 0.608, 35-44 0.601, 45-61 0.633, 62+ 0.672, < 25 0.821

## Monitoring
`creditdecide predict` and `creditdecide drift` compare new data with the validation sample saved in `reference.json` using the population stability index: < 0.10 stable, 0.10-0.25 check other metrics, > 0.25 investigate before relying on the scores. When outcomes become available, rerun `predict` on a file that includes the target column to compare realized capture and precision with the figures above.

## Validation coverage (SR 11-7 elements)
- Conceptual soundness: documented data checks, leakage-safe splits, baselines, selection on validation data only.
- Ongoing monitoring: PSI on every input and on the score; schema and category checks at scoring time.
- Outcomes analysis: test-split results above, with bootstrap intervals when computed; realized-outcome comparison when labels arrive.
- Not covered: independent review, champion-challenger testing, back-testing over many periods.

## Limitations
- Reason codes approximate the model by setting one factor at a time to typical values from a reference population; their accuracy is measured above, not guaranteed.
- Calibration is fitted on one validation sample; recheck it with `predict` on labeled data, since calibration drifts before ranking does.
- Results describe one historical sample; populations change. Bootstrap intervals cover the choice of test customers, not retraining.

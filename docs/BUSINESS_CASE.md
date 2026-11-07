# Business case: a monthly default-risk review for a card portfolio

This document follows one problem from the business question to a deployed, monitored and
explainable rule, and explains the reasoning at each step. The figures come from the bundled
data (seed 0) and are reproduced by:

```bash
creditdecide run --dataset credit-default --out budget                      # 10% budget
creditdecide run --dataset credit-default --out cost \
    --cost-fp 8000 --lgd 0.6 --exposure-column BILL_AMT1                   # one cut-off
creditdecide run --dataset credit-default --out el \
    --cost-fp 8000 --lgd 0.6 --exposure-column BILL_AMT1 --decision-rule expected_loss
creditdecide run --dataset credit-default --out tuned --tune                # ~4 min
```

All amounts are in New Taiwan dollars (NT$). The data describe 30,000 card holders in Taiwan in
2005. Ranges in brackets are 95% bootstrap intervals over the 6,002 test accounts (section 5).

## 1. The business problem

A card issuer sees each account's credit limit, its last six monthly bills and payments, and its
repayment status. Some accounts will miss next month's payment, and the issuer can act before
that happens with a call from the collections team, a payment plan or a lower limit. Acting
costs money and goodwill, and the team can only handle so many accounts. The questions a
manager would ask are:

1. Which accounts should we act on this month?
2. If we can handle 10% of accounts, how many of next month's defaults will we reach, and how
   many of the people we contact would have paid anyway?
3. Is a model worth it, or would a simple rule do as well?
4. How many defaults, and how much loss, should we plan for?
5. Why was this account flagged, and do we owe the customer an explanation?
6. Are we treating customers comparably across sex, age and marital status?
7. How will we know when the rule stops working?

## 2. Translating it into a technical problem

| Business element | Technical choice | Reason |
|---|---|---|
| Unit of decision | one account per month | the action is taken per account |
| Outcome | `DEFAULT` = payment missed next month | this is what the action tries to prevent; it is observed one month later, which delays monitoring |
| Inputs | limit, bills, payments, repayment status, education | all known at decision time |
| Excluded inputs | sex, marital status, age | prohibited (sex, marital status) or restricted (age) bases under the US Equal Credit Opportunity Act and Regulation B; kept for auditing only |
| Capacity | review budget (fraction of accounts) | this is how operations teams are actually limited |
| Unequal costs | cost of a false alarm vs loss given default × balance | finance's view; uncertain, so tested with a sensitivity analysis |
| What makes a model good | ranking (ROC AUC), capture and precision at the operating point, expected cost, and the margin over a simple rule | 22% of accounts default, so a model that says "no default" to everyone is 78% accurate and useless |
| Planning numbers | calibrated probabilities, checked against outcomes | staffing and provisioning need counts and amounts, not ranks |
| How sure we are | 95% bootstrap intervals on test results and on each comparison | a manager should not act on a difference that is smaller than its noise |
| Explanations | up to four reason codes per flagged account, with measured accuracy | the team needs to know what to discuss, and some actions legally require specific reasons |

## 3. Data quality first

`creditdecide quality --dataset credit-default` finds:

- 35 duplicate rows, and 108 rows that share all their columns (except ID and the label) with
  another row, with 21 of those groups having **conflicting labels**. Since sex, marital status
  and age are not model inputs, more rows look identical to the model: 833 rows in 170 groups,
  109 of them with conflicting labels. Most are accounts with six months of zero bills and
  zero payments. Identical customers with different outcomes are noise that no model can
  remove, and if they were split between training and test they would inflate the test scores,
  so the split keeps each group together.
- Undocumented codes: EDUCATION 0, 5 and 6 (345 rows), MARRIAGE 0 (54 rows), and repayment
  status 0 and −2, which are missing from the data documentation but make up the majority of
  every repayment-status column. They are kept (they carry signal) and flagged in the report.
- No missing values.

A lender would next ask the data owner what the undocumented codes mean before the model goes
live. The package raises the question but cannot answer it.

## 4. Solution

- Three models are compared on the same split: a majority-class baseline, logistic regression
  and a random forest (tuning adds gradient boosting).
- The scoring model is chosen by validation ROC AUC. The random forest wins (0.782 on
  validation, against 0.746 for logistic regression).
- Its probabilities are checked for calibration on the validation split, the operating rule is
  chosen on validation, and the test split is used once.
- The best one-feature rule at the same volume is reported as the challenger.
- Flagged accounts get reason codes, and outcomes are audited by protected group.
- Every test figure gets a 95% bootstrap interval.

## 5. Outcomes

### Capacity policy: act on 10% of accounts

| | validation | test [95% interval] |
|---|---|---|
| accounts flagged | 10.0% | 10.0% |
| defaulters caught (capture rate) | 31.3% | 31.5% [29.0%, 33.8%] |
| flagged accounts that default (precision) | 68.7% | 69.9% [66.3%, 73.5%] |
| lift over random selection | 3.12 | 3.15 [2.97, 3.33] |

The validation estimate held on unseen data to within 0.2 percentage points, well inside the
test interval, which is what makes the number usable for planning: the team should expect to
reach roughly 29% to 34% of next month's defaulters. Cumulative gains on the test split:

| reviewed (top deciles) | 10% | 20% | 30% | 50% |
|---|---|---|---|---|
| defaulters reached | 31.5% | 51.3% | 63.4% | 78.6% |

A manager can read the capacity trade-off directly from this table. For example, doubling the
review team from 10% to 20% of accounts reaches 51% of defaulters instead of 32%.

### Is the model better than a simple rule?

Every run searches all numeric inputs for a one-feature cut-off near the model's validation
volume with the highest validation precision, and then evaluates it once on test against the
model's top accounts at exactly the same volume.

| test split, same volume | flagged | rule: defaulters reached | model: defaulters reached | model minus rule |
|---|---|---|---|---|
| 10% budget: rule `PAY_0 >= 2` (two or more months late last month) | 10.9% | 34.1% (69.4% precision) | 33.7% (68.7% precision) | −0.4 points [−1.7, +0.9] |
| rule `PAY_0 >= 1` (any delay last month) | 22.6% | 51.4% (50.4% precision) | 54.8% (53.8% precision) | +3.5 points [+1.8, +5.2] |

**At a 10% budget the model is no better than a one-line rule** that any collections team
already uses. The interval also limits how much better the model could be: at most about 0.9
points, or 12 more defaulters per 6,000 accounts. The model adds value at larger volumes (24 to
69 more defaulters per 6,000 accounts at 22.6%), in ranking customers *within* a delinquency
band (ROC AUC 0.71 among accounts less than two months late, 0.59 among those two or more months
late), and most of all when the decision is about money rather than counts (next section). The
`PAY_0 >= 1` row is computed from `samples.csv` (snippet in FINDINGS.md), and the 10% row is the
package's own benchmark.

So the recommendation is not simply to deploy the model. At 10% capacity I would run the simple
rule, which is cheaper to operate and easy to explain, and use the model if capacity grows or if
the goal is to reduce losses.

### Cost policy: one cut-off or a per-account rule?

Assume that acting on an account that would have paid costs `cost_fp` (lost margin, goodwill,
staff time), that a missed default loses `LGD × current bill`, and that acting prevents the loss.
These are **illustrative assumptions**. Each lender estimates loss given default for cards from
its own recovery data, so no regulatory figure is used. Two rules apply them:

- **One cut-off** on the score, re-optimized on validation for each pair of costs.
- **Per-account expected loss**: act when `p × LGD × balance > cost_fp × (1 − p)`, with `p` the
  calibrated probability. It has nothing to fit, because the costs define it, and it uses each
  account's balance, which a single cut-off ignores.

"Saving" compares each rule with the better of acting on nobody and acting on everybody (test
split):

| cost of a false alarm | LGD | one cut-off: flagged / saving | per account: flagged / saving |
|---|---|---|---|
| 2,000 | 0.3 | 85.5% / 8.3% | 40.9% / 49.0% |
| 2,000 | 0.6 | 85.5% / −0.9% | 54.6% / 37.1% |
| 2,000 | 0.9 | 93.5% / 4.4% | 60.9% / 31.2% |
| 8,000 | 0.3 | 14.3% / 35.2% | 15.3% / 50.7% |
| 8,000 | 0.6 | 18.9% / 43.1% | 24.3% / 63.9% |
| 8,000 | 0.9 | 25.7% / 22.4% | 33.2% / 55.1% |
| 32,000 | 0.3 | 10.2% / 5.3% | 4.8% / 24.6% |
| 32,000 | 0.6 | 12.7% / 22.2% | 9.2% / 35.5% |
| 32,000 | 0.9 | 13.5% / 31.2% | 12.7% / 44.9% |

The per-account rule saves more in all nine scenarios. At the base case (8,000, 0.6) the
savings are 43.1% [35.8%, 48.2%] for the cut-off and 63.9% [60.0%, 66.2%] for the per-account
rule, and the difference of 20.8 points [15.2, 27.7] is NT$7.8 million [5.4, 10.3] on the 6,002
test accounts. So the advantage is not a product of the sample. The per-account rule reaches
**fewer** defaulters than the cut-off (45.7% vs 49.5%) and still saves more, because it
concentrates on large balances (the median current bill of flagged accounts is NT$90,289,
against NT$25,487 under the cut-off). When losses scale with the balance, the number of
defaulters caught is the wrong thing to maximize. With one cut-off, cheap action means acting on
almost everyone and the model barely matters (−0.9% at 2,000 and 0.6), while the per-account
rule still finds a 37% saving there.

The trade-off a manager has to accept is that the per-account rule acts on **646 customers in
good standing** (not late last month), 100 of whom then defaulted. Acting on current customers
costs goodwill that the cost figure may not capture and, for some actions, brings legal duties
(see the section on reason codes).

### Probabilities: how many defaults and how much loss to plan for

Ranking decides *which* accounts to act on, but planning needs *how many*. The random forest's
probabilities were checked on the validation split by 5-fold cross-validated log loss. A
sigmoid correction improved the log loss by 0.0003, well inside one standard error (0.0036), so
no correction is applied (the one-standard-error rule). On test:

| | expected | occurred | forecast error [95% interval] |
|---|---|---|---|
| defaults among all 6,002 clients | 1,337 | 1,332 | +0.4% [−3.5%, +4.6%] |
| defaults among the 599 flagged at 10% | 414 | 419 | −1.1% [−6.0%, +4.6%] |
| defaults among the 1,461 flagged by the per-account rule | 593 | 609 | −2.7% [−7.6%, +2.5%] |
| loss (0.6 × current bill of defaulters) | NT$36.8 million | NT$37.8 million | −2.7% [−9.6%, +4.6%] |

The expected calibration error is 0.016. None of the forecast errors is distinguishable from
zero, but their ranges differ. The default count is good to within about 5%, while the loss
forecast is only good to within about 10% because a few large balances carry much of the loss,
so finance should be given the wider range.

Calibration matters most when a model is *not* calibrated. A common way to handle class
imbalance, `class_weight="balanced"`, inflates the probabilities: on the same data such a
random forest forecasts **2,498 defaults when 1,332 occurred**. The calibration step selects a
sigmoid correction and brings the forecast to 1,296 (error 0.194 → 0.015) without changing the
ranking (`python scripts/check_calibration.py`). Calibration is monotone, so it never changes
which accounts a cut-off rule flags. It changes the numbers used for planning and the
per-account rule.

### Reason codes: why was this account flagged?

For each flagged account the package names up to four factors, such as "Repayment status last
month" or "Bill amounts over the last six months". Each factor is set in turn to the typical
values of accounts that were *just not* flagged, the account is scored again, and the factors
that lower its score the most are its reasons. This follows the first method in the Regulation
B official interpretation (comment 9(b)(2)-5: the factors on which the applicant fell furthest
below applicants at or slightly above the passing score), and four is the number the
interpretation calls helpful (comment 9(b)(2)-1). Correlated inputs (six bills, six payments)
are grouped into one factor so that their joint effect is not understated.

Because this approximates the model, its accuracy is measured, as the Consumer Financial
Protection Bureau expects for complex models (Circular 2022-03):

| rule | flagged | stated reasons change the decision | top reason alone | primary reason |
|---|---|---|---|---|
| 10% budget | 599 | 100% | 82.1% | repayment status last month (82.6%) |
| one cut-off, cost | 1,133 | 98.0% | 64.9% | last month's status (51.4%), earlier status (39.8%) |
| per-account, cost | 1,461 | 95.8% | 65.0% | bill amounts (45.2%), last month's status (31.8%) |

Setting an account's top reason to typical values lowers its score by 0.275 on average, against
0.078 for a randomly chosen factor (10% budget).

**Do we owe an explanation?** Under Regulation B, an unfavorable change in the terms of an
account (a lower limit, for example) that does not apply to substantially all accounts of its
class is adverse action and requires a notice with specific reasons. But actions taken *in
connection with a delinquency* on that account are excluded (12 CFR 1002.2(c)). That makes the
share of flagged accounts that are already late important:

| rule | flagged | already late last month | in good standing |
|---|---|---|---|
| 10% budget | 599 | 99.5% | 3 |
| one cut-off, cost | 1,133 | 87.6% | 141 |
| per-account, cost | 1,461 | 55.8% | 646 |

At a 10% budget the rule is in practice a delinquency rule, and notices would rarely be
required. The money-saving per-account rule acts on 646 current customers. If the action is a
limit cut, each of them needs a notice with specific reasons, and the reason codes above are
what make that rule usable. A compliance team, not this package, decides which actions count. I
am not a lawyer, and the regulation's text is the reference.

Reason codes are also useful where no notice is required, since a collections agent calling a
flagged customer needs to know what to talk about.

### Does tuning help?

`--tune` searches each model family's hyperparameters by 5-fold group cross-validation inside
the training split (validation and test are not touched), and adds gradient boosting:

| model | default CV AUC | tuned CV AUC (sd) | gain |
|---|---|---|---|
| logistic regression | 0.7440 | 0.7453 (0.0136) | +0.0013 |
| random forest | 0.7769 | 0.7807 (0.0094) | +0.0038 |
| gradient boosting | 0.7736 | 0.7804 (0.0095) | +0.0068 |

Every gain is within one standard deviation, and well below the width of the test AUC intervals
(about 0.03). Validation then picks gradient boosting, which on test matches the default random
forest (ROC AUC 0.782 for both) and reaches 31.8% of defaulters at 70.3% precision at the 10%
budget, against 31.5% and 69.9%. Within the tuned run, gradient boosting and the tuned random
forest differ by −0.001 [−0.005, +0.002] in test AUC. The search took about 3.5 minutes on two
CPU cores, 3 of them for the random forest.

These gradient boosting figures need scikit-learn 1.8 or later. scikit-learn 1.8 changed its
histogram gradient boosting, so the same data, settings and seed give a different model with
1.7, the last version for Python 3.10. There, gradient boosting gains +0.0071 (0.7737 to 0.7808,
sd 0.0093), is again picked on validation, and scores test ROC AUC 0.783, reaching 31.4% of
defaulters at 70.0% precision; it differs from the tuned random forest by +0.000 [−0.003,
+0.003]. The other two rows, and every run without `--tune`, are the same on both versions. The model is limited by the information in the data, not by its settings, and a
new model family would also need its own validation work. I would keep the defaults and spend
the effort on new data (bureau data, transaction histories) instead.

### Fairness audit (test split, 10% budget)

Sex, marital status and age are not model inputs, but outcomes are audited by group:

| group | n | default rate | flagged | non-defaulters flagged | adverse impact ratio |
|---|---|---|---|---|---|
| female | 3,623 | 21.2% | 9.6% | 3.4% | 1.00 |
| male | 2,379 | 23.7% | 10.6% | 4.5% | 0.99 |
| married | 2,674 | 24.0% | 10.1% | 3.6% | 0.94 |
| single | 3,241 | 20.9% | 10.0% | 3.9% | 0.94 |
| other marital status | 72 | 18.1% | 4.2% | 3.4% | 1.00 |
| age < 25 | 575 | 27.7% | 15.1% | 5.8% | 0.93 |
| age 25–34 | 2,610 | 21.4% | 9.6% | 4.1% | 0.99 |
| age 35–44 | 1,796 | 21.3% | 8.9% | 3.1% | 1.00 |
| age 45–61 | 978 | 22.6% | 10.0% | 3.4% | 0.99 |

The undocumented marital-status code "0" (15 people) and the age group 62+ (43) are in the
CSV but marked too small to interpret, since the audit treats groups under 50 as too small. The
"0" group's ratio is 0.77, below 0.8, but on 15 people it says nothing reliable. The "other"
marital group (72 people) is reported, though its figures are also noisy. The adverse impact
ratio divides each group's rate of *not* being flagged by the highest group's rate. All groups
large enough to interpret are above
0.8, the "four-fifths" reference from the US Uniform Guidelines on Employee Selection
Procedures (29 CFR 1607.4(D)). The guidelines say a rate below this will "generally" be
regarded as evidence of adverse impact, and that smaller differences can still count when they
are significant in statistical and practical terms, so it is a reference point, not a
definitive test.

The proxy check is the more important finding. The model's inputs predict whether a customer is
**under 25 with ROC AUC 0.82** (sex: 0.55; marital status: about 0.63), so excluding age did not
remove the age information. Under-25s are flagged more often (15.1%), and more of their
non-defaulters are flagged (5.8% vs 3.1% to 4.5%). Part of that follows from their higher
default rate, but a lender would look into which inputs carry the age signal (credit limit is a
likely candidate, since it shows the largest shift for young customers in the drift check) and
get a fair-lending review before deployment. These are descriptive statistics, not a legal
determination.

The decision rule changes the picture. Under the per-account cost rule all interpretable groups
stay above 0.8, but the pattern moves: married customers fall to 0.84 (from 0.94), and
under-25s go from the most-flagged age group to the least-flagged (22.3% vs 23.1% to 25.2%),
because the rule follows balances and young customers have smaller ones. A fairness review has
to be repeated whenever the decision rule changes, not only when the model does.

## 6. Monitoring after deployment

- **Every scoring run** (`creditdecide predict`) checks the schema (missing columns, missing or
  infinite values, unseen categories) and then computes the population stability index (PSI)
  of every input and of the score against the validation sample saved with the model. The usual
  convention is: below 0.10 stable, 0.10 to 0.25 check other metrics, above 0.25 do not rely on
  the scores until reviewed.
- **Example:** scoring only clients under 30 flags a moderate shift in credit limit (PSI 0.146)
  and changes the flag rate to 12.0%. When their outcomes are known, `predict` on the labeled
  file reports a capture of 35.5% at 69.3% precision, against 31.5% and 69.9% at training.
- **Outcomes analysis:** defaults are observed one month after scoring. Running `predict` again
  on the scored month once `DEFAULT` is known compares realized capture, precision **and
  calibration** (expected vs actual defaults, calibration error) with the model card. A new
  month has its own sampling noise on top of the uncertainty in the test estimate, so for a
  month of about 6,000 accounts the expected range of the capture rate is wider than the test
  interval: about 28% to 35% (31.5% ± 1.96 × √2 × 1.3 points). Calibration usually degrades
  before ranking does, so the planning figures need rechecking first.
- **A limitation found while testing:** multiplying every bill and limit by 1.5 gave a PSI of
  0.80 for the credit limit but only about 0.07 for the bill amounts. Decile-based PSI is coarse
  for heavy-tailed variables, so it should be used together with simple summary statistics
  (medians, quantiles) and the score PSI, not on its own.
- **Retraining triggers to agree with the business:** a score PSI above 0.25, a realized capture
  rate outside the expected range for a month of that size (28% to 35% for about 6,000
  accounts), expected and actual defaults differing by more than an agreed tolerance, or a
  change in products, pricing or customer base. A single month just outside the range is a
  reason to look closer, not proof that the model has broken.

## 7. Governance

The US Federal Reserve and OCC guidance on model risk management, SR 11-7 (2011), names three
validation elements. Coverage here:

| SR 11-7 element | Covered | Not covered |
|---|---|---|
| Conceptual soundness | documented data checks, design choices, baselines, simple-rule challenger, tuning inside training data only, selection on validation only, model card | independent review |
| Ongoing monitoring | schema checks and PSI at every scoring run; calibration and outcomes when labels arrive | automated alerting, a live challenger model |
| Outcomes analysis | test-split results with bootstrap intervals; realized-outcome and calibration comparison when labels arrive | back-testing over many months |

For adverse decisions, reason codes with measured accuracy are provided (Regulation B, CFPB
Circular 2022-03). Whether a given action is adverse action is a compliance decision.

## 8. Glossary

- **Capture rate (recall):** the share of all defaulters that the rule flags.
- **Precision:** the share of flagged accounts that default.
- **Lift:** precision divided by the overall default rate; 3 means three times better than
  random selection.
- **Gains table:** capture rate as a function of how many top-ranked accounts are reviewed.
- **Exposure, LGD:** the balance at risk, and the fraction of it lost if the customer defaults.
- **Calibration:** whether predicted probabilities match observed rates (of the accounts given
  20%, about 20% default).
- **Expected calibration error (ECE):** the average gap between predicted probability and
  observed rate across ten equal-size groups of accounts.
- **Brier score, log loss:** accuracy measures for probabilities; lower is better.
- **One-standard-error rule:** among options whose cross-validated score is within one standard
  error of the best, choose the simplest.
- **Bootstrap interval:** the range that covers the middle 95% of a figure when the test
  accounts are redrawn with replacement many times. For a difference, an interval that
  excludes zero means the sign of the difference is clear.
- **Reason code:** a factor named as a principal reason for a decision about an account.
- **Adverse action:** under Regulation B, for example a credit denial, or an unfavorable change
  in an account's terms that is not applied to substantially all accounts of its class; actions
  in connection with a delinquency are excluded.
- **PSI:** population stability index, a binned measure of the difference between a reference
  distribution and a new one.
- **Adverse impact ratio:** a group's favorable-outcome rate divided by the highest group's.
- **Proxy:** an input that carries information about a protected attribute.
- **Champion-challenger:** comparing the live model against an alternative (here, the best
  one-feature rule) on the same accounts.

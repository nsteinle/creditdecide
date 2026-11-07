# Bundled data

`src/creditdecide/data/credit_default.csv.gz`

- **Title:** Default of Credit Card Clients
- **Creator:** I.-C. Yeh
- **Source:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **DOI:** 10.24432/C55S3H
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- **Reference:** Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients. *Expert Systems
  with Applications*, 36(2), 2473–2480.

## Changes made

Produced from the official archive (SHA-256
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`) by
`scripts/build_credit_default.py`:

1. the first header row (`X1` … `X23`, `Y`) was dropped, keeping the descriptive column names;
2. the column `default payment next month` was renamed `DEFAULT`;
3. the table was stored as gzip-compressed CSV.

No values were changed, added or removed (30,000 rows, 25 columns). The SHA-256 of the
bundled file is recorded in `credit_default.json` and checked on every load.

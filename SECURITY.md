# Security

Please report vulnerabilities privately to nsastrophysics@gmail.com rather than in a public
issue. I aim to reply within a week.

What the package does to stay safe:

- There is no network access at run time, and the bundled dataset's SHA-256 is checked every
  time it is loaded.
- Saved models: the SHA-256 of every file is recorded in `manifest.json` and checked on load.
  The scikit-learn pipeline is stored in the skops format, and loading refuses any type outside
  a three-item allow-list, and any file the manifest does not expect. Calibration maps are
  stored as plain numbers in the manifest. No pickle files are written or read. Only load model
  directories from sources you trust.
- SQL runs in an in-memory DuckDB database inside the Python process, over tables the package
  registers itself. External access (files, network) and extension auto-loading are switched
  off for every connection, and column names are quoted, so a column name cannot change a
  query.
- User CSV files are checked (named columns present, numeric features, finite values, explicit
  handling of missing values, class counts) before any model runs.
- Output is written only to the directory you name, and existing files are not overwritten
  without `--overwrite`.
- Releases are published from GitHub Actions with PyPI Trusted Publishing, so no tokens are
  stored.

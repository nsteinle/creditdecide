# Releasing creditdecide

Releases go from GitHub Actions to PyPI with **Trusted Publishing**. PyPI trusts a specific
workflow in a specific repository, so there is no API token that could leak. Test on TestPyPI
first, because PyPI never accepts the same version number twice, even after a deletion.

## 0. Before the repository is public

Turn on two-factor authentication on GitHub, PyPI and TestPyPI.

## 1. GitHub repository

```bash
cd creditdecide
git remote add origin git@github.com:nsteinle/creditdecide.git   # create the empty repo first
git push -u origin main
```

Check the **Actions** tab. `ci` must pass (tests on Python 3.10, 3.12 and 3.13, the full-data
run, `twine check` and the Docker build).

Then in **Settings → Environments**, create `testpypi` and `pypi`. On `pypi`, add yourself as
a required reviewer, so that a release waits for your approval before uploading.

## 2. TestPyPI

1. https://test.pypi.org → Account settings → Publishing → *Add a new pending publisher*:
   project `creditdecide`, owner `nsteinle`, repository `creditdecide`, workflow `publish.yml`,
   environment `testpypi`.
2. GitHub → Actions → `publish` → *Run workflow* (branch `main`). The `testpypi` job uploads.
3. Verify in a fresh environment:

```bash
python -m venv /tmp/lc && source /tmp/lc/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ "creditdecide[plot]==0.1.0"
creditdecide datasets
creditdecide run --dataset credit-default --out /tmp/lc-check --plot
creditdecide predict --model-dir /tmp/lc-check/model --csv /tmp/lc-check/holdout_test.csv \
    --out /tmp/lc-scored --id-column ID --reasons 4
deactivate
```

`--extra-index-url` lets pip get NumPy, DuckDB and the other dependencies from the real PyPI.
To test again, raise the version (for example to `0.1.0rc2`) in
`src/creditdecide/_version.py` and `CITATION.cff`.

## 3. PyPI

1. https://pypi.org → Publishing → *Add a new pending publisher* with the same details but
   environment `pypi`.
2. Make sure the version in `src/creditdecide/_version.py` is the one you want (`0.1.0`) and
   is committed on `main`.
3. GitHub → Releases → *Draft a new release* → tag `v0.1.0` (it must equal the package
   version, or the workflow stops) → *Publish release*.
4. Approve the `pypi` environment when GitHub asks.
5. Verify: `pip install "creditdecide[plot]"` in a fresh environment, then the commands above.

## 4. Every later release

1. Update the version in `src/creditdecide/_version.py` and `CITATION.cff` and add a
   `CHANGELOG.md` entry.
2. Push to `main`; wait for `ci`.
3. Optionally, run TestPyPI with a pre-release version (for example `0.1.1rc1`).
4. Publish a GitHub release tagged `v<version>`.

## Rebuilding the bundled data (maintainers only)

Download the official archive yourself, then:

```bash
pip install xlrd
python scripts/build_credit_default.py "default+of+credit+card+clients.zip"
```

The script refuses any archive whose SHA-256 differs from the verified one, and it reproduces
the bundled file byte for byte.

## Operating a saved model (batch scoring)

A saved model is a directory (`<run>/model`). To use it on a new month of data:

```bash
creditdecide predict --model-dir model --csv accounts_2026_11.csv --out scored_2026_11 \
    --id-column ID
```

1. Inputs are checked against the schema in `manifest.json`. Missing columns, missing or
   infinite values and non-numeric values stop the run, and unseen categories give warnings.
2. `scores.csv` holds the score, the calibrated probability and the 0/1 flag for each row, and
   `drift.csv` holds the PSI of every input and of the score. Look into it before acting on the
   scores if any PSI is above 0.25. Add `--reasons 4` to write reason codes for flagged rows.
3. When the outcome is known (one month later for this target), run `predict` again on the same
   rows with the `DEFAULT` column included. `outcomes.json` compares the realized capture,
   precision and calibration (expected vs actual defaults) with the model card.
4. Retrain when the score PSI is above 0.25, the realized capture falls outside the range
   expected for a month of that size (BUSINESS_CASE.md, section 6), expected and actual
   defaults differ by more than an agreed tolerance, or the product or customer base changes.
   Keep each model directory as a fixed version and don't edit it.

With Docker (the image runs as a non-root user, so pass your user ID to let it write to the
mounted folder):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" creditdecide \
    predict --model-dir model --csv new.csv --out scored
```

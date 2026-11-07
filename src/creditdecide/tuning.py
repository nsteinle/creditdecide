"""Hyperparameter tuning on the training split only.

For each supervised scoring model family, a random sample of candidates from a small,
documented grid is scored by ROC AUC (the same criterion used to choose between families on
validation) with 5-fold stratified *group* cross-validation inside the training split: rows with
identical feature vectors stay in the same fold, and preprocessing is refitted inside every fold.
The library defaults are evaluated on the same folds, so the report shows what tuning bought.
The best configuration of each family is then refitted on the whole training split.

The folds (``group_folds``) and the candidates (``sample_candidates``) are drawn here rather than
by scikit-learn, with NumPy's legacy ``RandomState``, whose output NumPy keeps fixed across
versions. scikit-learn 1.8 changed how ``StratifiedGroupKFold`` assigns folds, so with it the
search, and the model it picks, depended on the installed scikit-learn.

The validation split (model selection, calibration, decision threshold) and the test split
(final report) are never seen by the search.

Families: logistic regression (regularization strength), random forest (depth, leaf size,
features per split) and histogram gradient boosting (learning rate, tree size, leaf size, L2
penalty; boosting rounds by early stopping on an internal 10% of each training fold).
Gradient boosting is only trained when tuning is requested, so default runs are unchanged.

Inside the search each random forest runs on one thread, and the folds and candidates run in
parallel instead (``n_jobs=-1``). A multi-threaded forest's scores can differ in the last bit
between runs (see creditdecide.forest), so this keeps the cross-validated AUCs, and therefore
the chosen settings, the same on every run. The best configuration is refitted with the
family's default ``n_jobs``.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

N_FOLDS = 5


def group_folds(y, groups, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified group k-fold: every group in one fold, label mix kept close to overall.

    Each group gets its most frequent label (the smallest on a tie). Groups are visited in a
    seeded random order and each goes to the fold that holds the fewest rows of its label so far
    (the lowest-numbered fold on a tie). Returns (train, test) row indices per fold."""
    y, groups = np.asarray(y), np.asarray(groups)
    _, g = np.unique(groups, return_inverse=True)
    _, lab = np.unique(y, return_inverse=True)
    counts = np.zeros((g.max() + 1, lab.max() + 1), int)
    np.add.at(counts, (g, lab), 1)
    stratum, size = counts.argmax(axis=1), counts.sum(axis=1)
    filled = np.zeros((counts.shape[1], n_splits), int)
    fold_of = np.empty(len(size), int)
    for k in np.random.RandomState(seed).permutation(len(size)):
        f = int(np.argmin(filled[stratum[k]]))
        fold_of[k] = f
        filled[stratum[k], f] += size[k]
    fold = fold_of[g]
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(n_splits)]


def sample_candidates(space: dict, n_iter: int, seed: int) -> list[dict]:
    """``n_iter`` distinct points of the grid ``space`` (all of it if smaller), in grid order.
    The grid is enumerated with the keys sorted, the last key varying fastest."""
    keys = sorted(space)
    grid = [dict(zip(keys, values, strict=True))
            for values in itertools.product(*(space[k] for k in keys))]
    pick = np.sort(np.random.RandomState(seed).permutation(len(grid))[:min(n_iter, len(grid))])
    return [grid[i] for i in pick]


def search_spaces(seed: int, rf_n_estimators: int) -> dict:
    """(default estimator, parameter space) per family."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000),
            {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=10,
                                   random_state=seed, n_jobs=-1),
            {"max_depth": [6, 8, 10, 14, None], "min_samples_leaf": [1, 5, 20, 50],
             "max_features": ["sqrt", 0.3, 0.5]},
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20,
                                           random_state=seed),
            {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
             "min_samples_leaf": [20, 50, 100, 200], "l2_regularization": [0.0, 0.1, 1.0, 10.0]},
        ),
    }


def tune(df_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray, preprocessor,
         seed: int, n_iter: int = 12, rf_n_estimators: int = 200
         ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Returns (best unfitted estimator per family, summary table, full search results)."""
    folds = group_folds(y_train, groups, N_FOLDS, seed)
    best, summary, details = {}, [], []
    for family, (default, space) in search_spaces(seed, rf_n_estimators).items():
        t0 = time.time()
        model = clone(default)
        if isinstance(model, RandomForestClassifier):
            model.set_params(n_jobs=1)
        pipe = Pipeline([("pre", clone(preprocessor)), ("model", model)])
        base = cross_val_score(pipe, df_train, y_train, scoring="roc_auc", cv=folds, n_jobs=-1)
        n_grid = int(np.prod([len(v) for v in space.values()]))
        candidates = [{f"model__{k}": [v] for k, v in c.items()}
                      for c in sample_candidates(space, n_iter, seed)]
        search = GridSearchCV(pipe, candidates, scoring="roc_auc", cv=folds, refit=False,
                              n_jobs=-1)
        search.fit(df_train, y_train)
        params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        best[family] = clone(default).set_params(**params)
        i = search.best_index_
        summary.append({
            "model": family,
            "default_cv_auc": float(base.mean()), "default_cv_auc_sd": float(base.std()),
            "tuned_cv_auc": float(search.cv_results_["mean_test_score"][i]),
            "tuned_cv_auc_sd": float(search.cv_results_["std_test_score"][i]),
            "best_params": params, "candidates": min(n_iter, n_grid), "grid_size": n_grid,
            "seconds": round(time.time() - t0, 1),
        })
        res = pd.DataFrame(search.cv_results_)
        keep = [c for c in res.columns if c.startswith("param_")] + [
            "mean_test_score", "std_test_score", "rank_test_score"]
        details.append(res[keep].assign(model=family))
    table = pd.DataFrame(summary)
    table["gain"] = table["tuned_cv_auc"] - table["default_cv_auc"]
    return best, table, pd.concat(details, ignore_index=True)

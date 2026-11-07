"""Verification: does calibration repair a model whose probabilities are known to be wrong?

The bundled random forest is already well calibrated, so the default run cannot show the
calibration step doing anything. This script trains the same random forest with
class_weight="balanced" (a common way to handle class imbalance, which inflates predicted
probabilities), then applies the package's calibration to it on the same splits.

    python scripts/check_calibration.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from creditdecide import CREDIT_DEFAULT_SPEC, Config, load_credit_default
from creditdecide.calibration import apply_map, calibration_metrics, choose_and_fit
from creditdecide.forest import predict_on_one_thread
from creditdecide.pipeline import group_split
from creditdecide.preprocess import make_preprocessor


def main() -> None:
    df, spec, cfg = load_credit_default(), CREDIT_DEFAULT_SPEC, Config(seed=0)
    y = df[spec.target].to_numpy()
    sp = group_split(df, spec.feature_columns(df), y, cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    pre = make_preprocessor(df, spec).fit(df.iloc[tr])
    X = pre.transform(df)
    rf = predict_on_one_thread(RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=0,
        n_jobs=-1).fit(X[tr], y[tr]))
    s = rf.predict_proba(X)[:, 1]
    params, table = choose_and_fit(s[va], y[va])
    print(table.round(4).to_string(index=False))
    print(f"chosen: {params['method']}")
    for name, p in (("raw (class-weighted)", s[te]), ("calibrated", apply_map(params, s[te]))):
        m = calibration_metrics(y[te], p)
        print(f"{name:22s} ECE {m['ece']:.3f}  Brier {m['brier']:.4f}  log loss "
              f"{m['log_loss']:.4f}  expected defaults {m['expected_positives']:,.0f} "
              f"(actual {m['actual_positives']:,})")
    order_kept = np.array_equal(np.argsort(s[te], kind="stable"),
                                np.argsort(apply_map(params, s[te]), kind="stable"))
    print(f"ranking unchanged by calibration: {order_kept}")


if __name__ == "__main__":
    main()

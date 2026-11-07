"""Figures. Requires: pip install "creditdecide[plot]". No LaTeX needed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_roc_pr(samples: pd.DataFrame, target: str, path: str | Path,
                split: str = "test") -> Path:
    """ROC and precision-recall curves for the probabilistic models (binary targets)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt = _plt()
    s = samples[samples["split"] == split]
    y = s[target].to_numpy()
    pos = sorted(np.unique(y))[1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for col in [c for c in s.columns if c.startswith("p_")]:
        fpr, tpr, _ = roc_curve(y == pos, s[col])
        prec, rec, _ = precision_recall_curve(y == pos, s[col])
        a1.plot(fpr, tpr, label=col[2:])
        a2.plot(rec, prec, label=col[2:])
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    a2.axhline(np.mean(y == pos), color="k", ls="--", lw=1, label="prevalence")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC ({split})")
    a2.set(xlabel="recall", ylabel="precision", title=f"precision-recall ({split})")
    for a in (a1, a2):
        a.legend(fontsize=9)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reliability(table: pd.DataFrame, path: str | Path) -> Path:
    """Reliability diagram: mean predicted probability vs observed rate per equal-count bin,
    for the raw and the calibrated scores (creditdecide.calibration.reliability_table)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, (name, t) in enumerate(table.groupby("scores", sort=False)):
        ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", color=f"C{i}", label=name)
    hi = float(max(table["mean_predicted"].max(), table["observed_rate"].max())) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="perfect calibration")
    ax.set(xlabel="mean predicted probability", ylabel="observed rate", xlim=(0, hi),
           ylim=(0, hi), title="reliability (test split, 10 equal-count bins)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

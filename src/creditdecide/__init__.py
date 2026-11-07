"""creditdecide: from a labeled table to a monitored, explainable decision rule.

Starts with a data-quality report, then trains a logistic regression and a random forest
(optionally tuned, which adds gradient boosting). The best model is benchmarked against the best
one-feature rule, its probabilities are calibrated and turned into a budget- or cost-based
decision rule, every flagged row gets reason codes, the test results get bootstrap confidence
intervals, and the model is saved as a checksum-verified directory with drift monitoring and a
fairness audit. The data-quality checks, gains table, outcomes join and drift bins are DuckDB
SQL (creditdecide.sql), and ``creditdecide tableau`` writes a run's results as Tableau-ready
tables.
"""

from creditdecide._version import __version__
from creditdecide.datasets import (
    CREDIT_DEFAULT_SPEC,
    FeatureSpec,
    load_credit_default,
    load_csv,
)
from creditdecide.decision import DecisionPolicy
from creditdecide.modelio import ScoringModel, load_model
from creditdecide.pipeline import MODELS, SCORING_MODELS, Config, Result, run
from creditdecide.quality import quality_report

__all__ = [
    "CREDIT_DEFAULT_SPEC",
    "MODELS",
    "SCORING_MODELS",
    "Config",
    "DecisionPolicy",
    "FeatureSpec",
    "Result",
    "ScoringModel",
    "__version__",
    "load_credit_default",
    "load_csv",
    "load_model",
    "quality_report",
    "run",
]

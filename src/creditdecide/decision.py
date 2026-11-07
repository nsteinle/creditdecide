"""Turn model scores into decisions.

A score only ranks accounts. A lender still has to choose which accounts to act on (review,
call, lower a limit, decline), and that choice depends on capacity or on what each kind of
mistake costs. Two policies are supported, and both pick the operating threshold on the VALIDATION
split so the test split measures what would happen in deployment:

``budget``  act on the highest-scoring fraction of accounts (e.g. the review team can handle
            10% of accounts a month). Needs no cost assumptions.
``cost``    act wherever it minimizes expected cost. A false alarm costs ``cost_fp``; a missed
            positive costs ``cost_fn`` (a constant) or ``lgd * exposure`` per account (loss given
            default times the balance at risk). True positives and true negatives cost 0, so any
            fixed handling cost of a true positive should be netted into the two inputs.

Cost inputs are business assumptions, not facts; ``cost_sensitivity`` shows how the decision
changes as they vary.

Two decision rules apply a cost policy:

``threshold``      one score cut-off for every account, chosen on validation to minimize cost
                   (the default, and the only rule for a budget policy);
``expected_loss``  per account: act when the expected loss avoided exceeds the expected cost of
                   acting, p * missed_cost > cost_fp * (1 - p), with p a CALIBRATED probability
                   (creditdecide.calibration). Uses each account's own exposure, which a single
                   score cut-off ignores; only as good as the calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecisionPolicy:
    kind: str = "budget"            # "budget" or "cost"
    budget: float | None = 0.10     # fraction of accounts acted on (budget policy)
    cost_fp: float | None = None    # cost of acting on an account that would not default
    cost_fn: float | None = None    # constant cost of a missed default ...
    lgd: float | None = None        # ... or loss given default (0-1) times exposure_column
    exposure_column: str | None = None

    def __post_init__(self):
        if self.kind == "budget":
            if self.budget is None or not 0 < self.budget < 1:
                raise ValueError("budget must be a fraction strictly between 0 and 1")
        elif self.kind == "cost":
            if self.cost_fp is None or self.cost_fp < 0:
                raise ValueError("cost policy needs cost_fp >= 0")
            per_row = self.lgd is not None or self.exposure_column is not None
            if per_row == (self.cost_fn is not None):
                raise ValueError("give either cost_fn, or both lgd and exposure_column")
            if per_row and (self.lgd is None or self.exposure_column is None
                            or not 0 < self.lgd <= 1):
                raise ValueError("lgd (0 < lgd <= 1) and exposure_column go together")
            if self.cost_fn is not None and self.cost_fn < 0:
                raise ValueError("cost_fn must be >= 0")
        else:
            raise ValueError("kind must be 'budget' or 'cost'")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def missed_cost(self, df: pd.DataFrame | None, n: int) -> np.ndarray:
        """Per-row cost of missing a positive."""
        if self.kind != "cost":
            return np.zeros(n)
        if self.cost_fn is not None:
            return np.full(n, float(self.cost_fn))
        if df is None or self.exposure_column not in df:
            raise ValueError(f"exposure column {self.exposure_column!r} not available")
        return self.lgd * np.clip(df[self.exposure_column].to_numpy(float), 0, None)


def _confusion(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def expected_cost(y, flag, cost_fp: float, missed: np.ndarray) -> float:
    y = np.asarray(y)
    return float(cost_fp * np.sum(flag & (y == 0)) + np.sum(missed[~flag & (y == 1)]))


def choose_threshold(y_val, score_val, policy: DecisionPolicy,
                     missed_val: np.ndarray | None = None) -> float:
    """Operating threshold chosen on validation data: flag accounts with score >= threshold."""
    y_val, s = np.asarray(y_val), np.asarray(score_val, float)
    if policy.kind == "budget":
        return float(np.quantile(s, 1 - policy.budget, method="higher"))
    missed = missed_val if missed_val is not None else np.full(len(s), float(policy.cost_fn))
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted, m_sorted = s[order], y_val[order], missed[order]
    # cost(k) for flagging the k highest scores = cost_fp * (negatives among them)
    #                                              + missed cost of positives not flagged
    fp_cum = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    saved = np.concatenate([[0], np.cumsum(np.where(y_sorted == 1, m_sorted, 0.0))])
    costs = policy.cost_fp * fp_cum + (saved[-1] - saved)
    # Candidate thresholds: every distinct score, plus "flag nobody": the smallest number above
    # every possible score (scores are probabilities, so above 1); finite, so valid JSON.
    nobody = float(np.nextafter(max(1.0, float(s.max())), np.inf))
    cands = np.unique(np.concatenate([s, [nobody]]))
    k = np.searchsorted(-s_sorted, -cands, side="right")  # number of scores >= candidate
    c = costs[k]
    best = np.flatnonzero(np.isclose(c, c.min(), rtol=1e-12, atol=1e-9))
    return float(cands[best].max())  # ties: the highest threshold (fewest accounts flagged)


def expected_loss_margin(p, missed, cost_fp: float) -> np.ndarray:
    """Expected loss avoided by acting minus expected cost of acting, per account."""
    p = np.asarray(p, float)
    return p * np.asarray(missed, float) - cost_fp * (1.0 - p)


def summarize(y, score, threshold: float | None, policy: DecisionPolicy | None = None,
              missed: np.ndarray | None = None, flag: np.ndarray | None = None) -> dict:
    """Outcome of applying a rule: volumes, capture rate, precision and, for a cost policy,
    expected cost against the two trivial strategies (flag nobody, flag everybody). Flags are
    ``score >= threshold`` unless given explicitly (expected-loss rule)."""
    y, s = np.asarray(y), np.asarray(score, float)
    flag = (s >= threshold) if flag is None else np.asarray(flag, bool)
    c = _confusion(y, flag)
    n, pos = len(y), int(np.sum(y == 1))
    n_flag = int(flag.sum())
    nan = float("nan")
    out = {
        "n": n,
        "threshold": None if threshold is None else float(threshold),
        "flagged": n_flag,
        "flagged_fraction": n_flag / n,
        "positives": pos,
        "capture_rate": c["tp"] / pos if pos else nan,  # recall of the positive class
        "precision": c["tp"] / n_flag if n_flag else nan,
        "false_alarm_rate": c["fp"] / (n - pos) if n > pos else nan,
        "lift": (c["tp"] / n_flag) / (pos / n) if n_flag and pos else nan,
        **c,
    }
    if policy is not None and policy.kind == "cost":
        m = missed if missed is not None else policy.missed_cost(None, n)
        cost = expected_cost(y, flag, policy.cost_fp, m)
        none = expected_cost(y, np.zeros(n, bool), policy.cost_fp, m)
        allf = expected_cost(y, np.ones(n, bool), policy.cost_fp, m)
        best_trivial = min(none, allf)
        out.update({
            "expected_cost": cost, "cost_per_account": cost / n,
            "cost_flag_nobody": none, "cost_flag_everybody": allf,
            "saving_fraction_vs_flag_nobody": (none - cost) / none if none else 0.0,
            # the fair benchmark: the better of the two strategies that need no model
            "saving_fraction_vs_best_trivial": ((best_trivial - cost) / best_trivial
                                                if best_trivial else 0.0),
        })
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}


def gains_table(y, score, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gains and lift by score decile (decile 1 = highest scores), computed in
    DuckDB SQL (creditdecide.sql, gains.sql)."""
    from creditdecide import sql

    return sql.gains(y, score, n_bins)


def cost_sensitivity(y_val, s_val, y_test, s_test, exposure_val, exposure_test,
                     cost_fps, lgds, p_test=None) -> pd.DataFrame:
    """Re-optimize the threshold on validation for each (cost_fp, lgd) pair and report the
    test outcome, to show how much the decision depends on the cost assumptions. With
    calibrated test probabilities ``p_test``, the expected-loss rule is reported alongside
    (it has nothing to re-optimize, since the costs define it)."""
    rows = []

    def row(rule, cfp, lgd, t, r):
        return {"rule": rule, "cost_fp": cfp, "lgd": lgd, "threshold": t,
                "flagged_fraction": r["flagged_fraction"],
                "capture_rate": r["capture_rate"], "precision": r["precision"],
                "saving_vs_flag_nobody": r["saving_fraction_vs_flag_nobody"],
                "saving_vs_best_trivial": r["saving_fraction_vs_best_trivial"]}

    for cfp in cost_fps:
        for lgd in lgds:
            pol = DecisionPolicy(kind="cost", budget=None, cost_fp=cfp, lgd=lgd,
                                 exposure_column="_exposure")
            mv = lgd * np.clip(np.asarray(exposure_val, float), 0, None)
            mt = lgd * np.clip(np.asarray(exposure_test, float), 0, None)
            t = choose_threshold(y_val, s_val, pol, mv)
            rows.append(row("threshold", cfp, lgd, t, summarize(y_test, s_test, t, pol, mt)))
            if p_test is not None:
                flag = expected_loss_margin(p_test, mt, cfp) > 0
                rows.append(row("expected_loss", cfp, lgd, None,
                                summarize(y_test, s_test, None, pol, mt, flag=flag)))
    return pd.DataFrame(rows)


def simple_rule_benchmark(df: pd.DataFrame, numeric: list[str], y: np.ndarray,
                          score: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
                          target_fraction: float) -> dict:
    """The challenger every scoring model should beat: a one-feature cut-off rule.

    For every numeric input and both directions, the cut-off whose validation flag rate is
    closest to ``target_fraction`` (the model's validation flag rate) is found. Among rules
    flagging within +/-50% of that volume, the one with the highest validation precision wins
    (precision, not capture, so a rule cannot win by flagging more accounts). It is evaluated
    once on the test split and compared with the model's ranking at the same test volume (its
    top accounts by score).
    """
    y = np.asarray(y)
    best = None
    for col in numeric:
        x = df[col].to_numpy(float)
        for sign in (1.0, -1.0):
            v = sign * x[val_idx]
            cuts = np.unique(v)
            frac = np.array([(v >= c).mean() for c in cuts])
            c = cuts[np.argmin(np.abs(frac - target_fraction))]
            flag_v = v >= c
            if abs(flag_v.mean() - target_fraction) > 0.5 * target_fraction:
                continue
            tp = (flag_v & (y[val_idx] == 1)).sum()
            prec_v = tp / flag_v.sum()
            if best is None or prec_v > best["validation_precision"]:
                best = {"column": col, "direction": ">=" if sign > 0 else "<=",
                        "cutoff": float(sign * c), "sign": sign,
                        "validation_flagged_fraction": float(flag_v.mean()),
                        "validation_precision": float(prec_v),
                        "validation_capture_rate": float(
                            tp / max((y[val_idx] == 1).sum(), 1))}
    if best is None:
        return {}
    yt = y[test_idx]
    flag_t = best["sign"] * df[best["column"]].to_numpy(float)[test_idx] >= best["sign"] * best[
        "cutoff"]
    n = int(flag_t.sum())
    top = np.zeros(len(test_idx), bool)
    top[np.argsort(-np.asarray(score)[test_idx], kind="stable")[:n]] = True
    pos = max(int(yt.sum()), 1)

    def stats(f):
        tp = int((f & (yt == 1)).sum())
        return {"flagged_fraction": float(f.mean()), "capture_rate": tp / pos,
                "precision": tp / f.sum() if f.sum() else float("nan")}

    best.pop("sign")
    rule, model = stats(flag_t), stats(top)
    return {"rule": f"{best['column']} {best['direction']} {best['cutoff']:g}", **best,
            "test_rule": rule, "test_model_same_volume": model,
            "capture_gain_of_model": model["capture_rate"] - rule["capture_rate"]}

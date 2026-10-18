"""Offline cascade simulation from cached per-item tables (no model calls).

Budget regime: escalate the fraction r of items with the lowest confidence.
Threshold regime: escalate items whose confidence is below c.
Both use expected values under random tie-breaking, like metrics.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import _tie_mean


def budget_curve(conf, small_correct, frontier_correct, rates=None) -> pd.DataFrame:
    conf = np.asarray(conf, float)
    s = np.asarray(small_correct, float)
    f = np.asarray(frontier_correct, float)
    n = len(conf)
    order = np.argsort(conf, kind="stable")  # least confident first
    key = conf[order]
    # Expected prefix sums of small/frontier correctness over the m least-confident items.
    S = np.concatenate([[0.0], np.cumsum(_tie_mean(key, s[order]))])
    F = np.concatenate([[0.0], np.cumsum(_tie_mean(key, f[order]))])
    ms = np.arange(n + 1)
    acc = (s.sum() - S + F) / n
    curve = pd.DataFrame({"escalated": ms, "call_rate": ms / n, "accuracy": acc})
    if rates is None:
        return curve
    pick = np.clip(np.round(np.asarray(rates) * n).astype(int), 0, n)
    return curve.iloc[pick].reset_index(drop=True)


def threshold_point(conf, small_correct, frontier_correct, c: float) -> dict:
    conf = np.asarray(conf, float)
    s = np.asarray(small_correct, float)
    f = np.asarray(frontier_correct, float)
    esc = conf < c
    kept = ~esc
    return {
        "threshold": c,
        "call_rate": float(esc.mean()),
        "retained_accuracy": float(s[kept].mean()) if kept.any() else float("nan"),
        "system_accuracy": float(np.where(esc, f, s).mean()),
    }


def oracle_accuracy(small_correct, frontier_correct) -> tuple[float, float]:
    """Best possible cascade: escalate exactly the items the small model gets wrong and the
    frontier gets right. Returns (accuracy, call_rate)."""
    s = np.asarray(small_correct, bool)
    f = np.asarray(frontier_correct, bool)
    esc = ~s & f
    return float((s | f).mean()), float(esc.mean())


def contingency(small_correct, frontier_correct) -> pd.DataFrame:
    s = pd.Series(np.asarray(small_correct, bool), name="small")
    f = pd.Series(np.asarray(frontier_correct, bool), name="frontier")
    t = pd.crosstab(s.map({True: "small right", False: "small wrong"}),
                    f.map({True: "frontier right", False: "frontier wrong"}))
    return t.reindex(index=["small right", "small wrong"], columns=["frontier right", "frontier wrong"],
                     fill_value=0)

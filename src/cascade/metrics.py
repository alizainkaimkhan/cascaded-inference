"""Calibration, discrimination and selective-prediction metrics.

Convention: `conf` is a confidence in the small model's answer (higher = more sure) and
`correct` is a 0/1 array for that answer. Ties in `conf` are common (the permutation signal
takes only 11 values), so every ranking metric here returns the *expected* value under
uniformly random tie-breaking rather than depending on sort order.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def accuracy(correct) -> float:
    return float(np.mean(np.asarray(correct, dtype=float)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def ece_equal_mass(conf, correct, n_bins: int = 15) -> float:
    """Top-label ECE with equal-mass (quantile) bins: sum_b (n_b/N) |acc_b - conf_b|.

    Items with identical confidence always share a bin (assigned by their average rank),
    so the result never depends on how ties happen to be ordered. With many ties some
    bins are larger than N/n_bins or empty."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    bins = np.minimum(((_average_ranks(conf) - 1) * n_bins / n).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / n)


def brier_multiclass(probs, gold_idx) -> float:
    """Mean over items of sum_k (p_k - 1[k = gold])^2, in [0, 2]."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), np.asarray(gold_idx)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_toplabel(conf, correct) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(correct, float)) ** 2))


# ---------------------------------------------------------------------------
# Ranking: AUROC, risk-coverage, AURC
# ---------------------------------------------------------------------------

def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(conf, correct) -> float:
    """P(conf of a random correct item > conf of a random wrong item), ties count 1/2."""
    conf = np.asarray(conf, float)
    pos = np.asarray(correct, bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _average_ranks(conf)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _tie_mean(sorted_keys: np.ndarray, sorted_vals: np.ndarray) -> np.ndarray:
    """Replace each value by the mean of its tie group. Prefix sums of the result are the
    expected prefix sums under random ordering within ties."""
    out = sorted_vals.astype(float).copy()
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    for g in np.split(np.arange(len(sorted_keys)), boundaries):
        out[g] = sorted_vals[g].mean()
    return out


def risk_coverage(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and selective risk (error rate among the k most confident), k = 1..n."""
    conf = np.asarray(conf, float)
    err = 1.0 - np.asarray(correct, float)
    order = np.argsort(-conf, kind="stable")
    e = _tie_mean(-conf[order], err[order])
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(e) / k


def aurc(conf, correct) -> float:
    return float(risk_coverage(conf, correct)[1].mean())


def aurc_optimal(correct) -> float:
    """AURC of an oracle ranking that puts every correct item first."""
    c = np.sort(np.asarray(correct, float))[::-1]
    k = np.arange(1, len(c) + 1)
    return float((np.cumsum(1 - c) / k).mean())


def eaurc(conf, correct) -> float:
    """Excess AURC: AURC minus the oracle's, removing the dependence on base accuracy."""
    return aurc(conf, correct) - aurc_optimal(correct)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(stat, *arrays, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI of stat(*arrays) resampling items with replacement. Returns (point, lo, hi)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        v = stat(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(stat(*arrays)), float(lo), float(hi)

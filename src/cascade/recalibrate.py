"""Post-hoc recalibration fit on the calibration split only, applied unchanged to test.

Temperature scaling (Guo et al., 2017): divide the four option logits by one scalar T > 0
chosen to minimise multiclass negative log-likelihood on the calibration split.

What it can and cannot change:
  * It never changes the argmax, so accuracy is identical.
  * It rescales confidence, so it changes ECE, Brier, NLL and the threshold regime
    ("escalate when confidence < c") -- its intended effect.
  * With four options it can also slightly REORDER items: the top-option probability
    depends on all four logits, so dividing by T can swap two items whose logit vectors
    have different shapes. Ranking metrics (AUROC, AURC, budget-regime cascade accuracy)
    usually move only a little, but not by exactly zero, so the comparison report
    computes them for both raw and scaled confidence rather than assuming they match.
"""
from __future__ import annotations

import numpy as np


def _nll(logits: np.ndarray, gold_idx: np.ndarray, beta: float) -> float:
    """Mean multiclass NLL of softmax(beta * logits), beta = 1/T. Convex in beta."""
    z = beta * logits
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(z)), gold_idx]))


def nll(logits, gold_idx, temperature: float = 1.0) -> float:
    return _nll(np.asarray(logits, float), np.asarray(gold_idx, int), 1.0 / temperature)


def fit_temperature(logits, gold_idx, beta_lo: float = 1e-3, beta_hi: float = 20.0,
                    tol: float = 1e-6) -> float:
    """Return T = 1/beta minimising NLL. Golden-section search on beta (NLL is convex in
    beta), so no optimiser dependency and no local-minimum risk."""
    z = np.asarray(logits, float)
    y = np.asarray(gold_idx, int)
    g = (np.sqrt(5) - 1) / 2
    a, b = beta_lo, beta_hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = _nll(z, y, c), _nll(z, y, d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _nll(z, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _nll(z, y, d)
    beta = (a + b) / 2
    if beta <= beta_lo * 1.01 or beta >= beta_hi * 0.99:
        raise RuntimeError(f"temperature fit hit the search bound (beta={beta:.4g}); check the logits")
    return 1.0 / beta

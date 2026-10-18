import numpy as np
import pytest

from cascade.cascade import budget_curve, oracle_accuracy, threshold_point
from cascade.metrics import (aurc, aurc_optimal, auroc, brier_multiclass, brier_toplabel, eaurc,
                             ece_equal_mass, risk_coverage, softmax)


def random_order_mean(fn, conf, correct, n=4000, seed=0):
    """Brute force: average fn over random tie-breaking, by jittering conf infinitesimally."""
    rng = np.random.default_rng(seed)
    vals = [fn(conf + rng.uniform(0, 1e-9, len(conf)), correct) for _ in range(n)]
    return float(np.mean(vals))


# --- AUROC ------------------------------------------------------------------

def test_auroc_perfect_reversed_and_constant():
    correct = np.array([1, 1, 1, 0, 0])
    assert auroc([0.9, 0.8, 0.7, 0.2, 0.1], correct) == 1.0
    assert auroc([0.1, 0.2, 0.3, 0.8, 0.9], correct) == 0.0
    assert auroc([0.5] * 5, correct) == 0.5


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    conf = rng.integers(0, 5, 300) / 4  # heavy ties
    correct = rng.random(300) < 0.6
    pos, neg = conf[correct], conf[~correct]
    pairwise = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert auroc(conf, correct) == pytest.approx(pairwise)


# --- AURC ---------------------------------------------------------------------

def test_aurc_hand_computed():
    # order by conf: right, wrong, right -> risks 0/1, 1/2, 1/3
    assert aurc([0.9, 0.5, 0.1], [1, 0, 1]) == pytest.approx((0 + 1 / 2 + 1 / 3) / 3)


def test_aurc_oracle_and_eaurc_zero_for_perfect_ranking():
    correct = np.array([1, 1, 0, 1, 0, 0, 1])
    conf = correct + 0.01 * np.arange(7)
    assert aurc(conf, correct) == pytest.approx(aurc_optimal(correct))
    assert eaurc(conf, correct) == pytest.approx(0.0)


def test_aurc_constant_conf_equals_error_rate():
    correct = np.array([1, 0, 1, 1, 0, 1, 1, 1])
    assert aurc(np.full(8, 0.3), correct) == pytest.approx(1 - correct.mean())


def test_aurc_ties_equal_random_tiebreak_expectation():
    rng = np.random.default_rng(2)
    conf = rng.integers(0, 11, 60) / 10  # like the 11-valued permutation signal
    correct = (rng.random(60) < 0.3 + 0.6 * conf).astype(int)
    assert aurc(conf, correct) == pytest.approx(random_order_mean(aurc, conf, correct), abs=2e-3)


def test_risk_coverage_endpoints():
    correct = np.array([1, 0, 1, 1])
    cov, risk = risk_coverage([0.9, 0.2, 0.8, 0.7], correct)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(0.25) and risk[0] == 0.0


# --- calibration -------------------------------------------------------------

def test_ece_zero_when_calibrated_within_bins():
    conf = np.repeat([0.2, 0.5, 0.8], 100)
    correct = np.concatenate([np.arange(100) < 20, np.arange(100) < 50, np.arange(100) < 80])
    assert ece_equal_mass(conf, correct, n_bins=3) == pytest.approx(0.0)


def test_ece_overconfident():
    assert ece_equal_mass(np.full(100, 0.9), np.arange(100) < 60, n_bins=15) == pytest.approx(0.3)


def test_brier():
    probs = np.array([[1, 0, 0, 0], [0.25] * 4])
    assert brier_multiclass(probs, [0, 1]) == pytest.approx((0 + (0.75**2 + 3 * 0.25**2)) / 2)
    assert brier_toplabel([1.0, 0.0], [1, 1]) == pytest.approx(0.5)


def test_softmax_temperature():
    p = softmax(np.array([[2.0, 1.0, 0.0, -1.0]]), temperature=1e6)
    assert np.allclose(p, 0.25, atol=1e-5)


# --- cascade -------------------------------------------------------------------

def test_budget_curve_endpoints_and_perfect_signal():
    s = np.array([1, 1, 0, 0, 1])
    f = np.array([1, 0, 1, 1, 1])
    conf = np.array([0.9, 0.8, 0.1, 0.2, 0.7])  # wrong items least confident
    c = budget_curve(conf, s, f)
    assert c["accuracy"].iloc[0] == pytest.approx(s.mean())
    assert c["accuracy"].iloc[-1] == pytest.approx(f.mean())
    assert c["accuracy"].iloc[2] == pytest.approx(oracle_accuracy(s, f)[0])


def test_budget_curve_ties_match_random_tiebreak():
    rng = np.random.default_rng(3)
    conf = rng.integers(0, 4, 40) / 3
    s = (rng.random(40) < 0.6).astype(int)
    f = (rng.random(40) < 0.85).astype(int)

    def acc_at_10(cf, _):
        return budget_curve(cf, s, f)["accuracy"].iloc[10]

    assert budget_curve(conf, s, f)["accuracy"].iloc[10] == pytest.approx(
        random_order_mean(acc_at_10, conf, None), abs=3e-3)


def test_threshold_point():
    r = threshold_point([0.95, 0.5, 0.92, 0.3], [1, 0, 0, 1], [1, 1, 1, 0], c=0.9)
    assert r["call_rate"] == 0.5
    assert r["retained_accuracy"] == 0.5
    assert r["system_accuracy"] == pytest.approx((1 + 1 + 0 + 0) / 4)


def test_ece_ignores_tie_order():
    conf = np.array([0.9] * 50 + [0.6] * 50)
    correct = np.array([1] * 45 + [0] * 5 + [1] * 30 + [0] * 20)
    rng = np.random.default_rng(0)
    p = rng.permutation(100)
    assert ece_equal_mass(conf, correct) == pytest.approx(ece_equal_mass(conf[p], correct[p]))
    assert ece_equal_mass(conf, correct) == pytest.approx(0.0)

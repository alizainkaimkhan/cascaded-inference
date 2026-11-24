import numpy as np
import pytest

from cascade.metrics import aurc, auroc, ece_equal_mass, softmax
from cascade.recalibrate import fit_temperature, nll


def _sample(n, T_true, seed=0):
    """Logits z with labels drawn from softmax(z / T_true): the NLL-optimal temperature is T_true."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 4.0, (n, 4))
    p = softmax(z, T_true)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    return z, y


@pytest.mark.parametrize("T_true", [0.7, 1.0, 2.5, 6.0])
def test_fit_temperature_recovers_generating_temperature(T_true):
    z, y = _sample(20000, T_true)
    assert fit_temperature(z, y) == pytest.approx(T_true, rel=0.06)


def test_temperature_lowers_nll_and_ece_but_not_accuracy_or_ranking():
    z, y = _sample(8000, 3.0, seed=1)          # model logits are 3x too sharp: overconfident
    T = fit_temperature(z, y)
    assert nll(z, y, T) < nll(z, y, 1.0)
    p1, pT = softmax(z, 1.0), softmax(z, T)
    correct = p1.argmax(1) == y
    assert np.array_equal(p1.argmax(1), pT.argmax(1))                     # accuracy unchanged
    c1, cT = p1.max(1), pT.max(1)
    assert ece_equal_mass(cT, correct) < ece_equal_mass(c1, correct) / 2  # calibration fixed
    # With K > 2 options the top-option probability depends on all K logits, so dividing by T
    # can swap the order of two items whose logit vectors differ in shape: ranking changes
    # are small but NOT zero, and the report computes ranking metrics for both versions.
    assert abs(auroc(cT, correct) - auroc(c1, correct)) < 0.02
    assert abs(aurc(cT, correct) - aurc(c1, correct)) < 0.02


def test_temperature_preserves_ranking_for_two_options():
    rng = np.random.default_rng(3)
    z = np.c_[rng.normal(0, 3, 4000), np.zeros(4000)]
    y = (rng.random(4000) < softmax(z, 2.0)[:, 1]).astype(int)
    p1, pT = softmax(z, 1.0), softmax(z, fit_temperature(z, y))
    correct = p1.argmax(1) == y
    assert auroc(pT.max(1), correct) == pytest.approx(auroc(p1.max(1), correct))


def test_fit_temperature_raises_at_search_bound():
    z = np.tile([[5.0, 0, 0, 0]], (50, 1))
    y = np.zeros(50, int)                                   # always right: wants T -> 0
    with pytest.raises(RuntimeError):
        fit_temperature(z, y)

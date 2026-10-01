import numpy as np
import pytest

from src.scaling import _fit_power_law, scaling_law


def test_scaling_law_decreases_with_more_data():
    D = np.array([100.0, 1000.0, 10000.0])
    L = scaling_law(D, L_inf=0.5, b=10.0, beta=0.4)
    assert np.all(np.diff(L) < 0)
    assert np.all(L > 0.5)


def test_power_law_fit_recovers_known_synthetic_curve():
    pytest.importorskip("scipy.optimize")
    D = np.geomspace(100, 100000, 12)
    loss = scaling_law(D, L_inf=0.7, b=20.0, beta=0.45)
    fit = _fit_power_law(D, loss)
    assert fit["L_inf"] == pytest.approx(0.7, rel=1e-3)
    assert fit["b"] == pytest.approx(20.0, rel=1e-3)
    assert fit["beta"] == pytest.approx(0.45, rel=1e-3)


@pytest.mark.parametrize(
    ("D", "loss", "message"),
    [
        ([100, 1000, 10000], [2.0, 1.5, 1.0], "four distinct"),
        ([100, 1000, 10000, 100000], [2.0, 1.5, 0.0, 0.8], "strictly positive"),
    ],
)
def test_power_law_fit_rejects_invalid_inputs(D, loss, message):
    pytest.importorskip("scipy.optimize")
    with pytest.raises(ValueError, match=message):
        _fit_power_law(np.array(D), np.array(loss))

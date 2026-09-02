# Licence: BSD-3-Clause
import numpy as np
import pytest

from canapy.segmentation import otsu, otsu_threshold

# Otsu 1979, worked by hand: six levels, 40 values, deliberately bimodal.
TOY_LEVELS = [0, 1, 2, 3, 4, 5]
TOY_COUNTS = [10, 8, 2, 2, 7, 11]
TOY_TOTAL_VARIANCE = 4.1494
TOY_BETWEEN_VARIANCE = 3.7056


@pytest.fixture()
def toy_values():
    return np.repeat(TOY_LEVELS, TOY_COUNTS).astype(float)


def test_otsu_cuts_in_the_valley(toy_values):
    threshold = otsu_threshold(toy_values, n_bins=len(TOY_LEVELS))
    assert 2.0 <= threshold <= 3.0


def test_otsu_maximises_between_class_variance(toy_values):
    _, eta_squared = otsu(toy_values, n_bins=len(TOY_LEVELS))

    total = float(toy_values.var())
    assert total == pytest.approx(TOY_TOTAL_VARIANCE, abs=1e-4)
    assert eta_squared * total == pytest.approx(TOY_BETWEEN_VARIANCE, abs=1e-4)


def test_eta_squared_is_a_share_of_variance(toy_values):
    _, eta_squared = otsu(toy_values)
    assert 0.0 <= eta_squared <= 1.0


def test_bimodal_scores_higher_than_unimodal():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0, 1, 5000), rng.normal(20, 1, 5000)])
    unimodal = rng.normal(0, 1, 10000)

    _, eta_bimodal = otsu(bimodal)
    _, eta_unimodal = otsu(unimodal)

    assert eta_bimodal > 0.9
    assert eta_unimodal < 0.8
    assert eta_bimodal > eta_unimodal


def test_constant_values_explain_nothing():
    _, eta_squared = otsu(np.full(100, 3.0))
    assert eta_squared == 0.0

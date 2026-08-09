"""Tests for the noise generator.

The first test is the one that matters. The previous generator produced noise
by reordering the measured residual values, so training noise and test noise
were built from the same numbers. This suite pins the property the fix
established: the generator takes three parameters and no data.
"""

import inspect

import numpy as np
import pytest

import noise
from noise import ar1_series, envelope_for, make_noise


def test_the_generator_takes_parameters_not_data():
    """make_noise must not accept a measured residual array.

    This is the guard against reintroducing the leak: as long as the only
    inputs are a reference curve and three floats, no measured noise value can
    reach the training set.
    """
    assert list(inspect.signature(make_noise).parameters) == [
        "reference",
        "phi",
        "slope",
        "intercept",
        "rng",
    ]
    assert "residual" not in inspect.getsource(noise.make_noise)


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.49, 0.8])
def test_ar1_reproduces_the_requested_autocorrelation(phi):
    series = ar1_series(200_000, phi, np.random.default_rng(0))
    measured = np.corrcoef(series[:-1], series[1:])[0, 1]
    assert measured == pytest.approx(phi, abs=0.01)


def test_ar1_has_unit_variance_on_average():
    rng = np.random.default_rng(1)
    variances = [ar1_series(4000, 0.5, rng).var(ddof=1) for _ in range(50)]
    assert np.mean(variances) == pytest.approx(1.0, abs=0.05)


def test_ar1_realisations_are_not_all_identically_scaled():
    """Normalising by the theoretical std, not the realised one, keeps the
    natural spread between realisations. Removing it would make the benchmark's
    bootstrap intervals too narrow."""
    rng = np.random.default_rng(2)
    stds = np.array([ar1_series(500, 0.5, rng).std(ddof=1) for _ in range(200)])
    assert stds.std(ddof=1) > 0.01


@pytest.mark.parametrize("phi", [-1.0, 1.0, 1.5])
def test_unstable_ar1_coefficient_is_rejected(phi):
    with pytest.raises(ValueError):
        ar1_series(10, phi, np.random.default_rng(0))


def test_envelope_grows_with_the_signal_level():
    envelope = envelope_for(np.array([0.0, 5.0, 10.0]), slope=0.02, intercept=0.07)
    np.testing.assert_allclose(envelope, [0.07, 0.17, 0.27])


def test_envelope_uses_magnitude_so_negative_tilts_are_not_special():
    np.testing.assert_allclose(
        envelope_for(np.array([6.0]), 0.02, 0.07),
        envelope_for(np.array([-6.0]), 0.02, 0.07),
    )


def test_non_positive_envelope_is_rejected():
    with pytest.raises(ValueError):
        envelope_for(np.zeros(5), slope=0.02, intercept=-0.1)


def test_noise_level_tracks_the_envelope():
    """At a high tilt the noise should be visibly larger than at rest."""
    rng = np.random.default_rng(4)
    at_rest = make_noise(np.zeros(20_000), 0.5, 0.02, 0.07, rng).std(ddof=1)
    at_tilt = make_noise(np.full(20_000, 10.0), 0.5, 0.02, 0.07, rng).std(ddof=1)
    assert at_rest == pytest.approx(0.07, rel=0.1)
    assert at_tilt == pytest.approx(0.27, rel=0.1)


def test_noise_matches_the_reference_length():
    rng = np.random.default_rng(5)
    assert make_noise(np.zeros(137), 0.4, 0.02, 0.07, rng).shape == (137,)


def test_same_seed_reproduces_the_same_noise():
    a = make_noise(np.ones(50), 0.4, 0.02, 0.07, np.random.default_rng(11))
    b = make_noise(np.ones(50), 0.4, 0.02, 0.07, np.random.default_rng(11))
    np.testing.assert_allclose(a, b)


def test_two_dimensional_reference_is_rejected():
    with pytest.raises(ValueError):
        make_noise(np.zeros((4, 4)), 0.4, 0.02, 0.07, np.random.default_rng(0))
"""Tests for the classical denoising filters."""

import numpy as np
import pytest

from filters import (
    Denoiser,
    ExponentialMovingAverage,
    LinearKalman,
    MovingAverage,
    published_baselines,
)


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(0)
    clean = np.linspace(0.0, 8.0, 400)
    return clean + rng.normal(0.0, 0.2, 400)


def test_the_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Denoiser()


@pytest.mark.parametrize("denoiser", published_baselines(), ids=lambda d: d.name)
def test_every_baseline_implements_the_interface(denoiser):
    assert isinstance(denoiser, Denoiser)


@pytest.mark.parametrize("denoiser", published_baselines(), ids=lambda d: d.name)
def test_output_length_matches_input(denoiser, noisy_signal):
    assert denoiser.apply(noisy_signal).shape == noisy_signal.shape


@pytest.mark.parametrize("denoiser", published_baselines(), ids=lambda d: d.name)
def test_constant_signal_passes_through_unchanged(denoiser):
    """A signal with no variation has no noise to remove."""
    constant = np.full(200, 3.5)
    np.testing.assert_allclose(denoiser.apply(constant), constant, atol=1e-8)


@pytest.mark.parametrize("denoiser", published_baselines(), ids=lambda d: d.name)
def test_no_baseline_makes_the_signal_rougher(denoiser, noisy_signal):
    roughness = np.var(np.diff(denoiser.apply(noisy_signal)))
    assert roughness <= np.var(np.diff(noisy_signal))


def test_ema_with_alpha_one_is_the_identity():
    signal = np.array([1.0, 5.0, 2.0, 9.0])
    np.testing.assert_allclose(ExponentialMovingAverage(alpha=1.0).apply(signal), signal)


def test_moving_average_of_window_one_is_the_identity():
    signal = np.array([1.0, 5.0, 2.0, 9.0])
    np.testing.assert_allclose(MovingAverage(window=1).apply(signal), signal)


def test_published_kalman_barely_filters():
    """Q=7 with R=0.01 gives a gain near 1, so the filter is near-transparent.

    This documents a property of the published parameters, not a bug.
    """
    assert LinearKalman(process_noise=7.0, measurement_noise=0.01).steady_state_gain > 0.99


def test_lower_process_noise_smooths_more(noisy_signal):
    responsive = LinearKalman(process_noise=7.0).apply(noisy_signal)
    smoothing = LinearKalman(process_noise=0.01).apply(noisy_signal)
    assert np.var(np.diff(smoothing)) < np.var(np.diff(responsive))


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        ExponentialMovingAverage(alpha=alpha)


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        MovingAverage().apply(np.zeros((4, 4)))
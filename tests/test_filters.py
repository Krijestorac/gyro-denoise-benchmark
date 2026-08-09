"""Tests for the classical denoising filters."""

import numpy as np
import pytest

from filters import (
    ButterworthLowPass,
    Denoiser,
    ExponentialMovingAverage,
    LinearKalman,
    MovingAverage,
    published_baselines,
)

ALL_BASELINES = published_baselines(zero_phase=False) + published_baselines(
    zero_phase=True
)


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(0)
    clean = np.linspace(0.0, 8.0, 400)
    return clean + rng.normal(0.0, 0.2, 400)


@pytest.fixture
def bump_and_noisy_bump():
    """A clean symmetric bump and a noisy copy, for lag measurements.

    A bump rather than a ramp: cross-correlation needs a signal with one
    unambiguous peak, and a ramp ending on a long plateau does not have one.
    """
    rng = np.random.default_rng(3)
    index = np.arange(500)
    clean = 8.0 * np.exp(-((index - 250.0) ** 2) / (2.0 * 60.0**2))
    return clean, clean + rng.normal(0.0, 0.3, clean.size)


def test_the_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Denoiser()


@pytest.mark.parametrize("denoiser", ALL_BASELINES, ids=repr)
def test_every_baseline_implements_the_interface(denoiser):
    assert isinstance(denoiser, Denoiser)


@pytest.mark.parametrize("denoiser", ALL_BASELINES, ids=repr)
def test_output_length_matches_input(denoiser, noisy_signal):
    assert denoiser.apply(noisy_signal).shape == noisy_signal.shape


@pytest.mark.parametrize("denoiser", ALL_BASELINES, ids=repr)
def test_constant_signal_passes_through_unchanged(denoiser):
    """A signal with no variation has no noise to remove."""
    constant = np.full(200, 3.5)
    np.testing.assert_allclose(denoiser.apply(constant), constant, atol=1e-8)


@pytest.mark.parametrize("denoiser", ALL_BASELINES, ids=repr)
def test_no_baseline_makes_the_signal_rougher(denoiser, noisy_signal):
    roughness = np.var(np.diff(denoiser.apply(noisy_signal)))
    assert roughness <= np.var(np.diff(noisy_signal))


@pytest.mark.parametrize("denoiser", ALL_BASELINES, ids=repr)
def test_phase_label_matches_the_flag(denoiser):
    """Results tables read `phase` directly, so it must never drift."""
    assert denoiser.phase == ("zero-phase" if denoiser.zero_phase else "causal")


# --- the property the whole benchmark depends on ---------------------------


def cross_correlation_lag(clean, filtered):
    """Lag in samples that best aligns `filtered` with `clean`."""
    a = clean - clean.mean()
    b = filtered - filtered.mean()
    return int(np.argmax(np.correlate(b, a, mode="full")) - (a.size - 1))


@pytest.mark.parametrize(
    "build",
    [
        lambda zp: MovingAverage(window=31, zero_phase=zp),
        lambda zp: ExponentialMovingAverage(alpha=0.05, zero_phase=zp),
        lambda zp: LinearKalman(process_noise=1e-4, zero_phase=zp),
        lambda zp: ButterworthLowPass(order=4, cutoff_hz=2.0, zero_phase=zp),
    ],
    ids=["MovingAverage", "EMA", "Kalman", "Butterworth"],
)
def test_zero_phase_mode_removes_the_lag(build, bump_and_noisy_bump):
    """EVERY filter must lose its lag in zero-phase mode, not just the IIR ones.

    This is the bug that invalidated the first version of the benchmark: only
    Bessel and Butterworth had a zero-phase mode, so the moving average, the
    EMA and the Kalman filter were compared against a non-causal network while
    still lagging. This test fails if that regresses.
    """
    clean, noisy = bump_and_noisy_bump
    causal_lag = cross_correlation_lag(clean, build(False).apply(noisy))
    zero_phase_lag = cross_correlation_lag(clean, build(True).apply(noisy))

    assert causal_lag > 0, "the causal filter should lag"
    assert abs(zero_phase_lag) <= 1, "the zero-phase filter should not"


@pytest.mark.parametrize(
    "build",
    [
        lambda zp: MovingAverage(window=31, zero_phase=zp),
        lambda zp: ExponentialMovingAverage(alpha=0.05, zero_phase=zp),
        lambda zp: LinearKalman(process_noise=1e-4, zero_phase=zp),
        lambda zp: ButterworthLowPass(order=4, cutoff_hz=2.0, zero_phase=zp),
    ],
    ids=["MovingAverage", "EMA", "Kalman", "Butterworth"],
)
def test_zero_phase_beats_causal_on_a_known_clean_signal(build, bump_and_noisy_bump):
    """Removing the lag must actually lower the error, or the fix is cosmetic.

    The margins here are a factor of four to seven on this signal, which is the
    size of the advantage the old benchmark was silently handing to the network
    by leaving three of five baselines causal.
    """
    clean, noisy = bump_and_noisy_bump

    def error(denoiser):
        return np.sqrt(np.mean((denoiser.apply(noisy) - clean) ** 2))

    assert error(build(True)) < error(build(False))


# --- individual filter properties ------------------------------------------


def test_ema_with_alpha_one_is_the_identity():
    signal = np.array([1.0, 5.0, 2.0, 9.0])
    np.testing.assert_allclose(
        ExponentialMovingAverage(alpha=1.0).apply(signal), signal
    )


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


def test_invalid_cutoff_is_rejected():
    with pytest.raises(ValueError):
        ButterworthLowPass(cutoff_hz=60.0, sampling_rate_hz=100.0)


def test_two_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        MovingAverage().apply(np.zeros((4, 4)))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        MovingAverage().apply(np.array([]))
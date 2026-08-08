"""Tests for the synthetic tilt curve generator."""

import numpy as np
import pytest

from signals import PEAK_LEVEL_RANGE, make_curve_batch, random_tilt_curve, tilt_curve


def test_curve_starts_and_ends_at_rest():
    curve = tilt_curve(length=500, onset=20, rise=150, hold=80, fall=150, peak_level=8.0)
    assert curve[0] == 0.0
    assert curve[-1] == 0.0


def test_curve_reaches_requested_peak():
    curve = tilt_curve(length=500, onset=20, rise=150, hold=80, fall=150, peak_level=8.0)
    assert curve.max() == pytest.approx(8.0)


def test_curve_is_smooth():
    """No step should exceed what a smooth ramp of this size can produce."""
    curve = tilt_curve(length=500, onset=20, rise=150, hold=80, fall=150, peak_level=8.0)
    largest_step = np.abs(np.diff(curve)).max()
    assert largest_step < 2.0 * 8.0 / 150


def test_random_curves_respect_the_configured_peak_range():
    rng = np.random.default_rng(0)
    peaks = make_curve_batch(50, rng).max(axis=1)
    assert peaks.min() >= PEAK_LEVEL_RANGE[0]
    assert peaks.max() <= PEAK_LEVEL_RANGE[1]


def test_same_seed_reproduces_the_same_curve():
    a = random_tilt_curve(np.random.default_rng(7))
    b = random_tilt_curve(np.random.default_rng(7))
    np.testing.assert_allclose(a, b)


def test_different_draws_give_different_curves():
    rng = np.random.default_rng(1)
    a, b = random_tilt_curve(rng), random_tilt_curve(rng)
    assert not np.allclose(a, b)


def test_segments_longer_than_the_signal_are_rejected():
    with pytest.raises(ValueError):
        tilt_curve(length=100, onset=50, rise=50, hold=50, fall=50, peak_level=1.0)
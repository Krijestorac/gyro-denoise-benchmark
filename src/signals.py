"""
Synthetic tilt manoeuvres used as training targets.

Only one real recording exists. Training a model on that single curve with
different noise added would let it memorise the shape and reproduce it from
memory, scoring well on the test recording while having learned nothing about
denoising.

This module produces a family of plausible tilt manoeuvres instead: rest, a
smooth rise, a hold, a smooth return to rest. Onset, durations and peak level
are randomised, so the model has to learn the operation rather than the answer.
The real recording is never used here and is reserved entirely for testing.

Transitions use a smoothstep polynomial, which has a continuous first
derivative at both ends. A linear ramp would introduce corners that no physical
tilt produces and that a denoiser would learn to reproduce as artefacts.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LENGTH = 500

# Ranges chosen to bracket the real recording (peak 8.75 degrees over 500
# samples at 100 Hz) without reproducing it.
PEAK_LEVEL_RANGE = (3.0, 12.0)
ONSET_RANGE = (5, 60)
RISE_FRACTION_RANGE = (0.25, 0.45)
FALL_FRACTION_RANGE = (0.25, 0.45)


def smoothstep(x: np.ndarray) -> np.ndarray:
    """Smooth 0-to-1 transition with zero gradient at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tilt_curve(
    length: int,
    onset: int,
    rise: int,
    hold: int,
    fall: int,
    peak_level: float,
) -> np.ndarray:
    """Build one tilt manoeuvre from explicit segment lengths."""
    if min(onset, rise, hold, fall) < 0:
        raise ValueError("Segment lengths must be non-negative.")
    if onset + rise + hold + fall > length:
        raise ValueError(
            f"Segments total {onset + rise + hold + fall} but length is {length}."
        )

    curve = np.zeros(length, dtype=np.float64)

    rise_end = onset + rise
    if rise > 0:
        curve[onset:rise_end] = peak_level * smoothstep(
            np.linspace(0.0, 1.0, rise, endpoint=False)
        )

    hold_end = rise_end + hold
    curve[rise_end:hold_end] = peak_level

    fall_end = hold_end + fall
    if fall > 0:
        curve[hold_end:fall_end] = peak_level * (
            1.0 - smoothstep(np.linspace(0.0, 1.0, fall, endpoint=False))
        )

    return curve


def random_tilt_curve(
    rng: np.random.Generator, length: int = DEFAULT_LENGTH
) -> np.ndarray:
    """Sample one tilt manoeuvre with randomised timing and amplitude."""
    peak_level = float(rng.uniform(*PEAK_LEVEL_RANGE))
    onset = int(rng.integers(*ONSET_RANGE))

    available = length - onset - 10
    rise = int(available * rng.uniform(*RISE_FRACTION_RANGE))
    fall = int(available * rng.uniform(*FALL_FRACTION_RANGE))
    hold = available - rise - fall

    return tilt_curve(length, onset, rise, hold, fall, peak_level)


def make_curve_batch(
    n: int, rng: np.random.Generator, length: int = DEFAULT_LENGTH
) -> np.ndarray:
    """Return an array of shape (n, length) of independent tilt manoeuvres."""
    return np.stack([random_tilt_curve(rng, length) for _ in range(n)])
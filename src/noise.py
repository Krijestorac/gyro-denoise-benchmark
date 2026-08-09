"""
Training noise generation.

Only one recording exists, so training noise has to be manufactured. The
generator has one hard constraint: it must not reuse any value from the part of
the recording the methods are scored on. Otherwise the network is trained on
the exact noise it will be tested against, and every number that follows is
optimistic.

The model is deliberately simple: a first-order autoregressive process scaled
by a signal-dependent envelope.

    e[k]     = phi * e[k-1] + w[k],   w ~ N(0, 1)
    noise[k] = e[k] * (slope * |reference[k]| + intercept)

Three parameters, all estimated from the FIT HALF of the recording only (see
data_prep.py):

    phi        lag-1 autocorrelation of the residual
    slope      how fast the noise level grows with the signal level
    intercept  the noise floor at rest

Nothing else crosses from the recording into the generator. No measured value
is reused, only three numbers, and those three come from samples nothing is
scored on.

Why this replaced the previous method
-------------------------------------
The previous version used the iterative amplitude-adjusted Fourier transform
(Schreiber & Schmitz, 1996). IAAFT reproduces a target's power spectrum and
amplitude distribution exactly, and it does so by reordering the target's own
values. The target was the residual of the recording the model was scored
against, so every training example carried the exact amplitude distribution and
power spectrum of the test noise. Checked directly, the surrogate was a
permutation of the test residual (identical sorted arrays) with a power-spectrum
correlation of 0.9999.

Known limitation, stated plainly
--------------------------------
The innovations are Gaussian. The measured residual is mildly right-skewed
(about +0.4) and heavy-tailed (excess kurtosis about +0.7), and this generator
reproduces neither. Synthetic noise is therefore slightly easier than the real
thing. That penalty applies equally to every method in the benchmark, so it
does not favour one over another, but absolute error figures on synthetic data
should be read as a lower bound.

Distributional realism was traded for a clean train/test boundary. That is the
right trade when the alternative is a benchmark nobody can believe.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def ar1_series(length: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """One realisation of a unit-variance AR(1) process.

    `lfilter([1], [1, -phi], w)` is the recursion e[k] = phi*e[k-1] + w[k].
    Dividing by the theoretical standard deviation sqrt(1/(1 - phi^2)) rather
    than the realised one leaves the natural variation between realisations
    intact; normalising by the realised value would force every draw to unit
    variance and quietly understate the spread in the benchmark's error bars.
    """
    if not -1.0 < phi < 1.0:
        raise ValueError(f"AR(1) coefficient must lie in (-1, 1), got {phi}.")
    if length < 1:
        raise ValueError("Length must be positive.")

    innovations = rng.standard_normal(length)
    series = lfilter([1.0], [1.0, -phi], innovations)
    return series / np.sqrt(1.0 / (1.0 - phi * phi))


def envelope_for(
    reference: np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    """Target noise standard deviation at each sample.

    MEMS error is conventionally split into a constant floor plus a term that
    grows with the measurement. The intercept is the floor, the slope is the
    growth. See data_prep.fit_noise_envelope for how well that holds on this
    recording and for the confound it cannot fully resolve.
    """
    envelope = slope * np.abs(reference) + intercept
    if np.any(envelope <= 0):
        raise ValueError("Fitted envelope is non-positive; check the fit.")
    return envelope


def make_noise(
    reference: np.ndarray,
    phi: float,
    slope: float,
    intercept: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one noise realisation shaped to the given reference curve."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1:
        raise ValueError("Reference must be one-dimensional.")
    return ar1_series(reference.size, phi, rng) * envelope_for(
        reference, slope, intercept
    )
"""
Training noise generation.

Only one recording exists, so training noise has to be manufactured. Two
measured properties rule out the obvious choice of white Gaussian noise:
consecutive samples are correlated (lag-1 autocorrelation about 0.51), and the
distribution is skewed with heavy tails. Training on noise that is easier than
the real thing would make every benchmark number optimistic.

The method used is the iterative amplitude-adjusted Fourier transform
(Schreiber & Schmitz, 1996), a surrogate data method. It alternates between
imposing the measured power spectrum and imposing the measured amplitude
distribution until both hold. The output is a reordering of the measured
values with a different, statistically equivalent time structure.

The generated noise is then scaled by the envelope fitted in data_prep.py, so
that it grows with the signal level the way the real sensor noise does.

Known limitation: because the output is a reordering of the 500 measured
values, no amplitude outside the observed range can ever appear.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ITERATIONS = 200


def iaaft_surrogate(
    residual: np.ndarray,
    rng: np.random.Generator,
    n_iterations: int = DEFAULT_ITERATIONS,
) -> np.ndarray:
    """Return one independent noise realisation matching the residual.

    Starts from a random shuffle, then repeatedly imposes the target spectrum
    (which distorts the distribution) and rank-orders back onto the target
    values (which restores it). Stops early once the ordering settles.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 1:
        raise ValueError("Residual must be one-dimensional.")

    target_magnitudes = np.abs(np.fft.rfft(residual))
    sorted_values = np.sort(residual)

    candidate = rng.permutation(residual)
    previous_ranks: np.ndarray | None = None

    for _ in range(n_iterations):
        spectrum = np.fft.rfft(candidate)
        candidate = np.fft.irfft(
            target_magnitudes * np.exp(1j * np.angle(spectrum)), n=residual.size
        )
        ranks = np.argsort(np.argsort(candidate))
        candidate = sorted_values[ranks]

        if previous_ranks is not None and np.array_equal(ranks, previous_ranks):
            break
        previous_ranks = ranks

    return candidate


def scale_to_envelope(
    noise: np.ndarray,
    reference: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """Scale noise so its local amplitude follows the signal level.

    The surrogate has a single standard deviation across its whole length. The
    real noise does not: it is roughly `slope * level + intercept`. Dividing by
    the surrogate's own standard deviation and multiplying by the target
    envelope imposes that relationship.
    """
    if noise.shape != reference.shape:
        raise ValueError(
            f"Noise shape {noise.shape} does not match reference {reference.shape}."
        )
    envelope = slope * np.abs(reference) + intercept
    if np.any(envelope <= 0):
        raise ValueError("Fitted envelope is non-positive; check the fit.")
    return noise / noise.std(ddof=1) * envelope


def make_noise(
    residual: np.ndarray,
    reference: np.ndarray,
    slope: float,
    intercept: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one noise realisation shaped to the given reference curve."""
    surrogate = iaaft_surrogate(residual, rng)
    if reference.size != surrogate.size:
        surrogate = np.interp(
            np.linspace(0.0, 1.0, reference.size),
            np.linspace(0.0, 1.0, surrogate.size),
            surrogate,
        )
    return scale_to_envelope(surrogate, reference, slope, intercept)
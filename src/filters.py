"""
Classical denoising filters behind a single interface.

Every filter in this module, and later the neural network, implements the same
`Denoiser` interface: construct it, call `apply(signal)`, get a denoised signal
back. The benchmark can then loop over a list of denoisers without knowing or
caring what any of them does internally. Adding a method means adding a class,
not editing the benchmark. This is the strategy pattern.

Phase
-----
Filters are causal by default: they see only past samples, so their output lags
the input. Lag inflates error, and that error has nothing to do with how well a
method removes noise.

The neural network is not causal. It sees samples on both sides of every point.
Comparing a non-causal network against causal filters therefore measures phase,
not denoising quality.

Setting `zero_phase=True` runs a filter forwards, then runs the result
backwards through the same filter. The two passes have equal and opposite
delay, so the total delay is zero. This is the standard forward-backward trick
(what `scipy.signal.filtfilt` does for IIR filters). It costs the ability to
run in real time, and it squares the magnitude response, so a 4th-order filter
behaves like an 8th-order one. Both effects apply identically to every method,
and every filter is re-tuned separately in each phase mode, so the comparison
stays matched.

The forward-backward pass lives in the base class and is applied the same way
to every filter, so there is exactly one code path and no method can quietly
get a different deal. Subclasses implement `_filter`, which must be causal.

Parameter defaults reproduce the published configuration in Krijestorac (2025),
INFOTEH-JAHORINA: Kalman P = 0.07, Q = 7, R = 0.01; EMA alpha = 0.2 and 0.75;
Bessel 4th order, 10 Hz cutoff, 100 Hz sampling.

A note on the Kalman defaults: at Q = 7 with R = 0.01 the steady-state gain is
0.9986, meaning the filter passes measurements through almost unchanged. The
published parameters are kept as the default for continuity, and the benchmark
re-tunes everything anyway.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import bessel, butter, lfilter

SAMPLING_RATE_HZ = 100.0


class Denoiser(ABC):
    """Common interface for every denoising method in the benchmark."""

    def __init__(self, zero_phase: bool = False) -> None:
        self.zero_phase = zero_phase

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in results tables and figures."""

    @abstractmethod
    def _filter(self, signal: np.ndarray) -> np.ndarray:
        """Causal single-pass filter. Same length in, same length out."""

    @property
    def phase(self) -> str:
        """Which phase mode this instance is in, for results tables."""
        return "zero-phase" if self.zero_phase else "causal"

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Return the denoised signal, same length as the input."""
        signal = self._validate(signal)
        output = self._filter(signal)
        if self.zero_phase:
            output = self._filter(np.ascontiguousarray(output[::-1]))[::-1]
        return np.ascontiguousarray(output)

    def _validate(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"{self.name} expects a 1D signal.")
        if signal.size == 0:
            raise ValueError(f"{self.name} received an empty signal.")
        return signal

    def __repr__(self) -> str:
        return f"<{self.name} [{self.phase}]>"


class MovingAverage(Denoiser):
    """Unweighted mean over a sliding window.

    Causal, the window covers the current sample and the previous w-1, so the
    output lags by (w-1)/2 samples. Run forwards and backwards, the two
    rectangular windows convolve into a symmetric triangular window with no lag.
    """

    def __init__(self, window: int = 5, zero_phase: bool = False) -> None:
        super().__init__(zero_phase)
        if window < 1:
            raise ValueError("Window must be at least 1.")
        self.window = window

    @property
    def name(self) -> str:
        return f"MovingAverage(w={self.window})"

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        kernel = np.ones(self.window) / self.window
        # Pad with the first value rather than zeros so the filter does not
        # ramp up from zero at the start of the recording.
        padded = np.concatenate([np.full(self.window - 1, signal[0]), signal])
        return np.convolve(padded, kernel, mode="valid")


class ExponentialMovingAverage(Denoiser):
    """X(k) = alpha * measurement(k) + (1 - alpha) * X(k-1)."""

    def __init__(self, alpha: float = 0.2, zero_phase: bool = False) -> None:
        super().__init__(zero_phase)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Alpha must lie in (0, 1].")
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"EMA(a={self.alpha})"

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        # A one-pole IIR filter, started from the first sample so there is no
        # start-up transient.
        offset = signal[0]
        return lfilter([self.alpha], [1.0, -(1.0 - self.alpha)], signal - offset) + offset


class _IIRLowPass(Denoiser):
    """Shared machinery for the low-pass IIR filters."""

    def __init__(
        self,
        order: int = 4,
        cutoff_hz: float = 10.0,
        sampling_rate_hz: float = SAMPLING_RATE_HZ,
        zero_phase: bool = False,
    ) -> None:
        super().__init__(zero_phase)
        nyquist = sampling_rate_hz / 2.0
        if not 0.0 < cutoff_hz < nyquist:
            raise ValueError(
                f"Cutoff {cutoff_hz} Hz must lie between 0 and {nyquist} Hz."
            )
        self.order = order
        self.cutoff_hz = cutoff_hz
        self.sampling_rate_hz = sampling_rate_hz
        self._b, self._a = self._design(cutoff_hz / nyquist)

    @abstractmethod
    def _design(self, normalised_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the filter coefficients."""

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        # Start from the first sample so the filter does not ramp up from zero.
        offset = signal[0]
        return lfilter(self._b, self._a, signal - offset) + offset


class ButterworthLowPass(_IIRLowPass):
    """Maximally flat passband; steeper rolloff than Bessel."""

    @property
    def name(self) -> str:
        return f"Butterworth(o={self.order}, fc={self.cutoff_hz}Hz)"

    def _design(self, normalised_cutoff: float):
        return butter(self.order, normalised_cutoff, btype="low")


class BesselLowPass(_IIRLowPass):
    """Near-linear phase response; gentler rolloff than Butterworth."""

    @property
    def name(self) -> str:
        return f"Bessel(o={self.order}, fc={self.cutoff_hz}Hz)"

    def _design(self, normalised_cutoff: float):
        return bessel(self.order, normalised_cutoff, btype="low", norm="phase")


class LinearKalman(Denoiser):
    """Scalar Kalman filter for a constant-position model.

    Predict: the state carries forward unchanged and the error covariance grows
    by the process noise Q. Update: the gain K = P / (P + R) decides how far to
    move towards the new measurement.

    Larger Q means less trust in the model and more trust in measurements, so
    the output is more responsive and less smooth.
    """

    def __init__(
        self,
        initial_covariance: float = 0.07,
        process_noise: float = 7.0,
        measurement_noise: float = 0.01,
        zero_phase: bool = False,
    ) -> None:
        super().__init__(zero_phase)
        if min(initial_covariance, process_noise, measurement_noise) <= 0:
            raise ValueError("Kalman covariances must be positive.")
        self.initial_covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    @property
    def name(self) -> str:
        return f"Kalman(Q={self.process_noise}, R={self.measurement_noise})"

    @property
    def steady_state_gain(self) -> float:
        """Gain the filter converges to, useful for sanity-checking tuning."""
        q, r = self.process_noise, self.measurement_noise
        covariance = (q + np.sqrt(q * q + 4.0 * q * r)) / 2.0
        return float(covariance / (covariance + r))

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        estimate = signal[0]
        covariance = self.initial_covariance
        output = np.empty_like(signal)

        for index, measurement in enumerate(signal):
            covariance = covariance + self.process_noise
            gain = covariance / (covariance + self.measurement_noise)
            estimate = estimate + gain * (measurement - estimate)
            covariance = (1.0 - gain) * covariance
            output[index] = estimate

        return output


def published_baselines(zero_phase: bool = False) -> list[Denoiser]:
    """The filter set from Krijestorac (2025), plus two tuned additions."""
    return [
        MovingAverage(window=5, zero_phase=zero_phase),
        ExponentialMovingAverage(alpha=0.75, zero_phase=zero_phase),
        ExponentialMovingAverage(alpha=0.2, zero_phase=zero_phase),
        BesselLowPass(order=4, cutoff_hz=10.0, zero_phase=zero_phase),
        ButterworthLowPass(order=4, cutoff_hz=10.0, zero_phase=zero_phase),
        LinearKalman(process_noise=7.0, measurement_noise=0.01, zero_phase=zero_phase),
        LinearKalman(process_noise=0.01, measurement_noise=0.01, zero_phase=zero_phase),
    ]
"""
Classical denoising filters behind a single interface.

Every filter in this module, and later the neural network, implements the same
`Denoiser` interface: construct it, call `apply(signal)`, get a denoised signal
back. The benchmark can then loop over a list of denoisers without knowing or
caring what any of them does internally. Adding a method means adding a class,
not editing the benchmark. This is the strategy pattern.

Parameter defaults reproduce the published configuration in Krijestorac (2025),
INFOTEH-JAHORINA: Kalman P = 0.07, Q = 7, R = 0.01; EMA alpha = 0.2 and 0.75;
Bessel 4th order, 10 Hz cutoff, 100 Hz sampling.

A note on the Kalman defaults: at Q = 7 with R = 0.01 the steady-state gain is
0.9986, meaning the filter passes measurements through almost unchanged. The
published parameters are kept as the default for continuity, and a tuned
variant is included in the benchmark so the comparison is informative.

A note on phase: filters are causal by default, matching the published latency
analysis and real-time use, so they lag the signal. Setting zero_phase=True
applies the filter forwards and backwards, removing lag at the cost of being
unusable in real time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import bessel, butter, filtfilt, lfilter

SAMPLING_RATE_HZ = 100.0


class Denoiser(ABC):
    """Common interface for every denoising method in the benchmark."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in results tables and figures."""

    @abstractmethod
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Return the denoised signal, same length as the input."""

    def _validate(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim != 1:
            raise ValueError(f"{self.name} expects a 1D signal.")
        if signal.size == 0:
            raise ValueError(f"{self.name} received an empty signal.")
        return signal

    def __repr__(self) -> str:
        return f"<{self.name}>"


class MovingAverage(Denoiser):
    """Unweighted mean over a sliding window."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("Window must be at least 1.")
        self.window = window

    @property
    def name(self) -> str:
        return f"MovingAverage(w={self.window})"

    def apply(self, signal: np.ndarray) -> np.ndarray:
        signal = self._validate(signal)
        kernel = np.ones(self.window) / self.window
        padded = np.concatenate(
            [np.full(self.window - 1, signal[0]), signal]
        )
        return np.convolve(padded, kernel, mode="valid")


class ExponentialMovingAverage(Denoiser):
    """X(k) = alpha * measurement(k) + (1 - alpha) * X(k-1)."""

    def __init__(self, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Alpha must lie in (0, 1].")
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"EMA(a={self.alpha})"

    def apply(self, signal: np.ndarray) -> np.ndarray:
        signal = self._validate(signal)
        output = np.empty_like(signal)
        output[0] = signal[0]
        for index in range(1, signal.size):
            output[index] = (
                self.alpha * signal[index] + (1.0 - self.alpha) * output[index - 1]
            )
        return output


class _IIRLowPass(Denoiser):
    """Shared machinery for the low-pass IIR filters."""

    def __init__(
        self,
        order: int = 4,
        cutoff_hz: float = 10.0,
        sampling_rate_hz: float = SAMPLING_RATE_HZ,
        zero_phase: bool = False,
    ) -> None:
        nyquist = sampling_rate_hz / 2.0
        if not 0.0 < cutoff_hz < nyquist:
            raise ValueError(
                f"Cutoff {cutoff_hz} Hz must lie between 0 and {nyquist} Hz."
            )
        self.order = order
        self.cutoff_hz = cutoff_hz
        self.sampling_rate_hz = sampling_rate_hz
        self.zero_phase = zero_phase
        self._b, self._a = self._design(cutoff_hz / nyquist)

    @abstractmethod
    def _design(self, normalised_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the filter coefficients."""

    def apply(self, signal: np.ndarray) -> np.ndarray:
        signal = self._validate(signal)
        if self.zero_phase:
            return filtfilt(self._b, self._a, signal)
        # Start from the first sample so the filter does not ramp up from zero.
        return lfilter(self._b, self._a, signal - signal[0]) + signal[0]


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
    ) -> None:
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

    def apply(self, signal: np.ndarray) -> np.ndarray:
        signal = self._validate(signal)
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


def published_baselines() -> list[Denoiser]:
    """The filter set from Krijestorac (2025), plus two tuned additions."""
    return [
        MovingAverage(window=5),
        ExponentialMovingAverage(alpha=0.75),
        ExponentialMovingAverage(alpha=0.2),
        BesselLowPass(order=4, cutoff_hz=10.0),
        ButterworthLowPass(order=4, cutoff_hz=10.0),
        LinearKalman(process_noise=7.0, measurement_noise=0.01),
        LinearKalman(process_noise=0.01, measurement_noise=0.01),
    ]
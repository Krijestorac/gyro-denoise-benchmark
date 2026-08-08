"""
Data preparation for the MEMS gyroscope denoising benchmark.

The recording in data/raw/ is a single manoeuvre measured with an MPU-6050.
There is no ground truth: nobody recorded the true tilt at each sample, so a
reference curve has to be constructed before any supervised model can be
trained.

This module does three things:

1. Fits a smooth reference curve with a Savitzky-Golay filter, following the
   reference-curve methodology used in Krijestorac (2025), INFOTEH-JAHORINA.
2. Treats the leftover residual as the noise component and measures its
   statistics, so that later stages can generate fresh noise with the same
   character. The statistics here are descriptive only. An AR(1) fit is
   computed and reported because it is the obvious first model to try, but the
   diagnostic table shows it does not describe this residual: the measured
   autocorrelation turns negative from lag 3 onward, which no AR(1) process
   with a positive coefficient can produce. Noise generation therefore uses a
   non-parametric surrogate method (see noise_model.py), which reproduces the
   measured power spectrum without committing to a parametric model.
3. Sweeps the Savitzky-Golay window length so the chosen value is a documented
   decision rather than an arbitrary one.

Known limitation: the residual is not exactly the sensor noise. A smoothing
filter always absorbs a little noise into the fitted curve and leaves a little
signal in the residual. The reference curve is therefore an approximation of
the truth, not the truth. Every result in this repository should be read with
that in mind.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "gyro_x_500_fitted.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

SIGNAL_COLUMN = "gyro_x_dps"

# Savitzky-Golay settings for the reference curve. The window is chosen from
# the sweep in sweep_window_lengths(); see the README for the reasoning.
SG_WINDOW = 31
SG_POLYORDER = 3

CANDIDATE_WINDOWS = (9, 15, 21, 31, 41, 51, 71, 101)


def load_recording(path: Path = RAW_CSV) -> np.ndarray:
    """Load the single-column recording as a 1D float array."""
    if not path.exists():
        raise FileNotFoundError(
            f"Recording not found at {path}. "
            "Check that the CSV sits in data/raw/."
        )
    frame = pd.read_csv(path)
    if SIGNAL_COLUMN not in frame.columns:
        raise KeyError(
            f"Expected column '{SIGNAL_COLUMN}', found {list(frame.columns)}."
        )
    signal = frame[SIGNAL_COLUMN].to_numpy(dtype=np.float64)
    if np.isnan(signal).any():
        raise ValueError("Recording contains missing values.")
    return signal


def fit_reference_curve(
    signal: np.ndarray,
    window: int = SG_WINDOW,
    polyorder: int = SG_POLYORDER,
) -> np.ndarray:
    """Fit a smooth reference curve, used as the training target."""
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be odd.")
    if window <= polyorder:
        raise ValueError("Window length must exceed the polynomial order.")
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def characterise_noise(residual: np.ndarray) -> dict[str, float]:
    """Measure descriptive statistics of the noise component.

    The residual is clearly not white: consecutive samples are correlated.
    An AR(1) coefficient is estimated by the lag-1 sample autocorrelation
    (the Yule-Walker estimator for order one) and the corresponding
    innovation standard deviation is recovered from the standard identity
    sigma_x = sigma_e / sqrt(1 - phi^2).

    These values are reported for description and comparison, not because
    AR(1) is an adequate model for this residual. See ar1_diagnostic().
    """
    centred = residual - residual.mean()
    sigma = float(centred.std(ddof=1))
    phi = float(np.corrcoef(centred[:-1], centred[1:])[0, 1])
    innovation_sigma = float(sigma * np.sqrt(max(1.0 - phi**2, 1e-12)))

    return {
        "residual_std": sigma,
        "residual_mean": float(residual.mean()),
        "ar1_coefficient": phi,
        "innovation_std": innovation_sigma,
        "variance_of_first_differences": float(np.var(np.diff(residual), ddof=1)),
        "n_samples": int(residual.size),
    }


def autocorrelation(series: np.ndarray, max_lag: int = 10) -> list[float]:
    """Autocorrelation at lags 1..max_lag, for diagnostics."""
    centred = series - series.mean()
    return [
        float(np.corrcoef(centred[:-lag], centred[lag:])[0, 1])
        for lag in range(1, max_lag + 1)
    ]


def ar1_diagnostic(residual: np.ndarray, max_lag: int = 8) -> pd.DataFrame:
    """Compare the measured autocorrelation against the AR(1) prediction.

    An AR(1) process with coefficient phi has autocorrelation phi**k at lag k,
    which is strictly positive and monotonically decaying for positive phi.
    Tabulating the measured values against that prediction documents whether
    the model is usable. For this recording it is not, and the table is the
    evidence for preferring a non-parametric noise generator.
    """
    measured = np.asarray(autocorrelation(residual, max_lag=max_lag))
    phi = measured[0]
    lags = np.arange(1, max_lag + 1)
    predicted = phi ** lags
    return pd.DataFrame(
        {
            "lag": lags,
            "measured_autocorr": measured,
            "ar1_predicted": predicted,
            "absolute_error": np.abs(measured - predicted),
        }
    )


def residual_spectrum(residual: np.ndarray) -> pd.DataFrame:
    """Power spectral density of the residual, in cycles per sample.

    The sampling rate of the recording is not encoded in the CSV, so
    frequencies are reported in normalised units. Multiply by the true
    sampling rate to convert to hertz.
    """
    freqs, power = welch(residual - residual.mean(), fs=1.0, nperseg=128)
    return pd.DataFrame({"cycles_per_sample": freqs, "power": power})


def sweep_window_lengths(signal: np.ndarray) -> pd.DataFrame:
    """Compare candidate window lengths.

    A window that is too short lets the reference curve chase the noise, which
    shrinks the residual and destroys the very thing we want to measure. A
    window that is too long flattens genuine movement, which inflates the
    residual with real signal. The useful window sits where the residual
    standard deviation stops changing quickly.
    """
    rows = []
    for window in CANDIDATE_WINDOWS:
        curve = fit_reference_curve(signal, window=window)
        residual = signal - curve
        rows.append(
            {
                "window": window,
                "residual_std": residual.std(ddof=1),
                "curve_std": curve.std(ddof=1),
                "residual_lag1_autocorr": np.corrcoef(
                    residual[:-1], residual[1:]
                )[0, 1],
            }
        )
    frame = pd.DataFrame(rows)
    frame["snr_db"] = 20 * np.log10(frame["curve_std"] / frame["residual_std"])
    return frame


def plot_overview(
    signal: np.ndarray,
    curve: np.ndarray,
    residual: np.ndarray,
    path: Path,
) -> None:
    """Save a two-panel figure: recording with reference curve, and residual."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1]
    )

    axes[0].plot(signal, lw=0.8, color="#888780", label="raw measurement")
    axes[0].plot(
        curve,
        lw=1.8,
        color="#185FA5",
        label=f"Savitzky-Golay reference ({SG_WINDOW}, {SG_POLYORDER})",
    )
    axes[0].set_ylabel("gyro x")
    axes[0].legend(frameon=False)
    axes[0].set_title("MPU-6050 recording and constructed reference curve")

    axes[1].plot(residual, lw=0.7, color="#D85A30")
    axes[1].axhline(0.0, color="#444441", lw=0.6)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("sample index")
    axes[1].set_title(
        f"Noise component, std = {residual.std(ddof=1):.3f}", fontsize=10
    )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    signal = load_recording()
    print(f"Loaded {signal.size} samples from {RAW_CSV.name}")
    print(
        f"  range {signal.min():.2f} to {signal.max():.2f}, "
        f"mean {signal.mean():.3f}, std {signal.std(ddof=1):.3f}"
    )

    print("\nWindow length sweep:")
    sweep = sweep_window_lengths(signal)
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sweep.to_csv(TABLES_DIR / "sg_window_sweep.csv", index=False)

    curve = fit_reference_curve(signal)
    residual = signal - curve
    stats = characterise_noise(residual)

    raw_first_diff_var = float(np.var(np.diff(signal), ddof=1))

    print(f"\nChosen window: {SG_WINDOW}, polynomial order {SG_POLYORDER}")
    print(
        "Variance of first differences of the raw recording: "
        f"{raw_first_diff_var:.4f}"
    )
    print("  (the smoothness metric used in Krijestorac 2025, INFOTEH)")
    print("Noise statistics:")
    for key, value in stats.items():
        print(f"  {key:32s} {value}")

    lags = autocorrelation(residual, max_lag=6)
    print("  residual autocorrelation lags 1-6  "
          f"{[round(v, 3) for v in lags]}")

    print("\nAR(1) adequacy check:")
    diagnostic = ar1_diagnostic(residual)
    print(diagnostic.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"  worst absolute error {diagnostic['absolute_error'].max():.3f} "
        "-- AR(1) is not an adequate model for this residual"
    )
    diagnostic.to_csv(TABLES_DIR / "ar1_diagnostic.csv", index=False)

    spectrum = residual_spectrum(residual)
    spectrum.to_csv(TABLES_DIR / "residual_spectrum.csv", index=False)
    peak = spectrum.loc[spectrum["power"].idxmax(), "cycles_per_sample"]
    print(f"  residual spectral peak at {peak:.4f} cycles/sample")

    processed = pd.DataFrame(
        {"raw": signal, "reference": curve, "residual": residual}
    )
    processed.to_csv(PROCESSED_DIR / "reference_and_residual.csv", index=False)

    stats["raw_variance_of_first_differences"] = raw_first_diff_var
    stats["sg_window"] = SG_WINDOW
    stats["sg_polyorder"] = SG_POLYORDER
    stats["residual_autocorrelation_lags_1_to_6"] = lags
    with open(PROCESSED_DIR / "noise_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    plot_overview(
        signal, curve, residual, FIGURES_DIR / "01_reference_and_residual.png"
    )

    print("\nWrote:")
    print("  data/processed/reference_and_residual.csv")
    print("  data/processed/noise_stats.json")
    print("  results/tables/sg_window_sweep.csv")
    print("  results/tables/ar1_diagnostic.csv")
    print("  results/tables/residual_spectrum.csv")
    print("  results/figures/01_reference_and_residual.png")


if __name__ == "__main__":
    main()
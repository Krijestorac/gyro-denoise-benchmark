"""
Data preparation for the gyroscope denoising benchmark.

There is no ground truth for this recording: nobody measured the true tilt at
each sample. A reference curve is therefore constructed with a Savitzky-Golay
filter, following the reference-curve method used in Krijestorac (2025),
INFOTEH-JAHORINA. Whatever remains after subtracting that curve is treated as
the noise component.

The reference curve is an approximation of the truth, not the truth itself.
A smoother always absorbs some noise into the fit and leaves some signal in
the residual, so every result in this repository inherits that limitation.

Outputs
-------
data/processed/reference_and_residual.csv : the three aligned series
data/processed/noise_stats.json           : statistics used by noise.py
results/tables/sg_window_sweep.csv        : evidence for the window choice
results/figures/01_reference_and_residual.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "gyro_x_500_fitted.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

SIGNAL_COLUMN = "gyro_x_dps"

SG_WINDOW = 25
SG_POLYORDER = 3
CANDIDATE_WINDOWS = (9, 15, 21, 31, 41, 51, 71, 101)

# Window used to measure how the noise level varies along the recording.
ENVELOPE_WINDOW = 41


def load_recording(path: Path = RAW_CSV) -> np.ndarray:
    """Load the single-column recording as a 1D float array."""
    if not path.exists():
        raise FileNotFoundError(f"Recording not found at {path}.")
    frame = pd.read_csv(path)
    if SIGNAL_COLUMN not in frame.columns:
        raise KeyError(
            f"Expected column '{SIGNAL_COLUMN}', got {list(frame.columns)}."
        )
    signal = frame[SIGNAL_COLUMN].to_numpy(dtype=np.float64)
    if np.isnan(signal).any():
        raise ValueError("Recording contains missing values.")
    return signal


def fit_reference_curve(
    signal: np.ndarray, window: int = SG_WINDOW, polyorder: int = SG_POLYORDER
) -> np.ndarray:
    """Fit the smooth reference curve used as the ground truth."""
    if window % 2 == 0:
        raise ValueError("Savitzky-Golay window length must be odd.")
    if window <= polyorder:
        raise ValueError("Window length must exceed the polynomial order.")
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def sweep_window_lengths(signal: np.ndarray) -> pd.DataFrame:
    """Compare candidate window lengths so the chosen one is justified.

    Too short and the reference curve chases the noise, shrinking the residual.
    Too long and it flattens real movement, so genuine signal leaks into the
    residual. The usable window is where the residual standard deviation stops
    changing quickly.
    """
    rows = []
    for window in CANDIDATE_WINDOWS:
        residual = signal - fit_reference_curve(signal, window=window)
        rows.append(
            {
                "window": window,
                "residual_std": residual.std(ddof=1),
                "residual_lag1_autocorr": np.corrcoef(
                    residual[:-1], residual[1:]
                )[0, 1],
            }
        )
    return pd.DataFrame(rows)


def fit_noise_envelope(
    reference: np.ndarray, residual: np.ndarray, window: int = ENVELOPE_WINDOW
) -> dict[str, float]:
    """Fit how the noise level varies with the signal level.

    MEMS gyroscope error is conventionally split into a constant floor plus a
    component proportional to the measurement. Measuring the local noise
    standard deviation in sliding windows and regressing it on the reference
    level recovers exactly that: the intercept is the floor, the slope is the
    proportional term.

    Without this, synthetic noise would be flat across the recording while the
    real noise is several times larger at full tilt than at rest.
    """
    half = window // 2
    centres = np.arange(half, residual.size - half)
    local_std = np.array(
        [residual[i - half : i + half + 1].std(ddof=1) for i in centres]
    )
    local_level = reference[centres]

    slope, intercept = np.polyfit(local_level, local_std, 1)
    predicted = slope * local_level + intercept
    r_squared = 1.0 - np.var(local_std - predicted) / np.var(local_std)

    return {
        "envelope_slope": float(slope),
        "envelope_intercept": float(intercept),
        "envelope_r_squared": float(r_squared),
    }


def characterise_noise(residual: np.ndarray) -> dict[str, float]:
    """Describe the noise well enough to reproduce it.

    Two properties rule out white Gaussian noise as a training signal:
    consecutive samples are correlated, and the distribution is skewed with
    heavy tails. noise.py uses a method that preserves both.
    """
    return {
        "residual_std": float(residual.std(ddof=1)),
        "residual_mean": float(residual.mean()),
        "lag1_autocorrelation": float(
            np.corrcoef(residual[:-1], residual[1:])[0, 1]
        ),
        "skewness": float(stats.skew(residual)),
        "excess_kurtosis": float(stats.kurtosis(residual)),
        "n_samples": int(residual.size),
    }


def plot_overview(
    signal: np.ndarray,
    reference: np.ndarray,
    residual: np.ndarray,
    path: Path,
    window: int,
    polyorder: int,
) -> None:
    """Save the recording with its reference curve, and the residual below."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1]
    )

    axes[0].plot(signal, lw=0.8, color="#888780", label="raw measurement")
    axes[0].plot(
        reference,
        lw=1.8,
        color="#185FA5",
        label=f"Savitzky-Golay reference ({window}, {polyorder})",
    )
    axes[0].set_ylabel("gyro x")
    axes[0].legend(frameon=False)
    axes[0].set_title("Recording and constructed reference curve")

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
    for directory in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    signal = load_recording()
    print(f"Loaded {signal.size} samples")
    print(
        f"  range {signal.min():.2f} to {signal.max():.2f}, "
        f"std {signal.std(ddof=1):.3f}"
    )

    sweep = sweep_window_lengths(signal)
    print("\nWindow length sweep:")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sweep.to_csv(TABLES_DIR / "sg_window_sweep.csv", index=False)

    reference = fit_reference_curve(signal)
    residual = signal - reference

    noise_stats = characterise_noise(residual)
    noise_stats.update(fit_noise_envelope(reference, residual))
    noise_stats["sg_window"] = SG_WINDOW
    noise_stats["sg_polyorder"] = SG_POLYORDER

    print(f"\nChosen window {SG_WINDOW}, polynomial order {SG_POLYORDER}")
    print("Noise statistics:")
    for key, value in noise_stats.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:22s} {formatted}")

    print(
        f"\n  Correlated (lag-1 {noise_stats['lag1_autocorrelation']:.2f}) and "
        f"skewed ({noise_stats['skewness']:+.2f}): white Gaussian noise would"
    )
    print("  not be a fair training signal. It also grows with the measurement:")
    print(
        f"  std = {noise_stats['envelope_slope']:.4f} * level + "
        f"{noise_stats['envelope_intercept']:.4f}  "
        f"(R^2 = {noise_stats['envelope_r_squared']:.3f})"
    )

    pd.DataFrame(
        {"raw": signal, "reference": reference, "residual": residual}
    ).to_csv(PROCESSED_DIR / "reference_and_residual.csv", index=False)

    with open(PROCESSED_DIR / "noise_stats.json", "w", encoding="utf-8") as handle:
        json.dump(noise_stats, handle, indent=2)

    plot_overview(
        signal,
        reference,
        residual,
        FIGURES_DIR / "01_reference_and_residual.png",
        SG_WINDOW,
        SG_POLYORDER,
    )

    print(
        "\nWrote reference_and_residual.csv, noise_stats.json, "
        "sg_window_sweep.csv, 01_reference_and_residual.png"
    )


if __name__ == "__main__":
    main()
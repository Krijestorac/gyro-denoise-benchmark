"""
Data preparation for the gyroscope denoising benchmark.


1. A reference curve
--------------------
There is no ground truth for this recording: nobody measured the true tilt at
each sample. A reference curve is constructed with a Savitzky-Golay filter,
following the reference-curve method used in Krijestorac (2025),
INFOTEH-JAHORINA. Whatever remains after subtracting it is treated as noise.


2. A fit / test split
---------------------
The first FIT_FRACTION of the recording is used to estimate the three noise
parameters. The rest is never touched here and is the only part any method is
scored on in evaluate.py. Without this split the noise parameters come from the
same samples used for scoring.

Units
-----
The recording is an integrated tilt ANGLE in degrees, not an angular rate. The
legacy column header says `gyro_x_dps`, which is wrong: the values run 0 to
8.75 over five seconds in a single rise and fall, which is an angle. Both
header names are accepted so the data file does not have to be rewritten, but
everything downstream treats and labels the series as degrees.

This also qualifies a physics claim below: MEMS scale-factor error is
proportional to angular rate, not to accumulated angle, so the envelope fitted
against angle is a description that fits this recording rather than a statement
about the sensor.
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

# `angle_deg` is the correct name; `gyro_x_dps` is the legacy header and is
# accepted so the raw file does not have to be rewritten. See the Units note.
SIGNAL_COLUMNS = ("angle_deg", "gyro_x_dps")

SG_WINDOW = 25
SG_POLYORDER = 3
# The chosen window must appear in the sweep, otherwise the table that exists
# to justify the choice does not contain the choice.
CANDIDATE_WINDOWS = (9, 15, 21, 25, 31, 41, 51, 71, 101)

# Window used to measure how the noise level varies along the recording.
ENVELOPE_WINDOW = 41

# Fraction of the recording used to estimate noise parameters. The remainder is
# held out and is the only part evaluate.py scores on.
FIT_FRACTION = 0.5


def load_recording(path: Path = RAW_CSV) -> np.ndarray:
    """Load the single-column recording as a 1D float array."""
    if not path.exists():
        raise FileNotFoundError(f"Recording not found at {path}.")
    frame = pd.read_csv(path)
    for column in SIGNAL_COLUMNS:
        if column in frame.columns:
            signal = frame[column].to_numpy(dtype=np.float64)
            break
    else:
        raise KeyError(
            f"Expected one of {SIGNAL_COLUMNS}, got {list(frame.columns)}."
        )
    if np.isnan(signal).any():
        raise ValueError("Recording contains missing values.")
    return signal


def fit_reference_curve(
    signal: np.ndarray, window: int = SG_WINDOW, polyorder: int = SG_POLYORDER
) -> np.ndarray:
    """Fit the smooth reference curve used in place of a ground truth."""
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

    Read the lag-1 column honestly. It runs from about -0.10 at window 9 to
    about +0.63 at window 101. Residual autocorrelation is therefore mostly a
    property of THIS CHOICE, not of the sensor: a longer window leaves more real
    signal in the residual, and real signal is autocorrelated. The value at the
    chosen window is what noise.py reproduces, and it should be described that
    way rather than as a measured sensor property.
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
                "chosen": window == SG_WINDOW,
            }
        )
    return pd.DataFrame(rows)


def fit_noise_envelope(
    reference: np.ndarray, residual: np.ndarray, window: int = ENVELOPE_WINDOW
) -> dict[str, float]:
    """Fit how the local noise level varies along the recording.

    Regressing the local residual standard deviation on the reference level
    gives a floor (intercept) and a growth term (slope).

    The confound, measured rather than assumed
    ------------------------------------------
    This recording is a single rise and fall, so signal level is not obviously
    separable from signal speed, and a Savitzky-Golay fit's own error grows with
    speed and curvature. A good fit against level therefore does not by itself
    prove the noise scales with angle; it may be tracking fit error. The same
    regression is run against |d(reference)/dt| and both R-squared values are
    reported, so the reader can see how much the level story is worth.
    """
    half = window // 2
    centres = np.arange(half, residual.size - half)
    local_std = np.array(
        [residual[i - half : i + half + 1].std(ddof=1) for i in centres]
    )
    local_level = np.abs(reference[centres])
    local_speed = np.abs(np.gradient(reference))[centres]

    def linear_fit(predictor: np.ndarray) -> tuple[float, float, float]:
        slope, intercept = np.polyfit(predictor, local_std, 1)
        predicted = slope * predictor + intercept
        r_squared = 1.0 - np.var(local_std - predicted) / np.var(local_std)
        return float(slope), float(intercept), float(r_squared)

    slope, intercept, r_squared = linear_fit(local_level)
    _, _, speed_r_squared = linear_fit(local_speed)

    return {
        "envelope_slope": slope,
        "envelope_intercept": intercept,
        "envelope_r_squared": r_squared,
        "envelope_r_squared_vs_speed": speed_r_squared,
        "level_speed_correlation": float(
            np.corrcoef(local_level, local_speed)[0, 1]
        ),
    }


def characterise_noise(residual: np.ndarray) -> dict[str, float]:
    """Describe the residual well enough to reproduce it, and to say what the
    generator will miss.

    noise.py reproduces the standard deviation and the lag-1 autocorrelation.
    It does not reproduce the skewness or the excess kurtosis, both reported
    here so the omission is on the record.
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
    fit_end: int,
    path: Path,
    window: int,
    polyorder: int,
) -> None:
    """Save the recording with its reference curve, and the residual below."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1]
    )

    axes[0].plot(signal, lw=0.8, color="#888780", label="recording")
    axes[0].plot(
        reference,
        lw=1.8,
        color="#185FA5",
        label=f"Savitzky-Golay reference ({window}, {polyorder})",
    )
    axes[0].set_ylabel("tilt angle (degrees)")
    axes[0].legend(frameon=False, loc="lower center")
    axes[0].set_title("Recording and constructed reference curve")

    axes[1].plot(residual, lw=0.7, color="#D85A30")
    axes[1].axhline(0.0, color="#444441", lw=0.6)
    axes[1].set_ylabel("residual (deg)")
    axes[1].set_xlabel("sample index")
    axes[1].set_title(
        f"Residual, std = {residual.std(ddof=1):.3f} deg", fontsize=10
    )

    for axis in axes:
        axis.axvspan(0, fit_end, color="#185FA5", alpha=0.06)
        axis.axvline(fit_end, color="#185FA5", lw=1.0, ls="--")
        axis.spines[["top", "right"]].set_visible(False)

    top = axes[0].get_ylim()[1]
    axes[0].text(
        fit_end * 0.5, top * 0.97, "noise parameters fitted here",
        ha="center", va="top", fontsize=8, color="#185FA5",
    )
    axes[0].text(
        fit_end + (signal.size - fit_end) * 0.5, top * 0.97, "held out for scoring",
        ha="center", va="top", fontsize=8, color="#444441",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    for directory in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    signal = load_recording()
    fit_end = int(signal.size * FIT_FRACTION)
    print(f"Loaded {signal.size} samples from {RAW_CSV.name}")
    print(
        f"  range {signal.min():.2f} to {signal.max():.2f} deg, "
        f"std {signal.std(ddof=1):.3f}"
    )
    print(f"  samples 0..{fit_end - 1} used for fitting")
    print(f"  samples {fit_end}..{signal.size - 1} held out for scoring")

    sweep = sweep_window_lengths(signal)
    print("\nWindow length sweep:")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sweep.to_csv(TABLES_DIR / "sg_window_sweep.csv", index=False)

    reference = fit_reference_curve(signal)
    residual = signal - reference

    # Everything below is estimated on the fit half only.
    noise_stats = characterise_noise(residual[:fit_end])
    noise_stats.update(fit_noise_envelope(reference[:fit_end], residual[:fit_end]))
    noise_stats["sg_window"] = SG_WINDOW
    noise_stats["sg_polyorder"] = SG_POLYORDER
    noise_stats["fit_end"] = fit_end
    noise_stats["n_total_samples"] = int(signal.size)

    print(f"\nChosen window {SG_WINDOW}, polynomial order {SG_POLYORDER}")
    print("Noise parameters (fit half only):")
    for key, value in noise_stats.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        print(f"  {key:28s} {formatted}")

    print("\nWhat noise.py will and will not reproduce:")
    print(
        f"  reproduced      std {noise_stats['residual_std']:.4f}, "
        f"lag-1 {noise_stats['lag1_autocorrelation']:+.3f}"
    )
    print(
        f"  NOT reproduced  skewness {noise_stats['skewness']:+.3f}, "
        f"excess kurtosis {noise_stats['excess_kurtosis']:+.3f} "
        "(the generator is Gaussian)"
    )
    print(
        "\n  Read the lag-1 value as a property of the chosen smoothing window,"
        "\n  not of the sensor: see the sweep table above."
    )
    print(
        f"\n  Envelope: std = {noise_stats['envelope_slope']:.4f} * |angle| + "
        f"{noise_stats['envelope_intercept']:.4f}"
    )
    print(
        f"    R^2 against level {noise_stats['envelope_r_squared']:.3f}, "
        f"against speed {noise_stats['envelope_r_squared_vs_speed']:.3f}, "
        f"level/speed correlation {noise_stats['level_speed_correlation']:+.3f}"
    )
    if (
        noise_stats["envelope_r_squared_vs_speed"]
        >= noise_stats["envelope_r_squared"]
    ):
        print(
            "    Speed explains the noise level at least as well as angle does."
            "\n    Treat the envelope as a description, not a physical claim."
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
        fit_end,
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
"""
Benchmark every denoising method under matched conditions.

Two corrections make this comparison fair, and both change the outcome.

Phase
    The network sees samples on both sides of every point. A causal filter sees
    only the past, so it lags, and lag inflates error. Comparing the two
    directly rewards the network for an advantage that has nothing to do with
    denoising quality. The classical filters therefore run zero-phase here.

Tuning
    The network learned from 2000 examples of this noise. Filters using
    parameters chosen for a different recording are not a fair opponent. Each
    filter's parameter is tuned on synthetic data only, which is the same
    information the network had. The real recording is never used for tuning.

Results are reported on two sets: the real recording, which is the honest
headline, and a set of held-out synthetic signals, which gives enough repeats
to test whether the differences are stable or coincidence.

Run:  python src/evaluate.py
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

from filters import (
    BesselLowPass,
    ButterworthLowPass,
    Denoiser,
    ExponentialMovingAverage,
    LinearKalman,
    MovingAverage,
)
from model import NeuralDenoiser
from noise import make_noise
from signals import make_curve_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

N_TUNING = 60
N_TEST = 100
TUNING_SEED = 99
TEST_SEED = 123

CUTOFF_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0)
WINDOW_GRID = (3, 5, 7, 9, 11, 15, 21, 31)
ALPHA_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
PROCESS_NOISE_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 7.0)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def make_pairs(
    n: int, residual: np.ndarray, envelope: dict[str, float], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, clean) synthetic signal pairs."""
    rng = np.random.default_rng(seed)
    clean = make_curve_batch(n, rng)
    noisy = np.stack(
        [
            curve
            + make_noise(
                residual,
                curve,
                envelope["envelope_slope"],
                envelope["envelope_intercept"],
                rng,
            )
            for curve in clean
        ]
    )
    return noisy, clean


def mean_rmse(
    denoiser: Denoiser, noisy: np.ndarray, clean: np.ndarray
) -> float:
    return float(
        np.mean([rmse(denoiser.apply(n), c) for n, c in zip(noisy, clean)])
    )


def tune(
    build: callable, grid: tuple, noisy: np.ndarray, clean: np.ndarray
):
    """Pick the grid value giving the lowest mean error on the tuning set."""
    scored = [(mean_rmse(build(value), noisy, clean), value) for value in grid]
    best_score, best_value = min(scored)
    return build(best_value), best_value, best_score


def build_methods(
    noisy: np.ndarray, clean: np.ndarray
) -> tuple[list[Denoiser], pd.DataFrame]:
    """Tune every classical filter, then return them alongside the network."""
    records = []
    methods: list[Denoiser] = []

    for label, build, grid in (
        (
            "Bessel",
            lambda fc: BesselLowPass(order=4, cutoff_hz=fc, zero_phase=True),
            CUTOFF_GRID,
        ),
        (
            "Butterworth",
            lambda fc: ButterworthLowPass(order=4, cutoff_hz=fc, zero_phase=True),
            CUTOFF_GRID,
        ),
        ("MovingAverage", lambda w: MovingAverage(window=w), WINDOW_GRID),
        ("EMA", lambda a: ExponentialMovingAverage(alpha=a), ALPHA_GRID),
        ("Kalman", lambda q: LinearKalman(process_noise=q), PROCESS_NOISE_GRID),
    ):
        denoiser, value, score = tune(build, grid, noisy, clean)
        methods.append(denoiser)
        records.append(
            {"filter": label, "chosen_parameter": value, "tuning_rmse": score}
        )

    methods.append(NeuralDenoiser.load(MODELS_DIR / "conv_denoiser.pt"))
    return methods, pd.DataFrame(records)


def plot_results(
    raw: np.ndarray,
    reference: np.ndarray,
    methods: list[Denoiser],
    real_scores: pd.DataFrame,
    path: Path,
) -> None:
    best_classical = real_scores[
        ~real_scores["method"].str.startswith("ConvDenoiser")
    ].iloc[0]["method"]
    lookup = {denoiser.name: denoiser for denoiser in methods}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[1.5, 1])

    axes[0].plot(raw, lw=0.7, color="#C9C7C1", label="raw")
    axes[0].plot(reference, lw=2.0, color="#444441", label="reference")
    axes[0].plot(
        lookup[best_classical].apply(raw), lw=1.3, color="#185FA5",
        label=f"best classical: {best_classical}",
    )
    axes[0].plot(
        lookup["ConvDenoiser"].apply(raw), lw=1.3, color="#D85A30",
        label="ConvDenoiser",
    )
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("angle (degrees)")
    axes[0].set_title("Real recording, held out from training")
    axes[0].legend(frameon=False, fontsize=8)

    ordered = real_scores.sort_values("rmse")
    colours = [
        "#D85A30" if name.startswith("ConvDenoiser") else "#185FA5"
        for name in ordered["method"]
    ]
    axes[1].barh(range(len(ordered)), ordered["rmse"], color=colours)
    axes[1].set_yticks(range(len(ordered)))
    axes[1].set_yticklabels(ordered["method"], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE on the real recording")
    axes[1].set_title("Lower is better")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    for directory in (FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(PROCESSED_DIR / "reference_and_residual.csv")
    raw = frame["raw"].to_numpy()
    reference = frame["reference"].to_numpy()
    residual = frame["residual"].to_numpy()
    with open(PROCESSED_DIR / "noise_stats.json", encoding="utf-8") as handle:
        envelope = json.load(handle)

    print("Tuning classical filters on synthetic data only...")
    tuning_noisy, tuning_clean = make_pairs(
        N_TUNING, residual, envelope, TUNING_SEED
    )
    methods, tuning_table = build_methods(tuning_noisy, tuning_clean)
    print(tuning_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nRESULT 1: the real recording")
    real_scores = pd.DataFrame(
        [
            {"method": denoiser.name, "rmse": rmse(denoiser.apply(raw), reference)}
            for denoiser in methods
        ]
    ).sort_values("rmse", ignore_index=True)
    real_scores.loc[len(real_scores)] = {
        "method": "(raw, unfiltered)",
        "rmse": rmse(raw, reference),
    }
    print(real_scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nRESULT 2: {N_TEST} held-out synthetic signals")
    test_noisy, test_clean = make_pairs(N_TEST, residual, envelope, TEST_SEED)
    per_signal = {
        denoiser.name: np.array(
            [rmse(denoiser.apply(n), c) for n, c in zip(test_noisy, test_clean)]
        )
        for denoiser in methods
    }
    synthetic_scores = pd.DataFrame(
        [
            {"method": name, "mean_rmse": scores.mean(), "std_rmse": scores.std(ddof=1)}
            for name, scores in per_signal.items()
        ]
    ).sort_values("mean_rmse", ignore_index=True)
    print(synthetic_scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    winner = synthetic_scores.loc[0, "method"]
    runner_up = synthetic_scores.loc[1, "method"]
    test = stats.wilcoxon(per_signal[winner], per_signal[runner_up])
    difference = per_signal[runner_up] - per_signal[winner]

    print(f"\nPaired Wilcoxon signed-rank test: {winner} vs {runner_up}")
    print(f"  median difference {np.median(difference):+.5f}")
    print(f"  statistic {test.statistic:.1f}, p = {test.pvalue:.2e}")
    print(
        "  the difference is statistically significant"
        if test.pvalue < 0.05
        else "  the difference is not statistically significant"
    )

    real_scores.to_csv(TABLES_DIR / "benchmark_real.csv", index=False)
    synthetic_scores.to_csv(TABLES_DIR / "benchmark_synthetic.csv", index=False)
    tuning_table.to_csv(TABLES_DIR / "tuned_parameters.csv", index=False)
    plot_results(
        raw, reference, methods, real_scores, FIGURES_DIR / "03_benchmark.png"
    )

    print("\nWrote benchmark_real.csv, benchmark_synthetic.csv, "
          "tuned_parameters.csv, 03_benchmark.png")


if __name__ == "__main__":
    main()
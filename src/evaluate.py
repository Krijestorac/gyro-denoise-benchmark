"""
Benchmark every denoising method under matched conditions.

The comparison this file exists to get right
--------------------------------------------
The convolutional network is not causal: it sees samples on both sides of every
point. A causal filter sees only the past, so it lags, and lag inflates error.
The previous version of this file gave a zero-phase mode to the two IIR filters
and left the moving average, the EMA and the Kalman filter causal, then
described the result as "matched conditions". It was not matched, and the
network's margin over the moving average was mostly phase, not denoising.

So two tables are produced, and neither mixes modes:

  Table A, matched non-causal.  Every classical filter runs forwards and
      backwards (zero phase), re-tuned in that mode. The network runs as it is.
      This is the accuracy comparison, and it is the headline.

  Table B, causal only.  Every classical filter runs forwards only, re-tuned in
      that mode. The network is ABSENT, because it cannot run causally without
      being redesigned. This is the real-time comparison, and it is the one that
      matters for the embedded deployment in Krijestorac (2025).

Tuning
------
The network learned from 2000 examples of this noise, so filters using
parameters chosen for a different recording are not a fair opponent. Each
filter's parameter is tuned by grid search on synthetic data -- the same
information the network had -- and separately for each phase mode, because the
optimum moves: causally, the lag penalty pushes towards barely filtering at
all; zero-phase, it pushes towards heavy smoothing. The held-out half of the
real recording is never used for tuning.

Statistics
----------
The comparison is pre-specified: ConvDenoiser against each of the five
zero-phase baselines, five paired Wilcoxon tests, Holm-Bonferroni corrected.
The previous version tested whichever two methods happened to come first and
second in the ranking, using the same data that produced the ranking, which
makes the p-value uninterpretable.

Uncertainty is a percentile bootstrap 95% interval over the test signals. The
real recording is n = 1 and has no interval, which is one more reason it is not
the headline.

Run:  python src/evaluate.py     (after data_prep.py and train.py)
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
BOOTSTRAP_SEED = 7
BOOTSTRAP_RESAMPLES = 2000

# Grids reach further in both directions than before. A forward-backward pass
# smooths twice as hard, so the zero-phase optimum sits at a wider window than
# the causal one; causally, the lag penalty pushes the optimum the other way,
# towards barely filtering at all. `at_grid_edge` in the tuning table flags any
# parameter landing on a boundary, because that means the real optimum may lie
# outside the grid.
CUTOFF_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0)
WINDOW_GRID = (3, 5, 7, 9, 11, 15, 21, 31, 41, 51)
ALPHA_GRID = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
PROCESS_NOISE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 7.0)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def make_pairs(
    n: int, parameters: dict[str, float], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, clean) synthetic signal pairs."""
    rng = np.random.default_rng(seed)
    clean = make_curve_batch(n, rng)
    noisy = np.stack(
        [
            curve
            + make_noise(
                curve,
                parameters["phi"],
                parameters["slope"],
                parameters["intercept"],
                rng,
            )
            for curve in clean
        ]
    )
    return noisy, clean


def per_signal_rmse(
    denoiser: Denoiser, noisy: np.ndarray, clean: np.ndarray
) -> np.ndarray:
    return np.array([rmse(denoiser.apply(n), c) for n, c in zip(noisy, clean)])


def tune(build, grid, noisy: np.ndarray, clean: np.ndarray):
    """Pick the grid value giving the lowest mean error on the tuning set."""
    scored = [
        (float(per_signal_rmse(build(value), noisy, clean).mean()), value)
        for value in grid
    ]
    best_score, best_value = min(scored, key=lambda pair: pair[0])
    return build(best_value), best_value, best_score


def filter_specs(zero_phase: bool):
    """The five classical methods and the grid each one is tuned over.

    Note that `zero_phase` is passed to ALL FIVE. The bug this file was written
    to fix was passing it to only the first two.
    """
    return (
        (
            "Bessel",
            lambda fc: BesselLowPass(order=4, cutoff_hz=fc, zero_phase=zero_phase),
            CUTOFF_GRID,
        ),
        (
            "Butterworth",
            lambda fc: ButterworthLowPass(
                order=4, cutoff_hz=fc, zero_phase=zero_phase
            ),
            CUTOFF_GRID,
        ),
        (
            "MovingAverage",
            lambda w: MovingAverage(window=w, zero_phase=zero_phase),
            WINDOW_GRID,
        ),
        (
            "EMA",
            lambda a: ExponentialMovingAverage(alpha=a, zero_phase=zero_phase),
            ALPHA_GRID,
        ),
        (
            "Kalman",
            lambda q: LinearKalman(process_noise=q, zero_phase=zero_phase),
            PROCESS_NOISE_GRID,
        ),
    )


def tune_all(
    zero_phase: bool, noisy: np.ndarray, clean: np.ndarray
) -> tuple[list[Denoiser], list[dict]]:
    """Tune every classical filter in one phase mode."""
    methods, records = [], []
    for label, build, grid in filter_specs(zero_phase):
        denoiser, value, score = tune(build, grid, noisy, clean)
        methods.append(denoiser)
        records.append(
            {
                "filter": label,
                "phase": denoiser.phase,
                "chosen_parameter": value,
                "tuning_rmse": score,
                "grid_min": min(grid),
                "grid_max": max(grid),
                "at_grid_edge": value in (min(grid), max(grid)),
            }
        )
    return methods, records


def bootstrap_interval(
    scores: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean."""
    draws = rng.integers(0, scores.size, size=(BOOTSTRAP_RESAMPLES, scores.size))
    means = scores[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment for a family of tests."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compare_against_network(
    per_signal: dict[str, np.ndarray], network_name: str
) -> pd.DataFrame:
    """Pre-specified family: the network against every zero-phase baseline."""
    rows, raw_p = [], []

    for name in [n for n in per_signal if n != network_name]:
        difference = per_signal[name] - per_signal[network_name]
        test = stats.wilcoxon(per_signal[network_name], per_signal[name])
        raw_p.append(float(test.pvalue))
        rows.append(
            {
                "comparison": f"{network_name} vs {name}",
                "median_rmse_difference": float(np.median(difference)),
                "network_better_on": int(np.sum(difference > 0)),
                "of_signals": int(difference.size),
                "wilcoxon_statistic": float(test.statistic),
                "p_raw": float(test.pvalue),
            }
        )

    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(raw_p)
    frame["verdict"] = np.where(
        frame["p_holm"] >= 0.05,
        "no significant difference",
        np.where(
            frame["median_rmse_difference"] > 0, "network better", "network worse"
        ),
    )
    return frame.sort_values("p_holm", ignore_index=True)


def plot_results(
    raw: np.ndarray,
    reference: np.ndarray,
    fit_end: int,
    methods: list[Denoiser],
    synthetic: pd.DataFrame,
    path: Path,
) -> None:
    """Left: the recording with the held-out region marked. Right: synthetic."""
    lookup = {denoiser.name: denoiser for denoiser in methods}
    best_classical = synthetic[synthetic["method"] != "ConvDenoiser"].iloc[0][
        "method"
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), width_ratios=[1.5, 1])

    axes[0].plot(raw, lw=0.7, color="#C9C7C1", label="recording")
    axes[0].plot(reference, lw=2.0, color="#444441", label="SG reference")
    axes[0].plot(
        lookup[best_classical].apply(raw),
        lw=1.3,
        color="#185FA5",
        label=f"best classical: {best_classical}",
    )
    axes[0].plot(
        lookup["ConvDenoiser"].apply(raw),
        lw=1.3,
        color="#D85A30",
        label="ConvDenoiser",
    )
    axes[0].axvspan(0, fit_end, color="#888780", alpha=0.12)
    axes[0].axvline(fit_end, color="#444441", lw=1.0, ls="--")
    axes[0].text(
        fit_end * 0.5,
        axes[0].get_ylim()[1] * 0.97,
        "used to fit the noise model\n(not scored)",
        ha="center",
        va="top",
        fontsize=8,
        color="#444441",
    )
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("tilt angle (degrees)")
    axes[0].set_title("Real recording (target is itself a smoother: see README)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    ordered = synthetic.sort_values("mean_rmse", ascending=False)
    positions = np.arange(len(ordered))
    colours = [
        "#D85A30" if name == "ConvDenoiser" else "#185FA5"
        for name in ordered["method"]
    ]
    errors = np.vstack(
        [
            ordered["mean_rmse"] - ordered["ci_low"],
            ordered["ci_high"] - ordered["mean_rmse"],
        ]
    )
    axes[1].barh(positions, ordered["mean_rmse"], color=colours)
    axes[1].errorbar(
        ordered["mean_rmse"],
        positions,
        xerr=errors,
        fmt="none",
        ecolor="#444441",
        capsize=3,
        lw=1.0,
    )
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(ordered["method"], fontsize=8)
    axes[1].set_xlabel("RMSE on held-out synthetic signals (95% bootstrap CI)")
    axes[1].set_title("Matched non-causal comparison; lower is better")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(
    tuning: pd.DataFrame,
    synthetic: pd.DataFrame,
    causal: pd.DataFrame,
    real: pd.DataFrame,
    significance: pd.DataFrame,
    fit_end: int,
    n_total: int,
    path: Path,
) -> None:
    """Write results/tables/summary.md so the README never holds stale numbers."""

    def table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False, floatfmt=".4f")
        except ImportError:  # tabulate not installed
            return "```\n" + frame.to_string(index=False) + "\n```"

    path.write_text(
        "\n".join(
            [
                "# Results",
                "",
                "Generated by `python src/evaluate.py`. Do not edit by hand.",
                "",
                f"Synthetic test set: {N_TEST} held-out curves, seed {TEST_SEED}.  ",
                f"Real recording: samples {fit_end}-{n_total - 1} "
                f"({n_total - fit_end} of {n_total}); the first {fit_end} were "
                "used to fit the noise model and are not scored.",
                "",
                "## A. Matched non-causal comparison (headline)",
                "",
                "Every classical filter runs forwards and backwards and is tuned "
                "in that mode. The network runs as trained.",
                "",
                table(synthetic),
                "",
                "## B. Causal comparison (real-time)",
                "",
                "Forward pass only, re-tuned. The network is absent: it is "
                "non-causal by construction and cannot compete here without "
                "being redesigned.",
                "",
                table(causal),
                "",
                "## C. Significance, pre-specified family",
                "",
                "ConvDenoiser against each zero-phase baseline, paired Wilcoxon "
                "signed-rank, Holm-Bonferroni corrected across the five tests.",
                "",
                table(significance),
                "",
                "## D. Real recording, held-out half",
                "",
                "A sanity check, not evidence. The target is a Savitzky-Golay "
                "curve, so low-pass filters reproduce it by construction. "
                "n = 1, so there is no interval to report.",
                "",
                table(real),
                "",
                "## E. Tuned parameters",
                "",
                table(tuning),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    for directory in (FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(PROCESSED_DIR / "reference_and_residual.csv")
    raw = frame["raw"].to_numpy()
    reference = frame["reference"].to_numpy()
    with open(PROCESSED_DIR / "noise_stats.json", encoding="utf-8") as handle:
        stats_json = json.load(handle)

    if "fit_end" not in stats_json:
        raise KeyError(
            "noise_stats.json predates the fit/test split. Re-run data_prep.py."
        )
    fit_end = int(stats_json["fit_end"])
    parameters = {
        "phi": stats_json["lag1_autocorrelation"],
        "slope": stats_json["envelope_slope"],
        "intercept": stats_json["envelope_intercept"],
    }

    tuning_noisy, tuning_clean = make_pairs(N_TUNING, parameters, TUNING_SEED)
    test_noisy, test_clean = make_pairs(N_TEST, parameters, TEST_SEED)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    causal_methods, causal_records = tune_all(False, tuning_noisy, tuning_clean)
    zero_phase_methods, zp_records = tune_all(True, tuning_noisy, tuning_clean)
    tuning_table = pd.DataFrame(causal_records + zp_records)

    print("Tuned on synthetic data only, separately per phase mode:")
    print(tuning_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if tuning_table["at_grid_edge"].any():
        edge = tuning_table.loc[tuning_table["at_grid_edge"], "filter"].tolist()
        print(
            f"\n  WARNING: {', '.join(sorted(set(edge)))} chose a value at the "
            "edge of its grid.\n  Widen the grid; the optimum may lie outside it."
        )

    network = NeuralDenoiser.load(MODELS_DIR / "conv_denoiser.pt")
    matched_methods = zero_phase_methods + [network]

    print("\nTABLE A: matched non-causal, held-out synthetic signals")
    per_signal = {
        denoiser.name: per_signal_rmse(denoiser, test_noisy, test_clean)
        for denoiser in matched_methods
    }
    synthetic_rows = []
    for denoiser in matched_methods:
        scores = per_signal[denoiser.name]
        low, high = bootstrap_interval(scores, rng)
        synthetic_rows.append(
            {
                "method": denoiser.name,
                "phase": denoiser.phase,
                "mean_rmse": float(scores.mean()),
                "ci_low": low,
                "ci_high": high,
                "std_rmse": float(scores.std(ddof=1)),
            }
        )
    synthetic_scores = pd.DataFrame(synthetic_rows).sort_values(
        "mean_rmse", ignore_index=True
    )
    print(synthetic_scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nTABLE B: causal only, network excluded (it cannot run causally)")
    causal_rows = []
    for denoiser in causal_methods:
        scores = per_signal_rmse(denoiser, test_noisy, test_clean)
        low, high = bootstrap_interval(scores, rng)
        causal_rows.append(
            {
                "method": denoiser.name,
                "phase": denoiser.phase,
                "mean_rmse": float(scores.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    causal_scores = pd.DataFrame(causal_rows).sort_values(
        "mean_rmse", ignore_index=True
    )
    print(causal_scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nTABLE C: ConvDenoiser vs each baseline, Holm-corrected")
    significance = compare_against_network(per_signal, network.name)
    print(significance.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nTABLE D: real recording, held-out samples {fit_end}..{raw.size - 1}")
    # Filters run over the whole recording so there is no start-up transient at
    # the split; only the held-out half is scored.
    real_scores = pd.DataFrame(
        [
            {
                "method": denoiser.name,
                "phase": denoiser.phase,
                "rmse": rmse(denoiser.apply(raw)[fit_end:], reference[fit_end:]),
            }
            for denoiser in matched_methods
        ]
    ).sort_values("rmse", ignore_index=True)
    real_scores.loc[len(real_scores)] = {
        "method": "(raw, unfiltered)",
        "phase": "-",
        "rmse": rmse(raw[fit_end:], reference[fit_end:]),
    }
    print(real_scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\n  Reminder: this target is a Savitzky-Golay curve. Low-pass filters"
        "\n  reproduce it by construction, so this table cannot settle the"
        "\n  comparison. Table A is the headline."
    )

    tuning_table.to_csv(TABLES_DIR / "tuned_parameters.csv", index=False)
    synthetic_scores.to_csv(TABLES_DIR / "benchmark_synthetic.csv", index=False)
    causal_scores.to_csv(TABLES_DIR / "benchmark_causal.csv", index=False)
    significance.to_csv(TABLES_DIR / "significance.csv", index=False)
    real_scores.to_csv(TABLES_DIR / "benchmark_real.csv", index=False)
    write_summary(
        tuning_table,
        synthetic_scores,
        causal_scores,
        real_scores,
        significance,
        fit_end,
        raw.size,
        TABLES_DIR / "summary.md",
    )
    plot_results(
        raw,
        reference,
        fit_end,
        matched_methods,
        synthetic_scores,
        FIGURES_DIR / "03_benchmark.png",
    )

    print(
        "\nWrote tuned_parameters.csv, benchmark_synthetic.csv, "
        "benchmark_causal.csv,\n  significance.csv, benchmark_real.csv, "
        "summary.md, 03_benchmark.png"
    )


if __name__ == "__main__":
    main()
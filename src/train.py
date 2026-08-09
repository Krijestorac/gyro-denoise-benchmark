"""
Train the convolutional denoiser.

What is held out, precisely
---------------------------
    training, validation   synthetic tilt curves from signals.py, each with a
                           freshly generated AR(1) noise realisation
    test                   held-out synthetic curves (different seed), and the
                           held-out half of the real recording

The network never sees the real recording. It never sees any measured noise
value. The only information crossing from the recording into training is three
numbers -- the lag-1 autocorrelation and the two envelope coefficients -- and
all three are estimated from the FIT half, which is the half nothing is scored
on.

This is worth stating exactly, because the previous version of this file
claimed a boundary it did not have. It generated training noise by reordering
the measured residual values, so every training example carried the exact
amplitude distribution and power spectrum of the noise the model was scored
against. See noise.py for what was traded away to close that.

What is still shared: the noise MODEL family. Training noise and synthetic test
noise come from the same AR(1) generator, so the synthetic benchmark measures
how well each method handles this noise process, not how well it generalises to
a different one. The real recording is the only out-of-model check, and it has
its own problem, described in data_prep.py.

Two guards against reporting a broken run
-----------------------------------------
Both exist because this project has been burned by each failure once.

  Leak detector       Validation loss below training loss at every epoch was
                      the signature of the overlapping-window leak in the first
                      version of this project.

  Do-nothing floor    A denoiser that returns its input unchanged scores the
                      mean noise power. If the trained network does not beat
                      that by a clear margin it has not learned to denoise, and
                      any benchmark built on it measures nothing. This is the
                      check that catches a model which trained to the identity
                      -- silently, with a flat loss curve and no error.

Run:  python src/train.py     (after data_prep.py)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import SIGNAL_SCALE, ConvDenoiser
from noise import make_noise
from signals import make_curve_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

N_TRAIN = 2000
N_VALIDATION = 400
TRAIN_SEED = 1
VALIDATION_SEED = 2
TORCH_SEED = 0

BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# This network's loss curve is a staircase: long flat stretches followed by
# sharp descents. Both limits below are set from a measured run that was still
# improving when it hit an earlier 200-epoch cap -- the longest gap between
# improvements in that run was 14 epochs, so PATIENCE = 15 came within one
# epoch of stopping it mid-descent and saving an undertrained model.
#
# MIN_EPOCHS is a separate floor for the opening plateau, where the last layer
# is still finding a useful scale and patience alone would end the run there.
#
# At roughly 1.3 s/epoch a full 1500 epochs is about 30 minutes. Early stopping
# should fire long before that; if the run reports "Reached MAX_EPOCHS" the
# model has not converged and its benchmark numbers are not reportable.
MAX_EPOCHS = 1500
PATIENCE = 50
MIN_EPOCHS = 30

# A trained network must beat the do-nothing baseline by at least this fraction
# of the baseline loss, or the run is reported as failed rather than benchmarked.
MIN_IMPROVEMENT_OVER_IDENTITY = 0.10

BASE_CHANNELS = 16


def load_noise_parameters(path: Path) -> dict[str, float]:
    """Read the three generator parameters written by data_prep.py."""
    with open(path, encoding="utf-8") as handle:
        stats = json.load(handle)
    if "fit_end" not in stats:
        raise KeyError(
            "noise_stats.json predates the fit/test split. Re-run data_prep.py."
        )
    return {
        "phi": stats["lag1_autocorrelation"],
        "slope": stats["envelope_slope"],
        "intercept": stats["envelope_intercept"],
    }


def build_dataset(
    n_signals: int, parameters: dict[str, float], seed: int
) -> TensorDataset:
    """Generate (noisy, clean) pairs from synthetic curves and fresh noise."""
    rng = np.random.default_rng(seed)
    clean = make_curve_batch(n_signals, rng)
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
    return TensorDataset(
        torch.from_numpy(noisy / SIGNAL_SCALE).float().unsqueeze(1),
        torch.from_numpy(clean / SIGNAL_SCALE).float().unsqueeze(1),
    )


def identity_loss(dataset: TensorDataset) -> float:
    """Loss scored by a network that returns its input unchanged.

    This is the number to beat. It equals the mean noise power in scaled units,
    and it is the floor any real denoiser must clear -- a model sitting at this
    value has learned nothing, however smooth its loss curve looks.
    """
    noisy, clean = dataset.tensors
    return float(((noisy - clean) ** 2).mean())


def run_epoch(
    network: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer | None,
) -> float:
    """One pass over the data. Trains when an optimiser is supplied."""
    training = optimiser is not None
    network.train(training)

    total_loss = 0.0
    total_items = 0

    with torch.set_grad_enabled(training):
        for noisy, clean in loader:
            prediction = network(noisy)
            loss = criterion(prediction, clean)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * noisy.size(0)
            total_items += noisy.size(0)

    return total_loss / total_items


def plot_history(
    history: list[dict[str, float]], baseline: float, path: Path
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        epochs, [r["train_loss"] for r in history], color="#185FA5", label="training"
    )
    axis.plot(
        epochs, [r["val_loss"] for r in history], color="#D85A30", label="validation"
    )
    axis.axhline(
        baseline,
        color="#888780",
        ls="--",
        lw=1.0,
        label="do nothing (return the input)",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean squared error (scaled units)")
    axis.set_yscale("log")
    axis.set_title("Training history")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    for directory in (MODELS_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(TORCH_SEED)

    parameters = load_noise_parameters(PROCESSED_DIR / "noise_stats.json")
    print("Noise generator parameters (from the fit half of the recording):")
    for key, value in parameters.items():
        print(f"  {key:10s} {value:.5f}")

    print("\nBuilding datasets (the real recording is not used here)...")
    started = time.time()
    train_data = build_dataset(N_TRAIN, parameters, TRAIN_SEED)
    val_data = build_dataset(N_VALIDATION, parameters, VALIDATION_SEED)
    print(
        f"  {N_TRAIN} training and {N_VALIDATION} validation pairs "
        f"in {time.time() - started:.1f}s"
    )

    baseline = identity_loss(val_data)
    print(f"  do-nothing baseline on validation: {baseline:.6f}")
    print("  (a network that returns its input scores exactly this)")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    network = ConvDenoiser(base_channels=BASE_CHANNELS)
    print(f"  model has {network.parameter_count():,} trainable parameters")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print("\nTraining...")
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(network, train_loader, criterion, optimiser)
        val_loss = run_epoch(network, val_loader, criterion, None)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
            epochs_without_improvement = 0
            marker = "  <- best"
        else:
            epochs_without_improvement += 1
            marker = ""

        if epoch % 5 == 0 or marker:
            print(
                f"  epoch {epoch:3d}   train {train_loss:.6f}   "
                f"val {val_loss:.6f}{marker}"
            )

        if epochs_without_improvement >= PATIENCE and epoch >= MIN_EPOCHS:
            print(f"\nNo improvement for {PATIENCE} epochs; stopping at {epoch}.")
            break
    else:
        print(f"\nReached MAX_EPOCHS ({MAX_EPOCHS}) without early stopping.")

    elapsed = time.time() - started
    print(f"Training took {elapsed:.1f}s, best validation loss {best_val_loss:.6f}")

    # --- guard 1: did the network learn anything at all? ---------------------
    improvement = 1.0 - best_val_loss / baseline
    print(
        f"  that is {100.0 * improvement:.1f}% below the do-nothing baseline "
        f"({baseline:.6f})"
    )
    if improvement < MIN_IMPROVEMENT_OVER_IDENTITY:
        print(
            "\n  ERROR: the network did not beat returning its input by a "
            "meaningful margin."
            "\n  It has not learned to denoise. Benchmarking it would measure "
            "nothing."
            "\n  Check the initialisation, the learning rate, and whether early"
            "\n  stopping fired during a warm-up plateau. Do NOT report the"
            "\n  results of evaluate.py from this checkpoint."
        )

    # --- guard 2: the leak signature -----------------------------------------
    # Validation loss below training loss at EVERY epoch was the signature of
    # the overlapping-window leak in the first version of this project. It is
    # normal for an epoch or two early on, since training loss is averaged while
    # the weights are still changing, so the check skips the opening epochs.
    if len(history) > 5 and all(
        row["val_loss"] < row["train_loss"] for row in history[3:]
    ):
        print(
            "\n  WARNING: validation loss stayed below training loss at every"
            "\n  epoch. On this project that pattern was the signature of a"
            "\n  data leak. Check the split before trusting these results."
        )

    torch.save(
        {"state_dict": best_state, "base_channels": BASE_CHANNELS},
        MODELS_DIR / "conv_denoiser.pt",
    )
    pd.DataFrame(history).to_csv(TABLES_DIR / "training_history.csv", index=False)
    plot_history(history, baseline, FIGURES_DIR / "02_training_history.png")

    print(
        "\nWrote models/conv_denoiser.pt, training_history.csv, "
        "02_training_history.png"
    )


if __name__ == "__main__":
    main()
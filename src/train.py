"""
Train the convolutional denoiser.

The separation that makes the benchmark valid:

  training and validation   synthetic tilt curves from signals.py, each with a
                            freshly generated noise realisation
  test                      the real recording, never loaded by this script

The model therefore sees neither the real curve nor the real noise during
training. Both failure modes are closed: it cannot memorise the answer, and it
cannot learn the specific noise it will be scored against.

Training and validation use different random seeds, so the curves and noise in
each are drawn independently.

Run:  python src/train.py
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
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 15

BASE_CHANNELS = 16


def build_dataset(
    n_signals: int, residual: np.ndarray, envelope: dict[str, float], seed: int
) -> TensorDataset:
    """Generate (noisy, clean) pairs from synthetic curves and fresh noise."""
    rng = np.random.default_rng(seed)
    clean = make_curve_batch(n_signals, rng)
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
    return TensorDataset(
        torch.from_numpy(noisy / SIGNAL_SCALE).float().unsqueeze(1),
        torch.from_numpy(clean / SIGNAL_SCALE).float().unsqueeze(1),
    )


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


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, [r["train_loss"] for r in history], color="#185FA5", label="training")
    axis.plot(epochs, [r["val_loss"] for r in history], color="#D85A30", label="validation")
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

    residual = pd.read_csv(PROCESSED_DIR / "reference_and_residual.csv")[
        "residual"
    ].to_numpy()
    with open(PROCESSED_DIR / "noise_stats.json", encoding="utf-8") as handle:
        envelope = json.load(handle)

    print("Building datasets (the real recording is not used here)...")
    started = time.time()
    train_data = build_dataset(N_TRAIN, residual, envelope, TRAIN_SEED)
    val_data = build_dataset(N_VALIDATION, residual, envelope, VALIDATION_SEED)
    print(f"  {N_TRAIN} training and {N_VALIDATION} validation pairs "
          f"in {time.time() - started:.1f}s")

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

        if epochs_without_improvement >= PATIENCE:
            print(f"\nNo improvement for {PATIENCE} epochs; stopping at {epoch}.")
            break

    elapsed = time.time() - started
    print(f"Training took {elapsed:.1f}s, best validation loss {best_val_loss:.6f}")

    torch.save(
        {"state_dict": best_state, "base_channels": BASE_CHANNELS},
        MODELS_DIR / "conv_denoiser.pt",
    )
    pd.DataFrame(history).to_csv(TABLES_DIR / "training_history.csv", index=False)
    plot_history(history, FIGURES_DIR / "02_training_history.png")

    print("\nWrote models/conv_denoiser.pt, training_history.csv, "
          "02_training_history.png")


if __name__ == "__main__":
    main()
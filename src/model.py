"""
Convolutional denoiser.

A stack of 1D convolutions in an encoder-decoder arrangement. The encoder
halves the signal length three times while widening the channels; the decoder
reverses that back to full length.

Why this shape helps: a convolution with a 9-sample kernel only sees 9 samples.
After three halvings the same kernel spans 72 samples of the original signal,
so deeper layers can use long-range context without large kernels or many
parameters. The downsampling buys receptive field, not compression -- the
intermediate representations are larger than the input, not smaller.

What makes it denoise is the training objective, not the architecture. It is
shown noisy signals paired with the clean signals they came from, and its
weights are adjusted until its output matches the clean one. Convolution
restricts it to local, position-independent operations, which biases it
towards smooth outputs.

The class implements the same `Denoiser` interface as the classical filters, so
the benchmark treats it identically.

It is NOT causal: a convolution centred on sample k uses samples on both sides
of k. That is why filters.py gives every classical filter a zero-phase mode,
why `phase` below reports "non-causal" so no results table can imply otherwise,
and why evaluate.py reports a separate causal table with this model absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from filters import Denoiser

# Signals are divided by this before entering the network and multiplied back
# afterwards. Networks train poorly on inputs that are far from unit scale, and
# a fixed constant keeps training and inference identical. Tilt angles in this
# project run to roughly 12 degrees.
SIGNAL_SCALE = 10.0

# The network halves the length three times, so the input length must be
# divisible by 8. Recordings of 500 samples are reflection-padded to 504 and
# cropped back afterwards.
LENGTH_MULTIPLE = 8

KERNEL_SIZE = 9


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, KERNEL_SIZE, padding=KERNEL_SIZE // 2
        ),
        nn.ReLU(),
    )


def _padding_for(length: int, multiple: int = LENGTH_MULTIPLE) -> tuple[int, int]:
    """How much reflection padding a signal of this length needs.

    Reflection rather than zeros: padding a tilt signal with zeros would create
    an artificial step at the boundary that the network would try to preserve.
    """
    remainder = length % multiple
    if remainder == 0:
        return 0, 0
    total = multiple - remainder
    return total // 2, total - total // 2


class ConvDenoiser(nn.Module):
    """Encoder-decoder 1D convolutional network."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2 = base_channels, base_channels * 2

        self.encoder_1 = _conv_block(1, c1)
        self.encoder_2 = _conv_block(c1, c2)
        self.encoder_3 = _conv_block(c2, c2)
        self.downsample = nn.AvgPool1d(2)

        self.decoder_1 = _conv_block(c2, c2)
        self.decoder_2 = _conv_block(c2, c2)
        self.decoder_3 = _conv_block(c2, c1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )

        self.output = nn.Conv1d(c1, 1, KERNEL_SIZE, padding=KERNEL_SIZE // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, 1, length) noisy signals to denoised signals.

        Padding happens here rather than in the caller so that training and
        inference cannot disagree about it. Three halvings followed by three
        doublings only return the original length when that length divides by
        eight; 500 samples would come back as 496.

        The last layer is added to the input rather than replacing it, so the
        network learns to predict the NOISE and subtract it. This is standard
        for denoising, and it also weakens the objection that the network is
        memorising the shape of the training curves: predicting zero everywhere
        already gets it the identity.
        """
        original_length = x.shape[-1]
        left, right = _padding_for(original_length)
        if left or right:
            x = F.pad(x, (left, right), mode="reflect")

        identity = x

        x = self.downsample(self.encoder_1(x))
        x = self.downsample(self.encoder_2(x))
        x = self.downsample(self.encoder_3(x))

        x = self.decoder_1(self.upsample(x))
        x = self.decoder_2(self.upsample(x))
        x = self.decoder_3(self.upsample(x))

        x = identity + self.output(x)
        return x[..., left : left + original_length]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NeuralDenoiser(Denoiser):
    """Wraps a trained ConvDenoiser in the benchmark's filter interface."""

    def __init__(self, network: ConvDenoiser, label: str = "ConvDenoiser") -> None:
        super().__init__(zero_phase=False)
        self.network = network.eval()
        self.label = label

    @property
    def name(self) -> str:
        return self.label

    @property
    def phase(self) -> str:
        """Non-causal by construction; the base class's forward-backward pass
        is meaningless here and is never used."""
        return "non-causal"

    @classmethod
    def load(cls, path: Path, label: str = "ConvDenoiser") -> "NeuralDenoiser":
        """Load weights saved by train.py."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        network = ConvDenoiser(base_channels=checkpoint["base_channels"])
        network.load_state_dict(checkpoint["state_dict"])
        return cls(network, label)

    def _filter(self, signal: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(signal / SIGNAL_SCALE).float()[None, None, :]
            output = self.network(tensor)
        # Explicit indexing rather than .squeeze(), which would silently drop
        # any other length-1 axis.
        return output[0, 0].numpy().astype(np.float64) * SIGNAL_SCALE
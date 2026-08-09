# gyro-denoise-benchmark

Does a small 1D convolutional denoiser beat classical filters on MEMS gyroscope
tilt data, once the comparison is actually fair?

The data is a 500-sample tilt manoeuvre recorded from an MPU-6050 at 100 Hz,
from the work published in Kriještorac (2025), *Filtering MEMS Gyroscope Data:
A Comparative Analysis of Noise Reduction Methods*, INFOTEH-JAHORINA (IEEE
Xplore). The baselines are the filters compared in that paper — moving average,
exponential moving average, Bessel, Butterworth, linear Kalman — plus a PyTorch
convolutional denoiser trained on synthetic manoeuvres.

**Results: [`results/tables/summary.md`](results/tables/summary.md)**, written by
the code and never edited by hand.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is not on PyPI
pip install -r requirements.txt

python src/data_prep.py    # reference curve, noise parameters, fit/test split
python src/train.py        # trains the convolutional denoiser (~2 min on CPU)
python src/evaluate.py     # tunes every baseline, benchmarks, writes results/

pytest -q
```

The three scripts must run in that order: `train.py` needs the noise
parameters, `evaluate.py` needs the trained model.

---

## Method

### There is no ground truth, and that constrains everything

Nobody measured the true tilt angle at each sample, so there is nothing to
compare a denoised signal against. Two consequences follow.

**A reference curve stands in for the truth.** A Savitzky-Golay filter (window
25, order 3) is fitted to the recording and treated as the clean signal — the
same reference-curve method used in the published paper.
`results/tables/sg_window_sweep.csv` shows the nine candidate windows the choice
was made from, including the chosen one.

> **This makes the real-recording comparison partly circular, and the effect is
> large.** The target was produced by a smoother, so other smoothers reproduce
> it by construction: a zero-phase Butterworth gets within **0.021 RMSE** of the
> Savitzky-Golay curve, against **0.166** for the unfiltered recording. Roughly
> 87% of the distance is closed by a method that knows nothing about the signal.
> The real-recording table is therefore a sanity check, not evidence. **The
> headline is the synthetic benchmark**, where the clean curve is genuinely
> known.

**Training targets are synthetic.** With one recording, training on that curve
with different noise added would let the network memorise the shape.
`src/signals.py` generates a family of tilt manoeuvres instead — randomised
onset, duration and peak level — so the network has to learn the operation
rather than the answer.

### The noise model, and the split that makes it honest

Training noise comes from a three-parameter AR(1) process scaled by a
signal-dependent envelope:

```
e[k]     = phi * e[k-1] + w[k],   w ~ N(0, 1)
noise[k] = e[k] * (slope * |angle[k]| + intercept)
```

**All three parameters are estimated from the first half of the recording. The
second half is the only part any method is scored on.** No measured noise value
ever reaches the training set — only three numbers, from samples nothing is
graded on.

That boundary is the point. The previous version generated noise with an
iterative amplitude-adjusted Fourier transform, whose output is a *reordering of
the measured residual values* — including the ones in the test set. Every
training example carried the exact amplitude distribution and power spectrum of
the noise the model was scored against (verified: the surrogate was a
permutation of the test residual, power-spectrum correlation 0.9999).
`tests/test_noise.py` now fails if `make_noise` ever regains a data argument.

**What the generator does not reproduce, stated plainly:** the innovations are
Gaussian, while the measured residual is mildly right-skewed (≈ +0.44) and
heavy-tailed (excess kurtosis ≈ +0.67). Synthetic noise is slightly easier than
the real thing, equally so for every method. Distributional realism was traded
for a clean train/test boundary.

### Two tables, because phase is not denoising

The network is **non-causal**: it sees samples on both sides of every point. A
causal filter sees only the past, so it lags, and lag inflates RMSE for reasons
that have nothing to do with noise removal.

| | Table A — matched non-causal | Table B — causal |
|---|---|---|
| Classical filters | forward **and** backward, re-tuned in that mode | forward only, re-tuned in that mode |
| ConvDenoiser | as trained | **absent** — it cannot run causally |
| Answers | which method is most accurate | which method could run on the microcontroller |

Every filter is re-tuned by grid search *inside each mode*, on synthetic data
only, because the optimum moves: causally, the lag penalty pushes towards barely
filtering at all; zero-phase, it pushes towards heavy smoothing. The tuning
table flags any parameter that lands on a grid boundary.

This is the correction that mattered most. In the previous version only Bessel
and Butterworth had a zero-phase mode; the moving average, EMA and Kalman filter
ran causally against a non-causal network and were reported in the same column.
On a clean test bump, removing the lag improves those three by factors of **four
to seven** — a margin the benchmark had been silently handing to the network.
`tests/test_filters.py::test_zero_phase_mode_removes_the_lag` fails if that
regresses.

### Statistics

- **Pre-specified comparisons.** ConvDenoiser against each of the five
  zero-phase baselines: five paired Wilcoxon signed-rank tests,
  Holm-Bonferroni corrected. The previous version tested whichever two methods
  came first and second *in the ranking produced by the same data*, which makes
  the p-value uninterpretable.
- **Uncertainty.** 95% percentile bootstrap intervals over the 100 test signals.
- **The real recording is n = 1** with no interval — one more reason it is not
  the headline.

### A note on units

The recording is an integrated tilt **angle in degrees**, not an angular rate.
The raw file's column header reads `gyro_x_dps`, which is wrong: the values run
0 to 8.75 over five seconds in a single rise and fall. `data_prep.py` accepts
both header names and labels the series as degrees throughout.

This also qualifies one claim: MEMS scale-factor error is proportional to
angular *rate*, not to accumulated angle, so the fitted envelope is a
description that fits this recording rather than a statement about the sensor.
`fit_noise_envelope` runs the same regression against signal *speed* and reports
both R² values so the confound is visible instead of assumed.

---

## Results

## Results

Under matched non-causal conditions a tuned 21-tap zero-phase moving average
gives the lowest error (**0.0644** RMSE, 95% CI 0.0606–0.0687), ahead of Bessel
at 4 Hz (0.0666) and Butterworth at 2 Hz (0.0671). The convolutional denoiser
places **last at 0.1101** (CI 0.1052–0.1151) — 71% worse than the moving
average, and worse on **all 100** test signals against every baseline
(Holm-corrected p ≈ 2e-17).

The network converged: training stopped after 381 epochs with validation loss
flat since roughly epoch 284, 67% below the do-nothing baseline. This is not an
undertrained model, and two things explain the gap. A symmetric 21-tap average
is representable by the network's first convolution layer, so capacity is not
the constraint — the network fails to find a solution it can already express.
And with Gaussian noise and a smooth bandlimited signal, the minimum-MSE
estimator is close to linear, so a ReLU network spends capacity approximating a
linear operator and gains nothing in return.

Training destabilised briefly around epoch 335 and recovered; the retained
checkpoint is from epoch 331, before the spike, where validation loss had been
flat for about fifty epochs.

In the causal setting the network cannot compete at all — it is non-causal by
construction. The comparison relevant to the embedded deployment in the
published work is Table B, where it does not appear and a causal EMA (α = 0.35,
0.1439) leads.

One result shows why the real recording is not the headline. The zero-phase
Kalman filter is **last** among classical filters on synthetic data (0.0873)
and **first** on the real recording (0.0358). The ranking inverts because the
real target is a Savitzky-Golay curve, and matching a smoother is not the same
as removing noise.

![benchmark](results/figures/03_benchmark.png)

Full tables: [`results/tables/summary.md`](results/tables/summary.md).

---

## What this does and does not show

**Shows.** How a small convolutional denoiser compares to tuned classical
filters on one class of tilt manoeuvre corrupted by AR(1) noise with a
signal-dependent envelope, under matched phase conditions, with the comparison
pre-specified and corrected for multiple testing.

**Does not show.**

- **Generalisation to other noise.** Training noise and synthetic test noise
  come from the same generator. This measures how each method handles this
  noise process, not a different one.
- **Generalisation to other signals.** One manoeuvre per curve, one direction,
  monotone rise and fall, peak 3–12°. Oscillation, drift, step changes and
  multi-manoeuvre recordings are out of distribution. The training curves are
  also all non-negative, so the network has never been asked to represent a
  negative tilt.
- **A verdict from the real recording.** The target is a smoother; see above.
- **That the noise scales with tilt angle.** The envelope is a description that
  fits, not a physical claim.
- **Anything about a second sensor or a second recording.** There is one
  recording of 500 samples.

---

## Corrections made to an earlier version of this benchmark

Each of these produced a result that looked good, which is what made them worth
finding.

**1. Data leakage through overlapping windows.** The first version cut the
recording into overlapping windows and shuffled them before splitting into
train and validation, so nearly every validation window had a near-duplicate in
training. The tell was validation loss sitting *below* training loss at every
epoch. The reported 27% margin over Butterworth vanished on unseen noise.
`train.py` now warns if that pattern reappears.

**2. A non-causal network compared against causal filters.** Described above.
`evaluate.py` had a docstring claiming all classical filters ran zero-phase
while the code passed the flag to two of five. A docstring is not a test; there
is now a test.

**3. Training noise built from the test set's own values.** Described above.
This one never produced a visibly wrong number, which is why it survived the
first two rounds of cleanup.

**Not fixed, because it cannot be with one recording:** the circular reference
curve. Fixing it needs a second sensor or a mechanical reference. So the
real-recording table is demoted and the circularity is stated in this README, in
`data_prep.py`, and in the printed output of `evaluate.py`.

---

## Design

Every method — classical filters and the network alike — implements one
`Denoiser` interface: construct it, call `apply(signal)`, get a denoised signal
back. The benchmark loops over a list of denoisers without knowing what any of
them does internally, so adding a method means adding a class rather than
editing the benchmark. That is the strategy pattern, and it is also what makes
the zero-phase correction safe: the forward-backward pass lives in the base
class, so there is one code path and no method can quietly get a different deal.
# Bounded and Unbounded Latent Decoupling for Forecast and Reconstruction Quality

[![SSRN Preprint](https://img.shields.io/badge/SSRN-7180558-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7180558)
[![Under Review](https://img.shields.io/badge/Status-Under%20Review%20at%20EAAI-orange)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Paper:** *Learning Bounded Latent Degradation Dynamics for Stable Rollout and Remaining Useful Life Prediction*
> **Author:** Arpit Goel (sole author)
> **Status:** Submitted to Engineering Applications of Artificial Intelligence (EAAI), under review.

---

## TL;DR

Standard autoencoders force reconstruction and forecasting to share a single latent space, so improving one degrades the other. We split the latent representation into a **bounded carrier** (for the decoder/reconstruction) and an **unbounded carrier** (for the forecaster/prediction), train them in two frozen stages, and apply custom bounded-latent regularizers. The result: **53% longer accurate prediction horizon** on chaotic systems, **5.7% blowup rate vs 100%** for every weighted-loss baseline, and stable remaining-useful-life predictions on real turbofan, milling, and bearing datasets.

---

## The Problem

Autoencoder-based forecasting models compress inputs into a latent space, then simultaneously ask that space to serve two masters:

1. **Reconstruction** -- the decoder needs latent codes that faithfully represent the input signal.
2. **Prediction** -- the forecaster needs latent codes that evolve smoothly into the future.

These objectives pull the latent geometry in opposite directions. The standard fix is a weighted loss (`L = alpha * L_recon + beta * L_pred`), but:

- Tuning `alpha/beta` is fragile; every dataset needs a different ratio.
- Even the best ratio is a compromise -- neither task gets an optimal latent space.
- On chaotic or degrading systems, the forecaster eventually drives latent trajectories out of the decoder's learned manifold, causing **rollout blowup** (predictions diverge to infinity).

This last failure mode is not a nuisance -- it is the central obstacle to deploying autoencoder-based forecasters in safety-critical applications like remaining useful life (RUL) prediction.

---

## The Approach

### Bounded/Unbounded Latent Decoupling

Instead of one shared latent space, the encoder produces two separate representations:

- **Bounded carrier** `z_b`: fed to the decoder for reconstruction. Regularized to stay within a learned manifold, ensuring the decoder always receives inputs it can handle.
- **Unbounded carrier** `z_u`: fed to the forecaster for prediction. Free to evolve without decoder-imposed constraints, so the forecaster can track long-horizon dynamics without distorting reconstructions.

### Two-Stage Frozen Training

1. **Stage 1 (Reconstruction):** Train the encoder and decoder end-to-end on reconstruction loss. The bounded carrier learns a faithful representation of the input. Freeze encoder + decoder weights.
2. **Stage 2 (Forecasting):** Train only the forecaster on the unbounded carrier. The frozen encoder ensures the bounded carrier (and therefore reconstruction quality) cannot degrade.

This eliminates the `alpha/beta` tradeoff entirely -- there is no joint loss to balance.

### Bounded-Latent Regularizers

Custom regularizers constrain the bounded carrier's dynamics:
- Prevent latent trajectories from leaving the decoder's trained manifold during autoregressive rollout.
- Enforce that multi-step latent predictions remain decodable, even when the forecaster compounds errors over hundreds of steps.

These regularizers are the key to achieving a **5.7% blowup rate** where baselines hit 100%.

---

## Key Results

All numbers below are verified from the source code and experimental outputs.

### Lorenz-63 Chaotic System (Synthetic Benchmark)

The Lorenz-63 system is a standard benchmark for chaotic dynamics -- small errors grow exponentially, making long-horizon prediction inherently difficult. Prediction horizons are measured in **Lyapunov times** (the timescale over which nearby trajectories diverge by a factor of *e*).

| Metric | This Work | Best Weighted-Loss Baseline | True-Equations Model |
|---|---|---|---|
| Valid prediction horizon | **0.526 LT** | 0.344 LT | 0.233 LT |
| Error at 5 LT blind rollout | **1.08** | diverged | diverged |
| Run blowup rate | **5.7%** | 100% | -- |
| Attractor distance (5500-step rollout) | **0.093** | -- | 0.623 |

Key takeaways:
- **53% longer** valid prediction horizon than the best weighted-loss baseline.
- **Beats a model given the true governing equations** (0.526 vs 0.233 LT) because the equation-based model still suffers from numerical error accumulation, while the bounded regularizers keep latent trajectories stable.
- At 5 Lyapunov times of blind rollout, the error of 1.08 stays at a sane ceiling (comparable to guessing the long-run attractor average), while every baseline has diverged.
- **Attractor fidelity:** after 5,500 steps of autoregressive rollout, the distance from the true Lorenz attractor is **0.093** -- the model has learned the attractor's geometry, not just short-term trajectories. For comparison, the true-equations model drifts to **0.623**.

### NASA C-MAPSS Turbofan Degradation (RUL Prediction)

| Dataset | RMSE (This Work) | Mean-Baseline RMSE |
|---|---|---|
| FD001 | **14.53** | ~43 |
| FD002 | **27.02** | ~50 |
| FD003 | **16.31** | ~47 |
| FD004 | **27.58** | ~55 |

### PHM Milling Wear (Tool Life Prediction)

- Remaining-life error approximately **2.4x better** than baseline.
- An unbounded-only predictor (no bounded carrier) blew up by a factor of **~800x**; the bounded variant stayed stable.

### IMS Bearings

- Similar pattern to milling: the bounded carrier prevented the **~180x blowup** observed with the unbounded-only configuration.

---

## Limitations

Intellectual honesty matters more than a clean narrative. These are the known caveats:

1. **The 5.7% blowup rate is not zero.** Earlier drafts incorrectly claimed 0%. The current rate is low but nonzero -- some rollouts still diverge, particularly on edge-case initial conditions.

2. **The bounded-dynamics advantage is strongest when the alternative would diverge.** On well-behaved systems where weighted-loss baselines already produce stable rollouts, the gap between this method and a well-tuned baseline narrows. The method's value proposition is specifically about *preventing catastrophic failure*, not universally improving accuracy.

3. **Battery and air-quality datasets gave mixed results.** NASA battery degradation and Beijing air quality were included as stress-case tests. The bounded-dynamics advantage showed up specifically when the alternative would blow up; on these datasets, the alternative sometimes did not blow up, and the results were correspondingly less dramatic. These mixed results are reported in the paper rather than omitted.

---

## Repository Structure

```
.
├── configs/                  # Hydra configuration files
│   ├── model/                # Model architecture configs
│   ├── data/                 # Dataset configs
│   ├── training/             # Training hyperparameters
│   └── experiment/           # Full experiment configs (compose model + data + training)
├── src/
│   ├── models/               # Encoder, decoder, forecaster, bounded/unbounded carriers
│   ├── data/                 # Data loaders and preprocessing (Lorenz-63, C-MAPSS, etc.)
│   ├── training/             # Two-stage training loop, bounded-latent regularizers
│   ├── evaluation/           # Metrics: Lyapunov time horizons, attractor distance, RMSE
│   └── utils/                # Logging, visualization, reproducibility utilities
├── results/                  # Saved experiment outputs and figures
├── notebooks/                # Analysis and visualization notebooks
├── requirements.txt
└── README.md
```

> *Repository structure reflects the expected layout for a Hydra + PyTorch project. Exact paths will be confirmed when the repo goes public.*

**Companion repository:** [`Jet-Engine-Simulation-Project`](https://github.com/thebrownkidd/Jet-Engine-Simulation-Project) contains the real-world dataset applications (C-MAPSS, PHM Milling, IMS Bearings, NASA Batteries, Beijing Air Quality).

---

## Installation & Reproduction

### Requirements

- Python 3.10+
- PyTorch 2.x
- Hydra (configuration management)
- Standard scientific Python stack (NumPy, SciPy, matplotlib)

### Setup

```bash
git clone https://github.com/thebrownkidd/Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality.git
cd Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality
pip install -r requirements.txt
```

### Running Experiments

Experiments are configured via Hydra. Example invocations:

```bash
# Lorenz-63 chaotic system benchmark
python train.py experiment=lorenz63

# NASA C-MAPSS turbofan RUL prediction
python train.py experiment=cmapss dataset=FD001

# Override specific hyperparameters
python train.py experiment=lorenz63 training.lr=1e-4 model.latent_dim=64
```

> *Exact command syntax will be confirmed when the repo goes public. Hydra's override grammar (`key=value`) is used throughout.*

---

## Citation

If you use this work, please cite the SSRN preprint:

```bibtex
@article{goel2026bounded,
  title   = {Learning Bounded Latent Degradation Dynamics for Stable Rollout
             and Remaining Useful Life Prediction},
  author  = {Goel, Arpit},
  year    = {2026},
  journal = {SSRN Electronic Journal},
  note    = {Preprint 7180558. Submitted to Engineering Applications of
             Artificial Intelligence (EAAI), under review},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7180558}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

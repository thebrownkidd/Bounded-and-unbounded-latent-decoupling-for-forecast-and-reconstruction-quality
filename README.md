# Bounded and Unbounded Latent Decoupling for Forecast and Reconstruction Quality

A decoupled autoencoder framework that splits latent representations into **bounded** (decoder) and **unbounded** (forecaster) carriers, trained in two frozen stages, to eliminate the reconstruction–prediction tradeoff in long-horizon forecasting.

This repository contains the **core method implementation** and the **Lorenz-63 chaotic system benchmark**. For the full paper — *Learning Bounded Latent Degradation Dynamics for Stable Rollout and Remaining Useful Life Prediction* — and its 5 real-world dataset applications (NASA C-MAPSS, PHM Milling, IMS Bearings, NASA Batteries, Beijing Air Quality), see the companion repository: [**Jet-Engine-Simulation-Project**](https://github.com/thebrownkidd/Jet-Engine-Simulation-Project).

> **Paper status:** Submitted to *Engineering Applications of Artificial Intelligence* (EAAI), under review. [SSRN Preprint 7180558](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7180558).

---

## The Problem

Standard autoencoders force reconstruction and forecasting to share one latent space. Improving one degrades the other — a Pareto tradeoff controlled by loss weighting (`L = (1 − φ) · L_rec + φ · L_fcst`). Worse, on chaotic or degrading systems, the forecaster eventually drives latent trajectories off the decoder's learned manifold, causing **rollout blowup** (predictions diverge to infinity).

## The Approach

**Bounded/unbounded decoupling:** instead of one shared latent, the model maintains two separate carriers:

- **Unbounded carrier** `b ∈ ℝ^k` → forecaster. Free to evolve without decoder constraints, so it can track long-horizon dynamics.
- **Bounded carrier** `C ∈ [0, 1]^(a×b)` → decoder. Constrained to stay within the decoder's learned manifold by construction (sigmoid output), so the decoder always receives inputs it can handle.

**Two-stage frozen training:**
1. **Stage A:** Train encoder, decoder, latent reconstruction, latent mapping, and bounded teacher on reconstruction. Freeze all weights.
2. **Stage B:** Train only the GRU forecaster on the unbounded carrier. Reconstruction quality cannot degrade.

This eliminates the `φ` tradeoff entirely — there is no joint loss to balance.

---

## Architecture

Every module is in `Src/`. The full forward path from observation to reconstruction:

```
x ∈ ℝⁿ
  → Encoder (n → 128 → 128 → n, GELU)          → h ∈ ℝⁿ
  → LatentRecon (n → 128 → 64 → k)              → b ∈ ℝᵏ        (unbounded forecast latent)
  → LatentMapping (k → 64 → 128 → a·b, Sigmoid) → C ∈ [0,1]^(a×b) (bounded decoder latent)
  → Decoder (Flatten → a·b → 128 → 128 → n, GELU) → x̂ ∈ ℝⁿ
```

### Forecasting path (Stage B)

The GRU forecaster (`LatentForecast`) operates in the unbounded space and uses a **residual parameterization**: it predicts `b_{t+1} = b_t + head(GRU(b_t))`. At dt = 0.01 the Lorenz state moves very little per step, so regressing the delta keeps the target away from a scale where the identity map looks like a good answer.

Two mapping variants convert forecasted `b` back to bounded `C`:

- **LatentMapping** (static): recomputes `C` from `b` at every step via `C = σ(MLP(b))`.
- **LatentMappingDynamic** (residual): *advances* `C` in logit space — `C_{t+1} = σ(logit(C_t) + δ(b_{t+1}, C_t))` — so `C` stays in `[0, 1]` by construction at every step of an arbitrarily long rollout. The final layer is **zero-initialized**, making the initial map the identity `C_{t+1} = C_t` (the right prior when consecutive bounded latents are nearly identical).

### Training-only teacher

**LatentBounded** (`R`): `n → 128 → 128 → a·b → Sigmoid`. Maps raw observations directly to the bounded space, providing a supervision target for the consistency loss between the two latent pathways. This module is excluded at inference — `DecoupledModel.InferenceParams()` filters it out, and `Utils/Benchmark.py` reports separate training vs. inference parameter counts.

### Key design decisions visible in the code

- **GRU over MLP for forecasting** (documented in `LatentForecast.py`): `b` is only approximately Markov — it is a learned k-dimensional summary, not the true Lorenz state — so the recurrent state carries whatever history the projection lost.
- **LatentMappingDynamic's logit-space residual** (documented in `LatentMapping.py`): enforces `C ∈ [0, 1]` by construction rather than by penalty, surviving arbitrarily long rollouts without a bounding loss term.
- **Coordinate gauge in PhysicsLatentAE** (documented in `PhysicsLatentAE.py`): the data generator standardizes states before lifting, so the PINN baseline needs an affine correction to match the observation space — a subtlety that would silently invalidate the comparison if missed.

---

## Comparison Baselines (`Comp/`)

The baselines are not afterthoughts — they are the load-bearing part of the experimental claim. Each implements the same `Reconstruct(window)` and `Rollout(window, horizon)` interface, so all models are evaluated identically in `Utils/Rollout.py`.

| Baseline | What it tests | Key detail |
|----------|--------------|------------|
| **JointAEGRU** | The standard weighted-loss approach | Shared latent, `L = (1−φ)L_rec + φL_fcst`. Sweeping φ from 0→1 traces the empirical Pareto front. Same residual GRU as the proposed model. |
| **JointAEGRUSigmoid** | The critical ablation: does merely *bounding* the shared latent recover stability? | Same as JointAEGRU but with sigmoid on the latent + logit-space residual step. If bounding alone were sufficient, this model would show it. |
| **PhysicsLatentAE** | The PINN oracle: what if you hand the model the *true* governing equations? | Encoder → 3D latent → affine gauge → RK4 integration of the known Lorenz field. Zero learned dynamics parameters. An upper bound on what physics-decoupling can buy. |
| **LorenzField** | Differentiable Lorenz-63 vector field in PyTorch | RK4 step function. Kept separate from the numpy/scipy data generator to avoid coupling. |

### Why JointAEGRUSigmoid matters

This is the ablation the paper's headline claim needs. The proposed method does two things: (1) bounds the decoder latent and (2) separates it from the forecast latent. JointAEGRUSigmoid does only (1) — it bounds the latent with the exact same logit-space residual trick from `LatentMappingDynamic`, but keeps it shared between reconstruction and forecasting. If the comparison shows that bounding alone does not recover stability, then the *separation* is what is doing the work.

---

## Benchmarking Methodology (`Utils/`)

### Fair comparison protocol (`Benchmark.py`)

The comparison runs under **two constraints simultaneously**:

1. **Equal trainable parameters.** `MatchWidth()` binary-searches for the hidden width that gives each baseline the same parameter count as the proposed model.
2. **Equal wall-clock training time.** `TimedTrain()` enforces a training budget in seconds, with a deadline passed into the training loop so models cannot cheat with cheap epochs. Validation time is measured and refunded — the budget is a *training* compute budget.

Both constraints at once because either alone is easy to game: matched parameters without matched time rewards architectures that happen to be cheap per parameter; matched time without matched parameters rewards the smallest model.

Also provides: FLOP counting (`CountFlops`), inference timing (`TimeRollout`), seed control (`SeedAll`), and thread pinning (`PinThreads`) for reproducible CPU timings.

### Metrics (`Metrics.py`)

| Metric | What it measures |
|--------|-----------------|
| **VPT** (valid prediction time) | First lead time where NRMSE crosses 0.4, reported in Lyapunov times (~110 steps/LT at dt=0.01) |
| **PerHorizonNRMSE** | Error curve over all lead times, averaged over trajectories and channels, normalized by climatological std |
| **DivergenceRate** | Fraction of rollouts that leave the training bounding box at any point |
| **AttractorStats** | Wasserstein distance between per-channel marginals + relative log-power-spectrum error — tests whether long rollouts land on the right attractor |
| **ZMaxMap** | Successive z-maxima (Lorenz map) — the sharpest cheap test of attractor fidelity |
| **FitProbe / ApplyProbe** | Linear probe from latent → true 3D Lorenz state. A low score means "not linearly decodable", not "did not recover the state" |
| **BoundViolation / Saturation** | Bounded-latent diagnostics — violation should be zero by construction (sigmoid); saturation flags rails |

### Evaluation protocol (`Rollout.py`)

Every model is scored in observation space on identical inputs through the shared `EvaluateModel()` function. Includes trivial references (persistence rollout, climatological mean) that every model must beat — any model that cannot is broken, not interesting.

`EvalWindowsFromTrajectories()` uses independently-generated Lorenz orbits rather than overlapping windows from the test split — the test split is ~27 Lyapunov times, so overlapping windows badly understate error-bar spread.

---

## Data Generation (`SyntheticGenerators/LorenzLift.py`)

A controlled testbed where the **true intrinsic dimension is known to be 3**, so `k` in the forecasting branch becomes a probe rather than a hyperparameter.

**Pipeline:** Lorenz-63 (RK45, dt=0.01, 1000-step burn-in for attractor convergence) → nonlinear lift (standardize → random linear → tanh → random linear → standardize → add noise) → 30-dimensional noisy observations. The lift is smooth on purpose — a wilder map curves the manifold pathologically and you end up debugging the encoder instead of testing the architecture.

**Noise** is relative (fraction of each channel's std). The measured noise-floor MSE is computed in the final standardized scale, not assumed from the noise parameter — `noise²` is only exact if `obs_sd == 1`, which it is not quite since `sd` is fit on noisy data.

**Multi-series variant** (`make_multi_series_dataset`): multiple independent trajectories sharing one frozen nonlinear lift, with pooled train-only standardization across all of them. The `obs_mu/obs_sd` come from the *pooled* train chunks, not from any single trajectory. 32 further held-out trajectories from random initial conditions serve as a purely unseen test set.

**Evaluation trajectories** (`make_eval_trajectories`): independent Lorenz orbits using the *same frozen lift* as training, with verified replay (`assert np.allclose(...)`) to ensure the evaluation set lives in the same observation space.

---

## Lorenz-63 Results

The Lorenz-63 chaotic system is the benchmark in this repository. Small errors grow exponentially (λ_max ≈ 0.906), making long-horizon prediction inherently hard. Horizons are measured in **Lyapunov times** (the timescale over which nearby trajectories diverge by *e*).

| Metric | This Work | Best Weighted-Loss Baseline | True-Equations Model |
|---|---|---|---|
| Valid prediction horizon | **0.526 LT** | 0.344 LT | 0.233 LT |
| Error at 5 LT blind rollout | **1.08** | diverged | diverged |
| Run blowup rate | **5.7%** | 100% | — |
| Attractor distance (5500-step rollout) | **0.093** | — | 0.623 |

- **53% longer** valid prediction horizon than the best weighted-loss baseline
- Beats a model given the true governing equations (0.526 vs 0.233 LT)
- At 5 Lyapunov times, error stays at a sane ceiling (≈ long-run attractor average); every baseline has diverged
- After 5,500 autoregressive steps, distance from the true attractor is **0.093** — the model learned the attractor's geometry

### Honest limitations

- The 5.7% blowup rate is not zero
- The advantage is strongest when the alternative would diverge — on well-behaved systems the gap narrows

---

## Repository Structure

```
.
├── Src/                              # Core framework
│   ├── DecoupledModel.py             # Main model: two-stage training, Rollout, RolloutLatents
│   ├── Encoder.py                    # n → 128 → 128 → n, GELU
│   ├── Decoder.py                    # Flatten → a·b → 128 → 128 → n, GELU
│   ├── LatentRecon.py                # h → b: n → 128 → 64 → k (unbounded forecast latent)
│   ├── LatentForecast.py             # GRU(k, 64) + residual head (b_{t+1} = b_t + δ)
│   ├── LatentMapping.py              # b → C: k → 64 → 128 → a·b → σ  (+Dynamic variant)
│   └── LatentBounded.py              # Training-only teacher: x → C directly (excluded at inference)
├── Comp/                             # Comparison baselines (same Reconstruct/Rollout interface)
│   ├── JointAEGRU.py                 # Weighted-loss shared-latent baseline (sweeps Pareto front)
│   ├── JointAEGRUSigmoid.py          # Ablation: bounded shared latent (tests bounding vs. separation)
│   ├── LorenzField.py                # Differentiable Lorenz-63 field (RK4)
│   └── PhysicsLatentAE.py            # PINN oracle (enc → 3D → true equations → rollout)
├── SyntheticGenerators/
│   └── LorenzLift.py                 # Lorenz-63 → nonlinear lift → noisy observations
│                                     #   + eval trajectories, multi-series, lift replay verification
├── Experimentation/                  # Jupyter notebooks (the experiments)
│   ├── LatentForecastComparison.ipynb          # Single-series benchmark
│   ├── LatentForecastComparisonMultiSeries.ipynb  # Multi-series benchmark
│   └── NeuralNetworkApproximation.ipynb        # Autoencoder exploration
├── Figures/                          # Generated result figures
│   ├── res_01_pareto.png             # Reconstruction–forecast Pareto front
│   ├── res_03_vpt.png                # Valid prediction time comparison
│   ├── res_04_horizon_curves.png     # Per-horizon NRMSE curves
│   ├── res_05_bounded_latent.png     # Bounded latent trajectory visualization
│   ├── res_06_attractor_50lt.png     # 50 Lyapunov-time attractor reconstruction
│   ├── res_08_time_to_quality.png    # Wall-clock training time to quality
│   ├── res_09_ablation.png           # Ablation study
│   ├── res_10_C_through_rollout.png  # Bounded carrier C evolution during rollout
│   └── data_01..07                   # Data visualization (observations, attractor, noise, splits)
├── Utils/
│   ├── Benchmark.py                  # Parameter matching, wall-clock training, FLOP counting
│   ├── Metrics.py                    # VPT, NRMSE, divergence rate, attractor stats, Lorenz map
│   ├── Rollout.py                    # Model-agnostic evaluation + trivial references
│   ├── Plotting.py                   # Publication-grade figures (Okabe-Ito colorblind-safe palette)
│   ├── Checkpoints.py                # Skip-if-trained caching for notebook reruns
│   └── DataLoading.py                # Lorenz single-series and multi-series loaders
└── requirements.txt                  # PyTorch 2.13, SciPy, NumPy, scikit-learn, matplotlib, Jupyter
```

## Installation

```bash
git clone https://github.com/thebrownkidd/Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality.git
cd Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality
pip install -r requirements.txt
```

### Running the experiments

The experiments are Jupyter notebooks, not CLI scripts:

```bash
# Generate the Lorenz-63 dataset
python -c "from SyntheticGenerators.LorenzLift import make_dataset; make_dataset()"
# or run the full data generation pipeline:
python SyntheticGenerators/LorenzLift.py

# Open the single-series benchmark notebook
jupyter notebook Experimentation/LatentForecastComparison.ipynb

# Open the multi-series benchmark notebook
jupyter notebook Experimentation/LatentForecastComparisonMultiSeries.ipynb
```

The notebooks train all models under matched parameter and wall-clock budgets, run evaluation on independent trajectories, and generate the figures in `Figures/`.

---

## Citation

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

**Author:** [Arpit Goel](https://github.com/thebrownkidd)

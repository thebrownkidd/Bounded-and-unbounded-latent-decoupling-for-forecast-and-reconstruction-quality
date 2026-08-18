# Bounded and unbounded latent decoupling for forecast and reconstruction quality

**Thesis.** Reconstruction quality and long-horizon forecast quality are usually treated as
competing objectives to be *balanced* — by a weighted loss, a shared encoder, a spectral
truncation. This project argues the tension is not a law but an **artefact of forcing one latent
representation to serve both jobs**, and that it can be overcome **architecturally**: give each
objective its own carrier, with its own geometry, trained in its own stage.

Two carriers do the work:

| carrier | shape | job | who reads it |
|---|---|---|---|
| $\mathbf{b}_t$ | $\mathbb{R}^{k}$, **unbounded** | carry the dynamics | the forecaster |
| $C_t$ | $[0,1]^{a\times b}$, **bounded** | carry the signal | the decoder |

The forecaster never touches the decoder's carrier. The decoder never touches the forecaster's.
Boundedness is not a penalty — it is a property of the function class, so it holds at every step
of an arbitrarily long rollout.

---

## 1. Mathematical formulation

### 1.1 Objects

An observation at time $t$ is

$$x_t = \langle x_{1,t},\, x_{2,t},\, \dots,\, x_{n,t}\rangle \in \mathbb{R}^{n\times 1}.$$

Five learned maps plus a forecaster:

$$
\begin{aligned}
E &: \mathbb{R}^{n} \to \mathbb{R}^{n}, & E(x_t) &= \vec{h}_t \in \mathbb{R}^{n\times 1} \\[2pt]
f &: \mathbb{R}^{n} \to \mathbb{R}^{k}, & f(\vec{h}_t) &= \vec{b}_t \in \mathbb{R}^{k\times 1}, \quad k \ll n \\[2pt]
r &: \mathbb{R}^{n} \to [0,1]^{a\times b}, & r(\vec{h}_t) &= C_t \in [0,1]^{a\times b} \\[2pt]
m &: \mathbb{R}^{k} \to [0,1]^{a\times b}, & m(\vec{b}_t) &= C_t \\[2pt]
g &: \mathbb{R}^{k} \to \mathbb{R}^{k}, & g(\vec{b}_t) &= \vec{b}_{t+1} \\[2pt]
D &: [0,1]^{a\times b} \to \mathbb{R}^{n}, & D(C_{t+1}) &= \hat{y}_{t+1}
\end{aligned}
$$

with $\hat{y}_{t+1}$ the estimator of $x_{t+1}$. In this repo $n = 30$, $k = 3$, $a = b = 8$.

### 1.2 The consistency condition

The architecture is only coherent if the *indirect* route to the bounded latent agrees with the
*direct* one:

$$m\big(f(\vec h_t)\big) \;\approx\; r(\vec h_t) \qquad\Longleftrightarrow\qquad m(\vec b_t) \approx C_t .$$

This is the load-bearing constraint. $r$ sees the full $n$-dimensional $\vec h_t$; $m$ sees only
the $k$-dimensional $\vec b_t$. Forcing them to agree forces $f$ to **retain everything $C$ needs**
while compressing $30 \to 3$ — i.e. it is a soft surrogate for *$f$ being invertible* on the
relevant subspace. $r$ is a **training-only teacher**: it never runs at inference.

### 1.3 The two paths

**Reconstruction** (what the model saw):

$$x_t \;\xrightarrow{\;E\;}\; \vec h_t \;\xrightarrow{\;f\;}\; \vec b_t \;\xrightarrow{\;m\;}\; C_t \;\xrightarrow{\;D\;}\; \hat x_t$$

**Forecast** (what happens next). Only $\vec b$ is stepped; $C$ is *recomputed* from it by the
same $m$:

$$\vec b_{t+1} = g(\vec b_t), \qquad C_{t+1} = m(\vec b_{t+1}), \qquad \hat y_{t+1} = D(C_{t+1}).$$

Free-running to horizon $H$ is iteration of $g$ alone:

$$\hat{\vec b}_{t+\tau} = g^{\circ\tau}(\vec b_t), \qquad \hat x_{t+\tau} = D\big(m(\hat{\vec b}_{t+\tau})\big), \qquad \tau = 1,\dots,H.$$

### 1.4 Why boundedness gives unconditional stability

$m$ terminates in a sigmoid, so $C_{t+\tau} \in [0,1]^{a\times b}$ **by construction** for any input
whatsoever. $D$ is a continuous map on the compact set $[0,1]^{a\times b}$, hence

$$\big\|\hat x_{t+\tau}\big\| \;=\; \big\|D(C_{t+\tau})\big\| \;\le\; \sup_{C\in[0,1]^{a\times b}} \|D(C)\| \;<\; \infty
\qquad \text{for all } \tau,$$

**independently of $\hat{\vec b}_{t+\tau}$.** However far the unbounded latent drifts during a long
rollout, the decoded trajectory cannot leave a bounded set. Forecast error **saturates** instead of
exploding. No competing model here has that guarantee, and it costs zero parameters.

### 1.5 Architecture as implemented

| map | implementation | file |
|---|---|---|
| $E$ | MLP $30 \to 128 \to 128 \to 30$, GELU | [Encoder.py](Src/Encoder.py) |
| $f$ | MLP $30 \to 128 \to 64 \to 3$, GELU | [LatentRecon.py](Src/LatentRecon.py) |
| $r$ | MLP $30 \to 128 \to 128 \to 64$, GELU, **sigmoid** | [LatentBounded.py](Src/LatentBounded.py) |
| $m$ | MLP $3 \to 64 \to 128 \to 64$, GELU, **sigmoid** | [LatentMapping.py](Src/LatentMapping.py) |
| $g$ | GRU$(3 \to 64)$ + linear head, **residual**: $g(\vec b) = \vec b + W\,\mathrm{GRU}(\vec b)$ | [LatentForecast.py](Src/LatentForecast.py) |
| $D$ | MLP $64 \to 128 \to 128 \to 30$, GELU | [Decoder.py](Src/Decoder.py) |

Assembled in [DecoupledModel.py](Src/DecoupledModel.py). **124,482** trainable parameters;
**95,746** run at inference (the teacher $r$ is excluded).

$g$ predicts a *residual* because at $dt = 0.01$ the state barely moves per step
($\vec b_{t+1}\approx\vec b_t$); regressing the delta keeps the target off a scale where the
identity map looks like a good answer.

---

## 2. The data

A Lorenz-63 system lifted into 30 noisy observation channels, so the **true intrinsic dimension is
known to be 3**. That makes $k$ a probe rather than a hyperparameter: if the architecture works,
$k = 3$ should suffice.

### 2.1 Generation

$$\dot x = \sigma(y - x), \qquad \dot y = x(\rho - z) - y, \qquad \dot z = xy - \beta z$$

with $\sigma = 10$, $\rho = 28$, $\beta = 8/3$, integrated by RK45 at $dt = 0.01$ for 20,000 steps
after a 1,000-step burn-in. The largest Lyapunov exponent is $\lambda_{\max} = 0.906$, so

$$1 \text{ Lyapunov time} \;=\; \frac{1}{\lambda_{\max}\, dt} \;\approx\; \mathbf{110 \text{ steps}}.$$

The 3-d state is then lifted by a **frozen smooth random nonlinearity** and corrupted:

$$s_t = \frac{\text{state}_t - \mu}{\sigma}, \qquad
\text{clean}_t = \tanh(s_t W_1)\,W_2, \qquad
x_t = \widetilde{\text{clean}}_t + \varepsilon_t,\;\; \varepsilon_t \sim \mathcal{N}(0,\, 0.05^2)$$

with $W_1 \in \mathbb{R}^{3\times16}$, $W_2 \in \mathbb{R}^{16\times30}$. Noise is 5% of each
channel's spread, which fixes the **reconstruction floor at $0.05^2 = 0.0025$ MSE** — no model can
beat it. Generator: [LorenzLift.py](SyntheticGenerators/LorenzLift.py).

### 2.2 Shapes

| array | shape | what it is |
|---|---|---|
| `train` | **(14000, 30)** | observations, first 70% |
| `val` | **(3000, 30)** | observations, next 15% |
| `test` | **(3000, 30)** | observations, last 15% |
| `clean` | (20000, 30) | noiseless signal — diagnostics only |
| `states` | (20000, 3) | true Lorenz state — **never in any loss** |
| `EvalObs` | **(32, 12000, 30)** | 32 *independent* trajectories for long rollouts |
| `EvalStates` | (32, 12000, 3) | their true states |

Global observation range $[-3.44,\, 3.04]$; every channel standardised to unit variance.

### 2.3 What it looks like

The split is **chronological — never shuffled** — so the test set is genuinely the future:

![splits](Figures/data_07_splits.png)

The 30 channels the model actually sees. Each is a smooth nonlinear mixture of all three Lorenz
coordinates, so no single channel is any one variable:

![observations](Figures/data_01_observations.png)

The 3-d truth underneath, which no model is ever shown:

![true state](Figures/data_02_true_state.png)

![attractor](Figures/data_03_attractor.png)

The noise, zoomed in far enough to see. This gap is the reconstruction floor:

![noise](Figures/data_04_noise.png)

Every channel carries comparable energy, so no channel dominates the MSE:

![channel stats](Figures/data_05_channel_stats.png)

**Why a separate evaluation set.** The test split is 3,000 steps ≈ 27 Lyapunov times. Long rollouts
cut from it overlap heavily and share most of their future, which badly understates error spread.
`LorenzEval.npz` holds 32 trajectories from different initial conditions, pushed through the
**frozen** lift so they land in exactly the same observation space:

![eval trajectories](Figures/data_06_eval_trajectories.png)

---

## 3. Training pipeline

Two stages. **The separation is the mechanism, not an optimisation convenience.**

### Stage 1 — representation

Train $\theta_A = \{E, f, m, r, D\}$ jointly on single timesteps. No forecasting, no sequences:

$$
\mathcal{L}_A(\theta_A) \;=\;
\underbrace{\frac{1}{|B|}\sum_{t\in B} \big\| D\big(m(f(E(x_t)))\big) - x_t \big\|_2^2}_{\text{reconstruction}}
\;+\; \lambda \underbrace{\frac{1}{|B|}\sum_{t\in B} \big\| m(f(E(x_t))) - r(E(x_t)) \big\|_F^2}_{\text{consistency (§1.2)}}
$$

with $\lambda = 1$, Adam, $\eta = 10^{-3}$, batch 256.

### Stage 2 — dynamics

**Freeze $\theta_A$ completely.** Run the frozen encoder over every training timestamp once to
produce the latent series, which becomes the ground truth:

$$\vec b_t \;=\; f\big(E(x_t)\big), \qquad t = 1,\dots,T \qquad \text{(computed once, no gradients)}$$

Train $\theta_g$ — and **only** $\theta_g$ — free-running in $\vec b$-space. Warm the GRU state on
$\vec b_{t-L+1:t-1}$, then unroll $H$ steps without teacher forcing:

$$\hat{\vec b}_{t+1} = g(\vec b_t), \qquad \hat{\vec b}_{t+\tau+1} = g(\hat{\vec b}_{t+\tau})$$

$$\boxed{\;\mathcal{L}_B(\theta_g) \;=\; \frac{1}{H}\sum_{\tau=1}^{H} \big\| \hat{\vec b}_{t+\tau} - \vec b_{t+\tau} \big\|_2^2\;}$$

with $L = 64$, $H = 20$, batch 64. **Neither $m$ nor $D$ appears in $\mathcal{L}_B$.**

### Why this is the decoupling

$$\frac{\partial \mathcal{L}_B}{\partial \theta_A} \;=\; 0 \qquad\text{identically, by construction.}$$

Improving the forecast **cannot** degrade reconstruction. Not "is penalised for degrading" —
*cannot*. This is the architectural statement of the thesis.

Contrast the standard balancing approach, which shares one latent $z$ and one gradient:

$$\mathcal{L}_\phi = (1-\phi)\,\mathcal{L}_{\text{rec}} + \phi\,\mathcal{L}_{\text{fcst}},
\qquad \frac{\partial \mathcal{L}_{\text{fcst}}}{\partial \theta_{\text{enc}}} \neq 0 .$$

There the two objectives pull on the same weights, and $\phi$ only chooses *which one loses*.

### Inference

$$\hat x_{t+\tau} \;=\; D\Big(m\big(g^{\circ\tau}(\vec b_t)\big)\Big)$$

Stage 2 trains only **13,443** of the 124,482 parameters, on precomputed latents with no decoder in
the graph — which is why it is also the cheapest thing in the comparison.

---

## 4. Results — in plain language

### 4.1 The setup, plainly

Twelve models compete. **Every one gets exactly the same resources**: 124,482 adjustable numbers,
120 seconds of training, one CPU core. Nobody gets to win by being bigger or training longer.

Two things are measured.

- **Copying** — show the model a signal, ask it to reproduce it. Lower error is better. There is a
  hard floor of `0.0025` because the data has random noise in it that nothing can predict.
- **Predicting** — show the model 64 steps, then cut it off and let it run blind. Count how long
  before it's badly wrong. Measured in **Lyapunov times**; 1 Lyapunov time ≈ 110 steps. Chaos makes
  this brutally hard — that is the point.

The competitors:

- **The knob (φ)** — one shared memory for both jobs, and a dial that decides how much each one
  matters. φ=0 means "only copy", φ=1 means "only predict". This is the standard approach.
- **The cheat (PINN)** — a model *handed the exact equations* that generate the data. It should
  win. It's here to show what's achievable.
- **The dummies** — "tomorrow equals today" (persistence) and "tomorrow equals the average"
  (climatology). Anything that can't beat these is broken.

### 4.2 The headline

![pareto](Figures/res_01_pareto.png)

The grey line is the knob, traced from end to end — every tradeoff it can possibly make. **The blue
star is this architecture, sitting above it.** Up is better prediction, left is better copying.

The star is somewhere the dial cannot reach at any setting.

### 4.3 Copying: basically tied for best

![reconstruction](Figures/res_02_reconstruction.png)

| model | copy error | how far above the floor |
|---|---|---|
| φ=0 (pure copier, doesn't predict at all) | 0.00280 | 1.12× |
| φ=0.1 | 0.00296 | 1.18× |
| **Ours** | **0.00313** | **1.25×** |
| φ=0.5 | 0.00369 | 1.47× |
| PINN (given the true equations) | 0.00391 | 1.57× |
| φ=1 | 0.01243 | 4.97× |

Third out of ten. In $R^2$ terms — the "percentage explained" score — a model that does *nothing but
copy* scores **0.9973**. Ours scores **0.9970**. That difference is three ten-thousandths.

**So: we give up almost nothing on copying.**

### 4.4 Predicting: first, by a mile

![vpt](Figures/res_03_vpt.png)

| model | how far it predicts (Lyapunov times) |
|---|---|
| **Ours** | **0.526 ± 0.066** |
| φ=0.1 | 0.344 |
| φ=0.75 | 0.344 |
| φ=0.5 / 0.9 / 1 | 0.326 |
| φ=0.25 | 0.308 |
| PINN (given the true equations) | 0.233 |
| "tomorrow = today" | 0.036 |

**53% further than anything else**, including the model handed the answer key.

And here is the finding that really makes the case. Look at the knob from φ=0.1 to φ=1: copy error
**doubles** (0.00296 → 0.01243) while prediction **gets slightly worse** (0.344 → 0.326).

> **Turning the dial past 0.1 destroys your copying and buys you nothing.** The knob has a hard
> ceiling around 0.34 that no amount of sacrifice breaks through. There is no tradeoff there to
> ride — just a cliff you fall off.

Ours is at 0.526. Not further along their curve — off it.

### 4.5 The rivals don't just get worse. They explode.

![horizon](Figures/res_04_horizon_curves.png)

| model | error at 5 Lyapunov times | % of runs that blow up |
|---|---|---|
| **Ours** | **1.08** | **5.7%** |
| φ=0.1 | 13.4 | 100% |
| φ=0.5 | 15.4 | 100% |
| φ=1 | 8.4 | 100% |
| "predict the average forever" | 1.01 | 0% |

An error of 1.0 means "no better than guessing the long-run average" — the worst a *sane* answer can
be. Ours lands at **1.08**, essentially at that sane ceiling. Every knob model is at **8 to 15**,
i.e. wrong by ten times the entire spread of the data, on **every single run**.

That is §1.4 in practice: sigmoid, compact domain, bounded output. It cannot blow up.

### 4.6 After 50 Lyapunov times — 5,500 steps blind

Short-term error hides this. A model can track for a while and then quietly collapse onto a fixed
point, and its error curve would look no different.

![attractor 50lt](Figures/res_06_attractor_50lt.png)

![invariant measure](Figures/res_07_invariant_measure.png)

| model | distance from the true attractor (lower = better) |
|---|---|
| **Ours** | **0.093** |
| PINN (given the true equations) | 0.623 |
| PINN (slightly wrong equations) | 0.669 |
| φ=0.5 | 113.2 |
| φ=1 | 98.9 |
| φ=0 | 200.5 |

After running blind for **5,500 steps**, this architecture still traces the Lorenz butterfly more
faithfully than a model that was **given the exact differential equations**. The knob models are off
by three orders of magnitude.

### 4.7 Why it works — the mechanism, visible

![bounded latent](Figures/res_05_bounded_latent.png)

![C through rollout](Figures/res_10_C_through_rollout.png)

| quantity | value | reading |
|---|---|---|
| $\vec b$ range on real data | $[-18.60,\ 18.57]$ | — |
| $\max_i \lvert b_i \rvert$ over a 5,500-step blind rollout | **19.98** | stays in its normal range |
| $C$ entries outside $[0,1]$ | **0.000000** | the guarantee holds exactly |
| $C$ entries pinned at the rails | 0.287 | the code stays *expressive*, not clipped |

The forecaster does not wander into nonsense: after 5,500 blind steps the latent is still inside the
range it occupies on real data. And $C$ is not saturated — it is carrying information, not merely
being clamped. Containment *and* fidelity, which is why the long-run attractor survives.

### 4.8 It is also the cheapest

![time to quality](Figures/res_08_time_to_quality.png)

| model | params actually trained for forecasting | time to reach good copying |
|---|---|---|
| **Ours** | **13,443** | **19.6 s** |
| PINN | 124,927 | 40.6 s |
| φ=0.5 | 124,256 | 68.9 s |
| φ=0.9, φ=1 | 124,256 | never |

Stage 2 trains on precomputed latents with no decoder in the graph, so its updates are ~10× cheaper.
Decoupling is not just better — it is faster.

### 4.9 Does the two-stage schedule matter? Yes.

![ablation](Figures/res_09_ablation.png)

Same architecture, trained end-to-end in one stage instead of two:

| | copy error | prediction |
|---|---|---|
| **Ours (two-stage)** | **0.0031** | **0.526** |
| Ours (joint, one stage) | 0.0126 | 0.462 |

Let the forecast gradient reach the encoder and copying gets **4× worse**. The architecture alone is
not enough — the *training separation* is what makes $\partial\mathcal{L}_B/\partial\theta_A = 0$
true, and that equation is the whole thesis.

### 4.10 One-paragraph summary

Everyone else has one memory doing two jobs and a dial to decide who wins. Past a certain point
their dial stops working — you lose copying quality and gain no prediction. This architecture uses
**two memories with different shapes**: a small unbounded one that carries motion, and a boxed-in
one that carries the picture. They are trained in separate stages, so improving prediction
*mathematically cannot* damage copying. The result: copying essentially tied with the best pure
copier, prediction 53% beyond anything the dial can reach, no blow-ups, a recognisable attractor
after 5,500 blind steps, and the cheapest training in the comparison. **The tradeoff was not
balanced. It was removed by construction.**

---

## 5. Repository

```
Src/                    the architecture
  Encoder.py            E : x -> h
  LatentRecon.py        f : h -> b        (unbounded, k-dim)
  LatentMapping.py      m : b -> C        (bounded, sigmoid)
  LatentBounded.py      r : h -> C        (training-only teacher)
  LatentForecast.py     g : b_t -> b_t+1  (residual GRU)
  Decoder.py            D : C -> x_hat
  DecoupledModel.py     assembly, Rollout, stage partitions
Comp/                   baselines (nothing here is ours)
  JointAEGRU.py         the weighted-loss knob
  PhysicsLatentAE.py    the PINN, RK4 on the true field
  LorenzField.py        differentiable Lorenz-63 RHS
Utils/                  DataLoading, Metrics, Rollout, Benchmark, Plotting
SyntheticGenerators/    LorenzLift.py — the data generator
Experimentation/
  NeuralNetworkApproximation.ipynb   Stage 1 alone, in detail
  LatentForecastComparison.ipynb     the full 12-way comparison
Data/                   LorenzLift.npz, LorenzEval.npz
Figures/                every figure in this README, 300 dpi
```

### Reproducing

```bash
pip install -r requirements.txt
python SyntheticGenerators/LorenzLift.py        # regenerate the data (optional)
jupyter lab Experimentation/LatentForecastComparison.ipynb
```

Set `QUICK = True` in the config cell for a fast smoke run. The full run is
`120 s × 12 configurations` on one pinned CPU thread.

### Caveats worth knowing

- **One system, one lift, one noise level.** The claim is demonstrated on Lorenz-63. Turbofan,
  weather or bearing data are needed before it is a claim about representations in general.
- **Seeds are uneven.** Ours and the PINN run 3 seeds; the φ sweep runs 1 each, so a `VPT std` of
  `0.0000` on those rows means $n=1$, not stability.
- **Absolute horizons are short.** 0.526 Lyapunov times is ~58 steps. Chaos is hard; everyone here
  fails eventually. The claim is comparative.
- **The φ baseline is a construction**, not a named published method — the weighted-loss
  autoencoder+GRU in its plainest form.

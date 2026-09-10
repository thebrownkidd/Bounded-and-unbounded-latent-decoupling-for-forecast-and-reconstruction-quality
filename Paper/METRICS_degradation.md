# Metric suite for degradation data (C-MAPSS, IMS bearing, PHM milling)

The Lorenz suite was built around a chaotic attractor with a known Lyapunov
exponent and a synthetic clean signal. Degradation data has neither. This is
what survives, what has to be reinterpreted, what dies, and what should be added.

## 1. Transfers unchanged

These never depended on chaos:

| metric | note |
|---|---|
| reconstruction MSE, R2, worst-channel R2 | unchanged |
| MAE and RMSE at lead `h` | unchanged |
| anomaly correlation (ACC) at lead `h` | correlation against the training climatology, no Lyapunov time involved |
| skill against persistence | ratio of MSEs, unit free |
| variance ratio | ratio of standard deviations, unit free |
| divergence | fraction of rollouts leaving the observed training range |

## 2. Transfers with one reinterpretation

The only thing Lyapunov time ever did in this project was **put lead time on a
dimensionless axis**. Replace it with the system's native clock and the metric is
still exactly as meaningful:

| Lorenz | degradation |
|---|---|
| VPT, in Lyapunov times | lead in **cycles** at which NRMSE first crosses 0.4 |
| ACC horizon, in Lyapunov times | lead in **cycles** at which ACC first drops below 0.6 |

`LAMBDA_MAX = 0.906` (`Utils/Metrics.py:16`) must not be applied here. The
implementation in `Experimentation/run_cmapss.py` computes both horizons directly
in cycles rather than reusing `Utils.Metrics.VPT`, precisely to avoid that.

## 3. Dies, and must not be reported

| metric | why |
|---|---|
| attractor Wasserstein, invariant measure, return map | run-to-failure trajectories are non-stationary by construction. There is no invariant measure to match. |
| noise floor, `x floor` | no ground truth clean signal, so no known floor. A smoothing-residual **estimate** is provided as `noise_floor_mse_est`, deliberately named so it is never mistaken for the Lorenz floor. It over-estimates wherever the signal has curvature at the smoothing scale. |
| capacity bound `(n-d)/n`, passthrough | both require the clean signal to separate copying from denoising. **These analyses stay Lorenz-only.** |
| long-horizon anything | C-MAPSS units are 128 to 360 cycles. The Lorenz rollouts are 550 to 5,500 steps. There is no long horizon here. |

## 4. Additions that suit degradation data

### 4a. Free-run growth
Norm of the decoded rollout at the final lead divided by the truth's norm.
1.0 is correct amplitude, above 1 is inflating. This is the previous project's
stability metric (`Jet-Engine-Simulation-Project/experiments/acml/acml_common.py:347`)
and it is the right stand-in for divergence on bounded-range sensor data.

### 4b. Monotonicity violation
Degradation is near-monotone. Fraction of sign flips in the mean-channel
trend of the rollout. A forecast that has decorrelated will oscillate and score
badly here even when its amplitude and divergence look fine, which is exactly the
failure mode the variance ratio caught on Lorenz for phi=0.1.

### 4c. RUL as downstream utility
Fit a linear head on the **frozen** forecast carrier `b` and report RUL RMSE.
This is the standard C-MAPSS metric and it does something no reconstruction or
forecast score does: it tests whether `b` carries health information at all,
rather than merely being reconstructable. Cheap, since the head is linear and
everything upstream is frozen.

## 5. The metric that actually measures decoupling

Everything above measures quality. None of it measures **decoupling**, which is
the paper's claim. On Lorenz the argument is comparative: our point sits off the
phi frontier. That needs a full sweep, which is expensive and which the reviewer
correctly says does not establish a frontier anyway.

There is a direct measurement available instead.

**Reconstruction sensitivity to forecast weight.** On a shared latent, turning up
the forecast weight degrades reconstruction, because the same carrier has to give
something up. Measure the slope

```
S = d(recon MSE) / d(phi)
```

over the sweep. A large positive `S` is the tradeoff, made quantitative on a
single dataset. For the three-phase split the corresponding quantity is the
change in reconstruction between the end of phase 1 (before any forecast
objective exists) and the end of phase 3, since the forecast gradient provably
never reaches `D` in phase 2. If that change is near zero while the baseline's
`S` is large, **that is the decoupling, measured directly rather than inferred
from a Pareto plot.**

Two things make this stronger than the frontier argument:

- It needs only a handful of phi points, not a dense searched frontier, because
  it is a slope rather than an envelope.
- It is immune to the reviewer's objection that seven hand-picked phi values do
  not establish a Pareto frontier. A slope does not claim to be a frontier.

This should be computed on every dataset, and it is the number to lead with on
the real data where the long-horizon story does not exist.

## 6. What this cannot show

On C-MAPSS the intrinsic dimension is unknown, so `k` has to be swept rather than
set. That removes the reviewer's "you gave it the right answer" objection but also
means the Lorenz result and the C-MAPSS result are not the same experiment. They
answer the same question on different terms and the paper must say so rather than
merging them into one table.

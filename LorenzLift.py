"""
Lorenz-63 -> nonlinear lift -> noisy high-dimensional observations.

Purpose: a first-experiment testbed where the TRUE intrinsic dimension is
known to be 3, so that k in the forecasting branch becomes a probe rather
than a hyperparameter.

Dependencies: numpy, scipy.
"""

import os

import numpy as np
from scipy.integrate import solve_ivp


def lorenz63(n_steps=20000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0,
             x0=(1.0, 1.0, 1.0), burn_in=1000, seed=0):
    """Integrate Lorenz-63. Returns (n_steps, 3) on the attractor.

    dt=0.01 gives ~110 steps per Lyapunov time (lambda ~ 0.906), so a
    50-step rollout is roughly half a Lyapunov time -- hard but not
    hopeless. Shrink dt for a finer-grained series.
    """
    def f(t, s):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    total = n_steps + burn_in
    t_eval = np.arange(total) * dt
    sol = solve_ivp(f, (0, t_eval[-1]), list(x0), t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y.T[burn_in:]  # drop transient, keep only the attractor


def lift(states, n_obs=30, noise=0.05, seed=0, hidden=16):
    """Map (T, 3) latent states -> (T, n_obs) noisy observations.

    Smooth random nonlinear map: linear -> tanh -> linear. Smooth on
    purpose. A wilder map curves the manifold pathologically and you end
    up debugging the encoder instead of testing the architecture.

    noise is relative: fraction of each channel's std, added as Gaussian
    observation noise. This is your main experimental dial.
    """
    rng = np.random.default_rng(seed)
    s = (states - states.mean(0)) / states.std(0)          # standardise latents

    W1 = rng.normal(0, 1.0, size=(3, hidden))
    W2 = rng.normal(0, 1.0, size=(hidden, n_obs))
    clean = np.tanh(s @ W1) @ W2

    clean = (clean - clean.mean(0)) / clean.std(0)         # standardise channels
    obs = clean + rng.normal(0, noise, size=clean.shape)
    return obs, clean


def make_dataset(n_steps=20000, n_obs=30, noise=0.05, dt=0.01, seed=0,
                 train_frac=0.7, val_frac=0.15):
    """Full pipeline with a chronological (non-shuffled) split.

    Returns dict with train/val/test observations, plus the clean signal
    and true 3-d latent states for diagnostics. Never train on those two.
    """
    states = lorenz63(n_steps=n_steps, dt=dt, seed=seed)
    obs, clean = lift(states, n_obs=n_obs, noise=noise, seed=seed)

    n = len(obs)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))

    # Standardise using TRAIN statistics only.
    mu, sd = obs[:i].mean(0), obs[:i].std(0)
    obs = (obs - mu) / sd

    return {
        "train": obs[:i], "val": obs[i:j], "test": obs[j:],
        "clean": clean, "states": states,      # diagnostics only
        "noise_floor": noise,                  # reconstruction MSE can't beat ~noise^2
    }


def windows(series, lookback=64, horizon=50):
    """Sliding windows -> (N, lookback, C) inputs and (N, horizon, C) targets."""
    T = len(series)
    n = T - lookback - horizon + 1
    idx = np.arange(n)[:, None]
    X = series[idx + np.arange(lookback)]
    Y = series[idx + lookback + np.arange(horizon)]
    return X, Y


if __name__ == "__main__":
    d = make_dataset()
    X, Y = windows(d["train"])
    print(f"train {d['train'].shape}  val {d['val'].shape}  test {d['test'].shape}")
    print(f"windows X {X.shape}  Y {Y.shape}")
    print(f"noise floor (MSE): {d['noise_floor'] ** 2:.5f}")
    train = d["train"]
    test = d["test"]
    val = d["val"]
    os.makedirs("Data", exist_ok=True)
    np.savez("Data/LorenzLift.npz", train=train, test=test, val=val, clean=d["clean"], states=d["states"])
    #also save a readme file with the parameters used to generate the dataset
    with open("Data/LorenzLift_README.txt", "w") as f:
        f.write(f"n_steps: {20000}\n")
        f.write(f"n_obs: {30}\n")
        f.write(f"noise: {0.05}\n")
        f.write(f"dt: {0.01}\n")
        f.write(f"train_frac: {0.7}\n")
        f.write(f"val_frac: {0.15}\n")
        f.write(f"test_frac: {0.15}\n")
        f.write(f"lookback: {64}\n")
        f.write(f"horizon: {50}\n")
    
    # Sanity check: the observations really do lie on a 3-d manifold.
    # Linear PCA will NOT show a clean cut at 3 -- the lift is nonlinear,
    # so 3 intrinsic dimensions smear across many linear components.
    # That smearing is the point: it is why a linear method struggles here.
    c = np.cov(d["clean"].T)
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    ev = ev / ev.sum()
    print("PCA var explained, first 8:", np.round(ev[:8], 4))
    print("components for 99% var:", int(np.searchsorted(np.cumsum(ev), 0.99) + 1))
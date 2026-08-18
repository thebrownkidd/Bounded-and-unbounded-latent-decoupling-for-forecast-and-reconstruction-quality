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


# --------------------------------------------------------------------------
# Held-out evaluation trajectories.
#
# The test split is 3000 steps, about 27 Lyapunov times. Long rollouts cut from
# it overlap heavily, so their errors are correlated and the spread across
# "different" starts is far narrower than it looks. Independent trajectories fix
# that -- but only if they land in exactly the same observation space, which
# means freezing the lift rather than redrawing it.
# --------------------------------------------------------------------------

def lift_params(states, n_obs=30, noise=0.05, seed=0, hidden=16, train_frac=0.7):
    """Recover every constant `lift` + `make_dataset` used, from the same seed.

    lift() draws W1, W2 and the noise from one default_rng(seed) in that order,
    so replaying the sequence reproduces the map exactly -- including the noise
    realisation, which is what makes the train-split mu/sd recoverable. Those
    are not stored in the npz and cannot be recovered from `train` alone, since
    standardisation erased them.
    """
    rng = np.random.default_rng(seed)
    state_mu, state_sd = states.mean(0), states.std(0)
    s = (states - state_mu) / state_sd

    W1 = rng.normal(0, 1.0, size=(3, hidden))
    W2 = rng.normal(0, 1.0, size=(hidden, n_obs))
    raw = np.tanh(s @ W1) @ W2

    clean_mu, clean_sd = raw.mean(0), raw.std(0)
    clean = (raw - clean_mu) / clean_sd
    obs = clean + rng.normal(0, noise, size=clean.shape)

    i = int(len(obs) * train_frac)
    obs_mu, obs_sd = obs[:i].mean(0), obs[:i].std(0)

    return {"W1": W1, "W2": W2, "state_mu": state_mu, "state_sd": state_sd,
            "clean_mu": clean_mu, "clean_sd": clean_sd,
            "obs_mu": obs_mu, "obs_sd": obs_sd, "noise": noise,
            "clean": clean, "train": (obs[:i] - obs_mu) / obs_sd}


def lift_with(states, params, rng):
    """Apply an already-fixed lift to new states. Returns (obs, clean).

    Every constant comes from `params`, none from `states` -- a fresh trajectory
    standardised by its own statistics would sit in a subtly different space and
    quietly invalidate the comparison.
    """
    s = (states - params["state_mu"]) / params["state_sd"]
    clean = (np.tanh(s @ params["W1"]) @ params["W2"] - params["clean_mu"]) / params["clean_sd"]
    obs = clean + rng.normal(0, params["noise"], size=clean.shape)
    return (obs - params["obs_mu"]) / params["obs_sd"], clean


def make_eval_trajectories(params, n_traj=32, n_steps=12000, dt=0.01,
                           burn_in=2000, seed=1234):
    """Independent Lorenz trajectories in the training observation space.

    Initial conditions are drawn well off the attractor and burned in, so the
    trajectories are independent of each other and of the training run rather
    than being distant points on the same orbit.
    """
    rng = np.random.default_rng(seed)
    obs, states = [], []
    for _ in range(n_traj):
        x0 = tuple(rng.uniform(-15.0, 15.0, size=3))
        st = lorenz63(n_steps=n_steps, dt=dt, x0=x0, burn_in=burn_in)
        o, _ = lift_with(st, params, rng)
        obs.append(o)
        states.append(st)
    return np.stack(obs), np.stack(states)


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
    
    # Independent held-out trajectories for long-rollout evaluation.
    # The two asserts are the load-bearing part: they prove the replayed lift is
    # the same map that produced LorenzLift.npz. If they fail, the eval set
    # lives in a different observation space and nothing measured on it is
    # comparable to anything measured on train/val/test.
    params = lift_params(d["states"])
    assert np.allclose(params["clean"], d["clean"], atol=1e-10), "lift replay mismatch"
    assert np.allclose(params["train"], train, atol=1e-10), "train replay mismatch"
    print("lift replay verified against LorenzLift.npz")

    eval_obs, eval_states = make_eval_trajectories(params)
    np.savez("Data/LorenzEval.npz", obs=eval_obs, states=eval_states,
             obs_mu=params["obs_mu"], obs_sd=params["obs_sd"],
             state_mu=params["state_mu"], state_sd=params["state_sd"])
    print(f"eval trajectories {eval_obs.shape}  states {eval_states.shape}")

    # Sanity check: the observations really do lie on a 3-d manifold.
    # Linear PCA will NOT show a clean cut at 3 -- the lift is nonlinear,
    # so 3 intrinsic dimensions smear across many linear components.
    # That smearing is the point: it is why a linear method struggles here.
    c = np.cov(d["clean"].T)
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    ev = ev / ev.sum()
    print("PCA var explained, first 8:", np.round(ev[:8], 4))
    print("components for 99% var:", int(np.searchsorted(np.cumsum(ev), 0.99) + 1))
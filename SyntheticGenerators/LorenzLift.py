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
    noise_raw = obs - clean                # the actual noise draw, pre-final-rescale

    n = len(obs)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))

    # Standardise using TRAIN statistics only.
    mu, sd = obs[:i].mean(0), obs[:i].std(0)
    obs = (obs - mu) / sd

    # The MSE floor no model can beat, MEASURED in the final standardised scale
    # rather than assumed from the noise parameter -- `noise**2` is only exact
    # if obs_sd == 1, which it is not quite, since sd is fit on noisy data.
    noise_floor_mse = float(((noise_raw[:i] / sd) ** 2).mean())

    return {
        "train": obs[:i], "val": obs[i:j], "test": obs[j:],
        "clean": clean, "states": states,      # diagnostics only
        "noise_floor": noise,                  # the config knob, not a usable MSE
        "noise_floor_mse": noise_floor_mse,    # the actual floor -- use this one
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


def lift_raw_with(states, params, rng):
    """As lift_with, but stops before the final obs_mu/obs_sd standardisation.

    Everything except the last rescale comes from `params`, none from `states`.
    Exists so a caller can pool several trajectories' TRAIN portions and fit one
    shared obs_mu/obs_sd across all of them, rather than being stuck with the
    single-trajectory scale baked into `params` by lift_params -- see
    make_multi_series_dataset.
    """
    s = (states - params["state_mu"]) / params["state_sd"]
    clean = (np.tanh(s @ params["W1"]) @ params["W2"] - params["clean_mu"]) / params["clean_sd"]
    obs = clean + rng.normal(0, params["noise"], size=clean.shape)
    return obs, clean


def lift_with(states, params, rng):
    """Apply an already-fixed lift to new states. Returns (obs, clean).

    Every constant comes from `params`, none from `states` -- a fresh trajectory
    standardised by its own statistics would sit in a subtly different space and
    quietly invalidate the comparison.
    """
    obs, clean = lift_raw_with(states, params, rng)
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


def make_multi_series_dataset(n_series=8, n_steps=20000, n_obs=30, noise=0.05, dt=0.01,
                              seed=0, train_frac=0.7, val_frac=0.15,
                              n_holdout=32, holdout_steps=12000, holdout_burn_in=2000,
                              hidden=16):
    """Several independent trajectories, sharing one frozen lift.

    Two pools, both in the same observation space:

    * `n_series` training-pool trajectories. EACH gets the same chronological
      70/15/15 split as the single-series pipeline; their train/val/test chunks
      are meant to be pooled by the caller (concatenated, per split).
    * `n_holdout` further trajectories, from different initial conditions and
      never chronologically split at all -- purely for testing on series the
      model has not seen in any form, matching the existing LorenzEval.npz role.

    EVERY standardisation constant -- state_mu/sd, clean_mu/sd, and obs_mu/sd --
    is fit on the POOLED TRAIN portion of the n_series trajectories only, then
    applied unchanged to val, test, and the n_holdout trajectories. Earlier
    versions of this function anchored state_mu/sd and clean_mu/sd to a single
    trajectory's FULL series via `lift_params` (i.e. including that trajectory's
    own val/test range), which let those ranges influence a constant every split
    then shared. That anchor is gone: nothing here is fit on anything but the
    pooled training portions, and holdout never contributes its own statistics.

    Returns a dict: "train"/"val"/"test" (n_series, T_i, n_obs), "clean"/"states"
    (n_series, n_steps, ...), "holdout_obs"/"holdout_states"
    (n_holdout, holdout_steps, ...), plus obs_mu, obs_sd, noise_floor.
    """
    rng = np.random.default_rng(seed)
    i, j = int(n_steps * train_frac), int(n_steps * (train_frac + val_frac))

    ics = rng.uniform(-15.0, 15.0, size=(n_series, 3))
    pool_states = [lorenz63(n_steps=n_steps, dt=dt, x0=tuple(x0), burn_in=1000, seed=seed)
                  for x0 in ics]

    # -- state standardisation: pooled TRAIN portions only
    pooled_train_states = np.concatenate([st[:i] for st in pool_states])
    state_mu, state_sd = pooled_train_states.mean(0), pooled_train_states.std(0)

    # -- the frozen nonlinearity itself: random, not fit on data, so no leak risk
    lift_rng = np.random.default_rng(seed)
    W1 = lift_rng.normal(0, 1.0, size=(3, hidden))
    W2 = lift_rng.normal(0, 1.0, size=(hidden, n_obs))

    def raw_clean(states):
        s = (states - state_mu) / state_sd
        return np.tanh(s @ W1) @ W2

    pool_raw_clean = [raw_clean(st) for st in pool_states]

    # -- clean-signal standardisation: pooled TRAIN portions only
    pooled_train_clean = np.concatenate([c[:i] for c in pool_raw_clean])
    clean_mu, clean_sd = pooled_train_clean.mean(0), pooled_train_clean.std(0)

    def clean_of(raw_c):
        return (raw_c - clean_mu) / clean_sd

    pool_clean = [clean_of(c) for c in pool_raw_clean]

    # -- noise, drawn once per pool trajectory in the clean-standardised scale
    obs_rng = np.random.default_rng(seed + 1)
    pool_raw = [(c + obs_rng.normal(0, noise, size=c.shape), c) for c in pool_clean]  # [(obs, clean), ...]

    # -- final obs standardisation: pooled TRAIN portions only (unchanged from before)
    pooled_train_raw = np.concatenate([raw[:i] for raw, _ in pool_raw])
    obs_mu, obs_sd = pooled_train_raw.mean(0), pooled_train_raw.std(0)

    def scale(a):
        return (a - obs_mu) / obs_sd

    train = np.stack([scale(raw[:i]) for raw, _ in pool_raw])
    val = np.stack([scale(raw[i:j]) for raw, _ in pool_raw])
    test = np.stack([scale(raw[j:]) for raw, _ in pool_raw])
    clean = np.stack(pool_clean)
    states = np.stack(pool_states)

    # Held-out trajectories: never chronologically split, standardised with the
    # SAME train-derived constants above (state_mu/sd, W1/W2, clean_mu/sd,
    # obs_mu/sd) -- never their own statistics.
    holdout_rng = np.random.default_rng(seed + 2)
    ho_states, ho_obs = [], []
    for _ in range(n_holdout):
        x0 = tuple(holdout_rng.uniform(-15.0, 15.0, size=3))
        st = lorenz63(n_steps=holdout_steps, dt=dt, x0=x0, burn_in=holdout_burn_in)
        c = clean_of(raw_clean(st))
        raw_obs = c + holdout_rng.normal(0, noise, size=c.shape)
        ho_obs.append(scale(raw_obs))
        ho_states.append(st)

    # The MSE floor no model can beat, MEASURED in the final standardised scale
    # and pooled across the training-pool trajectories -- see make_dataset.
    noise_train_raw = np.concatenate([raw[:i] - c[:i] for raw, c in pool_raw])
    noise_floor_mse = float(((noise_train_raw / obs_sd) ** 2).mean())

    return {"train": train, "val": val, "test": test, "clean": clean, "states": states,
           "holdout_obs": np.stack(ho_obs), "holdout_states": np.stack(ho_states),
           "obs_mu": obs_mu, "obs_sd": obs_sd, "noise_floor": noise,
           "noise_floor_mse": noise_floor_mse}


def save_multi_series_dataset(path="Data/LorenzLiftMulti.npz",
                              readme_path="Data/LorenzLiftMulti_README.txt",
                              **kwargs):
    """make_multi_series_dataset(**kwargs) -> saved .npz + README. Returns the dict."""
    d = make_multi_series_dataset(**kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, train=d["train"], val=d["val"], test=d["test"],
             clean=d["clean"], states=d["states"],
             holdout_obs=d["holdout_obs"], holdout_states=d["holdout_states"],
             obs_mu=d["obs_mu"], obs_sd=d["obs_sd"],
             noise_floor_mse=d["noise_floor_mse"])
    with open(readme_path, "w") as f:
        f.write(f"n_series (training pool): {d['train'].shape[0]}\n")
        f.write(f"per-series steps: train {d['train'].shape[1]}  "
               f"val {d['val'].shape[1]}  test {d['test'].shape[1]}\n")
        f.write(f"n_holdout (never trained on, no temporal split): "
               f"{d['holdout_obs'].shape[0]}\n")
        f.write(f"holdout steps: {d['holdout_obs'].shape[1]}\n")
        f.write(f"n_obs: {d['train'].shape[-1]}\n")
        f.write(f"noise_floor_mse (measured, use this): {d['noise_floor_mse']:.6f}\n")
        f.write("obs_mu/obs_sd: fit on the POOLED train chunk across every "
               "training-pool series.\n")
        f.write("Same frozen lift (W1, W2) as LorenzLift.npz would use at the "
               "same seed, so a single-series model and a multi-series model "
               "are NOT directly comparable on raw MSE -- obs_mu/obs_sd differ.\n")
    return d


if __name__ == "__main__":
    d = make_dataset()
    X, Y = windows(d["train"])
    print(f"train {d['train'].shape}  val {d['val'].shape}  test {d['test'].shape}")
    print(f"windows X {X.shape}  Y {Y.shape}")
    print(f"noise floor (measured MSE): {d['noise_floor_mse']:.6f}")
    train = d["train"]
    test = d["test"]
    val = d["val"]
    os.makedirs("Data", exist_ok=True)
    np.savez("Data/LorenzLift.npz", train=train, test=test, val=val, clean=d["clean"], states=d["states"],
             noise_floor_mse=d["noise_floor_mse"])
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
        f.write(f"noise_floor_mse (measured, use this): {d['noise_floor_mse']:.6f}\n")

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
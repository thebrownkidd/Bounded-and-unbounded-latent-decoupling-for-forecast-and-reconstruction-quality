"""
Four chaotic systems for the multi-system reproducibility experiment.

All four follow the same interface as LorenzLift.make_multi_series_dataset so
the experiment script can treat them identically.

Systems
-------
lorenz96_n8
    Lorenz-96, N=8 dimensions, F=8.  State is 8-D.  Lifted to 40 channels
    through a frozen tanh map, matching the Lorenz-63 pipeline exactly.
    λ_max ≈ 1.67,  dt=0.01,  steps/LT ≈ 60.

rossler
    Rössler attractor (a=0.2, b=0.2, c=5.7).  State is 3-D.  Lifted to
    30 channels.
    λ_max ≈ 0.071,  dt=0.25,  steps/LT ≈ 56.

ks_l22
    Kuramoto-Sivashinsky equation on [0, 22] with 64 Fourier collocation
    points.  State is 64-D and used directly (no tanh lift).
    λ_max ≈ 0.043,  dt=0.25,  steps/LT ≈ 93.

thomas
    Thomas cyclically symmetric attractor (b=0.208).  State is 3-D.  Lifted
    to 30 channels.
    λ_max ≈ 0.035,  dt=0.25,  steps/LT ≈ 114.

Dependencies: numpy, scipy.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Per-system metadata.  Keeps everything in one place so the experiment script
# never hard-codes a number.
# ---------------------------------------------------------------------------

SYSTEM_META = {
    "lorenz96_n8": {
        "name":        "Lorenz-96 N=8",
        "state_dim":   8,
        "n_obs":       40,     # lifted observation channels
        "k":           8,      # unbounded forecast latent  (= state dim)
        "dt":          0.01,
        "lam_max":     1.67,
        "steps_per_lt": 60,    # 1/(lam_max * dt), rounded
        "horizon":     300,    # 5 LT
        "climate":     3000,   # 50 LT
        "warm":        64,
        "unroll":      20,
        "noise":       0.05,
        "n_series":    8,
        "n_steps":     20000,
        "n_holdout":   32,
        "holdout_steps": 6000,
    },
    "rossler": {
        "name":        "Rossler",
        "state_dim":   3,
        "n_obs":       30,
        "k":           3,
        "dt":          0.25,
        "lam_max":     0.071,
        "steps_per_lt": 56,
        "horizon":     282,
        "climate":     2820,
        "warm":        64,
        "unroll":      20,
        "noise":       0.05,
        "n_series":    8,
        "n_steps":     20000,
        "n_holdout":   32,
        "holdout_steps": 5600,
    },
    "ks_l22": {
        "name":        "Kuramoto-Sivashinsky L=22",
        "state_dim":   64,
        "n_obs":       64,     # native spatial modes, no tanh lift
        "k":           5,      # KY attractor dim for L=22 is ≈ 3-5
        "dt":          0.25,
        "lam_max":     0.043,
        "steps_per_lt": 93,
        "horizon":     465,
        "climate":     4650,
        "warm":        64,
        "unroll":      20,
        "noise":       0.01,   # smaller: KS modes already have dynamical "noise"
        "n_series":    8,
        "n_steps":     20000,
        "n_holdout":   32,
        "holdout_steps": 9300,
    },
    "thomas": {
        "name":        "Thomas",
        "state_dim":   3,
        "n_obs":       30,
        "k":           3,
        "dt":          0.25,
        "lam_max":     0.035,
        "steps_per_lt": 114,
        "horizon":     571,
        "climate":     5714,
        "warm":        64,
        "unroll":      20,
        "noise":       0.05,
        "n_series":    8,
        "n_steps":     20000,
        "n_holdout":   32,
        "holdout_steps": 11400,
    },
}


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def lorenz96(N=8, F=8.0, n_steps=20000, dt=0.01, burn_in=1000, seed=0):
    """Lorenz-96 at dt=0.01.  Returns (n_steps, N) on the attractor."""
    rng = np.random.default_rng(seed)
    x0 = np.full(N, F) + rng.normal(0.0, 0.01, size=N)

    def f(t, x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

    total = n_steps + burn_in
    t_eval = np.arange(total) * dt
    sol = solve_ivp(f, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y.T[burn_in:]


def rossler(n_steps=20000, dt=0.25, a=0.2, b=0.2, c=5.7,
            burn_in=500, seed=0):
    """Rössler attractor.  Returns (n_steps, 3)."""
    rng = np.random.default_rng(seed)
    x0 = [rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 5)]

    def f(t, s):
        x, y, z = s
        return [-y - z, x + a * y, b + z * (x - c)]

    total = n_steps + burn_in
    t_eval = np.arange(total) * dt
    sol = solve_ivp(f, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y.T[burn_in:]


def kuramoto_sivashinsky(n_steps=20000, dt=0.25, N_fourier=64, L=22.0,
                         burn_in=1000, seed=0):
    """Kuramoto-Sivashinsky PDE: u_t + u*u_x + u_xx + u_xxxx = 0.

    Pseudo-spectral in space (N_fourier collocation points), integrated with
    scipy Radau (stiff) in Fourier space.  Returns (n_steps, N_fourier) in
    physical space.

    The mean mode û_0 is constant (it is a conserved quantity under periodic
    BCs when the initial mean is zero).  We initialise with zero mean.
    """
    rng = np.random.default_rng(seed)
    Nc = N_fourier // 2 + 1       # number of unique rfft coefficients

    # Wavenumbers for rfft
    k = np.fft.rfftfreq(N_fourier, d=L / N_fourier) * 2.0 * np.pi   # shape (Nc,)

    # Linear operator in Fourier space: u_xx -> -k^2, u_xxxx -> k^4
    # u_t = -u*u_x - u_xx - u_xxxx  =>  linear part = (k^2 - k^4)
    lin = k ** 2 - k ** 4   # shape (Nc,)

    def rhs(t, state):
        uhat_r = state[:Nc] + 1j * state[Nc:]
        u = np.fft.irfft(uhat_r, n=N_fourier)
        # Nonlinear: -(1/2) * d(u^2)/dx  =>  Fourier: -(ik/2) * rfft(u^2)
        nonlin = -0.5j * k * np.fft.rfft(u ** 2)
        duhat = lin * uhat_r + nonlin
        return np.concatenate([duhat.real, duhat.imag])

    # Initial condition: low-amplitude random Fourier modes
    uhat0 = np.zeros(Nc, dtype=complex)
    for m in rng.integers(1, 5, size=4):
        if m < Nc:
            uhat0[m] = rng.normal(0, 0.5) + 1j * rng.normal(0, 0.5)
    uhat0[0] = 0.0   # zero mean
    state0 = np.concatenate([uhat0.real, uhat0.imag])

    total = n_steps + burn_in
    t_eval = np.arange(total) * dt
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), state0, t_eval=t_eval,
                    method="Radau", rtol=1e-6, atol=1e-8)

    # Back to physical space
    states = np.empty((total, N_fourier))
    for i in range(total):
        uhat_i = sol.y[:Nc, i] + 1j * sol.y[Nc:, i]
        states[i] = np.fft.irfft(uhat_i, n=N_fourier)

    return states[burn_in:]


def thomas(n_steps=20000, dt=0.25, b=0.208, burn_in=1000, seed=0):
    """Thomas cyclically symmetric attractor.  Returns (n_steps, 3)."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=3).tolist()

    def f(t, s):
        x, y, z = s
        return [np.sin(y) - b * x,
                np.sin(z) - b * y,
                np.sin(x) - b * z]

    total = n_steps + burn_in
    t_eval = np.arange(total) * dt
    sol = solve_ivp(f, (0.0, t_eval[-1]), x0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y.T[burn_in:]


# ---------------------------------------------------------------------------
# Lift: frozen random tanh map (identical to LorenzLift, just generalised to
# arbitrary state_dim -> n_obs).  KS uses the identity lift (no tanh map).
# ---------------------------------------------------------------------------

def _tanh_lift_params(state_dim, n_obs, seed, hidden=32):
    """Draw frozen random weights for the tanh lift."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 1.0, size=(state_dim, hidden))
    W2 = rng.normal(0.0, 1.0, size=(hidden, n_obs))
    return W1, W2


def _apply_tanh_lift(states, W1, W2, state_mu, state_sd):
    """(T, state_dim) -> (T, n_obs) clean signal in standardised state space."""
    s = (states - state_mu) / state_sd
    return np.tanh(s @ W1) @ W2


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------

def _simulate_series(system, n_steps, dt, seed, burn_in):
    """Dispatch to the right integrator."""
    if system == "lorenz96_n8":
        return lorenz96(N=8, F=8.0, n_steps=n_steps, dt=dt,
                        burn_in=burn_in, seed=seed)
    elif system == "rossler":
        return rossler(n_steps=n_steps, dt=dt, burn_in=burn_in, seed=seed)
    elif system == "ks_l22":
        return kuramoto_sivashinsky(n_steps=n_steps, dt=dt,
                                    burn_in=burn_in, seed=seed)
    elif system == "thomas":
        return thomas(n_steps=n_steps, dt=dt, burn_in=burn_in, seed=seed)
    else:
        raise ValueError(f"Unknown system: {system!r}")


def make_dataset(system, seed=0, n_series=8, n_steps=20000, n_holdout=32,
                 holdout_steps=None, train_frac=0.7, val_frac=0.15,
                 lift_hidden=32, verbose=True):
    """Generate a multi-series dataset for `system`.

    Returns the same dict layout as LorenzLift.make_multi_series_dataset:
        train / val / test   (n_series, T_split, n_obs)
        holdout_obs          (n_holdout, holdout_steps, n_obs)
        holdout_states       (n_holdout, holdout_steps, state_dim)
        obs_mu / obs_sd      per-channel, fit on pooled TRAIN only
        noise_floor_mse      measured empirical floor

    For KS the 'states' == 'observations' (no tanh lift); for the other
    systems a frozen random tanh map lifts the state to n_obs channels.
    """
    meta = SYSTEM_META[system]
    state_dim = meta["state_dim"]
    n_obs = meta["n_obs"]
    noise = meta["noise"]
    dt = meta["dt"]
    use_lift = (system != "ks_l22")
    burn_in = max(1000, int(5 * meta["steps_per_lt"]))
    if holdout_steps is None:
        holdout_steps = meta["holdout_steps"]

    i_tr = int(n_steps * train_frac)
    i_va = int(n_steps * (train_frac + val_frac))

    # ------------------------------------------------------------------
    # 1. Simulate n_series training trajectories
    # ------------------------------------------------------------------
    if verbose:
        print(f"[{system}] simulating {n_series} training series ...")
    pool_states = []
    for s_idx in range(n_series):
        st = _simulate_series(system, n_steps, dt, seed=seed * 100 + s_idx,
                              burn_in=burn_in)
        pool_states.append(st)

    # ------------------------------------------------------------------
    # 2. Standardise states on pooled TRAIN portion
    # ------------------------------------------------------------------
    pooled_train_states = np.concatenate([st[:i_tr] for st in pool_states])
    state_mu = pooled_train_states.mean(0)
    state_sd = np.maximum(pooled_train_states.std(0), 1e-8)

    # ------------------------------------------------------------------
    # 3. Frozen lift (tanh for 3/4 systems, identity for KS)
    # ------------------------------------------------------------------
    if use_lift:
        W1, W2 = _tanh_lift_params(state_dim, n_obs, seed=seed, hidden=lift_hidden)

        def raw_clean(states):
            return _apply_tanh_lift(states, W1, W2, state_mu, state_sd)
    else:
        # KS: standardise the states and use them directly as clean signal.
        # state_mu / state_sd already computed above.
        def raw_clean(states):
            return (states - state_mu) / state_sd

    pool_raw_clean = [raw_clean(st) for st in pool_states]

    # Standardise clean signal on pooled TRAIN (may differ from state std for lift)
    pooled_train_clean = np.concatenate([c[:i_tr] for c in pool_raw_clean])
    clean_mu = pooled_train_clean.mean(0)
    clean_sd = np.maximum(pooled_train_clean.std(0), 1e-8)

    def clean_of(raw_c):
        return (raw_c - clean_mu) / clean_sd

    pool_clean = [clean_of(c) for c in pool_raw_clean]

    # ------------------------------------------------------------------
    # 4. Add observation noise
    # ------------------------------------------------------------------
    obs_rng = np.random.default_rng(seed + 1)
    pool_raw = [(c + obs_rng.normal(0.0, noise, size=c.shape), c)
                for c in pool_clean]

    # ------------------------------------------------------------------
    # 5. Final obs standardisation (pooled TRAIN only)
    # ------------------------------------------------------------------
    pooled_train_raw = np.concatenate([raw[:i_tr] for raw, _ in pool_raw])
    obs_mu = pooled_train_raw.mean(0)
    obs_sd = np.maximum(pooled_train_raw.std(0), 1e-8)

    def scale(a):
        return (a - obs_mu) / obs_sd

    train = np.stack([scale(raw[:i_tr]) for raw, _ in pool_raw])
    val   = np.stack([scale(raw[i_tr:i_va]) for raw, _ in pool_raw])
    test  = np.stack([scale(raw[i_va:]) for raw, _ in pool_raw])
    clean = np.stack(pool_clean)
    states = np.stack(pool_states)

    # Measured noise floor in final standardised scale (pooled TRAIN only)
    noise_train_raw = np.concatenate([raw[:i_tr] - c[:i_tr]
                                      for raw, c in pool_raw])
    noise_floor_mse = float(((noise_train_raw / obs_sd) ** 2).mean())

    # ------------------------------------------------------------------
    # 6. Held-out trajectories (never chronologically split)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[{system}] simulating {n_holdout} holdout trajectories ...")
    holdout_rng = np.random.default_rng(seed + 2)
    ho_states_list, ho_obs_list = [], []
    for h_idx in range(n_holdout):
        st = _simulate_series(system, holdout_steps, dt,
                              seed=seed * 10000 + h_idx + 1000,
                              burn_in=burn_in)
        c = clean_of(raw_clean(st))
        raw_obs = c + holdout_rng.normal(0.0, noise, size=c.shape)
        ho_obs_list.append(scale(raw_obs))
        ho_states_list.append(st)

    holdout_obs    = np.stack(ho_obs_list)
    holdout_states = np.stack(ho_states_list)

    return {
        "train":          train,
        "val":            val,
        "test":           test,
        "clean":          clean,
        "states":         states,
        "holdout_obs":    holdout_obs,
        "holdout_states": holdout_states,
        "obs_mu":         obs_mu,
        "obs_sd":         obs_sd,
        "noise_floor_mse": noise_floor_mse,
    }


def save_dataset(system, path=None, **kwargs):
    """make_dataset -> saved .npz.  Returns the dict."""
    if path is None:
        path = f"Data/{system}.npz"
    d = make_dataset(system, **kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path,
             train=d["train"], val=d["val"], test=d["test"],
             clean=d["clean"], states=d["states"],
             holdout_obs=d["holdout_obs"], holdout_states=d["holdout_states"],
             obs_mu=d["obs_mu"], obs_sd=d["obs_sd"],
             noise_floor_mse=d["noise_floor_mse"])
    return d


def load_dataset(path):
    """Load a saved .npz and return the same dict as make_dataset."""
    d = np.load(path)
    return {
        "train":          d["train"],
        "val":            d["val"],
        "test":           d["test"],
        "clean":          d["clean"],
        "states":         d["states"],
        "holdout_obs":    d["holdout_obs"],
        "holdout_states": d["holdout_states"],
        "obs_mu":         d["obs_mu"],
        "obs_sd":         d["obs_sd"],
        "noise_floor_mse": float(d["noise_floor_mse"]),
    }


if __name__ == "__main__":
    import sys
    systems = sys.argv[1:] if len(sys.argv) > 1 else list(SYSTEM_META)
    for sys_name in systems:
        path = f"Data/{sys_name}.npz"
        print(f"\n=== {sys_name} ===")
        d = save_dataset(sys_name, path=path)
        m = SYSTEM_META[sys_name]
        print(f"train {d['train'].shape}  val {d['val'].shape}  "
              f"test {d['test'].shape}")
        print(f"holdout {d['holdout_obs'].shape}")
        print(f"noise_floor_mse (measured): {d['noise_floor_mse']:.6f}")
        print(f"steps/LT: {m['steps_per_lt']}  "
              f"horizon: {m['horizon']} ({m['horizon']/m['steps_per_lt']:.1f} LT)  "
              f"climate: {m['climate']} ({m['climate']/m['steps_per_lt']:.1f} LT)")
        print(f"saved -> {path}")

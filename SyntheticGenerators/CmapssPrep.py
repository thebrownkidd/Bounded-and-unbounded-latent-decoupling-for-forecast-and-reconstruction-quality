"""Build a C-MAPSS dataset in the same npz layout the Lorenz pipeline expects.

The point of this file is that nothing downstream has to change. The models,
the training loops, the checkpointing and the evaluation all take
(n_series, T, n_obs) arrays and never ask where they came from.

Differences from the Lorenz generator that the caller must know about:

* There is no ground truth clean signal, so `noise_floor_mse` is an ESTIMATE
  obtained from the high frequency residual of a smoothed trajectory, not a
  known quantity. It is stored under `noise_floor_mse_est` so nobody mistakes
  it for the Lorenz floor.
* There is no Lyapunov exponent. Lead times are in CYCLES. Any metric that
  divides by a Lyapunov time is meaningless here and must not be reported.
* Trajectories are non-stationary by construction, they run to failure, so
  attractor statistics (invariant measure, Wasserstein) do not apply.

Split is at the UNIT level, which is the standard protocol for prognostics and
avoids the short-validation problem a temporal split would create on 128 cycle
units. Every scaling constant is fit on the training units only.
"""
import os
import numpy as np

CMAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Cmapss-degradation-modelling", "CMaps")

COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]


def _load(tag, split):
    path = os.path.join(CMAPS_DIR, f"{split}_{tag}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    return np.loadtxt(path)


def _live_columns(arr, tol=1e-8):
    """Channels that actually vary. FD001 has 7 constant ones that carry nothing."""
    sd = arr.std(0)
    idx = [i for i in range(2, arr.shape[1]) if sd[i] >= tol]
    return idx, [COLS[i] for i in idx]


def _tail_stack(arr, idx, length):
    """Per unit, the LAST `length` cycles, aligned at failure.

    Aligning at the end rather than the start keeps the degradation ramp, which
    is the part with any dynamics in it. Units shorter than `length` are dropped.
    """
    units, out, kept = arr[:, 0].astype(int), [], []
    for u in np.unique(units):
        g = arr[units == u]
        g = g[np.argsort(g[:, 1])]
        if len(g) < length:
            continue
        out.append(g[-length:, idx])
        kept.append(u)
    return np.stack(out), np.array(kept)


def _smooth(x, w):
    """Centred moving average along time, edges handled by reflection."""
    if w < 3:
        return x.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    p = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    k = np.ones(w) / w
    return np.stack([np.convolve(p[:, c], k, mode="valid")
                     for c in range(x.shape[1])], axis=1)


def _noise_floor_estimate(series, w=11):
    """Variance of the high frequency residual, in the standardised scale.

    A moving average is a crude trend model, so this OVER-estimates the noise
    slightly wherever the true signal has curvature at the smoothing scale. It
    is a reference point, not a bound, and is named accordingly.
    """
    res = [s - _smooth(s, w) for s in series]
    return float(np.mean(np.concatenate(res) ** 2))


def make_cmapss_dataset(tag="FD001", length=128, n_train=60, n_val=10,
                        n_holdout=30, seed=0, smooth_w=11):
    raw = _load(tag, "train")
    idx, names = _live_columns(raw)
    stack, units = _tail_stack(raw, idx, length)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stack))
    need = n_train + n_val + n_holdout
    if len(stack) < need:
        raise ValueError(f"{tag}: {len(stack)} usable units, need {need}")
    tr_i = order[:n_train]
    va_i = order[n_train:n_train + n_val]
    ho_i = order[n_train + n_val:need]

    # every constant fit on the training units only
    flat = stack[tr_i].reshape(-1, stack.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-8] = 1.0

    def scale(a):
        return (a - mu) / sd

    train = scale(stack[tr_i])
    val = scale(stack[va_i])
    holdout = scale(stack[ho_i])

    return {
        "train": train, "val": val, "test": val[:, :0],   # no temporal test split
        "holdout_obs": holdout,
        "obs_mu": mu, "obs_sd": sd,
        "noise_floor_mse_est": _noise_floor_estimate(train, smooth_w),
        "channels": np.array(names),
        "units_train": units[tr_i], "units_val": units[va_i],
        "units_holdout": units[ho_i],
        "tag": tag, "length": length, "smooth_w": smooth_w,
    }


def save_cmapss_dataset(path, **kw):
    d = make_cmapss_dataset(**kw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: v for k, v in d.items()})
    return d


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "Data", "CmapssFD001.npz")
    d = save_cmapss_dataset(out)
    print(f"wrote {out}")
    print(f"  channels ({len(d['channels'])}): {list(d['channels'])}")
    print(f"  train {d['train'].shape}  val {d['val'].shape}  holdout {d['holdout_obs'].shape}")
    print(f"  noise floor ESTIMATE (smoothing residual, w={d['smooth_w']}): "
          f"{d['noise_floor_mse_est']:.6f}")
    print(f"  train mean {d['train'].mean():+.6f} std {d['train'].std():.6f} "
          "(exact 0/1 by construction)")
    print(f"  holdout mean {d['holdout_obs'].mean():+.6f} std {d['holdout_obs'].std():.6f} "
          "(must NOT be exactly 0/1)")

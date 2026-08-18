"""The model-agnostic evaluation protocol.

Every model in the comparison exposes the same two methods --

    Reconstruct(window)          (B, L, n) -> (B, L, n)
    Rollout(window, horizon)     (B, L, n) -> (B, horizon, n)

-- and is scored here, in observation space, on identical inputs. What each
model does internally could hardly be more different (a GRU on a 3-d unbounded
latent, a shared latent with a weighted loss, RK4 on the true Lorenz field),
which is exactly why the comparison has to happen at the interface rather than
inside.
"""

import numpy as np
import torch

from .Metrics import (Mse, R2, NRMSE, PerHorizonNRMSE, VPT, DivergenceRate,
                      PerChannelR2)


def ReconMetrics(Truth, Pred, scale=None, noise_floor=None):
    Truth, Pred = np.asarray(Truth), np.asarray(Pred)
    flat_t = Truth.reshape(-1, Truth.shape[-1])
    flat_p = Pred.reshape(-1, Pred.shape[-1])
    out = {"recon_mse": Mse(flat_t, flat_p),
           "recon_nrmse": NRMSE(flat_t, flat_p, scale),
           "recon_r2": float(R2(flat_t, flat_p)),
           "recon_r2_worst": float(PerChannelR2(flat_t, flat_p).min())}
    if noise_floor is not None:
        out["recon_mse_over_floor"] = out["recon_mse"] / noise_floor
    return out


def ForecastMetrics(Truth, Pred, dt=0.01, threshold=0.4, scale=None, bounds=None):
    Truth, Pred = np.asarray(Truth), np.asarray(Pred)
    curve = PerHorizonNRMSE(Truth, Pred, scale)
    out = {"curve": curve,
           "vpt": VPT(curve, dt, threshold),
           "vpt_censored": bool(not (curve > threshold).any()),
           "fcst_nrmse_full": NRMSE(Truth, Pred, scale)}
    if bounds is not None:
        out["divergence"] = DivergenceRate(Pred, bounds[0], bounds[1])
    return out


@torch.no_grad()
def EvaluateModel(model, Warm, Future, dt=0.01, threshold=0.4, scale=None,
                  noise_floor=None, bounds=None, **rollout_kw):
    """Warm (B, L, n) and Future (B, H, n) torch tensors -> one metric dict."""
    was_training = model.training
    model.eval()
    recon = model.Reconstruct(Warm).cpu().numpy()
    pred = model.Rollout(Warm, Future.shape[1], **rollout_kw).cpu().numpy()
    if was_training:
        model.train()

    out = ReconMetrics(Warm.cpu().numpy(), recon, scale, noise_floor)
    out.update(ForecastMetrics(Future.cpu().numpy(), pred, dt, threshold,
                               scale, bounds))
    return out


# -- Trivial references -------------------------------------------------------
# Any model that cannot beat these is broken, not interesting. Cheap enough
# that there is no excuse for leaving them out of the results table.

def PersistenceRollout(Warm, horizon):
    """x_{t+h} = x_t. The bar a forecaster has to clear to have done anything."""
    last = np.asarray(Warm)[:, -1:, :]
    return np.repeat(last, horizon, axis=1)


def MeanRollout(Warm, horizon, mean):
    """Constant climatology. Its NRMSE tends to 1 by construction, which is
    what makes it the natural ceiling for the normalised error curve."""
    B = np.asarray(Warm).shape[0]
    return np.broadcast_to(np.asarray(mean), (B, horizon, len(mean))).copy()


def MakeWindows(series, warm, horizon, starts):
    """(T, n) series + start indices -> Warm (B, warm, n), Future (B, horizon, n)."""
    series = np.asarray(series)
    W = np.stack([series[s:s + warm] for s in starts])
    F = np.stack([series[s + warm:s + warm + horizon] for s in starts])
    return W, F


def MakeSeriesWindows(series_stack, length, stride=1, max_per_series=None):
    """(S, T, C) independent series -> (N, length, C) windows.

    Every window comes from a single series and never crosses a boundary --
    concatenating a pool of series first and sliding a window over the result
    would splice the tail of one trajectory onto the head of the next. Used for
    training windows, PINN's (t-1, t, t+1) triples (length=3, stride=1), and
    validation windows, whenever training pools more than one series.

    `max_per_series` caps how many windows are taken from each series (from the
    start), for validation-sized subsamples that should not scale with S.
    """
    series_stack = np.asarray(series_stack)
    S, T, C = series_stack.shape
    starts = np.arange(0, T - length + 1, stride)
    if max_per_series is not None:
        starts = starts[:max_per_series]
    windows = np.stack([series_stack[:, s:s + length] for s in starts], axis=1)
    return windows.reshape(-1, length, C)          # (S, W, length, C) -> (S*W, length, C)


def EvalWindowsFromTrajectories(trajs, warm, horizon, per_traj=1, seed=0):
    """(N, T, n) independent trajectories -> stacked warm/future windows.

    Independent trajectories matter here. The test split is ~27 Lyapunov times
    long, so overlapping windows cut from it produce rollouts that share most of
    their future and badly understate the spread of the error bars.
    """
    rng = np.random.default_rng(seed)
    trajs = np.asarray(trajs)
    N, T, n = trajs.shape
    need = warm + horizon
    if T < need:
        raise ValueError(f"trajectories are {T} steps, need {need}")

    Ws, Fs = [], []
    for i in range(N):
        starts = rng.choice(T - need + 1, size=per_traj, replace=False)
        W, F = MakeWindows(trajs[i], warm, horizon, starts)
        Ws.append(W)
        Fs.append(F)
    return np.concatenate(Ws), np.concatenate(Fs)

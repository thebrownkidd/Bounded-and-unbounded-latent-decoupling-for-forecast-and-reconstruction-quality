"""Metrics shared by every model in the forecasting comparison.

R2, FitProbe and ApplyProbe are lifted verbatim out of
Experimentation/NeuralNetworkApproximation.ipynb so the two notebooks cannot
drift apart on the definition of a score.

The forecast metrics follow the convention used throughout the chaotic-systems
literature: error is normalised by the climatological spread of the signal, and
horizons are reported in Lyapunov times rather than steps, so numbers stay
comparable across dt.
"""

import numpy as np
from scipy.stats import wasserstein_distance

LAMBDA_MAX = 0.906   # largest Lyapunov exponent of Lorenz-63 at the classic params


def StepsPerLyapunov(dt=0.01, lam=LAMBDA_MAX):
    """~110 at dt = 0.01."""
    return 1.0 / (lam * dt)


def ToLyapunov(steps, dt=0.01, lam=LAMBDA_MAX):
    return steps * dt * lam


# -- Pointwise ----------------------------------------------------------------

def R2(Y, YHat):
    """Global (not per-column) coefficient of determination."""
    Y, YHat = np.asarray(Y), np.asarray(YHat)
    return 1 - ((Y - YHat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def Mse(Y, YHat):
    return float(((np.asarray(Y) - np.asarray(YHat)) ** 2).mean())


def NRMSE(Y, YHat, scale=None):
    """RMSE normalised by the climatological std of Y."""
    Y, YHat = np.asarray(Y), np.asarray(YHat)
    s = Y.std() if scale is None else scale
    return float(np.sqrt(((Y - YHat) ** 2).mean()) / s)


def PerChannelR2(Y, YHat):
    Y, YHat = np.asarray(Y), np.asarray(YHat)
    sse = ((Y - YHat) ** 2).sum(0)
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - sse / sst


# -- Forecast -----------------------------------------------------------------

def PerHorizonNRMSE(Truth, Pred, scale=None):
    """(B, H, n) truth and prediction -> (H,) normalised error per lead time.

    Averaged over trajectories and channels, then normalised by a single
    climatological scale so the curve is comparable across models and the 0.4
    VPT threshold means the same thing for everyone.
    """
    Truth, Pred = np.asarray(Truth), np.asarray(Pred)
    s = Truth.std() if scale is None else scale
    return np.sqrt(((Truth - Pred) ** 2).mean(axis=(0, 2))) / s


def VPT(curve, dt=0.01, threshold=0.4, lam=LAMBDA_MAX):
    """Valid prediction time: first lead time where the error curve crosses
    `threshold`, in Lyapunov times.

    Returns the full horizon (and it is the caller's job to notice) when the
    curve never crosses -- that is a censored observation, not a score.
    """
    curve = np.asarray(curve)
    over = np.nonzero(curve > threshold)[0]
    steps = int(over[0]) if len(over) else len(curve)
    return ToLyapunov(steps, dt, lam)


def DivergenceRate(Pred, lo, hi):
    """Fraction of rollouts that leave the training bounding box at any point."""
    Pred = np.asarray(Pred)
    bad = (Pred < lo) | (Pred > hi)
    return float(bad.any(axis=(1, 2)).mean())


def BoundViolation(C, lo=0.0, hi=1.0):
    """Fraction of bounded-latent entries outside [lo, hi].

    For the residual-sigmoid LatentMappingDynamic this is 0 by construction,
    so any non-zero value is a bug rather than a result.
    """
    C = np.asarray(C)
    return float(((C < lo) | (C > hi)).mean())


def Saturation(C, lo=0.02, hi=0.98):
    """Fraction of bounded-latent entries pinned near the rails."""
    C = np.asarray(C)
    return float(((C < lo) | (C > hi)).mean())


# -- Long-horizon / climate ---------------------------------------------------

def AttractorStats(TrueSeq, PredSeq, nfft=None):
    """Do long rollouts still live on the right attractor?

    Short-horizon error says nothing about this: a model can track for two
    Lyapunov times and then settle onto a fixed point or a limit cycle. Both
    statistics below are computed on trajectories long past the point where
    pointwise error has saturated.

    Returns mean 1-D Wasserstein distance between per-channel marginals and a
    relative log-power-spectrum error.
    """
    T, P = np.asarray(TrueSeq), np.asarray(PredSeq)
    Tf = T.reshape(-1, T.shape[-1])
    Pf = P.reshape(-1, P.shape[-1])

    w = [wasserstein_distance(Tf[:, i], Pf[:, i]) for i in range(Tf.shape[1])]

    n = nfft or T.shape[-2]
    st = np.abs(np.fft.rfft(T, n=n, axis=-2)) ** 2
    sp = np.abs(np.fft.rfft(P, n=n, axis=-2)) ** 2
    st = st.mean(axis=0) if st.ndim == 3 else st
    sp = sp.mean(axis=0) if sp.ndim == 3 else sp
    eps = 1e-12
    spec = float(np.abs(np.log(st + eps) - np.log(sp + eps)).mean())

    return {"wasserstein": float(np.mean(w)),
            "wasserstein_max": float(np.max(w)),
            "spectrum_log_err": spec}


def ZMaxMap(z):
    """Successive maxima of the z channel -> (M-1, 2) pairs.

    The Lorenz map is the sharpest cheap test of attractor fidelity: the true
    system traces a thin tent, and a model that has learned the climate but not
    the geometry produces a visibly fattened or collapsed one.
    """
    z = np.asarray(z).ravel()
    if len(z) < 3:
        return np.empty((0, 2))
    interior = np.nonzero((z[1:-1] > z[:-2]) & (z[1:-1] > z[2:]))[0] + 1
    m = z[interior]
    return np.stack([m[:-1], m[1:]], axis=1) if len(m) > 1 else np.empty((0, 2))


# -- Linear probe onto the true state -----------------------------------------

def FitProbe(Bt, S):
    """Affine least-squares latent -> true Lorenz state. Fit on TRAIN only.

    states is never in a loss, so the latent is only identified up to an
    arbitrary rotation and scale; the probe absorbs that. It is linear, so a
    nonlinear reparameterisation of the attractor is under-credited -- read a
    low score as "not linearly decodable", not as "did not recover the state".
    """
    Design = np.concatenate([Bt, np.ones((len(Bt), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Design, S, rcond=None)
    return W


def ApplyProbe(W, Bt):
    return np.concatenate([Bt, np.ones((len(Bt), 1))], axis=1) @ W

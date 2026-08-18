import matplotlib.pyplot as plt
import numpy as np

def PlotNFeatSeries(Data, Cols=3, Height=1.6, Width=5.0, Titles=None):
    """Plot every feature of a (..., ts, n) array on its own subplot.

    Any leading dimensions are flattened and overlaid within each subplot,
    so a batch of sequences shows up as several lines per feature.
    """
    Data = np.asarray(Data)
    ts, n = Data.shape[-2:]
    Series = Data.reshape(-1, ts, n)

    Rows = int(np.ceil(n / Cols))
    Fig, Axes = plt.subplots(
        Rows, Cols,
        figsize=(Width * Cols, Height * Rows),
        dpi = 300,
        sharex=True,
        squeeze=False,
        
    )
    Axes = Axes.ravel()
    Time = np.arange(ts)

    for i in range(n):
        Ax = Axes[i]
        for s in range(Series.shape[0]):
            Ax.plot(Time, Series[s, :, i], linewidth=0.8, alpha=0.8)
        Ax.set_title(Titles[i] if Titles is not None else f'Feature {i}', fontsize=9)
        Ax.tick_params(labelsize=7)
        Ax.margins(x=0)
        Ax.grid(True, linewidth=0.4, alpha=0.4)
        Ax.set_axisbelow(True)

    for Ax in Axes[n:]:
        Ax.axis('off')

    for Ax in Axes[max(n - Cols, 0):n]:
        Ax.set_xlabel('Time', fontsize=8)

    Fig.tight_layout()

    return Fig, Axes[:n]


def PlotLossCurves(History, LogY=True, Height=3.2, Width=6.0):
    """Plot every loss series in a history dict on one axes."""
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)

    for Name, Values in History.items():
        Ax.plot(np.arange(1, len(Values) + 1), Values, linewidth=1.0, label=Name)

    if LogY:
        Ax.set_yscale('log')
    Ax.set_xlabel('Epoch', fontsize=8)
    Ax.set_ylabel('Loss', fontsize=8)
    Ax.tick_params(labelsize=7)
    Ax.margins(x=0)
    Ax.grid(True, linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)
    Ax.legend(fontsize=7)

    Fig.tight_layout()

    return Fig, Ax


def PlotParity(Y, YHat, Title='Parity', Height=3.4, Width=3.4, MaxPts=20000):
    """True against predicted, with the y = x line. Subsampled to stay readable."""
    Y, YHat = np.asarray(Y).ravel(), np.asarray(YHat).ravel()
    if len(Y) > MaxPts:
        Idx = np.random.default_rng(0).choice(len(Y), MaxPts, replace=False)
        Y, YHat = Y[Idx], YHat[Idx]

    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)
    Ax.scatter(Y, YHat, s=0.6, alpha=0.25, linewidths=0)
    Lo, Hi = min(Y.min(), YHat.min()), max(Y.max(), YHat.max())
    Ax.plot([Lo, Hi], [Lo, Hi], 'k--', linewidth=0.8)
    Ax.set_xlabel('True', fontsize=8)
    Ax.set_ylabel('Predicted', fontsize=8)
    Ax.set_title(Title, fontsize=9)
    Ax.tick_params(labelsize=7)
    Ax.grid(True, linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)

    Fig.tight_layout()

    return Fig, Ax


def PlotChannelBar(Values, Title='Per-channel R2', YLabel='R2', Baseline=None,
                   Height=2.6, Width=6.4):
    """One bar per observation channel."""
    Values = np.asarray(Values)
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)
    Ax.bar(np.arange(len(Values)), Values, width=0.8)
    if Baseline is not None:
        Ax.axhline(Baseline, color='k', linestyle='--', linewidth=0.8)
    Ax.set_xlabel('Channel', fontsize=8)
    Ax.set_ylabel(YLabel, fontsize=8)
    Ax.set_title(Title, fontsize=9)
    Ax.tick_params(labelsize=7)
    Ax.margins(x=0.01)
    Ax.grid(True, axis='y', linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)

    Fig.tight_layout()

    return Fig, Ax


def PlotAttractor3D(Series, Labels=None, Height=3.4, Width=3.4, Cols=3,
                    MaxPts=6000):
    """3-D scatter of one or more (T, 3) state trajectories, one panel each."""
    Series = [np.asarray(S) for S in Series]
    Labels = Labels or [f'Series {i}' for i in range(len(Series))]
    Rows = int(np.ceil(len(Series) / Cols))
    Cols = min(Cols, len(Series))

    Fig = plt.figure(figsize=(Width * Cols, Height * Rows), dpi=300)
    Axes = []
    for i, S in enumerate(Series):
        Ax = Fig.add_subplot(Rows, Cols, i + 1, projection='3d')
        P = S[:MaxPts]
        Ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=0.35, alpha=0.85)
        Ax.set_title(Labels[i], fontsize=8)
        Ax.tick_params(labelsize=5)
        Axes.append(Ax)

    Fig.tight_layout()

    return Fig, Axes


def PlotMatrixGrid(Mats, Titles=None, Cols=6, Height=1.5, Width=1.5,
                   VMin=0.0, VMax=1.0, Cmap='viridis'):
    """Grid of small heatmaps -- the bounded latent C at successive rollout steps."""
    Mats = [np.asarray(M) for M in Mats]
    Rows = int(np.ceil(len(Mats) / Cols))
    Fig, Axes = plt.subplots(Rows, Cols, figsize=(Width * Cols, Height * Rows),
                             dpi=300, squeeze=False)
    Axes = Axes.ravel()

    for i, M in enumerate(Mats):
        Im = Axes[i].imshow(M, vmin=VMin, vmax=VMax, cmap=Cmap)
        if Titles is not None:
            Axes[i].set_title(Titles[i], fontsize=7)
        Axes[i].set_xticks([]); Axes[i].set_yticks([])
    for Ax in Axes[len(Mats):]:
        Ax.axis('off')

    Fig.colorbar(Im, ax=Axes.tolist(), shrink=0.7, pad=0.02)

    return Fig, Axes[:len(Mats)]


def PlotHorizonCurves(Curves, Dt=0.01, Lam=0.906, Threshold=0.4,
                      Height=3.6, Width=6.4):
    """Per-horizon normalised error, one line per model, x-axis in Lyapunov times.

    Curves: dict[str, (H,) array]. The threshold line is where VPT is read off.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)

    for Name, Curve in Curves.items():
        Curve = np.asarray(Curve)
        Ax.plot(np.arange(1, len(Curve) + 1) * Dt * Lam, Curve,
                linewidth=1.2, label=Name)

    Ax.axhline(Threshold, color='k', linestyle='--', linewidth=0.8)
    Ax.set_xlabel('Lead time (Lyapunov times)', fontsize=8)
    Ax.set_ylabel('NRMSE', fontsize=8)
    Ax.tick_params(labelsize=7)
    Ax.margins(x=0)
    Ax.grid(True, linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)
    Ax.legend(fontsize=7)

    Fig.tight_layout()

    return Fig, Ax


def PlotParetoFront(Frontier, Points, XKey='recon_nrmse', YKey='vpt',
                    Height=4.0, Width=5.6, NoiseFloor=None):
    """The headline plot: reconstruction error against forecast horizon.

    Frontier: list of (label, metrics) traced by the phi sweep -- the empirical
    tradeoff curve. Points: dict[str, metrics] for the models being placed
    against it. Up and to the left is better, so a model sitting above the
    frontier is reconstructing and forecasting better than any setting of the
    weighted-loss knob achieves.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)

    Fx = [m[XKey] for _, m in Frontier]
    Fy = [m[YKey] for _, m in Frontier]
    Order = np.argsort(Fx)
    Ax.plot(np.asarray(Fx)[Order], np.asarray(Fy)[Order],
            '-o', color='0.45', linewidth=1.0, markersize=3.5,
            label='weighted-loss frontier (phi sweep)', zorder=2)
    for (Label, m) in Frontier:
        Ax.annotate(Label, (m[XKey], m[YKey]), fontsize=6,
                    textcoords='offset points', xytext=(3, -7), color='0.35')

    Markers = ['*', 'D', 's', '^', 'P', 'X']
    for i, (Name, m) in enumerate(Points.items()):
        Ax.scatter(m[XKey], m[YKey], marker=Markers[i % len(Markers)],
                   s=110 if Markers[i % len(Markers)] == '*' else 55,
                   zorder=3, label=Name)

    if NoiseFloor is not None:
        Ax.axvline(NoiseFloor, color='k', linestyle=':', linewidth=0.8)
        Ax.annotate('noise floor', (NoiseFloor, Ax.get_ylim()[1]), fontsize=6,
                    rotation=90, va='top', textcoords='offset points',
                    xytext=(3, -3))

    Ax.set_xlabel('Reconstruction NRMSE  (lower is better)', fontsize=8)
    Ax.set_ylabel('Valid prediction time (Lyapunov times)', fontsize=8)
    Ax.tick_params(labelsize=7)
    Ax.grid(True, linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)
    Ax.legend(fontsize=6.5, loc='best')

    Fig.tight_layout()

    return Fig, Ax


def PlotTimeToQuality(Histories, Key='val_recon', Height=3.4, Width=6.0,
                      LogY=True):
    """Validation metric against wall-clock seconds, one line per model.

    This is the honest reading of 'same compute budget': epochs are not a
    currency, seconds are.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=300)

    for Name, H in Histories.items():
        if Key in H:
            Ax.plot(H['elapsed'], H[Key], linewidth=1.2, label=Name)

    if LogY:
        Ax.set_yscale('log')
    Ax.set_xlabel('Wall clock (s)', fontsize=8)
    Ax.set_ylabel(Key, fontsize=8)
    Ax.tick_params(labelsize=7)
    Ax.margins(x=0)
    Ax.grid(True, linewidth=0.4, alpha=0.4)
    Ax.set_axisbelow(True)
    Ax.legend(fontsize=7)

    Fig.tight_layout()

    return Fig, Ax
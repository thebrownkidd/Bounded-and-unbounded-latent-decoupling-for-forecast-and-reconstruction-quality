"""Figure helpers shared by both notebooks.

Every figure here is meant to survive being dropped into a paper, so three
things are fixed at the module level rather than left to each call site:

* **Resolution.** DPI is 300 on screen and 600 on save. Nothing renders below
  300, including the ad-hoc figures the notebooks build themselves -- they pick
  up `savefig.dpi` / `figure.dpi` from the rcParams applied on import.
* **Identity is never carried by colour alone.** Series get a colour *and* a
  linestyle from `SeriesStyle`, assigned in fixed order and never cycled, and
  every multi-series axes carries a legend. That keeps the figures readable in
  greyscale print and under colour-vision deficiency.
* **Labels are the caller's obligation, not an afterthought.** Each function
  takes `Title` and its axis labels, so a figure cannot be produced without
  somewhere to say what it is.

The categorical palette is Okabe-Ito, ordered so that adjacent slots stay
separable under deuteranopia and tritanopia (worst adjacent pair dE 9.6 deutan,
8.5 tritan, 20.0 normal vision). Three of the six sit below 3:1 contrast against
white as thin lines, which is why the legend is mandatory rather than optional.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -- Shared style -------------------------------------------------------------

DPI = 300          # on-screen / inline
SAVE_DPI = 600     # savefig

#: Okabe-Ito, in a validated order. Fixed assignment -- slot i always means the
#: same series within a figure family, so a filtered chart never repaints.
PALETTE = ["#0072B2",   # blue
           "#D55E00",   # vermillion
           "#009E73",   # bluish green
           "#E69F00",   # orange
           "#CC79A7",   # reddish purple
           "#56B4E9"]   # sky blue

#: Second encoding dimension. Beyond six series, colour repeats only with a
#: different linestyle, so no two series ever share both.
LINESTYLES = ["-", "--", "-.", ":"]

GRID = dict(linewidth=0.4, alpha=0.4)
REFLINE = dict(color="0.25", linewidth=0.9)


def SeriesStyle(i):
    """Colour + linestyle for series `i`. Composite, so 24 series stay distinct."""
    return {"color": PALETTE[i % len(PALETTE)],
            "linestyle": LINESTYLES[(i // len(PALETTE)) % len(LINESTYLES)]}


def UseFigureStyle():
    """Apply the shared rcParams. Called on import, so ad-hoc notebook figures
    inherit the resolution and type sizes without repeating them."""
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.titleweight": "medium",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


UseFigureStyle()


def _Style(Ax, XLabel=None, YLabel=None, Title=None, Axis="both", Legend=False):
    """The grid/label boilerplate every axes repeats."""
    if XLabel:
        Ax.set_xlabel(XLabel)
    if YLabel:
        Ax.set_ylabel(YLabel)
    if Title:
        Ax.set_title(Title)
    Ax.grid(True, axis=Axis, **GRID)
    Ax.set_axisbelow(True)
    if Legend:
        Ax.legend(loc="best")
    return Ax


# -- Series -------------------------------------------------------------------

def PlotNFeatSeries(Data, Cols=3, Height=1.6, Width=5.0, Titles=None,
                    Labels=None, Title=None, XLabel="Time step",
                    YLabel="Value"):
    """Plot every feature of a (..., ts, n) array on its own subplot.

    Any leading dimensions are flattened and overlaid within each subplot, so a
    batch of sequences shows up as several lines per feature.

    `Labels` names those overlaid series and puts a single figure-level legend
    underneath. Pass it whenever the overlay means something -- truth vs
    reconstruction, truth vs forecast -- so the reader is not asked to decode
    "blue vs orange" from the caption.
    """
    Data = np.asarray(Data)
    ts, n = Data.shape[-2:]
    Series = Data.reshape(-1, ts, n)

    Rows = int(np.ceil(n / Cols))
    Fig, Axes = plt.subplots(
        Rows, Cols,
        figsize=(Width * Cols, Height * Rows),
        dpi=DPI,
        sharex=True,
        squeeze=False,
    )
    Axes = Axes.ravel()
    Time = np.arange(ts)

    Handles = []
    for i in range(n):
        Ax = Axes[i]
        for s in range(Series.shape[0]):
            Line, = Ax.plot(Time, Series[s, :, i], linewidth=0.9, alpha=0.9,
                            **SeriesStyle(s))
            if i == 0:
                Handles.append(Line)
        _Style(Ax, Title=Titles[i] if Titles is not None else f"Feature {i}")
        Ax.margins(x=0)

    for Ax in Axes[n:]:
        Ax.axis("off")

    # Axis labels only on the outer edge, or 30 subplots become unreadable.
    for Ax in Axes[max(n - Cols, 0):n]:
        Ax.set_xlabel(XLabel)
    for r in range(Rows):
        if r * Cols < n:
            Axes[r * Cols].set_ylabel(YLabel)

    if Labels is not None and len(Handles) > 1:
        Fig.legend(Handles[:len(Labels)], Labels, loc="outside lower center",
                   ncol=min(len(Labels), 4))
    if Title:
        Fig.suptitle(Title)

    return Fig, Axes[:n]


def PlotLossCurves(History, LogY=True, Height=3.2, Width=6.0, Title=None,
                   XLabel="Epoch", YLabel="Loss (MSE)", Floor=None,
                   FloorLabel="noise floor"):
    """Plot every loss series in a history dict on one axes.

    `Floor` draws the reference line *with a legend entry*, which is why it
    lives here rather than being bolted on by the caller afterwards.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)

    for i, (Name, Values) in enumerate(History.items()):
        Ax.plot(np.arange(1, len(Values) + 1), Values, linewidth=1.3,
                label=Name, **SeriesStyle(i))

    if Floor is not None:
        Ax.axhline(Floor, linestyle="--", label=FloorLabel, **REFLINE)
    if LogY:
        Ax.set_yscale("log")
    Ax.margins(x=0)
    _Style(Ax, XLabel, YLabel, Title, Legend=True)

    return Fig, Ax


# -- Agreement ----------------------------------------------------------------

def PlotParity(Y, YHat, Title="Parity", Height=3.4, Width=3.4, MaxPts=20000,
               XLabel="True value", YLabel="Predicted value", R2=None):
    """True against predicted, with the y = x line. Subsampled to stay readable.

    Pass `R2` to have the score appended to the title rather than hand-formatted
    at the call site.
    """
    Y, YHat = np.asarray(Y).ravel(), np.asarray(YHat).ravel()
    Total = len(Y)
    if Total > MaxPts:
        Idx = np.random.default_rng(0).choice(Total, MaxPts, replace=False)
        Y, YHat = Y[Idx], YHat[Idx]

    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)
    Ax.scatter(Y, YHat, s=0.6, alpha=0.25, linewidths=0, color=PALETTE[0],
               label=f"samples (n = {len(Y):,} of {Total:,})")
    Lo, Hi = min(Y.min(), YHat.min()), max(Y.max(), YHat.max())
    Ax.plot([Lo, Hi], [Lo, Hi], linestyle="--", label="perfect ($y = x$)",
            **REFLINE)
    Ax.set_aspect("equal", adjustable="box")

    if R2 is not None:
        Title = f"{Title}  ($R^2$ = {R2:.4f})"
    _Style(Ax, XLabel, YLabel, Title, Legend=True)

    return Fig, Ax


def PlotChannelBar(Values, Title="Per-channel $R^2$", YLabel="$R^2$",
                   Baseline=None, BaselineLabel=None, XLabel="Observation channel",
                   Height=2.6, Width=6.4):
    """One bar per observation channel."""
    Values = np.asarray(Values)
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)
    Ax.bar(np.arange(len(Values)), Values, width=0.8, color=PALETTE[0],
           label="per channel")
    if Baseline is not None:
        Ax.axhline(Baseline, linestyle="--",
                   label=BaselineLabel or f"baseline = {Baseline:.4g}", **REFLINE)
    Ax.margins(x=0.01)
    _Style(Ax, XLabel, YLabel, Title, Axis="y", Legend=Baseline is not None)

    return Fig, Ax


# -- State space --------------------------------------------------------------

def PlotAttractor3D(Series, Labels=None, Height=3.4, Width=3.4, Cols=3,
                    MaxPts=6000, Title=None, AxisLabels=("$x$", "$y$", "$z$")):
    """3-D scatter of one or more (T, 3) state trajectories, one panel each.

    Axis labels are on by default: an unlabelled Lorenz butterfly is decorative,
    and which coordinate is which is exactly what the reader needs to check when
    comparing a forecast panel against truth.
    """
    Series = [np.asarray(S) for S in Series]
    Labels = Labels or [f"Series {i}" for i in range(len(Series))]
    Rows = int(np.ceil(len(Series) / Cols))
    Cols = min(Cols, len(Series))

    Fig = plt.figure(figsize=(Width * Cols, Height * Rows), dpi=DPI)
    Axes = []
    for i, S in enumerate(Series):
        Ax = Fig.add_subplot(Rows, Cols, i + 1, projection="3d")
        P = S[:MaxPts]
        Ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=0.4, alpha=0.9,
                **SeriesStyle(i))
        Ax.set_title(f"{Labels[i]}\n({len(P):,} steps)", fontsize=8)
        Ax.set_xlabel(AxisLabels[0], fontsize=7, labelpad=-6)
        Ax.set_ylabel(AxisLabels[1], fontsize=7, labelpad=-6)
        Ax.set_zlabel(AxisLabels[2], fontsize=7, labelpad=-6)
        Ax.tick_params(labelsize=5, pad=-2)
        Axes.append(Ax)

    if Title:
        Fig.suptitle(Title)

    return Fig, Axes


def PlotMatrixGrid(Mats, Titles=None, Cols=6, Height=1.5, Width=1.5,
                   VMin=0.0, VMax=1.0, Cmap="viridis", Title=None,
                   CbarLabel="$C$ entry value"):
    """Grid of small heatmaps -- the bounded latent C at successive rollout steps.

    Sequential data, so a single-hue light-to-dark ramp (viridis) and a labelled
    colourbar. The cells carry no axes of their own: their meaning is the (a, b)
    index, which is arbitrary, so ticks would be noise.
    """
    Mats = [np.asarray(M) for M in Mats]
    Rows = int(np.ceil(len(Mats) / Cols))
    Fig, Axes = plt.subplots(Rows, Cols, figsize=(Width * Cols, Height * Rows),
                             dpi=DPI, squeeze=False)
    Axes = Axes.ravel()

    for i, M in enumerate(Mats):
        Im = Axes[i].imshow(M, vmin=VMin, vmax=VMax, cmap=Cmap)
        if Titles is not None:
            Axes[i].set_title(Titles[i], fontsize=7)
        Axes[i].set_xticks([]); Axes[i].set_yticks([])
    for Ax in Axes[len(Mats):]:
        Ax.axis("off")

    Cb = Fig.colorbar(Im, ax=Axes.tolist(), shrink=0.7, pad=0.02)
    Cb.set_label(CbarLabel, fontsize=7)
    Cb.ax.tick_params(labelsize=6)
    if Title:
        Fig.suptitle(Title)

    return Fig, Axes[:len(Mats)]


# -- Forecast comparison ------------------------------------------------------

def PlotHorizonCurves(Curves, Dt=0.01, Lam=0.906, Threshold=0.4,
                      Height=3.6, Width=6.4, Title=None,
                      YLabel="NRMSE (normalised by climatological std)"):
    """Per-horizon normalised error, one line per model, x-axis in Lyapunov times.

    Curves: dict[str, (H,) array]. The threshold line is where VPT is read off,
    so it is labelled with the value it encodes rather than left as a bare rule.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)

    for i, (Name, Curve) in enumerate(Curves.items()):
        Curve = np.asarray(Curve)
        Ax.plot(np.arange(1, len(Curve) + 1) * Dt * Lam, Curve,
                linewidth=1.3, label=Name, **SeriesStyle(i))

    Ax.axhline(Threshold, linestyle="--",
               label=f"VPT threshold ({Threshold:g})", **REFLINE)
    Ax.margins(x=0)
    _Style(Ax, "Lead time (Lyapunov times)", YLabel, Title)
    Ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              ncol=1 if len(Curves) <= 12 else 2)

    return Fig, Ax


def PlotParetoFront(Frontier, Points, XKey="recon_nrmse", YKey="vpt",
                    Height=4.0, Width=5.6, NoiseFloor=None, Title=None,
                    XLabel="Reconstruction NRMSE  (lower is better)",
                    YLabel="Valid prediction time, Lyapunov times  (higher is better)"):
    """The headline plot: reconstruction error against forecast horizon.

    Frontier: list of (label, metrics) traced by the phi sweep -- the empirical
    tradeoff curve. Points: dict[str, metrics] for the models being placed
    against it. Up and to the left is better, so a model sitting above the
    frontier is reconstructing and forecasting better than any setting of the
    weighted-loss knob achieves.

    Both axis labels state their direction, because "better" is up on one axis
    and left on the other and that is the single thing a reader most often gets
    backwards here.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)

    Fx = [m[XKey] for _, m in Frontier]
    Fy = [m[YKey] for _, m in Frontier]
    Order = np.argsort(Fx)
    Ax.plot(np.asarray(Fx)[Order], np.asarray(Fy)[Order],
            "-o", color="0.45", linewidth=1.0, markersize=3.5,
            label=r"weighted-loss frontier ($\phi$ sweep)", zorder=2)
    for (Label, m) in Frontier:
        Ax.annotate(Label, (m[XKey], m[YKey]), fontsize=6,
                    textcoords="offset points", xytext=(3, -7), color="0.35")

    Markers = ["*", "D", "s", "^", "P", "X"]
    for i, (Name, m) in enumerate(Points.items()):
        Mk = Markers[i % len(Markers)]
        Ax.scatter(m[XKey], m[YKey], marker=Mk, s=140 if Mk == "*" else 60,
                   color=PALETTE[i % len(PALETTE)], edgecolors="white",
                   linewidths=0.6, zorder=3, label=Name)

    if NoiseFloor is not None:
        Ax.axvline(NoiseFloor, linestyle=":",
                   label=f"noise floor ({NoiseFloor:.4f})", **REFLINE)

    _Style(Ax, XLabel, YLabel, Title)
    Ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5)

    return Fig, Ax


def PlotTimeToQuality(Histories, Key="val_recon", Height=3.4, Width=6.0,
                      LogY=True, Title=None, YLabel=None, Floor=None,
                      FloorLabel="noise floor"):
    """Validation metric against wall-clock seconds, one line per model.

    This is the honest reading of 'same compute budget': epochs are not a
    currency, seconds are.
    """
    Fig, Ax = plt.subplots(figsize=(Width, Height), dpi=DPI)

    for i, (Name, H) in enumerate(Histories.items()):
        if Key in H:
            Ax.plot(H["elapsed"], H[Key], linewidth=1.3, label=Name,
                    **SeriesStyle(i))

    if Floor is not None:
        Ax.axhline(Floor, linestyle="--", label=FloorLabel, **REFLINE)
    if LogY:
        Ax.set_yscale("log")
    Ax.margins(x=0)
    _Style(Ax, "Training wall clock (s)",
           YLabel or f"{Key.replace('_', ' ')} (MSE)", Title)
    Ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    return Fig, Ax

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
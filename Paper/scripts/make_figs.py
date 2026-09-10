import sys, json
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from Utils import PlotParetoFront, PlotHorizonCurves, PALETTE, UseFigureStyle

SP = "C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/"
J = json.load(open(SP + "paper_numbers.json"))
R, D = J["results"], J["data"]
FIG = ROOT / "Paper" / "figs"; FIG.mkdir(parents=True, exist_ok=True)
FLOOR, SCALE, N = D["noise_floor_mse"], D["scale"], D["n_obs"]


def save(fig, name):
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.96, bottom=0.20)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" ", name)


PHIS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
Frontier = [(f"{p:g}", R[f"phi={p:g}"]) for p in PHIS]
Points = {"Ours (three-phase)": R["Ours (three-phase)"],
          "AEGRU+sigmoid": R["AEGRU+sigmoid"],
          "PINN (true physics)": R["PINN (true physics)"]}
FloorNrmse = float(np.sqrt(FLOOR) / SCALE)

fig, ax = PlotParetoFront(Frontier, Points, NoiseFloor=FloorNrmse,
                          Width=2.9, Height=1.68, Title=None,
                          XLabel="Reconstruction NRMSE (lower)",
                          YLabel="VPT, Lyapunov times (higher)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.5, frameon=True)
save(fig, "paper_pareto.png")

Curves = {k: np.array(R[k]["curve"]) for k in
          ["Ours (three-phase)", "phi=0.1", "AEGRU+sigmoid",
           "PINN (true physics)", "climatology"] if "curve" in R[k]}
fig, ax = PlotHorizonCurves(Curves, Dt=D["dt"], Lam=0.906, Threshold=D["threshold"],
                            Width=2.9, Height=1.68, Title=None)
ax.set_yscale("log"); ax.set_ylabel("NRMSE (log scale)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.5)
save(fig, "paper_horizon.png")

# -- capacity figure: why wide-latent baselines dip under the floor
UseFigureStyle()
fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.3), dpi=300)
ax = axes[0]
# Restrict to the range where a linear code can already hold the 3-d signal.
# Below about d=12 rank-d PCA cannot represent the curved lift at all, so it
# sits far above the bound and would flatten the axis.
ds = [d for d in sorted(int(k) for k in J["pca_rank_over_floor"]) if 12 <= d <= 29]
meas = [J["pca_rank_over_floor"][str(d)] for d in ds]
pred = [(N - d) / N for d in ds]
ax.plot(ds, meas, marker="o", ms=3.0, lw=1.3, color=PALETTE[0],
        label="measured (rank-$d$ PCA)")
ax.plot(ds, pred, ls="--", lw=1.3, color="0.35", label=r"bound $(n-d)/n$")
ax.axhline(1.0, color=PALETTE[2], lw=1.1, ls=":", label="noise floor")
ax.axvline(16, color="0.6", lw=0.8, ls="-", alpha=0.7)
ax.text(16.3, 0.92, r"baseline latent", fontsize=5.0, color="0.4", rotation=90,
        va="top")
ax.set_xlabel("bottleneck width $d$")
ax.set_ylabel("recon MSE / noise floor")
ax.set_xlim(12, 29); ax.set_ylim(0, 1.15)
ax.legend(fontsize=5.2, loc="upper right"); ax.grid(True, lw=0.3, alpha=0.4)
ax.set_axisbelow(True)

ax = axes[1]
pt = J["passthrough"]
names = ["phi=0", "phi=0.5", "phi=1", "Ours (three-phase)"]
lbl = {"phi=0": r"$\phi$=0", "phi=0.5": r"$\phi$=0.5", "phi=1": r"$\phi$=1",
       "Ours (three-phase)": "Ours"}
vals = [pt[n]["passthrough"] for n in names]
cols = [PALETTE[1], PALETTE[1], PALETTE[1], PALETTE[0]]
ax.bar(range(len(names)), vals, color=cols, width=0.6)
ax.axhline(16 / N, ls="--", lw=1.0, color="0.35")
ax.text(0.05, 16 / N + 0.02, "rank-16 copy", fontsize=5.0, color="0.35")
ax.axhline(3 / N, ls=":", lw=1.0, color=PALETTE[2])
ax.text(0.05, 3 / N + 0.02, "pure denoiser", fontsize=5.0, color=PALETTE[2])
ax.set_xticks(range(len(names))); ax.set_xticklabels([lbl[n] for n in names], fontsize=6)
ax.set_ylabel("input passthrough"); ax.set_ylim(0, 0.62)
ax.grid(True, axis="y", lw=0.3, alpha=0.4); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "paper_capacity.png", dpi=300, bbox_inches="tight")
plt.close(fig); print("  paper_capacity.png")
print("figures written to", FIG)

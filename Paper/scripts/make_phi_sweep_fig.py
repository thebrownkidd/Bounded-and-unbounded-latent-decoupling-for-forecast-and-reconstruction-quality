"""
Polished φ-sweep figure for FMTS 2026 paper.
Shows the coupled Pareto frontier + our decoupled point + utopia diamond.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "phi_sweep.json"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

systems = list(data.keys())
panel_labels = ["(a)", "(b)", "(c)"]

for idx, sname in enumerate(systems):
    ax = axes[idx]
    res = data[sname]
    sweep = res["phi_sweep"]
    ours = res["ours"]

    phis = sorted(sweep.keys(), key=float)
    rs = [sweep[p]["rmse_r"] for p in phis]
    fs = [sweep[p]["rmse_f"] for p in phis]

    # Filter out extreme outliers (φ=1 often collapses) for plot range
    # but still show them
    rs_clean = [r for r, f in zip(rs, fs) if r < 2 * max(rs[0], rs[-1])]
    fs_clean = [f for r, f in zip(rs, fs) if r < 2 * max(rs[0], rs[-1])]

    # Utopia point: best recon × best forecast from sweep (excluding diverged)
    valid = [(r, f, p) for r, f, p in zip(rs, fs, phis)
             if r < 1.0 and f < 5.0]  # exclude collapsed runs
    if valid:
        best_r = min(v[0] for v in valid)
        best_f = min(v[1] for v in valid)
    else:
        best_r = min(rs)
        best_f = min(fs)

    # Use log scale on x if range is large
    use_log_x = (max(rs) / min(rs)) > 20

    if use_log_x:
        ax.set_xscale("log")

    # Plot sweep points (connected where sensible)
    # Sort by recon for connecting line
    valid_pts = [(r, f, p) for r, f, p in zip(rs, fs, phis)
                 if r < 2.0 and f < 5.0]
    valid_pts.sort(key=lambda x: x[0])
    if len(valid_pts) >= 2:
        vr = [v[0] for v in valid_pts]
        vf = [v[1] for v in valid_pts]
        ax.plot(vr, vf, '-', color="#bbb", lw=1.0, zorder=2)

    # Plot each φ point
    for i, phi in enumerate(phis):
        r, f = rs[i], fs[i]
        if r > 2.0 or f > 5.0:
            continue  # skip collapsed points from the visual
        ax.plot(r, f, 'o', color="#999", ms=7, zorder=3,
                markeredgecolor="#666", markeredgewidth=0.8)
        # Label with φ value
        phi_f = float(phi)
        label = f"$\\phi$={phi_f:.2f}" if phi_f not in (0, 1) else f"$\\phi$={phi_f:.0f}"
        # Offset labels to avoid overlap
        off_x, off_y = 6, 6
        if phi_f >= 0.75:
            off_y = -14
        if phi_f == 0:
            off_x = 6; off_y = -14
        ax.annotate(label, (r, f),
                    textcoords="offset points", xytext=(off_x, off_y),
                    fontsize=6, color="#666")

    # Utopia diamond
    ax.plot(best_r, best_f, 'D', color="#4477aa", ms=9, zorder=5,
            markeredgecolor="#224466", markeredgewidth=1.0)
    ax.annotate("utopia", (best_r, best_f),
                textcoords="offset points", xytext=(-8, 10),
                fontsize=7, color="#4477aa", fontweight="bold")

    # Our method (red star)
    ax.plot(ours["rmse_r"], ours["rmse_f"], '*', color="#d62728",
            ms=18, zorder=6, markeredgecolor="#8b0000", markeredgewidth=0.8)
    # Position label based on where our point is relative to frontier
    ours_off = (-12, 12)
    if ours["rmse_r"] < best_r * 0.5:
        ours_off = (8, -16)
    ax.annotate(f"Ours ($j$={ours['j']}, $k$={ours['k']})",
                (ours["rmse_r"], ours["rmse_f"]),
                textcoords="offset points", xytext=ours_off, fontsize=7.5,
                color="#d62728", fontweight="bold")

    # Shade the region between frontier and our point if inside
    # (visual indicator of improvement)

    ax.set_xlabel("Reconstruction RMSE", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Forecast RMSE", fontsize=9)
    ax.set_title(f"{panel_labels[idx]} {sname}", fontsize=10,
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7.5)

    # Legend on first panel
    if idx == 0:
        from matplotlib.lines import Line2D
        leg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
                   markeredgecolor='#666', ms=7,
                   label='Koopman AE ($\\phi$-sweep)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='#4477aa',
                   markeredgecolor='#224466', ms=8, label='Utopia (best of each)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#d62728',
                   markeredgecolor='#8b0000', ms=14, label='Ours (decoupled)'),
        ]
        ax.legend(handles=leg, fontsize=6.5, loc='upper right',
                  framealpha=0.85, edgecolor='#ddd')

fig.tight_layout(w_pad=2.5)
out = ROOT / "phi_sweep.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"→ {out}")
fig.savefig(ROOT / "phi_sweep.pdf", bbox_inches="tight")
print(f"→ {ROOT / 'phi_sweep.pdf'}")
print("Done.")

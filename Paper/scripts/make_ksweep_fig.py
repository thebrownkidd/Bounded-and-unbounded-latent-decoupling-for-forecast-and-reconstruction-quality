"""
Polished k-sweep figure for FMTS 2026 paper.
Shows the tradeoff frontier + matched-k improvement arrows.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "ksweep_frontier.json"))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

systems = ["Coupled Harmonic", "Brusselator", "Linear 5D"]
matched_k = {"Coupled Harmonic": 4, "Brusselator": 3, "Linear 5D": 4}
panel_labels = ["(a)", "(b)", "(c)"]

# Per-panel label offsets to avoid overlap
label_offsets = {
    "Coupled Harmonic": {
        2: (6, -12), 3: (6, 6), 4: (8, 6), 5: (-28, 8), 6: (8, 4), 8: (-8, -14),
        "ours": (-12, 14)
    },
    "Brusselator": {
        2: (6, 6), 3: (6, 6), 4: (6, -12), 5: (6, 6), 6: (6, -12), 8: (6, 6),
        "ours": (-48, 12)
    },
    "Linear 5D": {
        2: (-12, -14), 3: (6, 6), 4: (8, 8), 5: (-30, -14), 6: (6, 6), 8: (6, -12),
        "ours": (-12, 14)
    }
}

for idx, sname in enumerate(systems):
    ax = axes[idx]
    res = data[sname]
    sweep = res["ae_sweep"]
    ours = res["ours"]
    mk = matched_k[sname]
    offsets = label_offsets[sname]

    ks = sorted(sweep.keys(), key=int)
    rs = [sweep[k]["rmse_r"] for k in ks]
    fs = [sweep[k]["rmse_f"] for k in ks]

    # Use log scale for x-axis if range is large
    if max(rs) / min(rs) > 20:
        ax.set_xscale("log")

    # Plot baseline sweep line (light grey)
    ax.plot(rs, fs, '-', color="#bbb", lw=1.0, zorder=2)

    # Plot each k point
    for i, k in enumerate(ks):
        ki = int(k)
        is_matched = (ki == mk)
        if is_matched:
            ax.plot(rs[i], fs[i], 'o', color="#555", ms=9, zorder=4,
                    markeredgecolor="#222", markeredgewidth=1.5)
        else:
            ax.plot(rs[i], fs[i], 'o', color="#ccc", ms=6, zorder=3,
                    markeredgecolor="#999", markeredgewidth=0.7)

        off = offsets.get(ki, (6, 6))
        ax.annotate(f"$k$={ki}", (rs[i], fs[i]),
                    textcoords="offset points", xytext=off, fontsize=6.5,
                    color="#333" if is_matched else "#888",
                    fontweight="bold" if is_matched else "normal")

    # Plot ours (red star)
    ax.plot(ours["rmse_r"], ours["rmse_f"], '*', color="#d62728",
            ms=18, zorder=6, markeredgecolor="#8b0000", markeredgewidth=0.8)

    ours_off = offsets.get("ours", (-12, 14))
    ax.annotate(f"Ours ($j$={ours['j']}, $k$={ours['k']})",
                (ours["rmse_r"], ours["rmse_f"]),
                textcoords="offset points", xytext=ours_off, fontsize=7.5,
                color="#d62728", fontweight="bold")

    # Arrow from matched-k baseline to ours
    base_r = sweep[str(mk)]["rmse_r"]
    base_f = sweep[str(mk)]["rmse_f"]
    arrow = FancyArrowPatch(
        (base_r, base_f), (ours["rmse_r"], ours["rmse_f"]),
        arrowstyle='->', mutation_scale=14, lw=2.0,
        color='#d62728', alpha=0.65, zorder=5,
        connectionstyle='arc3,rad=-0.15')
    ax.add_patch(arrow)

    ax.set_xlabel("Reconstruction RMSE", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Forecast RMSE", fontsize=9)
    ax.set_title(f"{panel_labels[idx]} {sname}", fontsize=10, fontweight="bold",
                 loc="left")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7.5)

    # Grey/red legend
    if idx == 0:
        from matplotlib.lines import Line2D
        leg = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
                   markeredgecolor='#222', markeredgewidth=1.5, ms=8,
                   label=f'AE+DMD (matched $k$)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#ccc',
                   markeredgecolor='#999', ms=6, label='AE+DMD (other $k$)'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#d62728',
                   markeredgecolor='#8b0000', ms=14, label='Ours (decoupled)'),
        ]
        ax.legend(handles=leg, fontsize=6.5, loc='upper right',
                  framealpha=0.85, edgecolor='#ddd')

fig.tight_layout(w_pad=2.5)
out = ROOT / "ksweep_frontier.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"→ {out}")
fig.savefig(ROOT / "ksweep_frontier.pdf", bbox_inches="tight")
print(f"→ {ROOT / 'ksweep_frontier.pdf'}")
print("Done.")

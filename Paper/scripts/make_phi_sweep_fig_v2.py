"""
Polished φ-sweep figure for FMTS 2026 paper (multi-seed version).
Shows the coupled Pareto frontier (mean±std) + our decoupled point + utopia.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
phi_data = json.load(open(ROOT / "multiseed_phi.json"))
table_data = json.load(open(ROOT / "multiseed_table.json"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})

systems = ["Coupled Harmonic", "Brusselator", "Linear 5D"]
panel_labels = ["(a)", "(b)", "(c)"]

for idx, sname in enumerate(systems):
    ax = axes[idx]
    sweep = phi_data[sname]["phi_sweep"]
    ours = table_data[sname]["Resid+GRU"]

    phis = sorted(sweep.keys(), key=float)
    rs = [sweep[p]["rmse_r_mean"] for p in phis]
    fs = [sweep[p]["rmse_f_mean"] for p in phis]
    rs_std = [sweep[p]["rmse_r_std"] for p in phis]
    fs_std = [sweep[p]["rmse_f_std"] for p in phis]

    # Utopia: best recon mean × best forecast mean from sweep (excl. collapsed)
    valid = [(r, f, p) for r, f, p in zip(rs, fs, phis)
             if r < 1.0 and f < 5.0]
    if valid:
        best_r = min(v[0] for v in valid)
        best_f = min(v[1] for v in valid)
    else:
        best_r, best_f = min(rs), min(fs)

    use_log_x = (max(rs) / max(min(rs), 1e-6)) > 20
    if use_log_x:
        ax.set_xscale("log")

    # Connect valid points
    valid_pts = [(r, f, p) for r, f, p in zip(rs, fs, phis) if r < 2.0 and f < 5.0]
    valid_pts.sort(key=lambda x: x[0])
    if len(valid_pts) >= 2:
        vr = [v[0] for v in valid_pts]
        vf = [v[1] for v in valid_pts]
        ax.plot(vr, vf, '-', color="#bbb", lw=1.0, zorder=2)

    # Plot each φ point with error bars
    for i, phi in enumerate(phis):
        r, f = rs[i], fs[i]
        if r > 2.0 or f > 5.0:
            continue
        ax.errorbar(r, f, xerr=rs_std[i], yerr=fs_std[i],
                    fmt='o', color="#999", ms=7, zorder=3,
                    markeredgecolor="#666", markeredgewidth=0.8,
                    ecolor="#ccc", elinewidth=1.0, capsize=2)
        phi_f = float(phi)
        label = f"$\\phi$={phi_f:.2f}" if phi_f not in (0, 1) else f"$\\phi$={phi_f:.0f}"
        off_x, off_y = 6, 6
        if phi_f >= 0.75: off_y = -14
        if phi_f == 0: off_x = 6; off_y = -14
        ax.annotate(label, (r, f), textcoords="offset points",
                    xytext=(off_x, off_y), fontsize=6, color="#666")

    # Utopia diamond
    ax.plot(best_r, best_f, 'D', color="#4477aa", ms=9, zorder=5,
            markeredgecolor="#224466", markeredgewidth=1.0)
    ax.annotate("utopia", (best_r, best_f), textcoords="offset points",
                xytext=(-8, 10), fontsize=7, color="#4477aa", fontweight="bold")

    # Our method (red star with error bars)
    our_r, our_f = ours["rmse_r_mean"], ours["rmse_f_mean"]
    our_rs, our_fs = ours["rmse_r_std"], ours["rmse_f_std"]
    ax.errorbar(our_r, our_f, xerr=our_rs, yerr=our_fs,
                fmt='*', color="#d62728", ms=18, zorder=6,
                markeredgecolor="#8b0000", markeredgewidth=0.8,
                ecolor="#d62728", elinewidth=1.5, capsize=3)
    ours_off = (-12, 12)
    if our_r < best_r * 0.5: ours_off = (8, -16)
    ax.annotate("Ours",
                (our_r, our_f), textcoords="offset points",
                xytext=ours_off, fontsize=7.5,
                color="#d62728", fontweight="bold")

    ax.set_xlabel("Reconstruction RMSE", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Forecast RMSE", fontsize=9)
    ax.set_title(f"{panel_labels[idx]} {sname}", fontsize=10,
                 fontweight="bold", loc="left")
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=7.5)

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

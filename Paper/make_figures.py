"""Generate publication-quality figures for ICLR 2027 paper."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Global style ──────────────────────────────────────────────────
PALETTE = {
    'coupled':    '#4361ee',
    'decoupled':  '#e63946',
    'conflict':   '#d00000',
    'ok':         '#38b000',
    'bg':         '#fafafa',
    'grid':       '#e0e0e0',
    'text':       '#2b2d42',
    'accent':     '#ff9f1c',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#555',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'text.color': PALETTE['text'],
    'axes.labelcolor': PALETTE['text'],
})

SYSTEM_LABELS = {
    'Coupled Harmonic': 'Coupled\nHarmonic',
    'Linear 5D': 'Linear 5D',
    'Brusselator': 'Brusselator',
    'Duffing': 'Duffing',
    'Lorenz-96': 'Lorenz-96',
}


# ═══════════════════════════════════════════════════════════════════
#  1. Gradient Conflict — heatmap + distribution
# ═══════════════════════════════════════════════════════════════════

def make_gradient_conflict_figure():
    with open(ROOT / 'gradient_conflict.json', encoding='utf-8') as f:
        data = json.load(f)

    systems = list(data.keys())
    phis = sorted(data[systems[0]].keys(), key=float)

    fig, axes = plt.subplots(1, len(systems), figsize=(13, 3.2), sharey=True)
    fig.patch.set_facecolor('white')

    for idx, (ax, sname) in enumerate(zip(axes, systems)):
        means = []
        stds = []
        frac_negs = []
        all_cosines_by_phi = []

        for phi in phis:
            all_cos = []
            for seed in data[sname][phi]:
                all_cos.extend([t['cosine'] for t in data[sname][phi][seed]])
            means.append(np.mean(all_cos))
            stds.append(np.std(all_cos))
            frac_negs.append(np.mean([c < 0 for c in all_cos]))
            all_cosines_by_phi.append(all_cos)

        x = np.arange(len(phis))
        means = np.array(means)

        # Gradient fill bars — red for negative, green tint for positive
        for i, (m, s, fn) in enumerate(zip(means, stds, frac_negs)):
            color = PALETTE['conflict'] if m < 0 else PALETTE['ok']
            alpha = 0.3 + 0.5 * min(abs(m) / 0.25, 1.0)  # intensity by magnitude
            bar = ax.bar(x[i], m, width=0.7, color=color, alpha=alpha,
                         edgecolor=color, linewidth=1.2, zorder=3)
            # Error cap
            ax.errorbar(x[i], m, yerr=s, fmt='none', ecolor='#333',
                        capsize=4, capthick=1, linewidth=1, zorder=4)
            # Fraction negative label
            ax.text(x[i], -0.38, f'{fn:.0%}',
                    ha='center', va='top', fontsize=7.5, fontweight='bold',
                    color=PALETTE['conflict'],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        ax.axhline(0, color='#333', linewidth=1.0, linestyle='-', zorder=2)
        ax.axhspan(-0.5, 0, color=PALETTE['conflict'], alpha=0.04, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{float(p):.2f}' for p in phis], rotation=45, ha='right')
        ax.set_xlabel(r'Loss weight $\varphi$', fontsize=10)
        ax.set_title(SYSTEM_LABELS.get(sname, sname), pad=8)
        ax.set_ylim(-0.42, 0.12)
        ax.grid(axis='y', color=PALETTE['grid'], zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(
        r'$\cos\!\left(\nabla_{\theta} \mathcal{L}_{\mathrm{rec}},\; '
        r'\nabla_{\theta} \mathcal{L}_{\mathrm{fcst}}\right)$',
        fontsize=11)

    # Legend-like annotation
    fig.text(0.5, -0.06,
             'Red region = gradient conflict zone  |  '
             'Percentages = fraction of batches with conflicting gradients',
             ha='center', fontsize=8.5, style='italic', color='#555')

    fig.tight_layout(w_pad=1.5)
    fig.savefig(ROOT / 'gradient_conflict.pdf', facecolor='white')
    fig.savefig(ROOT / 'gradient_conflict.png', facecolor='white')
    print('  -> gradient_conflict.pdf/png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  2. Noise sweep — dual-panel (recon + forecast) per system
# ═══════════════════════════════════════════════════════════════════

def make_noise_sweep_figure():
    with open(ROOT / 'noise_spectral.json', encoding='utf-8') as f:
        data = json.load(f)

    systems = list(data.keys())
    sigmas = sorted([float(s) for s in data[systems[0]].keys()])
    sigma_labels = ['0', '0.01', '0.05', '0.1', '0.2']
    x_pos = np.arange(len(sigmas))

    fig, axes = plt.subplots(2, len(systems), figsize=(13, 5.5), sharey='row')
    fig.patch.set_facecolor('white')

    for col, sname in enumerate(systems):
        for row, (metric, label) in enumerate([
            ('rmse_f', 'Forecast RMSE'),
            ('rmse_r', 'Recon RMSE'),
        ]):
            ax = axes[row, col]
            ae_vals = np.array([data[sname][str(s)]['Coupled'][f'{metric}_mean'] for s in sigmas])
            ae_std = np.array([data[sname][str(s)]['Coupled'][f'{metric}_std'] for s in sigmas])
            gru_vals = np.array([data[sname][str(s)]['Decoupled'][f'{metric}_mean'] for s in sigmas])
            gru_std = np.array([data[sname][str(s)]['Decoupled'][f'{metric}_std'] for s in sigmas])

            ax.fill_between(x_pos, ae_vals - ae_std, ae_vals + ae_std,
                            color=PALETTE['coupled'], alpha=0.12, zorder=2)
            ax.fill_between(x_pos, gru_vals - gru_std, gru_vals + gru_std,
                            color=PALETTE['decoupled'], alpha=0.12, zorder=2)

            ax.plot(x_pos, ae_vals, 'o-', color=PALETTE['coupled'],
                    label='Coupled ($k$-dim)', linewidth=2, markersize=5,
                    markeredgecolor='white', markeredgewidth=1, zorder=4)
            ax.plot(x_pos, gru_vals, 's-', color=PALETTE['decoupled'],
                    label='Decoupled ($j$-dim)', linewidth=2, markersize=5,
                    markeredgecolor='white', markeredgewidth=1, zorder=4)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(sigma_labels)
            ax.grid(True, color=PALETTE['grid'], zorder=0)
            ax.set_axisbelow(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            if row == 1:
                ax.set_xlabel(r'Noise $\sigma$', fontsize=10)
            if row == 0:
                ax.set_title(SYSTEM_LABELS.get(sname, sname), pad=8)

    axes[0, 0].set_ylabel('Forecast RMSE', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Recon RMSE', fontsize=10, fontweight='bold')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.04), frameon=True,
               fancybox=True, shadow=False, edgecolor='#ccc',
               fontsize=9.5)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    fig.subplots_adjust(top=0.88)
    fig.savefig(ROOT / 'noise_sweep.pdf', facecolor='white')
    fig.savefig(ROOT / 'noise_sweep.png', facecolor='white')
    print('  -> noise_sweep.pdf/png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  3. Noise sweep — recon only (for appendix)
# ═══════════════════════════════════════════════════════════════════

def make_noise_recon_figure():
    with open(ROOT / 'noise_spectral.json', encoding='utf-8') as f:
        data = json.load(f)

    systems = list(data.keys())
    sigmas = sorted([float(s) for s in data[systems[0]].keys()])
    sigma_labels = ['0', '0.01', '0.05', '0.1', '0.2']
    x_pos = np.arange(len(sigmas))

    fig, axes = plt.subplots(1, len(systems), figsize=(13, 2.8), sharey=False)
    fig.patch.set_facecolor('white')

    for ax, sname in zip(axes, systems):
        ae_r = np.array([data[sname][str(s)]['Coupled']['rmse_r_mean'] for s in sigmas])
        ae_r_std = np.array([data[sname][str(s)]['Coupled']['rmse_r_std'] for s in sigmas])
        gru_r = np.array([data[sname][str(s)]['Decoupled']['rmse_r_mean'] for s in sigmas])
        gru_r_std = np.array([data[sname][str(s)]['Decoupled']['rmse_r_std'] for s in sigmas])

        ax.fill_between(x_pos, ae_r - ae_r_std, ae_r + ae_r_std,
                        color=PALETTE['coupled'], alpha=0.12)
        ax.fill_between(x_pos, gru_r - gru_r_std, gru_r + gru_r_std,
                        color=PALETTE['decoupled'], alpha=0.12)

        ax.plot(x_pos, ae_r, 'o-', color=PALETTE['coupled'], label='Coupled ($k$-dim)',
                linewidth=2, markersize=5, markeredgecolor='white', markeredgewidth=1)
        ax.plot(x_pos, gru_r, 's-', color=PALETTE['decoupled'], label='Decoupled ($j$-dim)',
                linewidth=2, markersize=5, markeredgecolor='white', markeredgewidth=1)

        ax.set_xlabel(r'Noise $\sigma$')
        ax.set_title(SYSTEM_LABELS.get(sname, sname), pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sigma_labels)
        ax.grid(True, color=PALETTE['grid'])
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel('Recon RMSE', fontweight='bold')
    axes[-1].legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')

    fig.tight_layout()
    fig.savefig(ROOT / 'noise_recon.pdf', facecolor='white')
    fig.savefig(ROOT / 'noise_recon.png', facecolor='white')
    print('  -> noise_recon.pdf/png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  4. C-MAPSS bar chart
# ═══════════════════════════════════════════════════════════════════

def make_cmapss_figure():
    fpath = ROOT / 'cmapss_resid.json'
    if not fpath.exists():
        print('  -- cmapss_resid.json not found, skipping')
        return

    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)

    metrics = ['rmse_r', 'rmse_f8', 'rmse_f16', 'rmse_f32', 'rmse_f64']
    labels = ['Recon', 'Fcst@8', 'Fcst@16', 'Fcst@32', 'Fcst@64']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(metrics))
    w = 0.35

    ae_means = [data['AE+DMD'][f'{m}_mean'] for m in metrics]
    ae_stds = [data['AE+DMD'][f'{m}_std'] for m in metrics]
    gru_means = [data['Resid+GRU'][f'{m}_mean'] for m in metrics]
    gru_stds = [data['Resid+GRU'][f'{m}_std'] for m in metrics]

    bars1 = ax.bar(x - w/2, ae_means, w, yerr=ae_stds, label='Coupled (AE+DMD)',
                   color=PALETTE['coupled'], alpha=0.8, capsize=4,
                   edgecolor='white', linewidth=1, zorder=3)
    bars2 = ax.bar(x + w/2, gru_means, w, yerr=gru_stds, label='Decoupled (Resid+GRU)',
                   color=PALETTE['decoupled'], alpha=0.8, capsize=4,
                   edgecolor='white', linewidth=1, zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h < 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7,
                        fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('C-MAPSS FD001 — Turbofan Degradation', pad=10)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc')
    ax.grid(axis='y', color=PALETTE['grid'], zorder=0)
    ax.set_axisbelow(True)

    # Annotation for the reconstruction win
    ax.annotate('24× lower', xy=(0 + w/2, gru_means[0] + 0.04),
                fontsize=8, fontweight='bold', color=PALETTE['decoupled'],
                ha='center',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    fig.tight_layout()
    fig.savefig(ROOT / 'cmapss_comparison.pdf', facecolor='white')
    fig.savefig(ROOT / 'cmapss_comparison.png', facecolor='white')
    print('  -> cmapss_comparison.pdf/png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  5. Phi-sweep Pareto frontier
# ═══════════════════════════════════════════════════════════════════

def make_phi_sweep_figure():
    fpath = ROOT / 'phi_sweep.json'
    if not fpath.exists():
        print('  -- phi_sweep.json not found, skipping')
        return

    with open(fpath, encoding='utf-8') as f:
        phi_data = json.load(f)

    spec_path = ROOT / 'spectral_fair.json'
    if spec_path.exists():
        with open(spec_path, encoding='utf-8') as f:
            spec_data = json.load(f)
    else:
        spec_data = {}

    systems = list(phi_data.keys())
    fig, axes = plt.subplots(1, len(systems), figsize=(13, 3.5))
    if len(systems) == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    for ax, sname in zip(axes, systems):
        ps = phi_data[sname]['phi_sweep']
        # Exclude phi=1.0 — pure forecast loss destroys the encoder
        phis = [p for p in sorted(ps.keys(), key=float) if float(p) < 1.0]
        rs = np.array([ps[p]['rmse_r_mean'] for p in phis])
        fs = np.array([ps[p]['rmse_f_mean'] for p in phis])
        rs_std = np.array([ps[p]['rmse_r_std'] for p in phis])
        fs_std = np.array([ps[p]['rmse_f_std'] for p in phis])

        ax.errorbar(rs, fs, xerr=rs_std, yerr=fs_std,
                    fmt='o', color='#888', markersize=6,
                    markeredgecolor='white', markeredgewidth=1, zorder=3,
                    capsize=3, capthick=1, ecolor='#bbb',
                    label=r'Coupled ($\varphi$-sweep)')

        for phi, r, f in zip(phis, rs, fs):
            pv = float(phi)
            if pv in (0.0, 0.5, 0.9):
                ax.annotate(f'$\\varphi$={pv}', (r, f), fontsize=6,
                            textcoords='offset points', xytext=(6, 4),
                            color='#666')

        if sname in spec_data:
            our_r = spec_data[sname]['decoupled_spectral']['rmse_r_mean']
            our_f = spec_data[sname]['decoupled_spectral']['rmse_f_mean']
            ax.plot(our_r, our_f, '*', color=PALETTE['decoupled'], markersize=18,
                    markeredgecolor='white', markeredgewidth=1, zorder=5,
                    label='Decoupled (ours)')

        ax.set_xlabel('Recon RMSE', fontsize=10)
        ax.set_title(SYSTEM_LABELS.get(sname, sname), pad=8)
        ax.grid(True, color=PALETTE['grid'])
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel('Forecast RMSE', fontsize=10, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.08), frameon=True,
               fancybox=True, shadow=False, edgecolor='#ccc', fontsize=9.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    fig.savefig(ROOT / 'phi_sweep.pdf', facecolor='white')
    fig.savefig(ROOT / 'phi_sweep.png', facecolor='white')
    print('  -> phi_sweep.pdf/png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating figures...')
    make_gradient_conflict_figure()
    make_noise_sweep_figure()
    make_noise_recon_figure()
    make_cmapss_figure()
    make_phi_sweep_figure()
    print('Done.')

"""
Diagnostic plots for the residual three-phase model on Lorenz-63.
  1. Training loss curves (phase 1 + phase 2)
  2. Carrier space: true C vs decoded C (select dims)
  3. Residual space: true ΔC vs m(f(ΔC))
  4. Latent b trajectory + DMD eigenvalue spectrum
  5. Forecast: truth vs predicted (x,y,z)
  6. Forecast carrier accumulation vs true carrier
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
SEED = 0; EPOCHS = 400; LR = 1e-3; BS = 512

# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════

class ResidualModel(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j = j; self.k = k
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        self.m = nn.Sequential(nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

# ═══════════════════════════════════════════════════════════════════
#  Data: Lorenz-63
# ═══════════════════════════════════════════════════════════════════

dt = 0.02
def ode(t, s):
    x, y, z = s
    return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]
sol = solve_ivp(ode, [0, 120], [1,1,1],
                t_eval=np.arange(0, 120, dt),
                method="RK45", rtol=1e-10, atol=1e-10)
raw = sol.y.T[1000:]
DELAYS = 5
obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
N_TRAIN, N_TEST = 3000, 500
train_obs = obs[:N_TRAIN]; test_obs = obs[N_TRAIN:N_TRAIN+N_TEST]
gt_test = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
train_n = (train_obs - mu) / sig
test_n = (test_obs - mu) / sig
n_obs = obs.shape[1]   # 15
j = 16; k = 4; h = 64
LT = 55; FCST_LEN = 400

Xt_all = torch.tensor(train_n, dtype=torch.float32)

# ═══════════════════════════════════════════════════════════════════
#  Train with loss logging
# ═══════════════════════════════════════════════════════════════════
SeedAll(SEED)
model = ResidualModel(n_obs, j, k, h=h)

# Phase 1: teacher AE
print("Phase 1: teacher AE")
for p in model.parameters(): p.requires_grad_(False)
for mod in [model.enc, model.dec]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

p1_losses = []
for ep in range(1, EPOCHS+1):
    model.train(); idx = torch.randperm(len(Xt_all)); ep_loss = []
    for i in range(0, len(Xt_all), BS):
        x = Xt_all[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.recon(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss.append(loss.item())
    p1_losses.append(np.mean(ep_loss))
    if ep % 100 == 0: print(f"  ep {ep}  loss={p1_losses[-1]:.6f}")

# Extract carriers
model.eval()
with torch.no_grad():
    C_all = model.carrier(Xt_all)
C_cur, C_nxt = C_all[:-1], C_all[1:]
dC = C_nxt - C_cur

# Phase 2: residual AE
print("\nPhase 2: residual AE (f + m)")
for p in model.parameters(): p.requires_grad_(False)
for mod in [model.f, model.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

p2_losses = []
for ep in range(1, EPOCHS+1):
    model.train(); idx = torch.randperm(len(dC)); ep_loss = []
    for i in range(0, len(dC), BS):
        dc = dC[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss.append(loss.item())
    p2_losses.append(np.mean(ep_loss))
    if ep % 100 == 0: print(f"  ep {ep}  loss={p2_losses[-1]:.6f}")

# Fit DMD
model.eval()
with torch.no_grad():
    B_all = model.f(dC).numpy()     # (N-1, k)
A_dmd = fit_dmd(B_all)
eigvals = np.linalg.eigvals(A_dmd)

# ═══════════════════════════════════════════════════════════════════
#  Collect all diagnostic data
# ═══════════════════════════════════════════════════════════════════

with torch.no_grad():
    # Reconstruction on test set
    recon_test = model.recon(torch.tensor(test_n, dtype=torch.float32)).numpy()
    recon_phys = (recon_test * sig + mu)[:, :3]

    # Carrier on test
    C_test = model.carrier(torch.tensor(test_n, dtype=torch.float32)).numpy()

    # Residual AE quality on train
    dC_hat = model.m(model.f(dC)).numpy()
    dC_np = dC.numpy()

    # Latent b on train
    B_train = model.f(dC).numpy()

    # Forecast
    Cp = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32)).numpy().ravel()
    Cc = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
    b0 = model.f(torch.tensor(Cc - Cp, dtype=torch.float32).unsqueeze(0)).numpy().ravel()

fc_obs = np.empty((FCST_LEN, n_obs))
fc_carriers = np.empty((FCST_LEN, j))
C = Cc.copy(); b = b0.copy()
with torch.no_grad():
    fc_obs[0] = model.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
    fc_carriers[0] = C.copy()
for t in range(1, FCST_LEN):
    b = A_dmd @ b
    with torch.no_grad():
        dc = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
    C = C + dc
    fc_carriers[t] = C.copy()
    with torch.no_grad():
        fc_obs[t] = model.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()

fc_phys = (fc_obs * sig + mu)[:, :3]

# True carrier on test (for comparison)
with torch.no_grad():
    C_test_true = model.carrier(torch.tensor(test_n, dtype=torch.float32)).numpy()


# ═══════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# ── Row 1: Loss curves ───────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(p1_losses, color="#1f77b4", lw=1.5)
ax1.set_title("Phase 1: Teacher AE Loss", fontsize=11)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE (log)")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(p2_losses, color="#2ca02c", lw=1.5)
ax2.set_title("Phase 2: Residual AE Loss", fontsize=11)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE (log)")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
theta = np.linspace(0, 2*np.pi, 100)
ax3.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=1)
ax3.scatter(eigvals.real, eigvals.imag, c="#d62728", s=80, zorder=5, edgecolors="k")
for i, ev in enumerate(eigvals):
    ax3.annotate(f"|λ|={abs(ev):.3f}", (ev.real, ev.imag),
                 textcoords="offset points", xytext=(8, 5), fontsize=8)
ax3.set_title("DMD Eigenvalues", fontsize=11)
ax3.set_xlabel("Re(λ)"); ax3.set_ylabel("Im(λ)")
ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

# ── Row 2: Carrier & residual space ──────────────────────────────
# Show first 4 carrier dims on test set (first 200 steps)
SHOW = 200
ax4 = fig.add_subplot(gs[1, 0:2])
for d in range(4):
    ax4.plot(C_test_true[:SHOW, d], '-', lw=1.2, label=f"C[{d}] true", alpha=0.8)
ax4.set_title(f"Carrier Trajectory (test, first {SHOW} steps, dims 0-3)", fontsize=11)
ax4.set_xlabel("Time step"); ax4.set_ylabel("Carrier value")
ax4.legend(fontsize=7, ncol=2); ax4.grid(True, alpha=0.3)

# Residual AE: true ΔC vs reconstructed (scatter, dim 0-3)
ax5 = fig.add_subplot(gs[1, 2])
for d in range(4):
    ax5.scatter(dC_np[::5, d], dC_hat[::5, d], s=3, alpha=0.4, label=f"dim {d}")
lims = [min(dC_np[:,:4].min(), dC_hat[:,:4].min()),
        max(dC_np[:,:4].max(), dC_hat[:,:4].max())]
ax5.plot(lims, lims, 'k--', lw=1, alpha=0.5)
ax5.set_title("Residual AE: true ΔC vs m(f(ΔC))", fontsize=11)
ax5.set_xlabel("True ΔC"); ax5.set_ylabel("Reconstructed ΔC")
ax5.legend(fontsize=7); ax5.grid(True, alpha=0.3); ax5.set_aspect("equal")

# ── Row 3: Latent b trajectory ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
for d in range(k):
    ax6.plot(B_train[:SHOW, d], lw=1.2, label=f"b[{d}]", alpha=0.8)
ax6.set_title(f"Residual Latent b Trajectory (train, first {SHOW} steps)", fontsize=11)
ax6.set_xlabel("Time step"); ax6.set_ylabel("b value")
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

# DMD 1-step prediction quality
B_pred = (A_dmd @ B_all[:-1].T).T  # (N-2, k)
B_true = B_all[1:]
ax7 = fig.add_subplot(gs[2, 2])
for d in range(k):
    ax7.scatter(B_true[::5, d], B_pred[::5, d], s=3, alpha=0.4, label=f"b[{d}]")
lims_b = [min(B_true.min(), B_pred.min()), max(B_true.max(), B_pred.max())]
ax7.plot(lims_b, lims_b, 'k--', lw=1, alpha=0.5)
ax7.set_title("DMD 1-step: true b_{t+1} vs A·b_t", fontsize=11)
ax7.set_xlabel("True b_{t+1}"); ax7.set_ylabel("Predicted A·b_t")
ax7.legend(fontsize=7); ax7.grid(True, alpha=0.3); ax7.set_aspect("equal")

# ── Row 4: Forecast ──────────────────────────────────────────────
labels = ["x", "y", "z"]
colors_xyz = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ax8 = fig.add_subplot(gs[3, 0:2])
for d in range(3):
    ax8.plot(gt_test[:SHOW, d], '-', color=colors_xyz[d], lw=1.5,
             label=f"true {labels[d]}", alpha=0.8)
    ax8.plot(fc_phys[:SHOW, d], '--', color=colors_xyz[d], lw=1.2,
             label=f"pred {labels[d]}", alpha=0.8)
ax8.axvline(LT, color="red", ls=":", lw=1.5, alpha=0.7, label=f"1 LT ({LT} steps)")
ax8.set_title(f"Forecast: Truth vs Predicted (first {SHOW} steps)", fontsize=11)
ax8.set_xlabel("Time step"); ax8.set_ylabel("Physical value")
ax8.legend(fontsize=7, ncol=3); ax8.grid(True, alpha=0.3)

# Forecast carrier vs true carrier (first 4 dims)
ax9 = fig.add_subplot(gs[3, 2])
for d in range(4):
    ax9.plot(C_test_true[:SHOW, d], '-', lw=1.2, alpha=0.6, color=f"C{d}")
    ax9.plot(fc_carriers[:SHOW, d], '--', lw=1.2, alpha=0.8, color=f"C{d}")
ax9.axvline(LT, color="red", ls=":", lw=1.5, alpha=0.7)
ax9.set_title("Carrier: true (solid) vs forecast (dashed)", fontsize=11)
ax9.set_xlabel("Time step"); ax9.set_ylabel("Carrier value")
ax9.grid(True, alpha=0.3)

fig.suptitle("Residual Three-phase Diagnostics — Lorenz-63\n"
             f"carrier_dim={j}, latent_dim={k}, DMD forecast",
             fontsize=14, y=0.995)

fig.savefig(OUT / "residual_diagnostics.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'residual_diagnostics.png'}")

# ── RMSE numbers ──────────────────────────────────────────────────
rmse_r = np.sqrt(np.mean((recon_phys - gt_test)**2))
rmse_f = np.sqrt(np.mean((fc_phys[:LT] - gt_test[:LT])**2))
print(f"Recon RMSE: {rmse_r:.4f}")
print(f"Forecast RMSE (1 LT): {rmse_f:.4f}")

# Residual AE quality
res_ae_rmse = np.sqrt(np.mean((dC_np - dC_hat)**2))
print(f"Residual AE RMSE (ΔC): {res_ae_rmse:.6f}")

# DMD 1-step quality
dmd_1step = np.sqrt(np.mean((B_true - B_pred)**2))
print(f"DMD 1-step RMSE (b): {dmd_1step:.6f}")

print("Done.")

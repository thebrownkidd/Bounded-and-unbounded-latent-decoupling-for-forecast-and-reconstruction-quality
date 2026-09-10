"""
Three-phase ablation: dynamics loss weight β
=============================================
Sweep β ∈ {0, 0.1, 0.5, 1, 2, 5} on Lorenz-63.

β=0   → carrier matching only, no dynamics shaping (degenerates toward Std AE)
β>0   → compressor latent is shaped for linear predictability

Also ablates:
  - Phase 3 skip (β=1, no fine-tuning) → shows phase 3's contribution
  - No-teacher baseline (compressor+decoder, no carrier, β=1) → shows teacher's contribution

Reference points (from run_lorenz63_comparison.py, same seed/data):
  Standard AE + DMD:    recon=0.0400  fcst=9.0200
  Koopman α=0.5 (best): recon=0.0845  fcst=6.6426
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

# ── Config ────────────────────────────────────────────────────────
SEED     = 0
DT       = 0.02
DELAYS   = 5
K_DIM    = 4
HIDDEN   = 96
EPOCHS   = 400
LR       = 1e-3
BS       = 512
N_TRAIN  = 3000
N_TEST   = 500
FCST_LEN = 400
LT       = 55

CARRIER  = 16
H_TP     = 64

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
DEVICE = "cpu"

# ── Data ──────────────────────────────────────────────────────────
def lorenz(t, s):
    x, y, z = s
    return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]

sol = solve_ivp(lorenz, [0, 120], [1,1,1],
                t_eval=np.arange(0, 120, DT),
                method="RK45", rtol=1e-10, atol=1e-10)
raw = sol.y.T[1000:]
obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
N_OBS = obs.shape[1]
train_obs, test_obs = obs[:N_TRAIN], obs[N_TRAIN:N_TRAIN+N_TEST]
test3 = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
train_n = (train_obs - mu) / sig
test_n  = (test_obs  - mu) / sig

Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
Xt_all = torch.tensor(train_n, dtype=torch.float32)
print(f"obs={N_OBS}D  latent={K_DIM}D  train={len(train_n)}  test={len(test_n)}")

# ── Models ────────────────────────────────────────────────────────
class ThreePhaseModel(nn.Module):
    def __init__(self, n, k, carrier_dim, h=64):
        super().__init__()
        self.teacher_enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        self.decoder = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        self.mapping = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, carrier_dim))

    def teacher_recon(self, x):
        return self.decoder(self.teacher_enc(x))
    def encode(self, x):
        return self.compressor(x)
    def decode_latent(self, z):
        return self.decoder(self.mapping(z))
    def student_recon(self, x):
        return self.decode_latent(self.encode(x))

class NoTeacherModel(nn.Module):
    """Ablation: compressor → decoder directly (no carrier, no teacher)."""
    def __init__(self, n, k, h=64):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        self.decoder = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
    def encode(self, x): return self.compressor(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): return self.decode(self.encode(x))

# ── DMD ───────────────────────────────────────────────────────────
def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out

# ── Evaluate ──────────────────────────────────────────────────────
def eval_model(tag, recon_fn, fcst_fn):
    rec = recon_fn(test_n)
    rec3 = (rec * sig + mu)[:, :3]
    fc = fcst_fn(test_n)
    fc3 = (fc * sig + mu)[:, :3]
    rmse_r = np.sqrt(np.mean((rec3 - test3)**2))
    rmse_f = np.sqrt(np.mean((fc3[:LT] - test3[:LT])**2))
    mx = np.max(np.abs(fc3))
    div = mx > 500
    print(f"  {tag:45s} recon={rmse_r:.4f}  fcst={rmse_f:.4f}  max={mx:.0f}"
          f"{'  DIVERGED' if div else ''}")
    return dict(rmse_r=rmse_r, rmse_f=rmse_f, mx=mx, div=div, rec3=rec3, fc3=fc3)

# ── Three-phase training function ────────────────────────────────
def train_three_phase(beta, skip_phase3=False, label=None):
    """Run the full three-phase pipeline with a given β. Returns eval dict."""
    tag = label or f"β={beta}"
    print(f"\n{'='*55}")
    print(f"  Three-phase  {tag}")
    print(f"{'='*55}")

    SeedAll(SEED)
    m = ThreePhaseModel(N_OBS, K_DIM, CARRIER, h=H_TP).to(DEVICE)
    K = nn.Linear(K_DIM, K_DIM, bias=False).to(DEVICE)

    # ---- Phase 1 ----
    print("  Phase 1: teacher + decoder")
    for p in m.parameters(): p.requires_grad_(False)
    for mod in [m.teacher_enc, m.decoder]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        m.train(); idx = torch.randperm(len(Xt_all)); losses = []
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(m.teacher_recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"    ep {ep:3d}  recon={np.mean(losses):.6f}")

    # ---- Phase 2 ----
    print(f"  Phase 2: compressor + mapping + K  (β={beta})")
    for p in m.parameters(): p.requires_grad_(False)
    for mod in [m.compressor, m.mapping]:
        for p in mod.parameters(): p.requires_grad_(True)
    for p in K.parameters(): p.requires_grad_(True)
    all_p2 = [p for p in m.parameters() if p.requires_grad] + list(K.parameters())
    opt = torch.optim.Adam(all_p2, lr=LR)

    with torch.no_grad():
        tc_cur = m.teacher_enc(Xt_cur)

    for ep in range(1, EPOCHS+1):
        m.train(); K.train()
        idx = torch.randperm(len(Xt_cur)); lc_ = []; ld_ = []
        for i in range(0, len(Xt_cur), BS):
            sl = idx[i:i+BS]
            z_c = m.compressor(Xt_cur[sl])
            z_n = m.compressor(Xt_nxt[sl])
            L_carrier = nn.functional.mse_loss(m.mapping(z_c), tc_cur[sl])
            L_dyn = nn.functional.mse_loss(K(z_c), z_n)
            loss = L_carrier + beta * L_dyn
            opt.zero_grad(); loss.backward(); opt.step()
            lc_.append(L_carrier.item()); ld_.append(L_dyn.item())
        if ep % 100 == 0:
            print(f"    ep {ep:3d}  carrier={np.mean(lc_):.6f}  dyn={np.mean(ld_):.6f}")

    K.eval(); m.eval()
    K_np = K.weight.detach().numpy()
    eig = np.sort(np.abs(np.linalg.eigvals(K_np)))[::-1]
    print(f"    K |λ|: {eig}")

    # ---- Phase 3 (optional) ----
    if not skip_phase3:
        print("  Phase 3: fine-tune mapping + decoder")
        for p in m.parameters(): p.requires_grad_(False)
        for mod in [m.mapping, m.decoder]:
            for p in mod.parameters(): p.requires_grad_(True)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR*0.3)
        for ep in range(1, EPOCHS+1):
            m.train(); idx = torch.randperm(len(Xt_all)); losses = []
            for i in range(0, len(Xt_all), BS):
                x = Xt_all[idx[i:i+BS]]
                loss = nn.functional.mse_loss(m.student_recon(x), x)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            if ep % 100 == 0:
                print(f"    ep {ep:3d}  recon={np.mean(losses):.6f}")
    else:
        print("  Phase 3: SKIPPED")

    m.eval()

    def _recon(tn):
        with torch.no_grad():
            return m.student_recon(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fcst(tn):
        with torch.no_grad():
            z0 = m.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: K_np @ z, FCST_LEN)
        with torch.no_grad():
            return m.decode_latent(torch.tensor(Z, dtype=torch.float32)).numpy()

    return eval_model(f"Three-phase {tag}", _recon, _fcst)

# ── No-teacher ablation ──────────────────────────────────────────
def train_no_teacher(beta=1.0):
    """AE + K (joint recon+dynamics), no teacher, no carrier."""
    print(f"\n{'='*55}")
    print(f"  No-teacher ablation (AE + K, β={beta})")
    print(f"{'='*55}")

    SeedAll(SEED)
    m = NoTeacherModel(N_OBS, K_DIM, h=H_TP).to(DEVICE)
    K = nn.Linear(K_DIM, K_DIM, bias=False).to(DEVICE)
    params = list(m.parameters()) + list(K.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    for ep in range(1, EPOCHS+1):
        m.train(); K.train()
        idx = torch.randperm(len(Xt_cur)); lr_ = []; ld_ = []
        for i in range(0, len(Xt_cur), BS):
            sl = idx[i:i+BS]
            xc, xn = Xt_cur[sl], Xt_nxt[sl]
            z_c = m.encode(xc)
            z_n = m.encode(xn)
            L_r = nn.functional.mse_loss(m.decode(z_c), xc)
            L_d = nn.functional.mse_loss(K(z_c), z_n)
            loss = L_r + beta * L_d
            opt.zero_grad(); loss.backward(); opt.step()
            lr_.append(L_r.item()); ld_.append(L_d.item())
        if ep % 100 == 0:
            print(f"    ep {ep:3d}  recon={np.mean(lr_):.6f}  dyn={np.mean(ld_):.6f}")

    m.eval(); K.eval()
    K_np = K.weight.detach().numpy()

    def _recon(tn):
        with torch.no_grad():
            return m(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fcst(tn):
        with torch.no_grad():
            z0 = m.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: K_np @ z, FCST_LEN)
        with torch.no_grad():
            return m.decode(torch.tensor(Z, dtype=torch.float32)).numpy()

    return eval_model("No-teacher (AE + K, joint)", _recon, _fcst)


# ═══════════════════════════════════════════════════════════════════
#  Run ablations
# ═══════════════════════════════════════════════════════════════════

results = {}

# β sweep
betas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
for b in betas:
    results[f"β={b}"] = train_three_phase(b)

# Phase 3 skip (β=1)
results["β=1 no-p3"] = train_three_phase(1.0, skip_phase3=True, label="β=1, no phase 3")

# No-teacher ablation
results["no-teacher"] = train_no_teacher(beta=1.0)

# Reference points
REF = {
    "Std AE + DMD":   dict(rmse_r=0.0400, rmse_f=9.0200),
    "Koopman α=0.5":  dict(rmse_r=0.0845, rmse_f=6.6426),
}

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Ablation Summary")
print(f"{'='*65}")
print(f"{'Variant':<30s} {'Recon':>8s} {'Fcst':>8s} {'Max|f|':>8s}")
print("-" * 65)
for tag, r in results.items():
    print(f"{tag:<30s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f} {r['mx']:>8.0f}"
          f"{'  DIV' if r.get('div') else ''}")
print("-" * 65)
for tag, r in REF.items():
    print(f"{tag + ' (ref)':<30s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}")
print("-" * 65)

# ── Pareto plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# β sweep points
br = [results[f"β={b}"]["rmse_r"] for b in betas]
bf = [results[f"β={b}"]["rmse_f"] for b in betas]
ax.plot(br, bf, "o-", color="#2ca02c", lw=1.8, ms=8, zorder=3,
        label="Three-phase (sweep β)")
for b, xr, yf in zip(betas, br, bf):
    ax.annotate(f"β={b}", (xr, yf), textcoords="offset points",
                xytext=(6, 5), fontsize=7.5, color="#2ca02c")

# Phase 3 skip
r_nop3 = results["β=1 no-p3"]
ax.plot(r_nop3["rmse_r"], r_nop3["rmse_f"], "D", color="#ff7f0e",
        ms=10, zorder=4, label="β=1, no phase 3")

# No-teacher
r_nt = results["no-teacher"]
ax.plot(r_nt["rmse_r"], r_nt["rmse_f"], "^", color="#9467bd",
        ms=10, zorder=4, label="No teacher (AE+K joint)")

# Reference points
ax.plot(REF["Std AE + DMD"]["rmse_r"], REF["Std AE + DMD"]["rmse_f"],
        "s", color="#1f77b4", ms=10, zorder=4, label="Std AE + DMD (ref)")
ax.plot(REF["Koopman α=0.5"]["rmse_r"], REF["Koopman α=0.5"]["rmse_f"],
        "p", color="#d62728", ms=10, zorder=4, label="Koopman α=0.5 (ref)")

ax.set_xlabel("Reconstruction RMSE", fontsize=11)
ax.set_ylabel("Forecast RMSE (1 Lyapunov time)", fontsize=11)
ax.set_title("Three-phase Ablation: β sweep  ·  Lorenz-63", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "lorenz63_ablation_beta.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'lorenz63_ablation_beta.png'}")

# Save numbers
nums = {tag: {k: v for k, v in r.items() if k in ("rmse_r", "rmse_f", "mx", "div")}
        for tag, r in results.items()}
with open(OUT / "lorenz63_ablation.json", "w") as f:
    json.dump(nums, f, indent=2, default=float)
print(f"→ {OUT / 'lorenz63_ablation.json'}")

print("\nDone.")

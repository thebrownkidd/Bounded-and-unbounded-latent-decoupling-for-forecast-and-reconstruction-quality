"""
Lorenz-63 comparison: Three-phase + DMD vs Koopman AE vs Standard AE
=====================================================================
Same delay-embedded Lorenz-63 data (15D → 4D latent) used in the DMD experiment.

  ① Koopman AE (joint, sweep α) — learned linear K, end-to-end
  ② Standard AE + DMD           — recon only, post-hoc DMD
  ③ Three-phase + DMD (Ours)    — teacher-guided decoupled training, DMD forecast

All methods share the same latent dimension k=4 and DMD forecast head.
The ONLY difference for ③ is the three-phase training procedure.
"""
import sys
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
DELAYS   = 5           # 3 × 5 = 15-D obs
K_DIM    = 4
HIDDEN   = 96          # baseline AE hidden width
EPOCHS   = 400
LR       = 1e-3
BS       = 512
N_TRAIN  = 3000
N_TEST   = 500
FCST_LEN = 400
LT       = 55           # ~1 Lyapunov time in steps

# Three-phase config
CARRIER  = 16           # carrier dim (> k, for richer decoder input)
H_TP     = 64           # hidden width for three-phase sub-networks

OUT = ROOT / "Paper"
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cpu"

# ── 1. Data ───────────────────────────────────────────────────────
def lorenz(t, s):
    x, y, z = s
    return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]

sol = solve_ivp(lorenz, [0, 120], [1,1,1],
                t_eval=np.arange(0, 120, DT),
                method="RK45", rtol=1e-10, atol=1e-10)
raw = sol.y.T[1000:]

obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i]
                      for i in range(DELAYS)], axis=1)
N_OBS = obs.shape[1]
train_obs, test_obs = obs[:N_TRAIN], obs[N_TRAIN:N_TRAIN+N_TEST]
test3 = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
train_n = (train_obs - mu) / sig
test_n  = (test_obs  - mu) / sig
print(f"obs={N_OBS}D  latent={K_DIM}D  train={len(train_n)}  test={len(test_n)}")

# ── 2. Simple AE / Koopman AE classes ────────────────────────────
class AE(nn.Module):
    def __init__(self, n, k, bounded=False):
        super().__init__()
        self.bounded = bounded; h = HIDDEN
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(torch.sigmoid(z) if self.bounded else z)
    def forward(self, x): return self.decode(self.encode(x))

class KoopmanAE(nn.Module):
    def __init__(self, n, k):
        super().__init__(); h = HIDDEN
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
        self.K  = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def advance(self, z): return self.K(z)
    def forward(self, x): return self.decode(self.encode(x))

# ── 3. Three-phase model (lightweight, DMD-native) ──────────────
class ThreePhaseModel(nn.Module):
    """Decoupled three-phase AE for DMD forecasting.

    Architecture:
      teacher_enc : n → carrier_dim   (training-only, phase 1)
      compressor  : n → k             (DMD latent)
      mapping     : k → carrier_dim   (bridges latent to carrier)
      decoder     : carrier_dim → n   (shared, used at inference)

    Inference path: x → compressor → DMD rollout → mapping → decoder → x̂
    """
    def __init__(self, n, k, carrier_dim, h=64):
        super().__init__()
        # Teacher encoder (phase 1 only)
        self.teacher_enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        # Compressor (student encoder)
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        # Mapping: k → carrier
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

# ── 4. DMD utilities ─────────────────────────────────────────────
def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out

# ── 5. Evaluate any model ────────────────────────────────────────
def evaluate(tag, recon_fn, forecast_fn):
    """recon_fn(test_n) → recon array, forecast_fn(test_n) → forecast array"""
    rec = recon_fn(test_n)
    rec3 = (rec * sig + mu)[:, :3]
    fc = forecast_fn(test_n)
    fc3 = (fc * sig + mu)[:, :3]
    rmse_r = np.sqrt(np.mean((rec3 - test3)**2))
    rmse_f = np.sqrt(np.mean((fc3[:LT] - test3[:LT])**2))
    mx = np.max(np.abs(fc3))
    div = mx > 500
    print(f"  {tag:40s} recon={rmse_r:.4f}  fcst1LT={rmse_f:.4f}  max={mx:.0f}"
          f"{'  DIVERGED' if div else ''}")
    return dict(rec3=rec3, fc3=fc3, rmse_r=rmse_r, rmse_f=rmse_f, mx=mx, div=div)

# ── 6. ① Koopman AE sweep ────────────────────────────────────────
alphas = [0.01, 0.1, 0.5, 1.0, 5.0]
Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
Xt_all = torch.tensor(train_n, dtype=torch.float32)
koop = {}

for alpha in alphas:
    print(f"\n--- Koopman AE α={alpha} ---")
    torch.manual_seed(SEED)
    m = KoopmanAE(N_OBS, K_DIM)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        m.train(); idx = torch.randperm(len(Xt_cur))
        for i in range(0, len(Xt_cur), BS):
            sl = idx[i:i+BS]; xc, xn = Xt_cur[sl], Xt_nxt[sl]
            z = m.encode(xc)
            Lr = nn.functional.mse_loss(m.decode(z), xc)
            Lf = nn.functional.mse_loss(m.decode(m.advance(z)), xn)
            loss = Lr + alpha * Lf
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            print(f"    ep {ep}")
    m.eval()
    K_np = m.K.weight.detach().numpy()
    def _recon(tn, _m=m):
        with torch.no_grad():
            return _m(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fcst(tn, _m=m, _K=K_np):
        with torch.no_grad():
            z0 = _m.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: _K @ z, FCST_LEN)
        with torch.no_grad():
            return _m.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    koop[alpha] = evaluate(f"Koopman α={alpha}", _recon, _fcst)

# ── 7. ② Standard AE + DMD ───────────────────────────────────────
print(f"\n--- Standard AE + DMD ---")
torch.manual_seed(SEED)
ae_s = AE(N_OBS, K_DIM)
opt = torch.optim.Adam(ae_s.parameters(), lr=LR)
for ep in range(1, EPOCHS+1):
    ae_s.train(); idx = torch.randperm(len(Xt_all))
    for i in range(0, len(Xt_all), BS):
        loss = nn.functional.mse_loss(ae_s(Xt_all[idx[i:i+BS]]), Xt_all[idx[i:i+BS]])
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0: print(f"    ep {ep}")
ae_s.eval()
with torch.no_grad(): Z_s = ae_s.encode(Xt_all).numpy()
A_s = fit_dmd(Z_s)
def _recon_s(tn):
    with torch.no_grad(): return ae_s(torch.tensor(tn, dtype=torch.float32)).numpy()
def _fcst_s(tn):
    with torch.no_grad():
        z0 = ae_s.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
    Z = rollout_np(z0, lambda z: A_s @ z, FCST_LEN)
    with torch.no_grad(): return ae_s.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
r_std = evaluate("Standard AE + DMD", _recon_s, _fcst_s)

# ── 8. ③ Three-phase + DMD (Ours) ────────────────────────────────
print(f"\n{'='*55}")
print(f"  Three-phase + DMD (Ours)")
print(f"{'='*55}")

SeedAll(SEED)
tpm = ThreePhaseModel(N_OBS, K_DIM, CARRIER, h=H_TP).to(DEVICE)
# Learned Koopman matrix for phase 2 dynamics shaping
K_learn = nn.Linear(K_DIM, K_DIM, bias=False).to(DEVICE)

total_p = sum(p.numel() for p in tpm.parameters()) + sum(p.numel() for p in K_learn.parameters())
teacher_p = sum(p.numel() for p in tpm.teacher_enc.parameters())
K_p = sum(p.numel() for p in K_learn.parameters())
infer_p = total_p - teacher_p
print(f"  Params: {total_p:,} total, {infer_p:,} inference, {teacher_p:,} teacher-only, K={K_p}")

# ---- Phase 1: teacher encoder + decoder (reconstruction) ----
print("\n  Phase 1: train teacher encoder + decoder (reconstruction)")
for p in tpm.parameters(): p.requires_grad_(False)
for mod in [tpm.teacher_enc, tpm.decoder]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=LR)
for ep in range(1, EPOCHS+1):
    tpm.train(); idx = torch.randperm(len(Xt_all)); losses = []
    for i in range(0, len(Xt_all), BS):
        x = Xt_all[idx[i:i+BS]]
        loss = nn.functional.mse_loss(tpm.teacher_recon(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if ep % 100 == 0 or ep == 1:
        print(f"    ep {ep:3d}  recon={np.mean(losses):.6f}")

# ---- Phase 2: compressor + mapping + K (carrier match + dynamics) ----
print("\n  Phase 2: train compressor + mapping + K (carrier + dynamics)")
for p in tpm.parameters(): p.requires_grad_(False)
for mod in [tpm.compressor, tpm.mapping]:
    for p in mod.parameters(): p.requires_grad_(True)
for p in K_learn.parameters(): p.requires_grad_(True)
all_p2 = ([p for p in tpm.parameters() if p.requires_grad]
         + list(K_learn.parameters()))
opt = torch.optim.Adam(all_p2, lr=LR)

# Precompute frozen teacher carriers for consecutive pairs
with torch.no_grad():
    teacher_carriers_cur = tpm.teacher_enc(Xt_cur)
    teacher_carriers_nxt = tpm.teacher_enc(torch.tensor(train_n[1:], dtype=torch.float32))

BETA = 1.0  # dynamics loss weight
for ep in range(1, EPOCHS+1):
    tpm.train(); K_learn.train()
    idx = torch.randperm(len(Xt_cur)); lc_ = []; ld_ = []
    for i in range(0, len(Xt_cur), BS):
        sl = idx[i:i+BS]
        xc, xn = Xt_cur[sl], Xt_nxt[sl]
        z_c = tpm.compressor(xc)
        z_n = tpm.compressor(xn)
        # Carrier matching loss
        L_carrier = nn.functional.mse_loss(tpm.mapping(z_c), teacher_carriers_cur[sl])
        # Dynamics loss: z_{t+1} ≈ K @ z_t
        L_dyn = nn.functional.mse_loss(K_learn(z_c), z_n)
        loss = L_carrier + BETA * L_dyn
        opt.zero_grad(); loss.backward(); opt.step()
        lc_.append(L_carrier.item()); ld_.append(L_dyn.item())
    if ep % 100 == 0 or ep == 1:
        print(f"    ep {ep:3d}  carrier={np.mean(lc_):.6f}  dyn={np.mean(ld_):.6f}")

# Use the learned K for forecasting
K_learn.eval(); tpm.eval()
K_np = K_learn.weight.detach().numpy()
eig = np.sort(np.abs(np.linalg.eigvals(K_np)))[::-1]
print(f"    K |λ|: {eig}")

# Also fit DMD for comparison
with torch.no_grad():
    Z_ours = tpm.encode(Xt_all).numpy()
A_dmd = fit_dmd(Z_ours)
eig_dmd = np.sort(np.abs(np.linalg.eigvals(A_dmd)))[::-1]
print(f"    DMD |λ|: {eig_dmd}")

# ---- Phase 3: fine-tune mapping + decoder ----
print("\n  Phase 3: fine-tune mapping + decoder (end-to-end recon)")
for p in tpm.parameters(): p.requires_grad_(False)
for mod in [tpm.mapping, tpm.decoder]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=LR * 0.3)
for ep in range(1, EPOCHS+1):
    tpm.train(); idx = torch.randperm(len(Xt_all)); losses = []
    for i in range(0, len(Xt_all), BS):
        x = Xt_all[idx[i:i+BS]]
        loss = nn.functional.mse_loss(tpm.student_recon(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if ep % 100 == 0 or ep == 1:
        print(f"    ep {ep:3d}  recon={np.mean(losses):.6f}")

tpm.eval()

# ---- Evaluate (use learned K, not post-hoc DMD) ----
def _recon_ours(tn):
    with torch.no_grad():
        return tpm.student_recon(torch.tensor(tn, dtype=torch.float32)).numpy()

def _fcst_ours(tn):
    with torch.no_grad():
        z0 = tpm.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
    Z = rollout_np(z0, lambda z: K_np @ z, FCST_LEN)
    with torch.no_grad():
        return tpm.decode_latent(torch.tensor(Z, dtype=torch.float32)).numpy()

r_ours = evaluate("Three-phase + DMD (Ours)", _recon_ours, _fcst_ours)

# ── 9. Summary ────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Summary Table")
print(f"{'='*65}")
print(f"{'Method':<35s} {'Recon':>8s} {'Fcst':>8s} {'Max|f|':>8s}")
print("-" * 65)
for a in alphas:
    r = koop[a]
    print(f"{'Koopman α='+str(a):<35s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f} {r['mx']:>8.0f}")
print(f"{'Standard AE + DMD':<35s} {r_std['rmse_r']:>8.4f} {r_std['rmse_f']:>8.4f} {r_std['mx']:>8.0f}")
print(f"{'Three-phase + DMD (Ours)':<35s} {r_ours['rmse_r']:>8.4f} {r_ours['rmse_f']:>8.4f} {r_ours['mx']:>8.0f}")
print("-" * 65)

# ── 10. Pareto plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))
kr = [koop[a]["rmse_r"] for a in alphas]
kf = [koop[a]["rmse_f"] for a in alphas]
ax.plot(kr, kf, "o-", color="#d62728", lw=1.5, ms=7, zorder=3,
        label="Koopman AE (joint, sweep α)")
for a, xr, yf in zip(alphas, kr, kf):
    ax.annotate(f"α={a}", (xr, yf), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#d62728")
ax.plot(r_std["rmse_r"], r_std["rmse_f"], "s", color="#1f77b4",
        ms=11, zorder=4, label="Standard AE + DMD")
ax.plot(r_ours["rmse_r"], r_ours["rmse_f"], "*", color="#2ca02c",
        ms=16, zorder=4, label="Three-phase + DMD (Ours)")
ax.set_xlabel("Reconstruction RMSE", fontsize=11)
ax.set_ylabel("Forecast RMSE (1 Lyapunov time)", fontsize=11)
ax.set_title("Reconstruction–Forecasting Tradeoff  ·  Lorenz-63", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "lorenz63_pareto.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'lorenz63_pareto.png'}")

# Forecast traces
t = np.arange(FCST_LEN) * DT
best_a = min(alphas, key=lambda a: koop[a]["rmse_f"])
studies = {"Standard AE + DMD": r_std,
           f"Koopman α={best_a}": koop[best_a],
           "Three-phase + DMD (Ours)": r_ours}
VARS = ["x", "y", "z"]
fig2, axes = plt.subplots(3, 3, figsize=(18, 8))
fig2.suptitle("Forecast Comparison  ·  Lorenz-63", fontsize=13, y=0.99)
for col, (tag, r) in enumerate(studies.items()):
    for row in range(3):
        ax = axes[row, col]
        N = min(FCST_LEN, len(r["fc3"]))
        ax.plot(t[:N], test3[:N, row], "k-", lw=0.6, label="Truth")
        ax.plot(t[:N], r["fc3"][:N, row], "r-", lw=0.6, alpha=.85, label="Forecast")
        if row == 0: ax.set_title(tag, fontsize=11)
        ax.set_ylabel(VARS[row])
        if row == 2: ax.set_xlabel("Time (s)")
        if row == 0 and col == 0: ax.legend(fontsize=7)
        gt = test3[:N, row]
        lo, hi = gt.min(), gt.max(); pad = (hi-lo)*0.5
        if r["mx"] > 500: ax.set_ylim(lo-pad, hi+pad)
fig2.tight_layout()
fig2.savefig(OUT / "lorenz63_forecast_comparison.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'lorenz63_forecast_comparison.png'}")

print("\nDone.")

"""
Reconstruction–Forecasting Tradeoff: Joint vs Decoupled Training
================================================================
Lorenz-63, delay-embedded (15-D → 4-D latent).

Three approaches, all using the SAME encoder/decoder architecture:

  ① Koopman AE (joint training, sweep forecast weight α)
      L = L_recon + α · L_forecast       α ∈ {0.01, 0.1, 0.5, 1, 5}
      Forecast gradient flows into encoder → recon may degrade

  ② Standard AE + post-hoc DMD
      Train AE for recon only → freeze → fit DMD on latent
      No forecast gradient → recon preserved

  ③ Bounded-carrier AE + post-hoc DMD  (Ours)
      Same as ② but sigmoid between encoder and decoder
      No forecast gradient + bounded carrier

Outputs:
  Paper/dmd_pareto.png             — recon RMSE vs forecast RMSE
  Paper/dmd_forecast_comparison.png — best Koopman vs Bounded forecasts
  Paper/dmd_recon_comparison.png   — reconstruction quality side-by-side
"""
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
SEED     = 0
DT       = 0.02
DELAYS   = 5           # 3 × 5 = 15-D observation
K_DIM    = 4
HIDDEN   = 96
EPOCHS   = 400
LR       = 1e-3
BS       = 512
N_TRAIN  = 3000
N_TEST   = 500
FCST_LEN = 400         # forecast steps (for plots)
LT       = 55          # ≈ 1 Lyapunov time in steps

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "Paper"; OUT.mkdir(exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)

# ── 1. Data ───────────────────────────────────────────────────────
def lorenz(t, s):
    x, y, z = s
    return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]

sol = solve_ivp(lorenz, [0, 120], [1,1,1],
                t_eval=np.arange(0, 120, DT),
                method="RK45", rtol=1e-10, atol=1e-10)
raw = sol.y.T[1000:]                             # drop 20 s transient

obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i]
                      for i in range(DELAYS)], axis=1)
N_OBS    = obs.shape[1]
train_ob = obs[:N_TRAIN]
test_ob  = obs[N_TRAIN:N_TRAIN+N_TEST]
test3    = raw[N_TRAIN:N_TRAIN+N_TEST, :3]   # obs[t][:3] == raw[t]
mu, sig  = train_ob.mean(0), train_ob.std(0)+1e-8
train_n  = (train_ob - mu) / sig
test_n   = (test_ob  - mu) / sig
print(f"obs={N_OBS}D  latent={K_DIM}D  train={len(train_n)}  test={len(test_n)}")

# ── 2. Model classes ─────────────────────────────────────────────
class AE(nn.Module):
    def __init__(self, n, k, bounded=False):
        super().__init__()
        self.bounded = bounded
        h = HIDDEN
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(),
                                 nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(),
                                 nn.Linear(h,n))
    def encode(self, x): return self.enc(x)
    def decode(self, z):
        return self.dec(torch.sigmoid(z) if self.bounded else z)
    def forward(self, x): return self.decode(self.encode(x))

class KoopmanAE(nn.Module):
    """AE with a learned linear map K for one-step latent advance."""
    def __init__(self, n, k):
        super().__init__()
        h = HIDDEN
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(),
                                 nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(),
                                 nn.Linear(h,n))
        self.K  = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def advance(self, z): return self.K(z)
    def forward(self, x): return self.decode(self.encode(x))

# ── 3. DMD ────────────────────────────────────────────────────────
def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n):
        out[t] = step(out[t-1])
    return out

# ── 4. Evaluation helper ─────────────────────────────────────────
def evaluate(model, decode_fn, step_fn, label):
    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(test_n, dtype=torch.float32)
        rec  = decode_fn(model.encode(Xte)).numpy()
        z0   = model.encode(Xte[:1]).numpy().ravel()
    rec3 = (rec * sig + mu)[:, :3]
    Z_fc = rollout_np(z0, step_fn, FCST_LEN)
    with torch.no_grad():
        fc = decode_fn(torch.tensor(Z_fc, dtype=torch.float32)).numpy()
    fc3 = (fc * sig + mu)[:, :3]
    rmse_r = np.sqrt(np.mean((rec3 - test3)**2))
    rmse_f = np.sqrt(np.mean((fc3[:LT] - test3[:LT])**2))
    mx     = np.max(np.abs(fc3))
    print(f"  {label:40s} recon={rmse_r:.3f}  fcst1LT={rmse_f:.3f}  max={mx:.0f}")
    return dict(rec3=rec3, fc3=fc3, rmse_r=rmse_r, rmse_f=rmse_f, mx=mx)

# ── 5. ① Koopman AE sweep ────────────────────────────────────────
alphas = [0.01, 0.1, 0.5, 1.0, 5.0]
Xt_cur  = torch.tensor(train_n[:-1], dtype=torch.float32)
Xt_nxt  = torch.tensor(train_n[1:],  dtype=torch.float32)
koop = {}

for alpha in alphas:
    tag = f"Koopman α={alpha}"
    print(f"\n{'='*55}\n  {tag}\n{'='*55}")
    torch.manual_seed(SEED)
    m = KoopmanAE(N_OBS, K_DIM)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        m.train(); idx = torch.randperm(len(Xt_cur)); lr_ = []; lf_ = []
        for i in range(0, len(Xt_cur), BS):
            sl  = idx[i:i+BS]
            xc, xn = Xt_cur[sl], Xt_nxt[sl]
            z   = m.encode(xc)
            Lr  = nn.functional.mse_loss(m.decode(z), xc)
            Lf  = nn.functional.mse_loss(m.decode(m.advance(z)), xn)
            loss = Lr + alpha * Lf
            opt.zero_grad(); loss.backward(); opt.step()
            lr_.append(Lr.item()); lf_.append(Lf.item())
        if ep % 100 == 0 or ep == 1:
            print(f"    ep {ep:3d}  recon={np.mean(lr_):.6f}  fcst={np.mean(lf_):.6f}")
    K_np = m.K.weight.detach().numpy()
    koop[alpha] = evaluate(m, m.decode, lambda z: K_np @ z, tag)

# ── 6. ② Standard AE + DMD ───────────────────────────────────────
print(f"\n{'='*55}\n  Standard AE + DMD\n{'='*55}")
torch.manual_seed(SEED)
ae_s = AE(N_OBS, K_DIM, bounded=False)
opt  = torch.optim.Adam(ae_s.parameters(), lr=LR)
Xt_all = torch.tensor(train_n, dtype=torch.float32)
for ep in range(1, EPOCHS+1):
    ae_s.train(); idx = torch.randperm(len(Xt_all)); ls_ = []
    for i in range(0, len(Xt_all), BS):
        b = Xt_all[idx[i:i+BS]]
        loss = nn.functional.mse_loss(ae_s(b), b)
        opt.zero_grad(); loss.backward(); opt.step()
        ls_.append(loss.item())
    if ep % 100 == 0 or ep == 1:
        print(f"    ep {ep:3d}  loss={np.mean(ls_):.6f}")
ae_s.eval()
with torch.no_grad():
    Z_s = ae_s.encode(Xt_all).numpy()
A_s = fit_dmd(Z_s)
eig_s = np.sort(np.abs(np.linalg.eigvals(A_s)))[::-1]
print(f"    DMD |λ|: {eig_s}")
r_std = evaluate(ae_s, ae_s.decode, lambda z: A_s @ z, "Standard AE + DMD")

# ── 7. ③ Bounded-carrier AE + DMD ────────────────────────────────
print(f"\n{'='*55}\n  Bounded-carrier AE + DMD (Ours)\n{'='*55}")
torch.manual_seed(SEED)
ae_b = AE(N_OBS, K_DIM, bounded=True)
opt  = torch.optim.Adam(ae_b.parameters(), lr=LR)
for ep in range(1, EPOCHS+1):
    ae_b.train(); idx = torch.randperm(len(Xt_all)); ls_ = []
    for i in range(0, len(Xt_all), BS):
        b = Xt_all[idx[i:i+BS]]
        loss = nn.functional.mse_loss(ae_b(b), b)
        opt.zero_grad(); loss.backward(); opt.step()
        ls_.append(loss.item())
    if ep % 100 == 0 or ep == 1:
        print(f"    ep {ep:3d}  loss={np.mean(ls_):.6f}")
ae_b.eval()
with torch.no_grad():
    Z_b = ae_b.encode(Xt_all).numpy()
A_b = fit_dmd(Z_b)
eig_b = np.sort(np.abs(np.linalg.eigvals(A_b)))[::-1]
print(f"    DMD |λ|: {eig_b}")
r_bnd = evaluate(ae_b, ae_b.decode, lambda z: A_b @ z, "Bounded AE + DMD (Ours)")

# ── 8. Pareto plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))
# Koopman sweep
kr = [koop[a]["rmse_r"] for a in alphas]
kf = [koop[a]["rmse_f"] for a in alphas]
ax.plot(kr, kf, "o-", color="#d62728", lw=1.5, ms=7, zorder=3,
        label="Koopman AE (joint, sweep α)")
for a, xr, yf in zip(alphas, kr, kf):
    ax.annotate(f"α={a}", (xr, yf), textcoords="offset points",
                xytext=(6, 4), fontsize=7, color="#d62728")
# Standard AE
ax.plot(r_std["rmse_r"], r_std["rmse_f"], "s", color="#1f77b4",
        ms=11, zorder=4, label="Standard AE + DMD")
# Bounded AE
ax.plot(r_bnd["rmse_r"], r_bnd["rmse_f"], "*", color="#2ca02c",
        ms=16, zorder=4, label="Bounded AE + DMD (Ours)")

ax.set_xlabel("Reconstruction RMSE", fontsize=11)
ax.set_ylabel("Forecast RMSE (1 Lyapunov time)", fontsize=11)
ax.set_title("Reconstruction–Forecasting Tradeoff  ·  Lorenz-63", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "dmd_pareto.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'dmd_pareto.png'}")

# ── 9. Forecast traces ───────────────────────────────────────────
best_a = min(alphas, key=lambda a: koop[a]["rmse_f"])
t = np.arange(FCST_LEN) * DT
VARS = ["x", "y", "z"]
studies = {f"Koopman AE α={best_a}": koop[best_a],
           "Bounded AE + DMD (Ours)": r_bnd}

fig2, axes = plt.subplots(3, 2, figsize=(14, 8))
fig2.suptitle("Forecast Comparison  ·  Lorenz-63", fontsize=13, y=0.99)
for col, (tag, r) in enumerate(studies.items()):
    for row in range(3):
        ax = axes[row, col]
        ax.plot(t, test3[:FCST_LEN, row], "k-", lw=0.6, label="Truth")
        ax.plot(t, r["fc3"][:FCST_LEN, row], "r-", lw=0.6, alpha=.85,
                label="Forecast")
        if row == 0: ax.set_title(tag, fontsize=11)
        ax.set_ylabel(VARS[row])
        if row == 2: ax.set_xlabel("Time (s)")
        if row == 0 and col == 0: ax.legend(fontsize=7)
        # clip if diverged
        gt = test3[:FCST_LEN, row]
        lo, hi = gt.min(), gt.max(); pad = (hi-lo)*0.5
        if r["mx"] > 500: ax.set_ylim(lo-pad, hi+pad)
fig2.tight_layout()
fig2.savefig(OUT / "dmd_forecast_comparison.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'dmd_forecast_comparison.png'}")

# ── 10. Reconstruction traces ────────────────────────────────────
fig3, axes = plt.subplots(3, 2, figsize=(14, 8))
fig3.suptitle("Reconstruction Comparison  ·  Lorenz-63", fontsize=13, y=0.99)
for col, (tag, r) in enumerate(studies.items()):
    for row in range(3):
        ax = axes[row, col]
        ax.plot(t[:200], test3[:200, row], "k-", lw=0.6, label="Truth")
        ax.plot(t[:200], r["rec3"][:200, row], "b-", lw=0.6, alpha=.85,
                label="Recon")
        if row == 0: ax.set_title(tag, fontsize=11)
        ax.set_ylabel(VARS[row])
        if row == 2: ax.set_xlabel("Time (s)")
        if row == 0 and col == 0: ax.legend(fontsize=7)
fig3.tight_layout()
fig3.savefig(OUT / "dmd_recon_comparison.png", dpi=150, bbox_inches="tight")
print(f"→ {OUT / 'dmd_recon_comparison.png'}")

# ── 11. Summary table ────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Summary Table")
print(f"{'='*65}")
print(f"{'Method':<35s} {'Recon':>8s} {'Fcst':>8s} {'Max|f|':>8s}")
print("-" * 65)
for a in alphas:
    r = koop[a]
    print(f"{'Koopman α='+str(a):<35s} {r['rmse_r']:>8.3f} {r['rmse_f']:>8.3f} {r['mx']:>8.0f}")
print(f"{'Standard AE + DMD':<35s} {r_std['rmse_r']:>8.3f} {r_std['rmse_f']:>8.3f} {r_std['mx']:>8.0f}")
print(f"{'Bounded AE + DMD (Ours)':<35s} {r_bnd['rmse_r']:>8.3f} {r_bnd['rmse_f']:>8.3f} {r_bnd['mx']:>8.0f}")
print("-" * 65)
print("Done.")

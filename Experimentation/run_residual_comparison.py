"""
Residual Three-phase: forecast on carrier DELTAS, not states
=============================================================
New architecture (Ours):
  E(x) → C ∈ ℝ^j          teacher encoder (carrier)
  D(C) → x̂                 decoder
  f(ΔC) → b ∈ ℝ^k, k≪j    compress carrier residual
  m(b) → ΔC                decode carrier residual
  DMD on b sequence         forecast in residual-latent space
  C_{t+1} = C_t + m(A·b_t) accumulate deltas

Recon uses E→D (pure AE, no forecast pressure).
Forecast uses DMD on residual latent + carrier accumulation.
The two are COMPLETELY decoupled.

Compared against:
  ① Standard AE + DMD    — post-hoc DMD on AE latent
  ② Koopman AE (α=0.5)   — joint training, learned K

Run on: Lorenz-63, Rössler, Lorenz-96.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
DEVICE = "cpu"
SEED = 0
EPOCHS = 400
LR = 1e-3
BS = 512

# ═══════════════════════════════════════════════════════════════════
#  Model definitions
# ═══════════════════════════════════════════════════════════════════

class AE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))

class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
        self.K = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def advance(self, z): return self.K(z)
    def forward(self, x): return self.decode(self.encode(x))

class ResidualModel(nn.Module):
    """Residual three-phase: teacher AE + residual forecaster.

    Inference recon:  x → enc → C → dec → x̂
    Inference fcst:   C_t, C_{t+1} → ΔC → f → b → DMD → b' → m → ΔC' → C + ΔC' → dec → x̂
    """
    def __init__(self, n, carrier_dim, k, h=64):
        super().__init__()
        # Teacher AE (used at inference too)
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        self.dec = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        # Residual compressor: ΔC → b
        self.f = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(),
            nn.Linear(h, k))
        # Residual decoder: b → ΔC
        self.m = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, carrier_dim))

    def recon(self, x):
        return self.dec(self.enc(x))

    def encode_carrier(self, x):
        return self.enc(x)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(name, train_n, test_n, last_train, gt_test, mu, sig,
               n_obs, gt_dim, lt_steps, fcst_len,
               k_dim, carrier_dim, h_base, h_tp,
               koopman_alpha=0.5):

    Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
    Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
    Xt_all = torch.tensor(train_n, dtype=torch.float32)

    def evaluate(tag, recon_fn, fcst_fn):
        rec = recon_fn(test_n)
        rec_p = (rec * sig + mu)[:, :gt_dim]
        fc = fcst_fn()
        fc_p = (fc * sig + mu)[:, :gt_dim]
        N = min(lt_steps, len(fc_p))
        rmse_r = np.sqrt(np.mean((rec_p - gt_test)**2))
        rmse_f = np.sqrt(np.mean((fc_p[:N] - gt_test[:N])**2))
        mx = np.max(np.abs(fc_p))
        div = mx > 500
        print(f"  {tag:45s} recon={rmse_r:.4f}  fcst={rmse_f:.4f}  max={mx:.0f}"
              f"{'  DIVERGED' if div else ''}")
        return dict(rmse_r=float(rmse_r), rmse_f=float(rmse_f),
                    mx=float(mx), div=bool(div))

    results = {}

    # ── ① Standard AE + DMD ──────────────────────────────────────
    print(f"\n  --- Standard AE + DMD (k={k_dim}) ---")
    torch.manual_seed(SEED)
    ae = AE(n_obs, k_dim, h_base)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        ae.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            loss = nn.functional.mse_loss(ae(Xt_all[idx[i:i+BS]]), Xt_all[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      ep {ep}")
    ae.eval()
    with torch.no_grad(): Z_s = ae.encode(Xt_all).numpy()
    A_s = fit_dmd(Z_s)

    def _r_std(tn):
        with torch.no_grad(): return ae(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _f_std():
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: A_s @ z, fcst_len)
        with torch.no_grad(): return ae.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    results["Std AE + DMD"] = evaluate("Std AE + DMD", _r_std, _f_std)

    # ── ② Koopman AE ─────────────────────────────────────────────
    alpha = koopman_alpha
    print(f"\n  --- Koopman AE α={alpha} (k={k_dim}) ---")
    torch.manual_seed(SEED)
    km = KoopmanAE(n_obs, k_dim, h_base)
    opt = torch.optim.Adam(km.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        km.train(); idx = torch.randperm(len(Xt_cur))
        for i in range(0, len(Xt_cur), BS):
            sl = idx[i:i+BS]; xc, xn = Xt_cur[sl], Xt_nxt[sl]
            z = km.encode(xc)
            Lr = nn.functional.mse_loss(km.decode(z), xc)
            Lf = nn.functional.mse_loss(km.decode(km.advance(z)), xn)
            loss = Lr + alpha * Lf
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      ep {ep}")
    km.eval(); K_np = km.K.weight.detach().numpy()

    def _r_km(tn):
        with torch.no_grad(): return km(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _f_km():
        with torch.no_grad():
            z0 = km.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: K_np @ z, fcst_len)
        with torch.no_grad(): return km.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    results[f"Koopman α={alpha}"] = evaluate(f"Koopman AE α={alpha}", _r_km, _f_km)

    # ── ③ Residual Three-phase + DMD ─────────────────────────────
    print(f"\n  --- Residual Three-phase + DMD (carrier={carrier_dim}, k={k_dim}) ---")
    SeedAll(SEED)
    rm = ResidualModel(n_obs, carrier_dim, k_dim, h=h_tp)

    tp = sum(p.numel() for p in rm.parameters())
    enc_p = sum(p.numel() for p in rm.enc.parameters())
    dec_p = sum(p.numel() for p in rm.dec.parameters())
    f_p = sum(p.numel() for p in rm.f.parameters())
    m_p = sum(p.numel() for p in rm.m.parameters())
    print(f"      Params: {tp:,} total (enc={enc_p:,} dec={dec_p:,} f={f_p:,} m={m_p:,})")

    # Phase 1: teacher AE
    print("      Phase 1: teacher AE (enc + dec)")
    for p in rm.parameters(): p.requires_grad_(False)
    for mod in [rm.enc, rm.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in rm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        rm.train(); idx = torch.randperm(len(Xt_all)); losses = []
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(rm.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  recon={np.mean(losses):.6f}")

    # Phase 2: compute carriers + residuals, train f+m, fit DMD
    print("      Phase 2: residual AE (f + m) + DMD")
    rm.eval()
    for p in rm.parameters(): p.requires_grad_(False)

    with torch.no_grad():
        C_all = rm.encode_carrier(Xt_all)          # (N, carrier_dim)
    delta_C = C_all[1:] - C_all[:-1]               # (N-1, carrier_dim)

    for p in rm.f.parameters(): p.requires_grad_(True)
    for p in rm.m.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in rm.parameters() if p.requires_grad], lr=LR)

    for ep in range(1, EPOCHS+1):
        rm.train(); idx = torch.randperm(len(delta_C)); losses = []
        for i in range(0, len(delta_C), BS):
            dc = delta_C[idx[i:i+BS]]
            dc_hat = rm.m(rm.f(dc))
            loss = nn.functional.mse_loss(dc_hat, dc)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  resid_ae={np.mean(losses):.6f}")

    rm.eval()
    with torch.no_grad():
        B_all = rm.f(delta_C).numpy()               # (N-1, k)
    A_res = fit_dmd(B_all)
    eig = np.sort(np.abs(np.linalg.eigvals(A_res)))[::-1]
    print(f"      DMD |λ|: {eig}")

    # ---- Evaluate ----
    def _r_res(tn):
        with torch.no_grad():
            return rm.recon(torch.tensor(tn, dtype=torch.float32)).numpy()

    def _f_res():
        with torch.no_grad():
            # Use last training point + first test point for initial transition
            x_prev = torch.tensor(last_train[np.newaxis], dtype=torch.float32)
            x_curr = torch.tensor(test_n[:1], dtype=torch.float32)
            C_prev = rm.encode_carrier(x_prev).numpy().ravel()
            C_curr = rm.encode_carrier(x_curr).numpy().ravel()
            dc = C_curr - C_prev
            b0 = rm.f(torch.tensor(dc, dtype=torch.float32).unsqueeze(0)).numpy().ravel()

        # Rollout: accumulate carrier deltas
        fc = np.empty((fcst_len, n_obs))
        C = C_curr.copy()
        b = b0.copy()
        # fc[0] = recon of test[0]
        with torch.no_grad():
            fc[0] = rm.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A_res @ b
            with torch.no_grad():
                dc = rm.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = rm.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc

    results["Residual 3-phase"] = evaluate("Residual Three-phase + DMD (Ours)", _r_res, _f_res)

    return results


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def gen_lorenz63():
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
    train_n = (train_obs-mu)/sig; test_n = (test_obs-mu)/sig
    return dict(name="Lorenz-63", train_n=train_n, test_n=test_n,
                last_train=train_n[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=55, fcst_len=400,
                k_dim=4, carrier_dim=16, h_base=96, h_tp=64)

def gen_rossler():
    a_, b_, c_ = 0.2, 0.2, 5.7; dt = 0.05
    def ode(t, s):
        x, y, z = s
        return [-(y+z), x + a_*y, b_ + z*(x - c_)]
    sol = solve_ivp(ode, [0, 800], [1, 1, 0],
                    t_eval=np.arange(0, 800, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[4000:]
    DELAYS = 5
    obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
    N_TRAIN, N_TEST = 3000, 500
    train_obs = obs[:N_TRAIN]; test_obs = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt_test = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    train_n = (train_obs-mu)/sig; test_n = (test_obs-mu)/sig
    return dict(name="Rössler", train_n=train_n, test_n=test_n,
                last_train=train_n[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=290, fcst_len=500,
                k_dim=4, carrier_dim=16, h_base=96, h_tp=64)

def gen_lorenz96():
    N_L96, F_L96 = 20, 8.0; dt = 0.01
    def l96(t, x):
        d = np.empty_like(x)
        for i in range(len(x)):
            d[i] = (x[(i+1)%N_L96] - x[(i-2)%N_L96]) * x[(i-1)%N_L96] - x[i] + F_L96
        return d
    x0 = F_L96 * np.ones(N_L96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 200], x0,
                    t_eval=np.arange(0, 200, dt),
                    method="RK45", rtol=1e-9, atol=1e-9)
    raw = sol.y.T[2000:]
    N_TRAIN, N_TEST = 4000, 500
    train_obs = raw[:N_TRAIN]; test_obs = raw[N_TRAIN:N_TRAIN+N_TEST]
    gt_test = raw[N_TRAIN:N_TRAIN+N_TEST]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    train_n = (train_obs-mu)/sig; test_n = (test_obs-mu)/sig
    return dict(name="Lorenz-96", train_n=train_n, test_n=test_n,
                last_train=train_n[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=20, gt_dim=20, lt_steps=67, fcst_len=200,
                k_dim=10, carrier_dim=18, h_base=96, h_tp=64)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

all_results = {}
t0 = time.time()

for gen_fn in [gen_lorenz63, gen_rossler, gen_lorenz96]:
    cfg = gen_fn()
    name = cfg.pop("name")
    print(f"\n{'#'*65}")
    print(f"  {name}")
    print(f"{'#'*65}")

    results = run_system(name, **cfg)
    all_results[name] = results

    # Print table
    print(f"\n  {'Method':<35s} {'Recon':>8s} {'Fcst':>8s}")
    print(f"  {'-'*55}")
    for tag, r in results.items():
        d = " DIV" if r["div"] else ""
        print(f"  {tag:<35s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d}")

# ═══════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  Cross-system Summary")
print(f"{'='*70}")
for sname, results in all_results.items():
    print(f"\n  {sname}:")
    print(f"    {'Method':<30s} {'Recon':>8s} {'Fcst':>8s}")
    print(f"    {'-'*50}")
    for tag, r in results.items():
        d = " DIV" if r["div"] else ""
        print(f"    {tag:<30s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d}")

# ── Pareto plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {"Std AE + DMD": "#1f77b4", "Residual 3-phase": "#2ca02c"}
markers = {"Std AE + DMD": "s", "Residual 3-phase": "*"}
ms_map = {"Std AE + DMD": 10, "Residual 3-phase": 14}

for ax, (sname, results) in zip(axes, all_results.items()):
    for tag, r in results.items():
        if r["div"]: continue
        c = colors.get(tag, "#d62728")
        mk = markers.get(tag, "o")
        ms = ms_map.get(tag, 8)
        ax.plot(r["rmse_r"], r["rmse_f"], mk, color=c, ms=ms, zorder=4, label=tag)
    ax.set_xlabel("Reconstruction RMSE")
    ax.set_ylabel("Forecast RMSE (1 LT)")
    ax.set_title(sname)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

fig.suptitle("Residual Three-phase vs Baselines", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "residual_comparison.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'residual_comparison.png'}")

# Save JSON
with open(OUT / "residual_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'residual_comparison.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

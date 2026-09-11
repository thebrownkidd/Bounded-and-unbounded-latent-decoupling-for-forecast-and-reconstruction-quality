"""
MLP Transition model: f(C_t, C_{t+1}) → b, M(b, C_t) → C_{t+1}
================================================================
f sees BOTH carriers and compresses the transition to k dims.
M reconstructs the next carrier from the latent + current carrier.
DMD forecasts in b-space; M(b_forecast, C_current) produces carriers.

No naked +. M is a learned transition function.

Compared against:
  ① Standard AE + DMD
  ② Koopman AE (α=0.5)
  ③ Original Three-phase + post-hoc DMD

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

class ThreePhaseModel(nn.Module):
    """Original three-phase: teacher AE + compressor + mapping."""
    def __init__(self, n, carrier_dim, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        self.dec = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        self.mapping = nn.Sequential(
            nn.Linear(k, h), nn.ELU(), nn.Linear(h, carrier_dim))

class TransitionModel(nn.Module):
    """MLP transition: f(C_t, C_{t+1}) → b, M(b, C_t) → C_{t+1}.

    Recon:   x → enc → C → dec → x̂  (pure AE, untouched)
    Forecast: f compresses carrier pair → b, DMD rolls out b,
              M(b_forecast, C_current) → next carrier → dec → x̂
    """
    def __init__(self, n, carrier_dim, k, h=64):
        super().__init__()
        j = carrier_dim
        # Teacher AE
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        # f: (C_t, C_{t+1}) → b ∈ ℝ^k
        self.f = nn.Sequential(
            nn.Linear(2 * j, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        # M: (b, C_t) → C_{t+1} ∈ ℝ^j
        self.M = nn.Sequential(
            nn.Linear(k + j, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, j))

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

    # ── ③ Original Three-phase + post-hoc DMD ────────────────────
    print(f"\n  --- Original Three-phase + DMD (carrier={carrier_dim}, k={k_dim}) ---")
    SeedAll(SEED)
    tp = ThreePhaseModel(n_obs, carrier_dim, k_dim, h=h_tp)

    # Phase 1: teacher AE
    for p in tp.parameters(): p.requires_grad_(False)
    for mod in [tp.enc, tp.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tp.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tp.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(tp.dec(tp.enc(x)), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      p1 ep {ep}")

    # Phase 2: compressor + mapping (carrier matching only)
    for p in tp.parameters(): p.requires_grad_(False)
    for mod in [tp.compressor, tp.mapping]:
        for p in mod.parameters(): p.requires_grad_(True)
    tp.eval()
    with torch.no_grad(): C_all = tp.enc(Xt_all)
    opt = torch.optim.Adam([p for p in tp.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tp.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            sl = idx[i:i+BS]
            b = tp.compressor(Xt_all[sl])
            loss = nn.functional.mse_loss(tp.mapping(b), C_all[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      p2 ep {ep}")

    # Phase 3: fine-tune mapping + decoder
    for p in tp.parameters(): p.requires_grad_(False)
    for mod in [tp.mapping, tp.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tp.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tp.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(tp.dec(tp.mapping(tp.compressor(x))), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      p3 ep {ep}")

    tp.eval()
    with torch.no_grad(): Z_tp = tp.compressor(Xt_all).numpy()
    A_tp = fit_dmd(Z_tp)

    def _r_tp(tn):
        with torch.no_grad():
            x = torch.tensor(tn, dtype=torch.float32)
            return tp.dec(tp.mapping(tp.compressor(x))).numpy()
    def _f_tp():
        with torch.no_grad():
            z0 = tp.compressor(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: A_tp @ z, fcst_len)
        with torch.no_grad(): return tp.dec(tp.mapping(
            torch.tensor(Z, dtype=torch.float32))).numpy()
    results["Orig 3-phase"] = evaluate("Original Three-phase + DMD", _r_tp, _f_tp)

    # ── ④ MLP Transition + DMD ───────────────────────────────────
    print(f"\n  --- MLP Transition + DMD (carrier={carrier_dim}, k={k_dim}) ---")
    SeedAll(SEED)
    tm = TransitionModel(n_obs, carrier_dim, k_dim, h=h_tp)

    nparams = sum(p.numel() for p in tm.parameters())
    print(f"      Params: {nparams:,} total")

    # Phase 1: teacher AE (enc + dec)
    print("      Phase 1: teacher AE")
    for p in tm.parameters(): p.requires_grad_(False)
    for mod in [tm.enc, tm.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(Xt_all)); losses = []
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(tm.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  recon={np.mean(losses):.6f}")

    # Phase 2: train f + M on carrier pairs
    print("      Phase 2: transition (f + M)")
    tm.eval()
    for p in tm.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        C_all = tm.encode_carrier(Xt_all)   # (N, j)
    C_cur = C_all[:-1]                       # (N-1, j)
    C_nxt = C_all[1:]                        # (N-1, j)

    for mod in [tm.f, tm.M]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)

    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(C_cur)); losses = []
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            cc, cn = C_cur[sl], C_nxt[sl]
            b = tm.f(torch.cat([cc, cn], dim=-1))
            c_hat = tm.M(torch.cat([b, cc], dim=-1))
            loss = nn.functional.mse_loss(c_hat, cn)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  transition={np.mean(losses):.6f}")

    # Fit DMD on b trajectory
    tm.eval()
    with torch.no_grad():
        B_all = tm.f(torch.cat([C_cur, C_nxt], dim=-1)).numpy()  # (N-1, k)
    A_b = fit_dmd(B_all)
    eig = np.sort(np.abs(np.linalg.eigvals(A_b)))[::-1]
    print(f"      DMD |λ|: {eig}")

    # Phase 3: fine-tune M + dec
    print("      Phase 3: fine-tune M + dec")
    for p in tm.parameters(): p.requires_grad_(False)
    for mod in [tm.M, tm.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)

    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(C_cur)); losses = []
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            cc, cn = C_cur[sl], C_nxt[sl]
            b = tm.f(torch.cat([cc, cn], dim=-1))  # f frozen
            c_hat = tm.M(torch.cat([b, cc], dim=-1))
            x_hat = tm.dec(c_hat)
            loss = nn.functional.mse_loss(x_hat, Xt_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  recon={np.mean(losses):.6f}")

    # ---- Evaluate ----
    def _r_tm(tn):
        with torch.no_grad():
            return tm.recon(torch.tensor(tn, dtype=torch.float32)).numpy()

    def _f_tm():
        with torch.no_grad():
            # Need C_{T-1} and C_T to get b_T
            x_prev = torch.tensor(last_train[np.newaxis], dtype=torch.float32)
            x_curr = torch.tensor(test_n[:1], dtype=torch.float32)
            C_prev = tm.encode_carrier(x_prev)
            C_curr = tm.encode_carrier(x_curr)

            b0 = tm.f(torch.cat([C_prev, C_curr], dim=-1)).numpy().ravel()

        # Rollout
        fc = np.empty((fcst_len, n_obs))
        C = C_curr.numpy().ravel()
        b = b0.copy()

        with torch.no_grad():
            fc[0] = tm.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()

        for t in range(1, fcst_len):
            b = A_b @ b
            with torch.no_grad():
                inp = torch.tensor(np.concatenate([b, C]), dtype=torch.float32).unsqueeze(0)
                C = tm.M(inp).numpy().ravel()
                fc[t] = tm.dec(torch.tensor(C, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc

    results["MLP Transition"] = evaluate("MLP Transition + DMD (Ours)", _r_tm, _f_tm)

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
colors = {"Std AE + DMD": "#1f77b4", "MLP Transition": "#2ca02c",
          "Orig 3-phase": "#ff7f0e"}
markers = {"Std AE + DMD": "s", "MLP Transition": "*",
           "Orig 3-phase": "^"}
ms_map = {"Std AE + DMD": 10, "MLP Transition": 14, "Orig 3-phase": 10}

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

fig.suptitle("MLP Transition vs Baselines", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "mlp_transition.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'mlp_transition.png'}")

with open(OUT / "mlp_transition.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'mlp_transition.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

"""
Smooth dynamical systems — no chaos, predictable dynamics.
============================================================
Systems:
  ① Van der Pol oscillator (μ=1.0)
  ② FitzHugh-Nagumo neuron model
  ③ Coupled harmonic oscillators (3 masses, 6D)
  ④ Damped Duffing oscillator

Methods:
  ① Std AE + DMD  (k same as carrier)
  ② Std AE + delay DMD
  ③ Resid + DMD + add
  ④ Resid + DMD + GRU

All use delay embedding (d=5) in observation space for 2D systems
to give the AE something to work with.
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
SEED = 0; EPOCHS = 400; LR = 1e-3; BS = 512
DELAY_D = 3  # delay embedding for DMD

# ═══════════════════════════════════════════════════════════════════
#  Models
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

class ResidualGRU(ResidualModel):
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.gru = nn.GRUCell(j, j)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def delay_embed(B, d):
    N, k = B.shape
    out = np.empty((N - d + 1, k * d))
    for i in range(d):
        out[:, i*k:(i+1)*k] = B[d-1-i:N-i]
    return out

def rollout_delay_dmd(A_del, z_aug, k, d, n_steps):
    out = np.empty((n_steps, k)); z = z_aug.copy(); out[0] = z[:k]
    for t in range(1, n_steps): z = A_del @ z; out[t] = z[:k]
    return out

# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_ae(model, Xt, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, epochs+1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            loss = nn.functional.mse_loss(model(Xt[idx[i:i+BS]]), Xt[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()

def train_resid_ae(model, dC):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(dC))
        for i in range(0, len(dC), BS):
            dc = dC[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()

def train_gru(model, dC, C_cur, C_nxt):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(C_cur))
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            with torch.no_grad():
                delta_hat = model.m(model.f(dC[sl]))
            c_hat = model.gru(delta_hat, C_cur[sl])
            loss = nn.functional.mse_loss(c_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(name, train_n, test_n, last_train, gt_test, mu, sig,
               n_obs, gt_dim, fcst_steps, k_dim, carrier_dim, h_size):

    Xt = torch.tensor(train_n, dtype=torch.float32)
    j = carrier_dim; k = k_dim; d = DELAY_D

    def evaluate(tag, recon_fn, fcst_fn):
        rec = recon_fn(test_n)
        rec_p = (rec * sig + mu)[:, :gt_dim]
        fc = fcst_fn()
        fc_p = (fc * sig + mu)[:, :gt_dim]
        N = min(fcst_steps, len(fc_p), len(gt_test))
        rmse_r = np.sqrt(np.mean((rec_p[:len(gt_test)] - gt_test)**2))
        rmse_f = np.sqrt(np.mean((fc_p[:N] - gt_test[:N])**2))
        mx = np.max(np.abs(fc_p[:N]))
        div = mx > 1000
        print(f"  {tag:40s} r={rmse_r:.6f}  f={rmse_f:.6f}"
              f"{'  DIV' if div else ''}")
        return dict(rmse_r=float(rmse_r), rmse_f=float(rmse_f),
                    mx=float(mx), div=bool(div))

    results = {}

    # ── Std AE + DMD ──────────────────────────────────────────────
    print(f"  --- Std AE + DMD (k={k}) ---")
    torch.manual_seed(SEED)
    ae = AE(n_obs, k, h_size); train_ae(ae, Xt); ae.eval()
    with torch.no_grad(): Z = ae.encode(Xt).numpy()
    A = fit_dmd(Z)

    def _r(tn):
        with torch.no_grad(): return ae(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f():
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
        out = np.empty((fcst_steps,k)); out[0]=z0
        for t in range(1,fcst_steps): out[t]=A@out[t-1]
        with torch.no_grad(): return ae.decode(torch.tensor(out,dtype=torch.float32)).numpy()
    results["Std AE+DMD"] = evaluate("Std AE + DMD", _r, _f)

    # ── Std AE + delay DMD ────────────────────────────────────────
    print(f"  --- Std AE + delay DMD ---")
    Zd = delay_embed(Z, d); Ad = fit_dmd(Zd)
    def _fd():
        with torch.no_grad():
            xi = torch.tensor(np.vstack([train_n[-(d-1):],test_n[:1]]),dtype=torch.float32)
            zi = ae.encode(xi).numpy()
        za = np.concatenate([zi[d-1-i] for i in range(d)])
        Br = rollout_delay_dmd(Ad, za, k, d, fcst_steps)
        with torch.no_grad(): return ae.decode(torch.tensor(Br,dtype=torch.float32)).numpy()
    results["Std AE+delDMD"] = evaluate("Std AE + delay DMD", _r, _fd)

    # ── Resid + DMD + add ─────────────────────────────────────────
    print(f"  --- Resid + DMD + add ---")
    SeedAll(SEED)
    r1 = ResidualModel(n_obs, j, k, h=64)
    train_teacher(r1, Xt)
    r1.eval()
    with torch.no_grad(): C1 = r1.carrier(Xt)
    dC1 = C1[1:] - C1[:-1]
    train_resid_ae(r1, dC1)
    r1.eval()
    with torch.no_grad(): B1 = r1.f(dC1).numpy()
    A1 = fit_dmd(B1)

    def _r1(tn):
        with torch.no_grad(): return r1.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f1():
        with torch.no_grad():
            Cp = r1.carrier(torch.tensor(last_train[None],dtype=torch.float32)).numpy().ravel()
            Cc = r1.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
            b0 = r1.f(torch.tensor(Cc-Cp,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        fc = np.empty((fcst_steps,n_obs)); C=Cc.copy(); b=b0.copy()
        with torch.no_grad():
            fc[0] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1,fcst_steps):
            b = A1 @ b
            with torch.no_grad():
                dc = r1.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+DMD+add"] = evaluate("Resid + DMD + add", _r1, _f1)

    # ── Resid + DMD + GRU ─────────────────────────────────────────
    print(f"  --- Resid + DMD + GRU ---")
    SeedAll(SEED)
    r2 = ResidualGRU(n_obs, j, k, h=64)
    train_teacher(r2, Xt)
    r2.eval()
    with torch.no_grad(): C2 = r2.carrier(Xt)
    C2c, C2n = C2[:-1], C2[1:]
    dC2 = C2n - C2c
    train_resid_ae(r2, dC2)
    train_gru(r2, dC2, C2c, C2n)
    r2.eval()
    with torch.no_grad(): B2 = r2.f(dC2).numpy()
    A2 = fit_dmd(B2)

    def _r2(tn):
        with torch.no_grad(): return r2.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f2():
        with torch.no_grad():
            Cp = r2.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = r2.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = r2.f(Cc-Cp).numpy().ravel()
        fc = np.empty((fcst_steps,n_obs))
        C = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1,fcst_steps):
            b = A2 @ b
            with torch.no_grad():
                dh = r2.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C = r2.gru(dh, C.unsqueeze(0)).squeeze(0)
                fc[t] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+DMD+GRU"] = evaluate("Resid + DMD + GRU", _r2, _f2)

    return results


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def make_delay(raw, delays, gt_dim):
    obs = np.concatenate([raw[i:len(raw)-delays+1+i] for i in range(delays)], axis=1)
    return obs, gt_dim

def gen_vanderpol():
    """Van der Pol oscillator: x'' - μ(1-x²)x' + x = 0, μ=1.0"""
    mu_vdp = 1.0; dt = 0.02
    def ode(t, s):
        x, y = s
        return [y, mu_vdp*(1 - x**2)*y - x]
    sol = solve_ivp(ode, [0, 200], [2.0, 0.0],
                    t_eval=np.arange(0, 200, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[1000:]  # skip transient
    DELAYS = 5
    obs, gd = make_delay(raw, DELAYS, 2)
    N_TR, N_TE = 3000, 500
    tr, te = obs[:N_TR], obs[N_TR:N_TR+N_TE]
    gt = raw[N_TR:N_TR+N_TE]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    # Period ≈ 2π for μ=0, longer for μ=1 (~6.66s → 333 steps)
    return dict(name="Van der Pol", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=gd, fcst_steps=500,
                k_dim=4, carrier_dim=8, h_size=64)

def gen_fitzhugh():
    """FitzHugh-Nagumo: v' = v - v³/3 - w + I, w' = ε(v + a - bw)"""
    a, b, eps, I_ext = 0.7, 0.8, 0.08, 0.5; dt = 0.1
    def ode(t, s):
        v, w = s
        return [v - v**3/3 - w + I_ext, eps*(v + a - b*w)]
    sol = solve_ivp(ode, [0, 2000], [-1.0, 1.0],
                    t_eval=np.arange(0, 2000, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    DELAYS = 5
    obs, gd = make_delay(raw, DELAYS, 2)
    N_TR, N_TE = 3000, 500
    tr, te = obs[:N_TR], obs[N_TR:N_TR+N_TE]
    gt = raw[N_TR:N_TR+N_TE]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="FitzHugh-Nagumo", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=gd, fcst_steps=500,
                k_dim=4, carrier_dim=8, h_size=64)

def gen_coupled_oscillators():
    """3 coupled harmonic oscillators (6D state: x1,v1,x2,v2,x3,v3).
    Springs: k12=1.0, k23=0.5, k_wall=0.3, masses=1."""
    dt = 0.05
    k12, k23, kw = 1.0, 0.5, 0.3
    def ode(t, s):
        x1,v1,x2,v2,x3,v3 = s
        a1 = -kw*x1 - k12*(x1-x2)
        a2 = -k12*(x2-x1) - k23*(x2-x3)
        a3 = -k23*(x3-x2) - kw*x3
        return [v1, a1, v2, a2, v3, a3]
    sol = solve_ivp(ode, [0, 500], [1,0, 0,0.5, -0.5,0],
                    t_eval=np.arange(0, 500, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    N_TR, N_TE = 4000, 500
    tr, te = raw[:N_TR], raw[N_TR:N_TR+N_TE]
    gt = raw[N_TR:N_TR+N_TE]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="Coupled Oscillators", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=6, gt_dim=6, fcst_steps=500,
                k_dim=4, carrier_dim=8, h_size=64)

def gen_duffing():
    """Damped Duffing: x'' + δx' + αx + βx³ = γcos(ωt)
    Non-chaotic regime: δ=0.3, α=-1, β=1, γ=0.37, ω=1.2"""
    delta, alpha, beta, gamma, omega = 0.3, -1.0, 1.0, 0.37, 1.2
    dt = 0.05
    def ode(t, s):
        x, v = s
        return [v, -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(ode, [0, 600], [0.5, 0.0],
                    t_eval=np.arange(0, 600, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    DELAYS = 5
    obs, gd = make_delay(raw, DELAYS, 2)
    N_TR, N_TE = 3000, 500
    tr, te = obs[:N_TR], obs[N_TR:N_TR+N_TE]
    gt = raw[N_TR:N_TR+N_TE]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="Duffing", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=gd, fcst_steps=500,
                k_dim=4, carrier_dim=8, h_size=64)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

all_results = {}
t0 = time.time()

for gen_fn in [gen_vanderpol, gen_fitzhugh, gen_coupled_oscillators, gen_duffing]:
    cfg = gen_fn()
    name = cfg.pop("name")
    print(f"\n{'#'*60}")
    print(f"  {name}  (obs={cfg['n_obs']}D, carrier={cfg['carrier_dim']}, k={cfg['k_dim']})")
    print(f"{'#'*60}")

    results = run_system(name, **cfg)
    all_results[name] = results

    print(f"\n  {'Method':<25s} {'Recon':>10s} {'Fcst':>10s}")
    print(f"  {'-'*50}")
    for tag, r in results.items():
        d_ = " DIV" if r["div"] else ""
        print(f"  {tag:<25s} {r['rmse_r']:>10.6f} {r['rmse_f']:>10.6f}{d_}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Cross-system Summary — Smooth Systems")
print(f"{'='*65}")
for sname, results in all_results.items():
    print(f"\n  {sname}:")
    print(f"    {'Method':<25s} {'Recon':>10s} {'Fcst':>10s}")
    print(f"    {'-'*50}")
    for tag, r in results.items():
        d_ = " DIV" if r["div"] else ""
        print(f"    {tag:<25s} {r['rmse_r']:>10.6f} {r['rmse_f']:>10.6f}{d_}")

# ── Pareto plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
colors = {"Std AE+DMD": "#888", "Std AE+delDMD": "#aaa",
          "Resid+DMD+add": "#1f77b4", "Resid+DMD+GRU": "#2ca02c"}
markers = {"Std AE+DMD": "s", "Std AE+delDMD": "D",
           "Resid+DMD+add": "o", "Resid+DMD+GRU": "*"}

for ax, (sname, results) in zip(axes, all_results.items()):
    for tag, r in results.items():
        if r["div"]: continue
        ax.plot(r["rmse_r"], r["rmse_f"],
                markers.get(tag,"o"), color=colors.get(tag,"#000"),
                ms=11, zorder=4, label=tag)
    ax.set_xlabel("Recon RMSE"); ax.set_ylabel("Fcst RMSE")
    ax.set_title(sname, fontsize=10)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.25)

fig.suptitle("Smooth Dynamical Systems — Residual vs Baselines", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "smooth_systems.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'smooth_systems.png'}")

with open(OUT / "smooth_systems.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'smooth_systems.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

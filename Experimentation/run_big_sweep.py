"""
Big system sweep: Residual models vs Std AE+DMD
================================================
Systems tuned for DMD (smooth, periodic, quasiperiodic).
Latent dims adjusted per system. k_forecast = k_ae for fairness.

Systems:
  1.  Van der Pol (μ=1)          2D+delays=10D, limit cycle
  2.  FitzHugh-Nagumo            2D+delays=10D, relaxation osc
  3.  Coupled harmonic (3 mass)  6D, linear
  4.  Damped Duffing             2D+delays=10D, forced osc
  5.  Lotka-Volterra             2D+delays=10D, predator-prey cycle
  6.  Glycolytic oscillator      2D+delays=10D, biochemical osc
  7.  Brusselator                2D+delays=10D, chemical osc
  8.  Stuart-Landau              2D+delays=10D, normal form limit cycle
  9.  Linear 5D system           5D, multi-freq linear
  10. Pendulum (damped+driven)   2D+delays=10D, smooth periodic
  11. Coupled Van der Pol        4D+delays=20D, coupled limit cycles
  12. Heat equation (1D, 20 pts) 20D, diffusion (decaying modes)
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

# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_ae(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
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

def train_resid(model, dC):
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

def train_gru(model, dC, Cc, Cn):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i+BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(cfg):
    name = cfg["name"]
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    last_train = train_n[-1]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps = cfg["fcst_steps"]
    k = cfg["k"]; j = cfg["j"]; h = cfg.get("h", 64)

    Xt = torch.tensor(train_n, dtype=torch.float32)

    def ev(tag, rfn, ffn):
        rec = rfn(test_n)
        rp = (rec * sig + mu)[:, :gt_dim]
        fc = ffn()
        fp = (fc * sig + mu)[:, :gt_dim]
        N = min(fcst_steps, len(fp), len(gt_test))
        rr = np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2))
        rf = np.sqrt(np.mean((fp[:N] - gt_test[:N])**2))
        mx = np.max(np.abs(fp[:N]))
        dv = mx > 1000
        return dict(rmse_r=float(rr), rmse_f=float(rf), mx=float(mx), div=bool(dv))

    res = {}

    # ── Std AE + DMD ──────────────────────────────────────────────
    torch.manual_seed(SEED)
    ae = AE(n_obs, k, h); train_ae(ae, Xt); ae.eval()
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
    res["AE+DMD"] = ev("AE+DMD", _r, _f)

    # ── Resid + DMD + add ─────────────────────────────────────────
    SeedAll(SEED)
    r1 = ResidualModel(n_obs, j, k, h=h)
    train_teacher(r1, Xt); r1.eval()
    with torch.no_grad(): C1 = r1.carrier(Xt)
    dC1 = C1[1:] - C1[:-1]
    train_resid(r1, dC1); r1.eval()
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
    res["Resid+add"] = ev("Resid+add", _r1, _f1)

    # ── Resid + DMD + GRU ─────────────────────────────────────────
    SeedAll(SEED)
    r2 = ResidualGRU(n_obs, j, k, h=h)
    train_teacher(r2, Xt); r2.eval()
    with torch.no_grad(): C2 = r2.carrier(Xt)
    C2c, C2n = C2[:-1], C2[1:]
    dC2 = C2n - C2c
    train_resid(r2, dC2)
    train_gru(r2, dC2, C2c, C2n); r2.eval()
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
    res["Resid+GRU"] = ev("Resid+GRU", _r2, _f2)

    return res


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd, gt_dim):
    obs = np.concatenate([raw[i:len(raw)-nd+1+i] for i in range(nd)], axis=1)
    return obs, gt_dim

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr+n_te]
    gt = raw[n_tr:n_tr+n_te, :gt_dim] if gt_dim < raw.shape[1] else raw[n_tr:n_tr+n_te]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, gt, mu, sig

def S(name, raw, gt_dim, obs_dim, n_tr, n_te, fcst, k, j, nd=0, h=64):
    if nd > 0:
        obs, gd = delay(raw, nd, gt_dim)
    else:
        obs = raw; gd = gt_dim
    trn, ten, gt, mu, sig = norm_split(obs, raw, gd, n_tr, n_te)
    return dict(name=name, train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=gd, fcst_steps=fcst, k=k, j=j, h=h)


def gen_all():
    systems = []

    # 1. Van der Pol
    mu_vdp = 1.0
    sol = solve_ivp(lambda t,s: [s[1], mu_vdp*(1-s[0]**2)*s[1]-s[0]],
                    [0,200], [2,0], t_eval=np.arange(0,200,0.02),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Van der Pol", sol.y.T[1000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 2. FitzHugh-Nagumo
    a,b,eps,I = 0.7,0.8,0.08,0.5
    sol = solve_ivp(lambda t,s: [s[0]-s[0]**3/3-s[1]+I, eps*(s[0]+a-b*s[1])],
                    [0,2000], [-1,1], t_eval=np.arange(0,2000,0.1),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("FitzHugh-Nagumo", sol.y.T[2000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 3. Coupled harmonic oscillators
    k12,k23,kw = 1.0,0.5,0.3
    def osc(t,s):
        x1,v1,x2,v2,x3,v3 = s
        return [v1, -kw*x1-k12*(x1-x2), v2, -k12*(x2-x1)-k23*(x2-x3),
                v3, -k23*(x3-x2)-kw*x3]
    sol = solve_ivp(osc, [0,500], [1,0,0,0.5,-0.5,0],
                    t_eval=np.arange(0,500,0.05), rtol=1e-10, atol=1e-10)
    systems.append(S("Coupled Harmonic", sol.y.T[200:], 6, 6, 4000, 500, 500, k=4, j=8))

    # 4. Damped Duffing
    de,al,be,ga,om = 0.3,-1,1,0.37,1.2
    sol = solve_ivp(lambda t,s: [s[1], -de*s[1]-al*s[0]-be*s[0]**3+ga*np.cos(om*t)],
                    [0,600], [0.5,0], t_eval=np.arange(0,600,0.05),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Duffing", sol.y.T[2000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 5. Lotka-Volterra
    a_,b_,c_,d_ = 1.5, 1.0, 3.0, 1.0
    sol = solve_ivp(lambda t,s: [a_*s[0]-b_*s[0]*s[1], -c_*s[1]+d_*s[0]*s[1]],
                    [0,200], [1.0,1.0], t_eval=np.arange(0,200,0.02),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Lotka-Volterra", sol.y.T[1000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 6. Glycolytic oscillator (Sel'kov model)
    a_g, b_g = 0.08, 0.6
    sol = solve_ivp(lambda t,s: [-s[0]+a_g*s[1]+s[0]**2*s[1],
                                  b_g-a_g*s[1]-s[0]**2*s[1]],
                    [0,500], [1.0,1.0], t_eval=np.arange(0,500,0.05),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Glycolytic (Sel'kov)", sol.y.T[2000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 7. Brusselator
    A_br, B_br = 1.0, 3.0
    sol = solve_ivp(lambda t,s: [A_br-(B_br+1)*s[0]+s[0]**2*s[1],
                                  B_br*s[0]-s[0]**2*s[1]],
                    [0,200], [1.0,1.0], t_eval=np.arange(0,200,0.01),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Brusselator", sol.y.T[2000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 8. Stuart-Landau (normal form, polar → cartesian)
    mu_sl, omega_sl = 0.1, 2*np.pi
    sol = solve_ivp(lambda t,s: [mu_sl*s[0]-omega_sl*s[1]-(s[0]**2+s[1]**2)*s[0],
                                  omega_sl*s[0]+mu_sl*s[1]-(s[0]**2+s[1]**2)*s[1]],
                    [0,100], [0.5,0], t_eval=np.arange(0,100,0.01),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Stuart-Landau", sol.y.T[1000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 9. Linear 5D (3 frequencies)
    w1, w2, w3 = 1.0, np.sqrt(2), np.pi
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],
                    [0,0,0,w2,0],[0,0,-w2,0,0],
                    [0,0,0,0,-0.05]])  # 2 osc + 1 decay
    from scipy.linalg import expm
    Ad = expm(A5 * 0.05)  # discrete-time step
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1,0,0.5,0.5,1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i-1]
    systems.append(S("Linear 5D", raw5[200:], 5, 5, 4000, 500, 500, k=4, j=8))

    # 10. Damped driven pendulum (non-chaotic)
    g_p, l_p, b_p, A_p, w_p = 9.81, 1.0, 0.5, 0.5, 2.0/3
    sol = solve_ivp(lambda t,s: [s[1], -g_p/l_p*np.sin(s[0])-b_p*s[1]+A_p*np.cos(w_p*t)],
                    [0,500], [0.5,0], t_eval=np.arange(0,500,0.05),
                    rtol=1e-10, atol=1e-10)
    systems.append(S("Driven Pendulum", sol.y.T[2000:], 2, 10, 3000, 500, 500, k=3, j=8, nd=5))

    # 11. Coupled Van der Pol
    mu_c, kc = 1.0, 0.2
    def cvdp(t, s):
        x1,y1,x2,y2 = s
        return [y1, mu_c*(1-x1**2)*y1 - x1 + kc*(x2-x1),
                y2, mu_c*(1-x2**2)*y2 - x2 + kc*(x1-x2)]
    sol = solve_ivp(cvdp, [0,400], [2,0,0.1,0.5],
                    t_eval=np.arange(0,400,0.02), rtol=1e-10, atol=1e-10)
    raw_cvdp = sol.y.T[2000:]
    nd_c = 5
    obs_c = np.concatenate([raw_cvdp[i:len(raw_cvdp)-nd_c+1+i] for i in range(nd_c)], axis=1)
    trn, ten, gt, mu_, sig_ = norm_split(obs_c, raw_cvdp, 4, 3000, 500)
    systems.append(dict(name="Coupled VdP", train_n=trn, test_n=ten, gt_test=gt,
                        mu=mu_, sig=sig_, n_obs=obs_c.shape[1], gt_dim=4,
                        fcst_steps=500, k=4, j=12, h=64))

    # 12. Heat equation (1D, 20 spatial points, diffusion)
    Nx = 20; dx = 1.0/(Nx+1); alpha_h = 0.01; dt_h = 0.001
    # Build discrete Laplacian
    L = np.zeros((Nx, Nx))
    for i in range(Nx):
        L[i,i] = -2
        if i > 0: L[i,i-1] = 1
        if i < Nx-1: L[i,i+1] = 1
    L *= alpha_h / dx**2
    Ad_h = expm(L * dt_h * 50)  # sample every 50 micro-steps
    N_h = 8000
    # Initial condition: sum of sines
    x_grid = np.linspace(dx, 1-dx, Nx)
    u0 = np.sin(np.pi*x_grid) + 0.5*np.sin(3*np.pi*x_grid) + 0.3*np.sin(5*np.pi*x_grid)
    raw_h = np.empty((N_h, Nx)); raw_h[0] = u0
    for i in range(1, N_h): raw_h[i] = Ad_h @ raw_h[i-1]
    systems.append(S("Heat Equation", raw_h[200:], Nx, Nx, 4000, 500, 500, k=6, j=12))

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

systems = gen_all()
all_results = {}
t0 = time.time()

for i, cfg in enumerate(systems):
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(systems)}] {name}  (obs={cfg['n_obs']}D  j={cfg['j']}  k={cfg['k']})")
    print(f"{'='*60}")

    res = run_system(cfg)
    all_results[name] = res

    # One-line per method
    for tag, r in res.items():
        d_ = " DIV" if r["div"] else ""
        print(f"    {tag:<15s}  r={r['rmse_r']:.6f}  f={r['rmse_f']:.6f}{d_}")

# ═══════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  MASTER TABLE — Pareto dominance vs Std AE+DMD")
print(f"{'='*80}")
print(f"  {'System':<22s} {'AE+DMD r/f':>16s}  {'Resid+add r/f':>16s}  {'Resid+GRU r/f':>16s}  Win?")
print(f"  {'-'*78}")

wins = 0; total = 0
for sname, res in all_results.items():
    ae = res["AE+DMD"]
    ra = res["Resid+add"]
    rg = res["Resid+GRU"]

    # Check Pareto dominance (either variant)
    add_win = (ra["rmse_r"] <= ae["rmse_r"]) and (ra["rmse_f"] <= ae["rmse_f"]) and not ra["div"]
    gru_win = (rg["rmse_r"] <= ae["rmse_r"]) and (rg["rmse_f"] <= ae["rmse_f"]) and not rg["div"]
    either = add_win or gru_win

    tag = ""
    if add_win and gru_win: tag = "✓ both"
    elif add_win: tag = "✓ add"
    elif gru_win: tag = "✓ GRU"
    else: tag = "✗"

    if either: wins += 1
    total += 1

    ae_s = f"{ae['rmse_r']:.4f}/{ae['rmse_f']:.4f}"
    ra_s = f"{ra['rmse_r']:.4f}/{ra['rmse_f']:.4f}"
    rg_s = f"{rg['rmse_r']:.4f}/{rg['rmse_f']:.4f}"
    if ra["div"]: ra_s += " D"
    if rg["div"]: rg_s += " D"

    print(f"  {sname:<22s} {ae_s:>16s}  {ra_s:>16s}  {rg_s:>16s}  {tag}")

print(f"\n  Pareto wins: {wins}/{total}")

# ── Plot ──────────────────────────────────────────────────────────
n_sys = len(all_results)
cols = 4; rows = (n_sys + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
axes = axes.flatten()

colors = {"AE+DMD": "#888", "Resid+add": "#1f77b4", "Resid+GRU": "#2ca02c"}
markers = {"AE+DMD": "s", "Resid+add": "o", "Resid+GRU": "*"}

for idx, (sname, res) in enumerate(all_results.items()):
    ax = axes[idx]
    for tag, r in res.items():
        if r["div"]: continue
        ax.plot(r["rmse_r"], r["rmse_f"],
                markers[tag], color=colors[tag], ms=10, zorder=4, label=tag)
    ax.set_xlabel("Recon", fontsize=8); ax.set_ylabel("Fcst", fontsize=8)
    ax.set_title(sname, fontsize=9)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.25)

for idx in range(len(all_results), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("12-System Sweep: Residual vs AE+DMD", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "big_sweep.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'big_sweep.png'}")

with open(OUT / "big_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'big_sweep.json'}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

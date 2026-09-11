"""
Multi-seed experiment for FMTS 2026 paper.
Runs 4 systems × 3 methods × N seeds for Table 1 + ablation.
Also runs φ-sweep × N seeds for Figure 1.

Outputs:
  Paper/multiseed_table.json   — mean±std for each system×method
  Paper/multiseed_phi.json     — mean±std for φ-sweep
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
PHIS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

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

class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
        self.A = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))
    def predict(self, z): return self.A(z)

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

def train_koopman(model, Xt, phi):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i+BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_lin = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_lin
            opt.zero_grad(); loss.backward(); opt.step()

# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw)-nd+1+i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr+n_te]
    gt = raw[n_tr:n_tr+n_te, :gt_dim]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, gt, mu, sig

def make_systems():
    systems = {}

    # 1. Coupled Harmonic (6D)
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1,v1,x2,v2,x3,v3 = s
        return [v1, -kw*x1-k12*(x1-x2), v2, -k12*(x2-x1)-k23*(x2-x3),
                v3, -k23*(x3-x2)-kw*x3]
    sol = solve_ivp(osc, [0,500], [1,0,0,0.5,-0.5,0],
                    t_eval=np.arange(0,500,0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=6, gt_dim=6, fcst_steps=500, j=8, k=4, h=64)

    # 2. Linear 5D
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],
                    [0,0,0,w2,0],[0,0,-w2,0,0],
                    [0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1,0,0.5,0.5,1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=5, gt_dim=5, fcst_steps=500, j=8, k=4, h=64)

    # 3. Brusselator (2D + 5 delays = 10D)
    A_br, B_br = 1.0, 3.0
    sol = solve_ivp(lambda t,s: [A_br-(B_br+1)*s[0]+s[0]**2*s[1],
                                  B_br*s[0]-s[0]**2*s[1]],
                    [0,200], [1.0,1.0], t_eval=np.arange(0,200,0.01),
                    rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    obs = delay(raw, 5)
    trn, ten, gt, mu, sig = norm_split(obs, raw, 2, 3000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

    # 4. Duffing (2D + 5 delays = 10D)
    de, al, be, ga, om = 0.3, -1, 1, 0.37, 1.2
    sol = solve_ivp(lambda t,s: [s[1], -de*s[1]-al*s[0]-be*s[0]**3+ga*np.cos(om*t)],
                    [0,600], [0.5,0], t_eval=np.arange(0,600,0.05),
                    rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    obs = delay(raw, 5)
    trn, ten, gt, mu, sig = norm_split(obs, raw, 2, 3000, 500)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

    return systems

# ═══════════════════════════════════════════════════════════════════
#  Evaluate
# ═══════════════════════════════════════════════════════════════════

def eval_metrics(rfn, ffn, test_n, gt_test, mu, sig, gt_dim, fcst_steps):
    rec = rfn(test_n)
    rp = (rec * sig + mu)[:, :gt_dim]
    fc = ffn()
    fp = (fc * sig + mu)[:, :gt_dim]
    N = min(fcst_steps, len(fp), len(gt_test))
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2)))
    rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N])**2)))
    return rr, rf

# ═══════════════════════════════════════════════════════════════════
#  Run one system, one seed, all 3 methods
# ═══════════════════════════════════════════════════════════════════

def run_table(sname, cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    last_train = train_n[-1]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)
    res = {}

    # ── AE+DMD ──
    torch.manual_seed(seed); np.random.seed(seed)
    ae = AE(n_obs, k, h); train_ae(ae, Xt); ae.eval()
    with torch.no_grad(): Z = ae.encode(Xt).numpy()
    A = fit_dmd(Z)
    def _r(tn, m=ae):
        with torch.no_grad(): return m(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f(m=ae, A_=A):
        with torch.no_grad():
            z0 = m.encode(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
        out = np.empty((fcst_steps,k)); out[0]=z0
        for t in range(1,fcst_steps): out[t]=A_@out[t-1]
        with torch.no_grad(): return m.decode(torch.tensor(out,dtype=torch.float32)).numpy()
    rr, rf = eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["AE+DMD"] = {"rmse_r": rr, "rmse_f": rf}

    # ── Resid+add ──
    SeedAll(seed)
    r1 = ResidualModel(n_obs, j, k, h=h)
    train_teacher(r1, Xt); r1.eval()
    with torch.no_grad(): C1 = r1.carrier(Xt)
    dC1 = C1[1:] - C1[:-1]
    train_resid(r1, dC1); r1.eval()
    with torch.no_grad(): B1 = r1.f(dC1).numpy()
    A1 = fit_dmd(B1)
    def _r1(tn, m=r1):
        with torch.no_grad(): return m.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f1(m=r1, A_=A1):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None],dtype=torch.float32)).numpy().ravel()
            Cc = m.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
            b0 = m.f(torch.tensor(Cc-Cp,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        fc = np.empty((fcst_steps,n_obs)); C_=Cc.copy(); b=b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(torch.tensor(C_,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1,fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dc = m.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C_ = C_ + dc
            with torch.no_grad():
                fc[t] = m.dec(torch.tensor(C_,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    rr, rf = eval_metrics(_r1, _f1, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+add"] = {"rmse_r": rr, "rmse_f": rf}

    # ── Resid+GRU ──
    SeedAll(seed)
    r2 = ResidualGRU(n_obs, j, k, h=h)
    train_teacher(r2, Xt); r2.eval()
    with torch.no_grad(): C2 = r2.carrier(Xt)
    C2c, C2n = C2[:-1], C2[1:]
    dC2 = C2n - C2c
    train_resid(r2, dC2)
    train_gru(r2, dC2, C2c, C2n); r2.eval()
    with torch.no_grad(): B2 = r2.f(dC2).numpy()
    A2 = fit_dmd(B2)
    def _r2(tn, m=r2):
        with torch.no_grad(): return m.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f2(m=r2, A_=A2):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = m.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = m.f(Cc-Cp).numpy().ravel()
        fc = np.empty((fcst_steps,n_obs))
        C_ = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1,fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dh = m.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc
    rr, rf = eval_metrics(_r2, _f2, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    return res

# ═══════════════════════════════════════════════════════════════════
#  Run φ-sweep for one system, one seed
# ═══════════════════════════════════════════════════════════════════

def run_phi(sname, cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)
    res = {}

    for phi in PHIS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = KoopmanAE(n_obs, k, h)
        train_koopman(model, Xt, phi); model.eval()
        with torch.no_grad(): Z = model.encode(Xt).numpy()
        A_dmd = fit_dmd(Z)
        def _r(tn, m=model):
            with torch.no_grad():
                return m(torch.tensor(tn, dtype=torch.float32)).numpy()
        def _f(m=model, A_=A_dmd):
            with torch.no_grad():
                z0 = m.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
            out = np.empty((fcst_steps, k)); out[0] = z0
            for t in range(1, fcst_steps): out[t] = A_ @ out[t-1]
            with torch.no_grad():
                return m.decode(torch.tensor(out, dtype=torch.float32)).numpy()
        rr, rf = eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
        res[str(phi)] = {"rmse_r": rr, "rmse_f": rf}

    return res

# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()

    # ── Part 1: Table results (4 systems × 3 methods × N seeds) ──
    print("=" * 60)
    print("  PART 1: Multi-seed table results")
    print("=" * 60)
    table_raw = {}  # {system: {method: [{rmse_r, rmse_f}, ...]}}
    for sname, cfg in systems.items():
        table_raw[sname] = {"AE+DMD": [], "Resid+add": [], "Resid+GRU": []}
        for seed in SEEDS:
            print(f"  {sname} seed={seed} ... ", end="", flush=True)
            r = run_table(sname, cfg, seed)
            for method in r:
                table_raw[sname][method].append(r[method])
            print(f"AE r={r['AE+DMD']['rmse_r']:.4f}/f={r['AE+DMD']['rmse_f']:.3f}  "
                  f"add r={r['Resid+add']['rmse_r']:.4f}/f={r['Resid+add']['rmse_f']:.3f}  "
                  f"GRU r={r['Resid+GRU']['rmse_r']:.4f}/f={r['Resid+GRU']['rmse_f']:.3f}")

    # Compute mean ± std
    table_stats = {}
    for sname in table_raw:
        table_stats[sname] = {}
        for method in table_raw[sname]:
            rs = [x["rmse_r"] for x in table_raw[sname][method]]
            fs = [x["rmse_f"] for x in table_raw[sname][method]]
            table_stats[sname][method] = {
                "rmse_r_mean": float(np.mean(rs)),
                "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)),
                "rmse_f_std": float(np.std(fs)),
                "rmse_r_all": rs,
                "rmse_f_all": fs,
            }

    # ── Part 2: φ-sweep (3 systems × 8 phis × N seeds) ──
    print("\n" + "=" * 60)
    print("  PART 2: Multi-seed φ-sweep")
    print("=" * 60)
    phi_systems = ["Coupled Harmonic", "Brusselator", "Linear 5D"]
    phi_raw = {}  # {system: {phi: [{rmse_r, rmse_f}, ...]}}
    for sname in phi_systems:
        cfg = systems[sname]
        phi_raw[sname] = {str(p): [] for p in PHIS}
        for seed in SEEDS:
            print(f"  {sname} φ-sweep seed={seed} ... ", end="", flush=True)
            r = run_phi(sname, cfg, seed)
            for p in r:
                phi_raw[sname][p].append(r[p])
            print("done")

    # Compute mean ± std for phi sweep
    phi_stats = {}
    for sname in phi_raw:
        phi_stats[sname] = {"phi_sweep": {}, "ours": table_stats[sname]["Resid+GRU"]}
        for p in phi_raw[sname]:
            rs = [x["rmse_r"] for x in phi_raw[sname][p]]
            fs = [x["rmse_f"] for x in phi_raw[sname][p]]
            phi_stats[sname]["phi_sweep"][p] = {
                "rmse_r_mean": float(np.mean(rs)),
                "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)),
                "rmse_f_std": float(np.std(fs)),
            }

    # ── Save ──
    with open(OUT / "multiseed_table.json", "w") as f:
        json.dump(table_stats, f, indent=2)
    print(f"\n→ {OUT / 'multiseed_table.json'}")

    with open(OUT / "multiseed_phi.json", "w") as f:
        json.dump(phi_stats, f, indent=2)
    print(f"→ {OUT / 'multiseed_phi.json'}")

    # ── Summary ──
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("\nTable summary (mean ± std):")
    for sname in table_stats:
        print(f"\n  {sname}:")
        for method in ["AE+DMD", "Resid+add", "Resid+GRU"]:
            s = table_stats[sname][method]
            print(f"    {method:12s}  r={s['rmse_r_mean']:.4f}±{s['rmse_r_std']:.4f}  "
                  f"f={s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")
    print("\nDone.")

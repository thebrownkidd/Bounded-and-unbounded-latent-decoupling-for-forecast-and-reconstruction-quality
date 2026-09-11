"""
Multi-seed Lorenz-96 experiment for FMTS 2026.
Same architecture as the other systems but j=12, k=10, 20D native.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]

# ── Models (same as run_multiseed.py) ──

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

# ── Training ──

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

# ── Lorenz-96 data ──

def make_lorenz96():
    N_L96, F_L96 = 20, 8.0; dt = 0.05
    def l96(t, x):
        d = np.empty_like(x)
        for i in range(len(x)):
            d[i] = (x[(i+1)%N_L96] - x[(i-2)%N_L96]) * x[(i-1)%N_L96] - x[i] + F_L96
        return d
    x0 = F_L96 * np.ones(N_L96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 500], x0,
                    t_eval=np.arange(0, 500, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    n_tr, n_te = 4000, 500
    tr, te = raw[:n_tr], raw[n_tr:n_tr+n_te]
    gt = raw[n_tr:n_tr+n_te]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(
        train_n=(tr-mu)/sig, test_n=(te-mu)/sig, gt_test=gt, mu=mu, sig=sig,
        n_obs=20, gt_dim=20, fcst_steps=500, j=12, k=10, h=64)

# ── Eval ──

def eval_metrics(rfn, ffn, test_n, gt_test, mu, sig, gt_dim, fcst_steps):
    rec = rfn(test_n)
    rp = (rec * sig + mu)[:, :gt_dim]
    fc = ffn()
    fp = (fc * sig + mu)[:, :gt_dim]
    N = min(fcst_steps, len(fp), len(gt_test))
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2)))
    rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N])**2)))
    return rr, rf

# ── Run ──

if __name__ == "__main__":
    cfg = make_lorenz96()
    t0 = time.time()
    results = {"AE+DMD": [], "Resid+add": [], "Resid+GRU": []}
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    last_train = train_n[-1]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]

    for seed in SEEDS:
        print(f"  Lorenz-96 seed={seed} ... ", end="", flush=True)
        Xt = torch.tensor(train_n, dtype=torch.float32)

        # AE+DMD
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
        results["AE+DMD"].append({"rmse_r": rr, "rmse_f": rf})

        # Resid+add
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
        results["Resid+add"].append({"rmse_r": rr, "rmse_f": rf})

        # Resid+GRU
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
                b = A2 @ b
                with torch.no_grad():
                    dh = m.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                    C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                    fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
            return fc
        rr, rf = eval_metrics(_r2, _f2, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
        results["Resid+GRU"].append({"rmse_r": rr, "rmse_f": rf})

        print(f"AE r={results['AE+DMD'][-1]['rmse_r']:.3f}/f={results['AE+DMD'][-1]['rmse_f']:.3f}  "
              f"add r={results['Resid+add'][-1]['rmse_r']:.3f}/f={results['Resid+add'][-1]['rmse_f']:.3f}  "
              f"GRU r={results['Resid+GRU'][-1]['rmse_r']:.3f}/f={results['Resid+GRU'][-1]['rmse_f']:.3f}")

    # Stats
    stats = {}
    for method in results:
        rs = [x["rmse_r"] for x in results[method]]
        fs = [x["rmse_f"] for x in results[method]]
        stats[method] = {
            "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
            "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
            "rmse_r_all": rs, "rmse_f_all": fs,
        }

    with open(OUT / "lorenz96_multiseed.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ {OUT / 'lorenz96_multiseed.json'}")
    print(f"Total time: {time.time()-t0:.0f}s")
    for method in stats:
        s = stats[method]
        print(f"  {method:12s}  r={s['rmse_r_mean']:.3f}±{s['rmse_r_std']:.3f}  "
              f"f={s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")
    print("Done.")

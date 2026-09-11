"""
Multi-seed: Linear accumulation for carrier update.
C_{t+1} = W·C_t + V·m(b_t)  — learned linear matrices, no activations.
DMD forecast head throughout. Runs all 5 systems × 5 seeds.
Also runs Resid+MLP for multi-seed comparison.
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

# ── Models ──

class ResidualBase(nn.Module):
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

class ResidualLinear(ResidualBase):
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.W = nn.Linear(j, j, bias=False)
        self.V = nn.Linear(j, j, bias=False)
        # Init W near identity, V near identity (close to naive add)
        nn.init.eye_(self.W.weight)
        nn.init.eye_(self.V.weight)
    def accumulate(self, delta, carrier):
        return self.W(carrier) + self.V(delta)

class ResidualMLP(ResidualBase):
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.accum = nn.Sequential(
            nn.Linear(2*j, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, j))
    def accumulate(self, delta, carrier):
        return self.accum(torch.cat([delta, carrier], dim=1))

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

# ── Training ──

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

def train_accum(model, dC, Cc, Cn, accum_params):
    """Train the accumulator (linear or MLP) with everything else frozen."""
    for p in model.parameters(): p.requires_grad_(False)
    for p in accum_params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(accum_params, lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i+BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.accumulate(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()

# ── Data generators ──

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

    # 5. Lorenz-96 (20D native)
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
    trn, ten, gt, mu, sig = norm_split(raw, raw, 20, 4000, 500)
    systems["Lorenz-96"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=20, gt_dim=20, fcst_steps=500, j=12, k=10, h=64)

    return systems

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

# ── Run one system, one seed, one method ──

def run_one(model_cls, sname, cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    last_train = train_n[-1]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)

    SeedAll(seed)
    model = model_cls(n_obs, j, k, h=h)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    Cc, Cn = C[:-1], C[1:]
    dC = Cn - Cc
    train_resid(model, dC)

    # Get accumulator params
    if isinstance(model, ResidualLinear):
        accum_params = list(model.W.parameters()) + list(model.V.parameters())
    else:
        accum_params = list(model.accum.parameters())

    train_accum(model, dC, Cc, Cn, accum_params); model.eval()
    with torch.no_grad(): B = model.f(dC).numpy()
    A_dmd = fit_dmd(B)

    def _r(tn, m=model):
        with torch.no_grad(): return m.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f(m=model, A_=A_dmd):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc_ = m.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = m.f(Cc_-Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc_.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dh = m.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C_ = m.accumulate(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc
    return eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)

# ── Main ──

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()

    methods = {
        "Resid+Linear": ResidualLinear,
        "Resid+MLP": ResidualMLP,
    }

    all_results = {}  # {method: {system: {mean, std, all}}}

    for method_name, model_cls in methods.items():
        all_results[method_name] = {}
        for sname, cfg in systems.items():
            rs, fs = [], []
            for seed in SEEDS:
                print(f"  {method_name} / {sname} / seed={seed} ... ", end="", flush=True)
                rr, rf = run_one(model_cls, sname, cfg, seed)
                rs.append(rr); fs.append(rf)
                print(f"r={rr:.4f}  f={rf:.3f}")
            all_results[method_name][sname] = {
                "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
                "rmse_r_all": rs, "rmse_f_all": fs,
            }

    with open(OUT / "linear_mlp_multiseed.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {OUT / 'linear_mlp_multiseed.json'}")
    print(f"Total time: {time.time()-t0:.0f}s")

    print("\n" + "="*60)
    print("  SUMMARY (mean ± std over 5 seeds)")
    print("="*60)
    for method_name in all_results:
        print(f"\n  {method_name}:")
        for sname in all_results[method_name]:
            s = all_results[method_name][sname]
            print(f"    {sname:20s}  r={s['rmse_r_mean']:.4f}±{s['rmse_r_std']:.4f}  "
                  f"f={s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")

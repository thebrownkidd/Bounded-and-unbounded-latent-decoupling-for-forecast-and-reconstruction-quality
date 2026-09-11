"""
Multi-seed unrolled GRU (L=10) on all 5 systems.
Checks Coupled Harmonic first — if it doesn't beat the AE+DMD baseline
(f=0.532 mean over 5 seeds), skips the rest.
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
EPOCHS = 400; LR = 1e-3; BS = 128; SEED_LIST = [0, 1, 2, 42, 123]
L = 10  # unroll length

class ResidualGRU(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,j))
        self.dec = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.f = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,k))
        self.m = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,j))
        self.gru = nn.GRUCell(j, j)
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

def fit_dmd(Z):
    return Z[1:].T @ np.linalg.pinv(Z[:-1].T)

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(EPOCHS):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            x = Xt[idx[i:i+512]]
            loss = nn.functional.mse_loss(model.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()

def train_resid(model, dC):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(EPOCHS):
        model.train(); idx = torch.randperm(len(dC))
        for i in range(0, len(dC), 512):
            dc = dC[idx[i:i+512]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()

def train_gru_unrolled(model, dC_all, C_all, L_steps):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    n_windows = len(C_all) - L_steps
    with torch.no_grad():
        all_dh = model.m(model.f(dC_all))
    for ep in range(EPOCHS):
        model.train()
        starts = torch.randint(0, n_windows, (BS,))
        C_t = C_all[starts]
        loss = torch.tensor(0.0)
        for step in range(L_steps):
            dh = all_dh[starts + step]
            C_next = model.gru(dh, C_t)
            loss = loss + nn.functional.mse_loss(C_next, C_all[starts + step + 1])
            C_t = C_next
        loss = loss / L_steps
        opt.zero_grad(); loss.backward(); opt.step()

# ── Data ──

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw)-nd+1+i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr+n_te]
    gt = raw[n_tr:n_tr+n_te, :gt_dim]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, gt, mu, sig

def make_systems():
    systems = {}
    k12,k23,kw = 1.0,0.5,0.3
    sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
        s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
        t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[200:]
    trn,ten,gt,mu,sig = norm_split(raw,raw,6,4000,500)
    systems["Coupled Harmonic"] = dict(train_n=trn,test_n=ten,gt_test=gt,mu=mu,sig=sig,
        n_obs=6,gt_dim=6,fcst_steps=500,j=8,k=4,h=64)

    w1,w2 = 1.0,np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5*0.05); N5=10000
    raw5 = np.empty((N5,5)); raw5[0]=[1,0,0.5,0.5,1]
    for i in range(1,N5): raw5[i]=Ad@raw5[i-1]
    raw = raw5[200:]
    trn,ten,gt,mu,sig = norm_split(raw,raw,5,4000,500)
    systems["Linear 5D"] = dict(train_n=trn,test_n=ten,gt_test=gt,mu=mu,sig=sig,
        n_obs=5,gt_dim=5,fcst_steps=500,j=8,k=4,h=64)

    sol = solve_ivp(lambda t,s: [1.0-(3.0+1)*s[0]+s[0]**2*s[1],3.0*s[0]-s[0]**2*s[1]],
        [0,200],[1.0,1.0],t_eval=np.arange(0,200,0.01),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]; obs = delay(raw,5)
    trn,ten,gt,mu,sig = norm_split(obs,raw,2,3000,500)
    systems["Brusselator"] = dict(train_n=trn,test_n=ten,gt_test=gt,mu=mu,sig=sig,
        n_obs=10,gt_dim=2,fcst_steps=500,j=8,k=3,h=64)

    sol = solve_ivp(lambda t,s: [s[1],-0.3*s[1]+s[0]-s[0]**3+0.37*np.cos(1.2*t)],
        [0,600],[0.5,0],t_eval=np.arange(0,600,0.05),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]; obs = delay(raw,5)
    trn,ten,gt,mu,sig = norm_split(obs,raw,2,3000,500)
    systems["Duffing"] = dict(train_n=trn,test_n=ten,gt_test=gt,mu=mu,sig=sig,
        n_obs=10,gt_dim=2,fcst_steps=500,j=8,k=3,h=64)

    N_L96,F_L96 = 20,8.0
    def l96(t,x):
        d=np.empty_like(x)
        for i in range(len(x)):
            d[i]=(x[(i+1)%N_L96]-x[(i-2)%N_L96])*x[(i-1)%N_L96]-x[i]+F_L96
        return d
    x0=F_L96*np.ones(N_L96); x0[0]+=0.01
    sol = solve_ivp(l96,[0,500],x0,t_eval=np.arange(0,500,0.05),method="RK45",rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]
    trn,ten,gt,mu,sig = norm_split(raw,raw,20,4000,500)
    systems["Lorenz-96"] = dict(train_n=trn,test_n=ten,gt_test=gt,mu=mu,sig=sig,
        n_obs=20,gt_dim=20,fcst_steps=500,j=12,k=10,h=64)

    return systems

def run_one(cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)

    SeedAll(seed)
    model = ResidualGRU(n_obs, j, k, h=h)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    dC = C[1:] - C[:-1]
    train_resid(model, dC); model.eval()
    train_gru_unrolled(model, dC, C, L); model.eval()
    with torch.no_grad(): B = model.f(dC).numpy()
    A_dmd = fit_dmd(B)

    with torch.no_grad():
        Cp = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
        Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
        b0 = model.f(Cc0 - Cp).numpy().ravel()
    fc = np.empty((fcst_steps, n_obs)); C_ = Cc0.squeeze(0); b = b0.copy()
    with torch.no_grad(): fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
    for t in range(1, fcst_steps):
        b = A_dmd @ b
        with torch.no_grad():
            dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
            C_ = model.gru(dh, C_.unsqueeze(0)).squeeze(0)
            fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
    fp = (fc * sig + mu)[:, :gt_dim]
    N = min(fcst_steps, len(gt_test))
    rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N]) ** 2)))
    rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
    rp = (rec * sig + mu)[:, :gt_dim]
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test) ** 2)))
    return rr, rf

# ── Main ──

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    # AE+DMD baseline means for gate check
    AE_FCST_MEAN = {"Coupled Harmonic": 0.532}

    results = {}
    order = ["Coupled Harmonic", "Linear 5D", "Brusselator", "Duffing", "Lorenz-96"]

    # Run Coupled Harmonic first
    sname = "Coupled Harmonic"
    cfg = systems[sname]
    rs, fs = [], []
    for seed in SEED_LIST:
        print(f"  {sname} seed={seed} ... ", end="", flush=True)
        rr, rf = run_one(cfg, seed)
        rs.append(rr); fs.append(rf)
        print(f"r={rr:.4f}  f={rf:.4f}")
    results[sname] = {
        "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
        "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
        "rmse_r_all": rs, "rmse_f_all": fs,
    }
    ch_f = float(np.mean(fs))
    print(f"\n  Coupled Harmonic mean forecast: {ch_f:.4f}  (AE+DMD baseline: {AE_FCST_MEAN[sname]:.4f})")

    if ch_f >= AE_FCST_MEAN[sname]:
        print(f"  *** GATE CHECK FAILED: {ch_f:.4f} >= {AE_FCST_MEAN[sname]:.4f}. Stopping. ***")
    else:
        print(f"  *** GATE CHECK PASSED: {ch_f:.4f} < {AE_FCST_MEAN[sname]:.4f}. Running remaining systems. ***\n")
        for sname in order[1:]:
            cfg = systems[sname]
            rs, fs = [], []
            for seed in SEED_LIST:
                print(f"  {sname} seed={seed} ... ", end="", flush=True)
                rr, rf = run_one(cfg, seed)
                rs.append(rr); fs.append(rf)
                print(f"r={rr:.4f}  f={rf:.4f}")
            results[sname] = {
                "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
                "rmse_r_all": rs, "rmse_f_all": fs,
            }

    with open(OUT / "unrolled_gru_multiseed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ {OUT / 'unrolled_gru_multiseed.json'}")
    print(f"Total time: {time.time()-t0:.0f}s")

    print("\n" + "="*60)
    print("  SUMMARY: Unrolled GRU (L=10, mean±std)")
    print("="*60)
    for sname in results:
        s = results[sname]
        print(f"  {sname:20s}  r={s['rmse_r_mean']:.4f}±{s['rmse_r_std']:.4f}  "
              f"f={s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")

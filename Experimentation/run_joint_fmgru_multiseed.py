"""
Joint f+m+GRU multi-seed on all 5 systems.
Phase 1: teacher AE → Phase 2: f+m warmup → Phase 3: GRU warmup → Phase 4: joint f+m+GRU unrolled (L=10)
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
EPOCHS = 400; LR = 1e-3; BS = 128; L = 10
SEED_LIST = [0, 1, 2, 42, 123]

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

def make_systems():
    systems = {}
    # Coupled Harmonic
    k12,k23,kw = 1.0,0.5,0.3
    sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
        s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
        t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[200:]
    tr,te = raw[:4000],raw[4000:4500]
    mu,sig = tr.mean(0),tr.std(0)+1e-8
    systems["Coupled Harmonic"] = dict(train_n=(tr-mu)/sig,test_n=(te-mu)/sig,gt_test=te,
        mu=mu,sig=sig,n_obs=6,gt_dim=6,fcst_steps=500,j=8,k=4,h=64)

    # Linear 5D
    w1,w2 = 1.0,np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5*0.05); N5=10000
    raw5 = np.empty((N5,5)); raw5[0]=[1,0,0.5,0.5,1]
    for i in range(1,N5): raw5[i]=Ad@raw5[i-1]
    raw = raw5[200:]
    tr,te = raw[:4000],raw[4000:4500]
    mu,sig = tr.mean(0),tr.std(0)+1e-8
    systems["Linear 5D"] = dict(train_n=(tr-mu)/sig,test_n=(te-mu)/sig,gt_test=te,
        mu=mu,sig=sig,n_obs=5,gt_dim=5,fcst_steps=500,j=8,k=4,h=64)

    # Brusselator
    sol = solve_ivp(lambda t,s: [1.0-(3.0+1)*s[0]+s[0]**2*s[1],3.0*s[0]-s[0]**2*s[1]],
        [0,200],[1.0,1.0],t_eval=np.arange(0,200,0.01),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]
    obs = np.concatenate([raw[i:len(raw)-4+i] for i in range(5)], axis=1)
    tr,te = obs[:3000],obs[3000:3500]
    gt = raw[3000:3500,:2]
    mu,sig = tr.mean(0),tr.std(0)+1e-8
    systems["Brusselator"] = dict(train_n=(tr-mu)/sig,test_n=(te-mu)/sig,gt_test=gt,
        mu=mu,sig=sig,n_obs=10,gt_dim=2,fcst_steps=500,j=8,k=3,h=64)

    # Duffing
    sol = solve_ivp(lambda t,s: [s[1],-0.3*s[1]+s[0]-s[0]**3+0.37*np.cos(1.2*t)],
        [0,600],[0.5,0],t_eval=np.arange(0,600,0.05),rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]
    obs = np.concatenate([raw[i:len(raw)-4+i] for i in range(5)], axis=1)
    tr,te = obs[:3000],obs[3000:3500]
    gt = raw[3000:3500,:2]
    mu,sig = tr.mean(0),tr.std(0)+1e-8
    systems["Duffing"] = dict(train_n=(tr-mu)/sig,test_n=(te-mu)/sig,gt_test=gt,
        mu=mu,sig=sig,n_obs=10,gt_dim=2,fcst_steps=500,j=8,k=3,h=64)

    # Lorenz-96
    N_L96,F_L96 = 20,8.0
    def l96(t,x):
        d=np.empty_like(x)
        for i in range(len(x)):
            d[i]=(x[(i+1)%N_L96]-x[(i-2)%N_L96])*x[(i-1)%N_L96]-x[i]+F_L96
        return d
    x0=F_L96*np.ones(N_L96); x0[0]+=0.01
    sol = solve_ivp(l96,[0,500],x0,t_eval=np.arange(0,500,0.05),method="RK45",rtol=1e-10,atol=1e-10)
    raw = sol.y.T[2000:]
    tr,te = raw[:4000],raw[4000:4500]
    mu,sig = tr.mean(0),tr.std(0)+1e-8
    systems["Lorenz-96"] = dict(train_n=(tr-mu)/sig,test_n=(te-mu)/sig,gt_test=te,
        mu=mu,sig=sig,n_obs=20,gt_dim=20,fcst_steps=500,j=12,k=10,h=64)

    return systems

def run_one(cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)

    SeedAll(seed)
    model = ResidualGRU(n_obs, j, k, h=h)

    # Phase 1: teacher AE
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
    model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    dC = C[1:] - C[:-1]

    # Phase 2: f+m warmup
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

    # Phase 3: GRU warmup (1-step)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    Cc, Cn = C[:-1], C[1:]
    for ep in range(EPOCHS):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), 512):
            sl = idx[i:i+512]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 4: joint f+m+GRU unrolled
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    for p in model.gru.parameters(): p.requires_grad_(True)
    joint_params = list(model.f.parameters()) + list(model.m.parameters()) + list(model.gru.parameters())
    opt = torch.optim.Adam(joint_params, lr=LR * 0.1)
    n_windows = len(C) - L
    for ep in range(EPOCHS):
        model.train()
        starts = torch.randint(0, n_windows, (BS,))
        C_t = C[starts].detach()
        loss = torch.tensor(0.0)
        for step in range(L):
            true_dc = dC[starts + step].detach()
            dh = model.m(model.f(true_dc))
            C_next = model.gru(dh, C_t)
            target = C[starts + step + 1].detach()
            loss = loss + nn.functional.mse_loss(C_next, target)
            C_t = C_next
        loss = loss / L
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # DMD + forecast
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
    rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N])**2)))
    rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
    rp = (rec * sig + mu)[:, :gt_dim]
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2)))
    return rr, rf

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    order = ["Coupled Harmonic", "Linear 5D", "Brusselator", "Duffing", "Lorenz-96"]
    results = {}

    for sname in order:
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
        print()

    with open(OUT / "joint_fmgru_multiseed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {OUT / 'joint_fmgru_multiseed.json'}")
    print(f"Total time: {time.time()-t0:.0f}s")

    print("\n" + "="*60)
    print("  SUMMARY: Joint f+m+GRU (L=10, mean+-std)")
    print("="*60)
    for sname in order:
        s = results[sname]
        print(f"  {sname:20s}  r={s['rmse_r_mean']:.4f}+-{s['rmse_r_std']:.4f}  "
              f"f={s['rmse_f_mean']:.3f}+-{s['rmse_f_std']:.3f}")

    # Compare with existing results
    print("\n  --- Comparison (existing Resid+GRU from multiseed_table.json) ---")
    try:
        with open(OUT / "multiseed_table.json") as f:
            old = json.load(f)
        for sname in ["Coupled Harmonic", "Linear 5D", "Brusselator", "Duffing"]:
            if sname in old and "resid_gru" in old[sname]:
                og = old[sname]["resid_gru"]
                nw = results[sname]
                print(f"  {sname:20s}  old_f={og['rmse_f_mean']:.3f}  new_f={nw['rmse_f_mean']:.3f}  "
                      f"{'BETTER' if nw['rmse_f_mean'] < og['rmse_f_mean'] else 'WORSE'}")
    except Exception:
        pass
    try:
        with open(OUT / "lorenz96_multiseed.json") as f:
            l96 = json.load(f)
        if "resid_gru" in l96:
            og = l96["resid_gru"]
            nw = results["Lorenz-96"]
            print(f"  {'Lorenz-96':20s}  old_f={og['rmse_f_mean']:.3f}  new_f={nw['rmse_f_mean']:.3f}  "
                  f"{'BETTER' if nw['rmse_f_mean'] < og['rmse_f_mean'] else 'WORSE'}")
    except Exception:
        pass

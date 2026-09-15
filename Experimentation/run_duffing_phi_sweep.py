"""
φ-sweep for Duffing only, multi-seed. Appends to multiseed_phi.json.
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
PHIS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.A = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))
    def predict(self, z): return self.A(z)

def fit_dmd(Z):
    return Z[1:].T @ np.linalg.pinv(Z[:-1].T)

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw)-nd+1+i] for i in range(nd)], axis=1)

# Duffing data
sol = solve_ivp(lambda t,s: [s[1],-0.3*s[1]+s[0]-s[0]**3+0.37*np.cos(1.2*t)],
    [0,600],[0.5,0],t_eval=np.arange(0,600,0.05),rtol=1e-10,atol=1e-10)
raw = sol.y.T[2000:]
obs = delay(raw, 5)
tr,te = obs[:3000],obs[3000:3500]
gt = raw[3000:3500,:2]
mu,sig = tr.mean(0),tr.std(0)+1e-8
train_n,test_n,gt_test = (tr-mu)/sig,(te-mu)/sig,gt
n_obs,gt_dim,k,h,fcst_steps = 10,2,3,64,500
Xt = torch.tensor(train_n, dtype=torch.float32)

def run_koopman_seed(phi, seed):
    SeedAll(seed)
    model = KoopmanAE(n_obs, k, h)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(EPOCHS):
        model.train(); idx = torch.randperm(len(Xt)-1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i+BS]
            x_t, x_tp1 = Xt[sl], Xt[sl+1]
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            L_lin = nn.functional.mse_loss(model.predict(z_t), model.encode(x_tp1).detach())
            loss = (1-phi)*L_rec + phi*L_lin
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): Z = model.encode(Xt).numpy()
    A_dmd = fit_dmd(Z)
    # Recon
    with torch.no_grad():
        rec = model(torch.tensor(test_n, dtype=torch.float32)).numpy()
    rp = (rec * sig + mu)[:,:gt_dim]
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2)))
    # Forecast
    with torch.no_grad():
        z0 = model.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
    zs = np.empty((fcst_steps, k)); zs[0] = z0
    for t in range(1,fcst_steps): zs[t] = A_dmd @ zs[t-1]
    with torch.no_grad():
        fc = model.decode(torch.tensor(zs, dtype=torch.float32)).numpy()
    fp = (fc * sig + mu)[:,:gt_dim]
    rf = float(np.sqrt(np.mean((fp[:len(gt_test)] - gt_test)**2)))
    return rr, rf

t0 = time.time()
results = {"phi_sweep": {}}

for phi in PHIS:
    rs, fs = [], []
    for seed in SEEDS:
        print(f"  Duffing phi={phi:.2f} seed={seed} ... ", end="", flush=True)
        rr, rf = run_koopman_seed(phi, seed)
        rs.append(rr); fs.append(rf)
        print(f"r={rr:.4f}  f={rf:.4f}")
    results["phi_sweep"][str(phi)] = {
        "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
        "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
        "rmse_r_all": rs, "rmse_f_all": fs,
    }

# Append to multiseed_phi.json
phi_path = OUT / "multiseed_phi.json"
all_data = json.load(open(phi_path))
all_data["Duffing"] = results
with open(phi_path, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\nAppended Duffing to {phi_path}")
print(f"Total time: {time.time()-t0:.0f}s")

"""
GRU-guided add on Coupled Harmonic, seed 0.
C_{t+1} = sigma(alpha) * (C_t + m(b_t)) + (1 - sigma(alpha)) * GRU(m(b_t), C_t)
Per-dimension learned blend. Train alpha + GRU jointly.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS=400; LR=1e-3; BS=512; SEED=0

class ResidualGRUAdd(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,j))
        self.dec = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.f = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,k))
        self.m = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,j))
        self.gru = nn.GRUCell(j, j)
        # Init alpha=0 -> sigmoid(0)=0.5 -> equal blend
        self.alpha = nn.Parameter(torch.zeros(j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)
    def accumulate(self, delta, carrier):
        add_result = carrier + delta
        gru_result = self.gru(delta, carrier)
        w = torch.sigmoid(self.alpha)
        return w * add_result + (1 - w) * gru_result

def fit_dmd(Z):
    return Z[1:].T @ np.linalg.pinv(Z[:-1].T)

# Data
k12,k23,kw = 1.0,0.5,0.3
sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
    s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
    t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
raw = sol.y.T[200:]
tr,te = raw[:4000],raw[4000:4500]
mu,sig = tr.mean(0),tr.std(0)+1e-8
train_n,test_n,gt_test = (tr-mu)/sig,(te-mu)/sig,te
Xt = torch.tensor(train_n, dtype=torch.float32)

SeedAll(SEED)
model = ResidualGRUAdd(6, 8, 4, h=64)

# Phase 1: teacher
for p in model.parameters(): p.requires_grad_(False)
for mod in [model.enc, model.dec]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    model.train(); idx = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BS):
        x = Xt[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.recon(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad(): C = model.carrier(Xt)
Cc, Cn = C[:-1], C[1:]
dC = Cn - Cc

# Phase 2: f+m
for p in model.parameters(): p.requires_grad_(False)
for mod in [model.f, model.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    model.train(); idx = torch.randperm(len(dC))
    for i in range(0, len(dC), BS):
        dc = dC[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
        opt.zero_grad(); loss.backward(); opt.step()

# Phase 3: GRU + alpha jointly
for p in model.parameters(): p.requires_grad_(False)
for p in model.gru.parameters(): p.requires_grad_(True)
model.alpha.requires_grad_(True)
accum_params = list(model.gru.parameters()) + [model.alpha]
opt = torch.optim.Adam(accum_params, lr=LR)
for ep in range(EPOCHS):
    model.train(); idx = torch.randperm(len(Cc))
    for i in range(0, len(Cc), BS):
        sl = idx[i:i+BS]
        with torch.no_grad(): dh = model.m(model.f(dC[sl]))
        loss = nn.functional.mse_loss(model.accumulate(dh, Cc[sl]), Cn[sl])
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()

# Print learned blend weights
w = torch.sigmoid(model.alpha).detach().numpy()
print(f"  Learned blend weights (add fraction per dim):")
print(f"    {np.array2string(w, precision=3)}")
print(f"    mean={w.mean():.3f}  min={w.min():.3f}  max={w.max():.3f}")

with torch.no_grad(): B = model.f(dC).numpy()
A_dmd = fit_dmd(B)

# Forecast
with torch.no_grad():
    Cp = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
    Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
    b0 = model.f(Cc0 - Cp).numpy().ravel()
fc = np.empty((500, 6)); C_ = Cc0.squeeze(0); b = b0.copy()
with torch.no_grad(): fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
for t in range(1, 500):
    b = A_dmd @ b
    with torch.no_grad():
        dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
        C_ = model.accumulate(dh, C_.unsqueeze(0)).squeeze(0)
        fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
fp = fc * sig + mu
rf = float(np.sqrt(np.mean((fp - gt_test) ** 2)))
rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
rr = float(np.sqrt(np.mean(((rec * sig + mu) - gt_test) ** 2)))
print(f"\n  GRU-guided add / Coupled Harmonic:  r={rr:.4f}  f={rf:.4f}")
print(f"  Reference: GRU f=0.610, AE+DMD f=0.641, add f=2.492")

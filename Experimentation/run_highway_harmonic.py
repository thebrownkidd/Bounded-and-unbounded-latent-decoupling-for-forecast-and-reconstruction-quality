"""Highway gate on Coupled Harmonic only, seed 0."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS=400; LR=1e-3; BS=512; SEED=0

class ResidualHighway(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,j))
        self.dec = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.f = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,k))
        self.m = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,j))
        self.gate = nn.Linear(2*j, j)
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)
    def accumulate(self, delta, carrier):
        g = torch.sigmoid(self.gate(torch.cat([delta, carrier], dim=1)))
        return g * (carrier + delta) + (1-g) * carrier

def fit_dmd(Z):
    return Z[1:].T @ np.linalg.pinv(Z[:-1].T)

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
m = ResidualHighway(6,8,4,h=64)

# Phase 1
for p in m.parameters(): p.requires_grad_(False)
for mod in [m.enc,m.dec]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    m.train(); idx=torch.randperm(len(Xt))
    for i in range(0,len(Xt),BS):
        x=Xt[idx[i:i+BS]]; loss=nn.functional.mse_loss(m.recon(x),x)
        opt.zero_grad(); loss.backward(); opt.step()
m.eval()
with torch.no_grad(): C=m.carrier(Xt)
Cc,Cn = C[:-1],C[1:]
dC = Cn-Cc

# Phase 2
for p in m.parameters(): p.requires_grad_(False)
for mod in [m.f,m.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    m.train(); idx=torch.randperm(len(dC))
    for i in range(0,len(dC),BS):
        dc=dC[idx[i:i+BS]]; loss=nn.functional.mse_loss(m.m(m.f(dc)),dc)
        opt.zero_grad(); loss.backward(); opt.step()

# Phase 3: gate
for p in m.parameters(): p.requires_grad_(False)
for p in m.gate.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam(m.gate.parameters(), lr=LR)
for ep in range(EPOCHS):
    m.train(); idx=torch.randperm(len(Cc))
    for i in range(0,len(Cc),BS):
        sl=idx[i:i+BS]
        with torch.no_grad(): dh=m.m(m.f(dC[sl]))
        loss=nn.functional.mse_loss(m.accumulate(dh,Cc[sl]),Cn[sl])
        opt.zero_grad(); loss.backward(); opt.step()
m.eval()

with torch.no_grad(): B=m.f(dC).numpy()
A_dmd = fit_dmd(B)

# Forecast
with torch.no_grad():
    Cp=m.carrier(torch.tensor(train_n[-1:],dtype=torch.float32))
    Cc0=m.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
    b0=m.f(Cc0-Cp).numpy().ravel()
fc=np.empty((500,6)); C_=Cc0.squeeze(0); b=b0.copy()
with torch.no_grad(): fc[0]=m.dec(C_.unsqueeze(0)).numpy().ravel()
for t in range(1,500):
    b=A_dmd@b
    with torch.no_grad():
        dh=m.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
        C_=m.accumulate(dh,C_.unsqueeze(0)).squeeze(0)
        fc[t]=m.dec(C_.unsqueeze(0)).numpy().ravel()
fp=fc*sig+mu
rf=float(np.sqrt(np.mean((fp-gt_test)**2)))
rec=m.recon(torch.tensor(test_n,dtype=torch.float32)).detach().numpy()
rr=float(np.sqrt(np.mean(((rec*sig+mu)-gt_test)**2)))
print(f"Highway / Coupled Harmonic:  r={rr:.4f}  f={rf:.4f}")
print(f"Reference: GRU f=0.610, AE+DMD f=0.641")

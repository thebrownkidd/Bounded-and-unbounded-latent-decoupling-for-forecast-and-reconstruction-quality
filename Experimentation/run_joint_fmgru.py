"""
Joint fine-tuning of f + m + GRU with multi-step unrolling on Coupled Harmonic, seed 0.
Phase 1: teacher AE (frozen after)
Phase 2: f+m standard training (warmup)
Phase 3: GRU standard training (warmup)
Phase 4: f+m+GRU joint unrolled training (L=10)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS=400; LR=1e-3; BS=128; SEED=0; L=10

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
model = ResidualGRU(6, 8, 4, h=64)

# ── Phase 1: teacher AE ──
print("Phase 1: Teacher AE")
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
rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
rr_ae = float(np.sqrt(np.mean(((rec*sig+mu) - gt_test)**2)))
print(f"  AE recon RMSE: {rr_ae:.4f}")

# ── Phase 2: f+m warmup ──
print("Phase 2: f+m warmup")
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
model.eval()
with torch.no_grad():
    fm_err = nn.functional.mse_loss(model.m(model.f(dC)), dC).sqrt().item()
print(f"  f/m roundtrip RMSE: {fm_err:.6f}")

# ── Phase 3: GRU warmup (standard 1-step) ──
print("Phase 3: GRU warmup (1-step)")
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
model.eval()

# Eval before joint training
with torch.no_grad():
    dh_all = model.m(model.f(dC))
    gru_onestep = nn.functional.mse_loss(model.gru(dh_all, Cc), Cn).sqrt().item()
print(f"  GRU one-step RMSE: {gru_onestep:.6f}")

# Quick forecast before joint training
with torch.no_grad(): B_pre = model.f(dC).numpy()
A_pre = fit_dmd(B_pre)
with torch.no_grad():
    Cp = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
    Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
    b0 = model.f(Cc0 - Cp).numpy().ravel()
fc = np.empty((500, 6)); C_ = Cc0.squeeze(0); b = b0.copy()
with torch.no_grad(): fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
for t in range(1, 500):
    b = A_pre @ b
    with torch.no_grad():
        dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
        C_ = model.gru(dh, C_.unsqueeze(0)).squeeze(0)
        fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
fp = fc * sig + mu
rf_pre = float(np.sqrt(np.mean((fp - gt_test)**2)))
print(f"  Pre-joint forecast RMSE: {rf_pre:.4f}")

# ── Phase 4: Joint f+m+GRU unrolled training ──
print(f"Phase 4: Joint f+m+GRU unrolled (L={L})")
for p in model.parameters(): p.requires_grad_(False)
for mod in [model.f, model.m]:
    for p in mod.parameters(): p.requires_grad_(True)
for p in model.gru.parameters(): p.requires_grad_(True)

joint_params = list(model.f.parameters()) + list(model.m.parameters()) + list(model.gru.parameters())
opt = torch.optim.Adam(joint_params, lr=LR * 0.1)  # lower LR for fine-tuning

n_windows = len(C) - L
for ep in range(EPOCHS):
    model.train()
    starts = torch.randint(0, n_windows, (BS,))
    C_t = C[starts].detach()  # detach from encoder graph
    loss = torch.tensor(0.0)
    for step in range(L):
        # Compute delta through f/m (trainable now)
        true_dc = dC[starts + step].detach()
        dh = model.m(model.f(true_dc))
        C_next = model.gru(dh, C_t)
        target = C[starts + step + 1].detach()
        loss = loss + nn.functional.mse_loss(C_next, target)
        C_t = C_next
    loss = loss / L
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()

# ── Final evaluation ──
print("\n--- After joint training ---")
with torch.no_grad():
    fm_err2 = nn.functional.mse_loss(model.m(model.f(dC)), dC).sqrt().item()
    dh_all2 = model.m(model.f(dC))
    gru_onestep2 = nn.functional.mse_loss(model.gru(dh_all2, Cc), Cn).sqrt().item()
    add_onestep2 = nn.functional.mse_loss(Cc + dh_all2, Cn).sqrt().item()
print(f"  f/m roundtrip RMSE: {fm_err2:.6f} (was {fm_err:.6f})")
print(f"  GRU one-step RMSE: {gru_onestep2:.6f} (was {gru_onestep:.6f})")
print(f"  add one-step RMSE: {add_onestep2:.6f}")

# DMD + forecast
with torch.no_grad(): B = model.f(dC).numpy()
A_dmd = fit_dmd(B)

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
        C_ = model.gru(dh, C_.unsqueeze(0)).squeeze(0)
        fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
fp = fc * sig + mu
rf = float(np.sqrt(np.mean((fp - gt_test)**2)))
rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
rr = float(np.sqrt(np.mean(((rec*sig+mu) - gt_test)**2)))

print(f"\nJoint f+m+GRU / Coupled Harmonic:  r={rr:.4f}  f={rf:.4f}")
print(f"Reference: GRU(standard) f={rf_pre:.4f}, AE+DMD f=0.641, add f=2.492")

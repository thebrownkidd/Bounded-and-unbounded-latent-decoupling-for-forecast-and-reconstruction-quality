"""
Multi-step unrolled GRU training on Coupled Harmonic.
Batched: sample B random windows of length L, unroll in parallel.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS=400; LR=1e-3; BS=128; SEED=0

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

def train_gru_unrolled(model, dC_all, C_all, L):
    """Batched unrolled GRU: sample B windows of length L, unroll in parallel."""
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    n_windows = len(C_all) - L
    # Precompute all deltas through f/m
    with torch.no_grad():
        all_dh = model.m(model.f(dC_all))  # (T-1, j)

    for ep in range(EPOCHS):
        model.train()
        starts = torch.randint(0, n_windows, (BS,))
        # Init: (BS, j)
        C_t = C_all[starts]
        loss = torch.tensor(0.0)
        for step in range(L):
            idx = starts + step
            dh = all_dh[idx]  # (BS, j) — precomputed, detached from f/m
            C_next = model.gru(dh, C_t)
            target = C_all[starts + step + 1]
            loss = loss + nn.functional.mse_loss(C_next, target)
            C_t = C_next  # feed own output, BPTT through
        loss = loss / L
        opt.zero_grad(); loss.backward(); opt.step()

# Data
k12, k23, kw = 1.0, 0.5, 0.3
sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
    s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
    t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
raw = sol.y.T[200:]
tr,te = raw[:4000],raw[4000:4500]
mu,sig = tr.mean(0),tr.std(0)+1e-8
train_n,test_n,gt_test = (tr-mu)/sig,(te-mu)/sig,te
Xt = torch.tensor(train_n, dtype=torch.float32)

def run_with_L(L_steps):
    SeedAll(SEED)
    model = ResidualGRU(6, 8, 4, h=64)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    dC = C[1:] - C[:-1]
    train_resid(model, dC); model.eval()
    train_gru_unrolled(model, dC, C, L_steps); model.eval()
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
    rf = float(np.sqrt(np.mean((fp - gt_test) ** 2)))
    rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
    rr = float(np.sqrt(np.mean(((rec * sig + mu) - gt_test) ** 2)))
    return rr, rf

ROLLOUT_LENGTHS = [1, 5, 10, 20, 50]
print("Unrolled GRU training on Coupled Harmonic (seed 0)")
print("=" * 50)
for L in ROLLOUT_LENGTHS:
    rr, rf = run_with_L(L)
    print(f"  L={L:3d} steps:  r={rr:.4f}  f={rf:.4f}")
print(f"\n  Reference: GRU (L=1) f=0.610, AE+DMD f=0.641")

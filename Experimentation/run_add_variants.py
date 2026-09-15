"""
Four add-based accumulator variants on Coupled Harmonic, seed 0.
1. Mean reversion: C_{t+1} = C_t + m(b_t) - lambda*(C_t - C_mean)
2. PCA projection: C_{t+1} = P*(C_t + m(b_t))
3. Encoder-decoder feedback: C_{t+1} = C_t + m(b_t) + alpha*(E(D(C_t)) - C_t)
4. Multi-step trained f/m: train f,m on L-step unrolled add
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS=400; LR=1e-3; BS=512; SEED=0

# ── Base model ──

class ResidualBase(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j = j
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,j))
        self.dec = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.f = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,k))
        self.m = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,j))
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
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()

def train_resid(model, dC):
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

# ── Data ──

k12,k23,kw = 1.0,0.5,0.3
sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
    s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
    t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
raw = sol.y.T[200:]
tr,te = raw[:4000],raw[4000:4500]
mu,sig = tr.mean(0),tr.std(0)+1e-8
train_n,test_n,gt_test = (tr-mu)/sig,(te-mu)/sig,te
n_obs,k,j,h = 6,4,8,64
Xt = torch.tensor(train_n, dtype=torch.float32)

def forecast_eval(model, accumulate_fn):
    with torch.no_grad():
        Cp = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
        Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
        b0 = model.f(Cc0 - Cp).numpy().ravel()
        B = model.f(model.carrier(Xt)[1:] - model.carrier(Xt)[:-1]).numpy()
    A_dmd = fit_dmd(B)
    fc = np.empty((500, n_obs)); C_ = Cc0.squeeze(0); b = b0.copy()
    with torch.no_grad(): fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
    for t in range(1, 500):
        b = A_dmd @ b
        with torch.no_grad():
            dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
            C_ = accumulate_fn(dh, C_.unsqueeze(0)).squeeze(0)
            fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
    fp = fc * sig + mu
    rf = float(np.sqrt(np.mean((fp - gt_test) ** 2)))
    rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
    rr = float(np.sqrt(np.mean(((rec * sig + mu) - gt_test) ** 2)))
    return rr, rf

# ═══════════════════════════════════════════════════
#  1. Mean reversion
# ═══════════════════════════════════════════════════

print("1. Mean reversion")
print("-" * 40)

SeedAll(SEED)
m1 = ResidualBase(n_obs, j, k, h=h)
train_teacher(m1, Xt); m1.eval()
with torch.no_grad(): C1 = m1.carrier(Xt)
dC1 = C1[1:] - C1[:-1]
train_resid(m1, dC1); m1.eval()
C_mean = C1.mean(dim=0).detach()

# Learn lambda per dim
lam = nn.Parameter(torch.zeros(j))  # sigmoid(0) = 0.5, start moderate

opt = torch.optim.Adam([lam], lr=LR)
Cc1, Cn1 = C1[:-1], C1[1:]
for ep in range(EPOCHS):
    idx = torch.randperm(len(Cc1))
    for i in range(0, len(Cc1), BS):
        sl = idx[i:i+BS]
        with torch.no_grad(): dh = m1.m(m1.f(dC1[sl]))
        lam_val = 0.1 * torch.sigmoid(lam)  # cap at 0.1
        pred = Cc1[sl] + dh - lam_val * (Cc1[sl] - C_mean)
        loss = nn.functional.mse_loss(pred, Cn1[sl])
        opt.zero_grad(); loss.backward(); opt.step()

lam_val = (0.1 * torch.sigmoid(lam)).detach()
print(f"  Learned lambda: {lam_val.numpy()}")

def accum_mr(delta, carrier):
    return carrier + delta - lam_val * (carrier - C_mean)

rr, rf = forecast_eval(m1, accum_mr)
print(f"  r={rr:.4f}  f={rf:.4f}\n")

# ═══════════════════════════════════════════════════
#  2. PCA projection
# ═══════════════════════════════════════════════════

print("2. PCA projection")
print("-" * 40)

SeedAll(SEED)
m2 = ResidualBase(n_obs, j, k, h=h)
train_teacher(m2, Xt); m2.eval()
with torch.no_grad(): C2 = m2.carrier(Xt).numpy()
dC2_t = torch.tensor(np.diff(C2, axis=0), dtype=torch.float32)
train_resid(m2, dC2_t); m2.eval()

# PCA on carriers — keep all j components but build projector
C2_centered = C2 - C2.mean(axis=0)
U, S, Vt = np.linalg.svd(C2_centered, full_matrices=False)
# Project onto top j dims (all of them — but this re-centers and removes noise)
P = Vt.T @ Vt  # j×j projector (rank j, so it's identity... need fewer dims)
# Actually use fewer dims — keep dims that explain 99% variance
cumvar = np.cumsum(S**2) / np.sum(S**2)
n_keep = np.searchsorted(cumvar, 0.99) + 1
print(f"  Keeping {n_keep}/{j} PCA dims (99% variance)")
P = Vt[:n_keep].T @ Vt[:n_keep]  # rank-n_keep projector
P_t = torch.tensor(P, dtype=torch.float32)
C_mean2 = torch.tensor(C2.mean(axis=0), dtype=torch.float32)

def accum_pca(delta, carrier):
    c_new = carrier + delta
    # Project centered, then uncenter
    centered = c_new - C_mean2
    projected = centered @ P_t.T
    return projected + C_mean2

rr, rf = forecast_eval(m2, accum_pca)
print(f"  r={rr:.4f}  f={rf:.4f}\n")

# ═══════════════════════════════════════════════════
#  3. Encoder-decoder feedback
# ═══════════════════════════════════════════════════

print("3. Encoder-decoder feedback")
print("-" * 40)

SeedAll(SEED)
m3 = ResidualBase(n_obs, j, k, h=h)
train_teacher(m3, Xt); m3.eval()
with torch.no_grad(): C3 = m3.carrier(Xt)
dC3 = C3[1:] - C3[:-1]
train_resid(m3, dC3); m3.eval()

# Learn alpha per dim
alpha_fb = nn.Parameter(torch.zeros(j))
opt = torch.optim.Adam([alpha_fb], lr=LR)
Cc3, Cn3 = C3[:-1], C3[1:]
for ep in range(EPOCHS):
    idx = torch.randperm(len(Cc3))
    for i in range(0, len(Cc3), BS):
        sl = idx[i:i+BS]
        with torch.no_grad(): dh = m3.m(m3.f(dC3[sl]))
        c_add = Cc3[sl] + dh
        # E(D(C_t)) feedback
        with torch.no_grad():
            ed = m3.enc(m3.dec(Cc3[sl]))
        a = 0.5 * torch.sigmoid(alpha_fb)  # cap at 0.5
        pred = c_add + a * (ed - Cc3[sl])
        loss = nn.functional.mse_loss(pred, Cn3[sl])
        opt.zero_grad(); loss.backward(); opt.step()

a_val = (0.5 * torch.sigmoid(alpha_fb)).detach()
print(f"  Learned alpha: {a_val.numpy()}")

def accum_fb(delta, carrier, mdl=m3):
    c_add = carrier + delta
    with torch.no_grad():
        ed = mdl.enc(mdl.dec(carrier))
    return c_add + a_val * (ed - carrier)

rr, rf = forecast_eval(m3, accum_fb)
print(f"  r={rr:.4f}  f={rf:.4f}\n")

# ═══════════════════════════════════════════════════
#  4. Multi-step trained f/m
# ═══════════════════════════════════════════════════

print("4. Multi-step trained f/m (L=10)")
print("-" * 40)

L_FM = 10

SeedAll(SEED)
m4 = ResidualBase(n_obs, j, k, h=h)
train_teacher(m4, Xt); m4.eval()
with torch.no_grad(): C4 = m4.carrier(Xt)
dC4 = C4[1:] - C4[:-1]

# First do standard f/m pretraining (100 epochs warmup)
for p in m4.parameters(): p.requires_grad_(False)
for mod in [m4.f, m4.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in m4.parameters() if p.requires_grad], lr=LR)
for ep in range(100):
    m4.train(); idx = torch.randperm(len(dC4))
    for i in range(0, len(dC4), BS):
        dc = dC4[idx[i:i+BS]]
        loss = nn.functional.mse_loss(m4.m(m4.f(dc)), dc)
        opt.zero_grad(); loss.backward(); opt.step()

# Then multi-step unrolled f/m training
n_windows = len(C4) - L_FM
opt = torch.optim.Adam([p for p in m4.parameters() if p.requires_grad], lr=LR*0.1)
for ep in range(EPOCHS):
    m4.train()
    starts = torch.randint(0, n_windows, (BS,))
    C_t = C4[starts].detach()
    loss = torch.tensor(0.0)
    for step in range(L_FM):
        true_dc = dC4[starts + step]
        reconstructed_dc = m4.m(m4.f(true_dc))
        C_t = C_t + reconstructed_dc
        loss = loss + nn.functional.mse_loss(C_t, C4[starts + step + 1])
    loss = loss / L_FM
    opt.zero_grad(); loss.backward(); opt.step()
m4.eval()

# Check f/m roundtrip after multi-step training
with torch.no_grad():
    fm_err = nn.functional.mse_loss(m4.m(m4.f(dC4)), dC4).sqrt().item()
    dc_norm = dC4.norm() / len(dC4)**0.5
print(f"  f/m roundtrip RMSE after multi-step: {fm_err:.6f} (ratio: {fm_err/dc_norm.item():.4f})")

def accum_add(delta, carrier):
    return carrier + delta

rr, rf = forecast_eval(m4, accum_add)
print(f"  r={rr:.4f}  f={rf:.4f}\n")

print("=" * 40)
print("Reference: GRU f=0.610, AE+DMD f=0.641, add f=2.492")

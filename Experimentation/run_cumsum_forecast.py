"""
Cumulative-sum forecast on Coupled Harmonic, seed 0.
Instead of DMD on per-step residuals b_t (requiring accumulation),
train f/m on cumulative carrier displacements S_t = C_t - C_ref,
fit DMD on s_t = f(S_t), forecast s_hat_t = A^t * s_0,
recover C_t = C_ref + m(s_hat_t). No accumulation loop.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from Utils.Benchmark import SeedAll

EPOCHS = 400; LR = 1e-3; BS = 512; SEED = 0

class CumsumResidual(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,j))
        self.dec = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
        self.f = nn.Sequential(nn.Linear(j,h),nn.ELU(),nn.Linear(h,k))
        self.m = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

def fit_dmd(Z):
    return Z[1:].T @ np.linalg.pinv(Z[:-1].T)

# Data
k12, k23, kw = 1.0, 0.5, 0.3
sol = solve_ivp(lambda t,s: [s[1],-kw*s[0]-k12*(s[0]-s[2]),s[3],-k12*(s[2]-s[0])-k23*(s[2]-s[4]),
    s[5],-k23*(s[4]-s[2])-kw*s[4]], [0,500],[1,0,0,0.5,-0.5,0],
    t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
raw = sol.y.T[200:]
tr, te = raw[:4000], raw[4000:4500]
mu, sig = tr.mean(0), tr.std(0) + 1e-8
train_n, test_n, gt_test = (tr - mu) / sig, (te - mu) / sig, te
Xt = torch.tensor(train_n, dtype=torch.float32)

SeedAll(SEED)
model = CumsumResidual(6, 8, 4, h=64)

# Phase 1: teacher AE
print("Phase 1: Teacher AE")
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
rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
rr = float(np.sqrt(np.mean(((rec * sig + mu) - gt_test) ** 2)))
print(f"  AE recon RMSE: {rr:.4f}")

# Phase 2: f/m on cumulative displacements
# Reference carrier = C[0] (first training carrier)
print("Phase 2: f/m on cumulative displacements")
C_ref = C[0:1].detach()  # (1, j)
S = C - C_ref  # cumulative displacement from reference, (T, j)

for p in model.parameters(): p.requires_grad_(False)
for mod in [model.f, model.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    model.train(); idx = torch.randperm(len(S))
    for i in range(0, len(S), BS):
        s = S[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.m(model.f(s)), s)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    fm_err = nn.functional.mse_loss(model.m(model.f(S)), S).sqrt().item()
    s_norm = S.norm() / len(S)**0.5
print(f"  f/m cumsum roundtrip RMSE: {fm_err:.6f} (ratio: {fm_err/s_norm.item():.4f})")

# DMD on compressed cumulative displacements
with torch.no_grad(): s_seq = model.f(S).numpy()  # (T, k)
A_dmd = fit_dmd(s_seq)

# Check DMD quality
s_pred = s_seq[:-1] @ A_dmd.T
dmd_err = float(np.sqrt(np.mean((s_pred - s_seq[1:]) ** 2)))
print(f"  DMD one-step RMSE in s-space: {dmd_err:.6f}")

# Check spectral radius
evals = np.linalg.eigvals(A_dmd)
print(f"  DMD spectral radius: {np.max(np.abs(evals)):.6f}")
print(f"  DMD eigenvalues: {np.abs(evals)}")

# Forecast: no accumulation loop!
print("\nForecasting (no accumulation)...")
with torch.no_grad():
    C_ref_test = model.carrier(torch.tensor(train_n[0:1], dtype=torch.float32))  # same ref as training
    Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
    s0 = model.f(Cc0 - C_ref_test).numpy().ravel()

fc = np.empty((500, 6))
s = s0.copy()
with torch.no_grad():
    # Step 0
    C_0 = Cc0.squeeze(0)
    fc[0] = model.dec(C_0.unsqueeze(0)).numpy().ravel()
for t in range(1, 500):
    s = A_dmd @ s
    with torch.no_grad():
        S_hat = model.m(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
        C_hat = C_ref_test + S_hat
        fc[t] = model.dec(C_hat).numpy().ravel()

fp = fc * sig + mu
rf = float(np.sqrt(np.mean((fp - gt_test) ** 2)))
print(f"\nCumsum forecast / Coupled Harmonic:  r={rr:.4f}  f={rf:.4f}")
print(f"Reference: GRU f=0.610, AE+DMD f=0.532 (multi-seed), add f=2.492")

# Also try: reference = last training carrier (closer to test)
print("\n--- Variant: C_ref = last training carrier ---")
with torch.no_grad():
    C_ref_last = model.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
    S_last = C - C_ref_last  # retrain f/m? No, just recompute s using existing f/m

# Retrain f/m with last carrier as reference
C_ref2 = C[-1:].detach()
S2 = C - C_ref2

for p in model.parameters(): p.requires_grad_(False)
for mod in [model.f, model.m]:
    for p in mod.parameters(): p.requires_grad_(True)
# Re-init f/m
model.f = nn.Sequential(nn.Linear(8,64),nn.ELU(),nn.Linear(64,4))
model.m = nn.Sequential(nn.Linear(4,64),nn.ELU(),nn.Linear(64,8))
opt = torch.optim.Adam(list(model.f.parameters()) + list(model.m.parameters()), lr=LR)
for ep in range(EPOCHS):
    model.train(); idx = torch.randperm(len(S2))
    for i in range(0, len(S2), BS):
        s = S2[idx[i:i+BS]]
        loss = nn.functional.mse_loss(model.m(model.f(s)), s)
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    fm_err2 = nn.functional.mse_loss(model.m(model.f(S2)), S2).sqrt().item()
print(f"  f/m cumsum roundtrip RMSE (ref=last): {fm_err2:.6f}")

with torch.no_grad(): s_seq2 = model.f(S2).numpy()
A_dmd2 = fit_dmd(s_seq2)
evals2 = np.linalg.eigvals(A_dmd2)
print(f"  DMD spectral radius: {np.max(np.abs(evals2)):.6f}")

with torch.no_grad():
    Cc0_2 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
    s0_2 = model.f(Cc0_2 - C_ref2).numpy().ravel()

fc2 = np.empty((500, 6))
s = s0_2.copy()
with torch.no_grad():
    fc2[0] = model.dec(Cc0_2).numpy().ravel()
for t in range(1, 500):
    s = A_dmd2 @ s
    with torch.no_grad():
        S_hat = model.m(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
        C_hat = C_ref2 + S_hat
        fc2[t] = model.dec(C_hat).numpy().ravel()

fp2 = fc2 * sig + mu
rf2 = float(np.sqrt(np.mean((fp2 - gt_test) ** 2)))
print(f"  Cumsum (ref=last) / Coupled Harmonic:  r={rr:.4f}  f={rf2:.4f}")

# Also try: DMD on raw cumulative b (from per-step f/m), no retraining
print("\n--- Variant: cumsum of per-step b, existing f/m ---")
SeedAll(SEED)
m3 = CumsumResidual(6, 8, 4, h=64)
# Train teacher
for p in m3.parameters(): p.requires_grad_(False)
for mod in [m3.enc, m3.dec]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in m3.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    m3.train(); idx = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BS):
        x = Xt[idx[i:i+BS]]
        loss = nn.functional.mse_loss(m3.recon(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
m3.eval()
with torch.no_grad(): C3 = m3.carrier(Xt)
dC3 = C3[1:] - C3[:-1]
# Train f/m on deltas (standard)
for p in m3.parameters(): p.requires_grad_(False)
for mod in [m3.f, m3.m]:
    for p in mod.parameters(): p.requires_grad_(True)
opt = torch.optim.Adam([p for p in m3.parameters() if p.requires_grad], lr=LR)
for ep in range(EPOCHS):
    m3.train(); idx = torch.randperm(len(dC3))
    for i in range(0, len(dC3), BS):
        dc = dC3[idx[i:i+BS]]
        loss = nn.functional.mse_loss(m3.m(m3.f(dc)), dc)
        opt.zero_grad(); loss.backward(); opt.step()
m3.eval()
# Get per-step b, then cumsum
with torch.no_grad(): b_seq = m3.f(dC3).numpy()
B_cumsum = np.cumsum(b_seq, axis=0)  # (T-1, k)
A_dmd3 = fit_dmd(B_cumsum)
evals3 = np.linalg.eigvals(A_dmd3)
print(f"  DMD spectral radius on cumsum(b): {np.max(np.abs(evals3)):.6f}")

with torch.no_grad():
    Cp3 = m3.carrier(torch.tensor(train_n[-1:], dtype=torch.float32))
    Cc0_3 = m3.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
    b0_3 = m3.f(Cc0_3 - Cp3).numpy().ravel()

# B_0 for test = cumsum up to end of train + first test step
B_last_train = B_cumsum[-1]  # cumsum at end of training
B_test_0 = B_last_train + b0_3  # add first test delta

fc3 = np.empty((500, 6))
B = B_test_0.copy()
with torch.no_grad(): fc3[0] = m3.dec(Cc0_3).numpy().ravel()
for t in range(1, 500):
    B = A_dmd3 @ B
    with torch.no_grad():
        # B represents cumsum of b from start; carrier = C_0 + m(B) roughly
        # But m was trained on deltas not cumsums, so this is m applied to cumsum
        # This won't be perfect — m(cumsum(b)) != cumsum(m(b))
        # Just decode: C_ref + m maps individual deltas. Here we hack it:
        # Actually: forecast the next B, then take the delta: dB = B_new - B_old
        pass

# Cleaner: forecast b_t from B via differencing
B = B_test_0.copy()
C_ = Cc0_3.squeeze(0)
with torch.no_grad(): fc3[0] = m3.dec(C_.unsqueeze(0)).numpy().ravel()
B_prev = B.copy()
for t in range(1, 500):
    B_next = A_dmd3 @ B
    b_step = B_next - B  # recover per-step delta from cumsum forecast
    with torch.no_grad():
        dh = m3.m(torch.tensor(b_step, dtype=torch.float32).unsqueeze(0))
        C_ = C_ + dh.squeeze(0)  # add accumulation
        fc3[t] = m3.dec(C_.unsqueeze(0)).numpy().ravel()
    B = B_next

fp3 = fc3 * sig + mu
rf3 = float(np.sqrt(np.mean((fp3 - gt_test) ** 2)))
rec3 = m3.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
rr3 = float(np.sqrt(np.mean(((rec3 * sig + mu) - gt_test) ** 2)))
print(f"  Cumsum-b + diff / Coupled Harmonic:  r={rr3:.4f}  f={rf3:.4f}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Cumsum (ref=first):   f={rf:.4f}")
print(f"  Cumsum (ref=last):    f={rf2:.4f}")
print(f"  Cumsum-b + diff:      f={rf3:.4f}")
print(f"  Standard GRU:         f=0.610")
print(f"  AE+DMD baseline:      f=0.532 (multi-seed)")

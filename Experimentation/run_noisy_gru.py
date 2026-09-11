"""
Noise-injected GRU training on Coupled Harmonic only.
Add Gaussian noise to C_t during GRU training to teach robustness to drift.
Tests multiple noise scales. Seed 0.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp

from Utils.Benchmark import SeedAll

EPOCHS = 400; LR = 1e-3; BS = 512; SEED = 0

class ResidualGRU(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        self.m = nn.Sequential(nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))
        self.gru = nn.GRUCell(j, j)
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

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

def train_gru_noisy(model, dC, Cc, Cn, noise_std):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i+BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            noisy_c = Cc[sl] + noise_std * torch.randn_like(Cc[sl])
            loss = nn.functional.mse_loss(model.gru(dh, noisy_c), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()

# ── Data ──

def make_coupled_harmonic():
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1,v1,x2,v2,x3,v3 = s
        return [v1, -kw*x1-k12*(x1-x2), v2, -k12*(x2-x1)-k23*(x2-x3),
                v3, -k23*(x3-x2)-kw*x3]
    sol = solve_ivp(osc, [0,500], [1,0,0,0.5,-0.5,0],
                    t_eval=np.arange(0,500,0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    n_tr, n_te = 4000, 500
    tr, te = raw[:n_tr], raw[n_tr:n_tr+n_te]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, te, mu, sig

# ── Run ──

if __name__ == "__main__":
    train_n, test_n, gt_test, mu, sig = make_coupled_harmonic()
    n_obs, k, j, h = 6, 4, 8, 64
    fcst_steps = 500
    Xt = torch.tensor(train_n, dtype=torch.float32)
    last_train = train_n[-1]

    NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

    results = {}
    for ns in NOISE_LEVELS:
        print(f"  noise_std={ns:.2f} ... ", end="", flush=True)
        SeedAll(SEED)
        model = ResidualGRU(n_obs, j, k, h=h)
        train_teacher(model, Xt); model.eval()
        with torch.no_grad(): C = model.carrier(Xt)
        Cc, Cn = C[:-1], C[1:]
        dC = Cn - Cc
        train_resid(model, dC)
        train_gru_noisy(model, dC, Cc, Cn, ns); model.eval()
        with torch.no_grad(): B = model.f(dC).numpy()
        A_dmd = fit_dmd(B)

        # Forecast
        with torch.no_grad():
            Cp = model.carrier(torch.tensor(last_train[None], dtype=torch.float32))
            Cc0 = model.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
            b0 = model.f(Cc0 - Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc0.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_dmd @ b
            with torch.no_grad():
                dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                C_ = model.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
        fp = fc * sig + mu
        N = min(fcst_steps, len(gt_test))
        rec = model.recon(torch.tensor(test_n, dtype=torch.float32)).detach().numpy()
        rp = rec * sig + mu
        rr = float(np.sqrt(np.mean((rp - gt_test)**2)))
        rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N])**2)))
        results[f"noise_{ns}"] = {"rmse_r": rr, "rmse_f": rf, "noise_std": ns}
        print(f"r={rr:.4f}  f={rf:.4f}")

    print(f"\n  Reference: GRU (no noise) f=0.610, AE+DMD f=0.641")
    print(f"  Done.")

"""
Quick test: b-space GRU + forecast decoder on Linear 5D.

Architecture:
  - Phase 1-2: Same carrier-residual (teacher AE + residual compressor)
  - Phase 3: GRU operates in k-dim b-space (not j-dim carrier space)
  - Forecast decoder: separate MLP maps GRU hidden state -> observations
  - Reconstruction: unchanged (carrier decoder)

Compare against AE+DMD and Resid+GRU (current).
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import expm
from Utils.Benchmark import SeedAll

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2]
FCST = 500
N_OBS = 5; J = 8; K = 4; H = 64

def make_linear5d():
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N = 10000; raw = np.empty((N, 5)); raw[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N): raw[i] = Ad @ raw[i - 1]
    raw = raw[200:]
    tr, te = raw[:4000], raw[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig

class CarrierResidBase(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(nn.Linear(j, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        self.m = nn.Sequential(nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

class ForecastDecoder(nn.Module):
    def __init__(self, k, n, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
    def forward(self, z): return self.net(z)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
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
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(dC))
        for i in range(0, len(dC), BS):
            dc = dC[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()

def run_bspace_gru(seed, train_n, test_n):
    SeedAll(seed)
    Xt = torch.tensor(train_n, dtype=torch.float32)

    # Phase 1-2: teacher + residual compressor
    model = CarrierResidBase(N_OBS, J, K, H)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    dC = C[1:] - C[:-1]
    train_resid(model, dC); model.eval()
    with torch.no_grad(): B = model.f(dC)  # (T-1, k)
    B_np = B.numpy()
    A = fit_dmd(B_np)

    # Phase 3: GRU in b-space (k-dim)
    # Predicts b_{t+1} from b_t via gated recurrence
    bspace_gru = nn.GRUCell(K, K)
    Bb_curr = B[:-1]  # b_t
    Bb_next = B[1:]    # b_{t+1}

    opt = torch.optim.Adam(bspace_gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        bspace_gru.train(); idx = torch.randperm(len(Bb_curr))
        for i in range(0, len(Bb_curr), BS):
            sl = idx[i:i+BS]
            pred = bspace_gru(Bb_curr[sl], Bb_curr[sl])
            loss = nn.functional.mse_loss(pred, Bb_next[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    bspace_gru.eval()

    # Phase 4: Forecast decoder (b-space hidden state -> observations)
    fdec = ForecastDecoder(K, N_OBS, H)
    opt = torch.optim.Adam(fdec.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        fdec.train(); idx = torch.randperm(len(B))
        for i in range(0, len(B), BS):
            sl = idx[i:i+BS]
            loss = nn.functional.mse_loss(fdec(B[sl]), Xt[1:][sl])
            opt.zero_grad(); loss.backward(); opt.step()
    fdec.eval()

    # Evaluate recon
    Xte = torch.tensor(test_n, dtype=torch.float32)
    with torch.no_grad():
        pred_r = model.recon(Xte).numpy()
    rmse_r = float(np.sqrt(np.mean((pred_r - test_n) ** 2)))

    # Evaluate forecast: rollout in b-space, decode via forecast decoder
    with torch.no_grad():
        c_last = model.carrier(Xte[0:1])
        dc_last = dC[-1:]
        b = model.f(dc_last).squeeze(0)  # (k,)

    out = np.empty((FCST, N_OBS))
    h_state = b.clone()
    for t in range(FCST):
        with torch.no_grad():
            # DMD step in b-space
            b_np = A @ b.numpy()
            b = torch.tensor(b_np, dtype=torch.float32)
            # GRU refinement
            h_state = bspace_gru(b.unsqueeze(0), h_state.unsqueeze(0)).squeeze(0)
            # Decode to observations
            out[t] = fdec(h_state.unsqueeze(0)).numpy().ravel()

    rmse_f = float(np.sqrt(np.mean((out - test_n[:FCST]) ** 2)))
    return rmse_r, rmse_f


if __name__ == "__main__":
    train_n, test_n = make_linear5d()
    print(f"Linear 5D: train {train_n.shape}, test {test_n.shape}\n")

    results = []
    for seed in SEEDS:
        t0 = time.time()
        rmse_r, rmse_f = run_bspace_gru(seed, train_n, test_n)
        dt = time.time() - t0
        results.append({"rmse_r": rmse_r, "rmse_f": rmse_f})
        print(f"  seed={seed}  R={rmse_r:.5f}  F={rmse_f:.5f}  ({dt:.0f}s)")

    rs = [x["rmse_r"] for x in results]
    fs = [x["rmse_f"] for x in results]
    print(f"\n  b-GRU+FDec    R={np.mean(rs):.5f}+/-{np.std(rs):.5f}  "
          f"F={np.mean(fs):.5f}+/-{np.std(fs):.5f}")
    print(f"\n  Compare:")
    print(f"  AE+DMD        R=0.00968  F=1.20774  (from arch_compare)")
    print(f"  Resid+GRU     R=0.00736  F=1.02516  (from arch_compare)")

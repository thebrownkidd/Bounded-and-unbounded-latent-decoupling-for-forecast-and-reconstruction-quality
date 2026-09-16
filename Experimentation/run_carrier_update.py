"""
Carrier updater: C_{t+τ} = CarrierUpdate(C_t, b_{t+1:t+τ}) then x̂ = D(C_{t+τ}).

Key insight: keep the powerful carrier decoder D (j-dim, well-trained),
but replace sequential GRU accumulation with a learned carrier update
that takes the initial carrier + aggregated b-trajectory.

The b-trajectory is computed analytically via DMD (non-autoregressive).
The carrier decoder is frozen from Phase 1.
"""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import expm
from Utils.Benchmark import SeedAll

EPOCHS = 300; LR = 1e-3; BS = 512
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

class CarrierUpdater(nn.Module):
    """Maps (C_t, cumulative_decoded_residual, tau_encoding) -> C_{t+tau}.

    Input: C_t (j) + cumsum of m(b_{1:tau}) (j) + tau/T (1) = 2j+1
    Output: C_{t+tau} (j)

    Residual connection: output = C_t + net(inputs)
    """
    def __init__(self, j, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * j + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, j))

    def forward(self, c_t, resid_cumsum, tau_enc):
        inp = torch.cat([c_t, resid_cumsum, tau_enc], dim=-1)
        return c_t + self.net(inp)

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

def run_carrier_update(seed, train_n, test_n):
    SeedAll(seed)
    Xt = torch.tensor(train_n, dtype=torch.float32)

    # Phase 1-2
    model = CarrierResidBase(N_OBS, J, K, H)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    dC = C[1:] - C[:-1]
    train_resid(model, dC); model.eval()
    with torch.no_grad():
        B = model.f(dC)           # (T-1, k)
        mB = model.m(B)           # decoded residuals (T-1, j)

    B_np = B.numpy()
    A_dmd = fit_dmd(B_np)

    # Build training data for carrier updater:
    # For random (anchor, tau) pairs, predict C_{anchor+tau} from
    # (C_anchor, cumsum(m(b_{anchor+1:anchor+tau})), tau/max_tau)
    T = len(C)
    max_window = min(100, T - 1)

    # Sample anchor-tau pairs efficiently
    train_anchors = []
    train_cumsums = []
    train_taus = []
    train_targets = []

    rng = np.random.default_rng(seed)
    anchors = rng.choice(T - 2, size=min(500, T - 2), replace=False)
    tau_samples = [1, 2, 5, 10, 20, 50, 100]

    for anchor in anchors:
        cum = torch.zeros(J)
        for tau in range(1, min(max_window, T - anchor)):
            cum = cum + mB[anchor + tau - 1]
            if tau in tau_samples or tau % 25 == 0:
                train_anchors.append(C[anchor])
                train_cumsums.append(cum.clone())
                train_taus.append(tau / max_window)
                train_targets.append(C[anchor + tau])

    train_anchors = torch.stack(train_anchors)
    train_cumsums = torch.stack(train_cumsums)
    train_taus = torch.tensor(train_taus, dtype=torch.float32).unsqueeze(1)
    train_targets = torch.stack(train_targets)

    print(f"    CarrierUpdater training pairs: {len(train_anchors)}")

    # Phase 3: Train carrier updater
    SeedAll(seed + 3000)
    updater = CarrierUpdater(J, H)
    opt = torch.optim.Adam(updater.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        updater.train(); idx = torch.randperm(len(train_anchors))
        for i in range(0, len(train_anchors), BS):
            sl = idx[i:i+BS]
            pred = updater(train_anchors[sl], train_cumsums[sl], train_taus[sl])
            loss = nn.functional.mse_loss(pred, train_targets[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    updater.eval()

    # Evaluate recon (same as always — carrier decoder)
    Xte = torch.tensor(test_n, dtype=torch.float32)
    with torch.no_grad():
        pred_r = model.recon(Xte).numpy()
    rmse_r = float(np.sqrt(np.mean((pred_r - test_n) ** 2)))

    # Evaluate forecast
    with torch.no_grad():
        c0 = model.carrier(Xte[0:1]).squeeze(0)  # initial carrier
        dc_last = dC[-1:]
        b = model.f(dc_last).numpy().ravel()

    out = np.empty((FCST, N_OBS))
    cum_resid = torch.zeros(J)
    for tau in range(1, FCST + 1):
        b = A_dmd @ b
        with torch.no_grad():
            mb = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).squeeze(0)
        cum_resid = cum_resid + mb
        tau_enc = torch.tensor([tau / max_window], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            c_tau = updater(c0.unsqueeze(0), cum_resid.unsqueeze(0), tau_enc)
            out[tau - 1] = model.dec(c_tau).numpy().ravel()

    rmse_f = float(np.sqrt(np.mean((out - test_n[:FCST]) ** 2)))
    return rmse_r, rmse_f


if __name__ == "__main__":
    train_n, test_n = make_linear5d()
    print(f"Linear 5D: train {train_n.shape}, test {test_n.shape}\n")

    results = []
    for seed in SEEDS:
        t0 = time.time()
        rmse_r, rmse_f = run_carrier_update(seed, train_n, test_n)
        dt = time.time() - t0
        results.append({"rmse_r": rmse_r, "rmse_f": rmse_f})
        print(f"  seed={seed}  R={rmse_r:.5f}  F={rmse_f:.5f}  ({dt:.0f}s)")

    rs = [x["rmse_r"] for x in results]
    fs = [x["rmse_f"] for x in results]
    print(f"\n  CarrierUpd    R={np.mean(rs):.5f}+/-{np.std(rs):.5f}  "
          f"F={np.mean(fs):.5f}+/-{np.std(fs):.5f}")
    print(f"\n  Compare:")
    print(f"  AE+DMD        R=0.00968  F=1.20774")
    print(f"  Resid+GRU     R=0.00736  F=1.02516")

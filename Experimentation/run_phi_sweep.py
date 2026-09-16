"""
Phi-sweep Pareto frontier with spectral DMD evaluation.

Trains a coupled Koopman AE with loss = (1-phi)*L_rec + phi*L_fcst
for various phi values, evaluates with spectral DMD.
Decoupled "ours" point comes from spectral_fair.json.

All 5 systems, 5 seeds, spectral DMD evaluation.
Outputs: Paper/phi_sweep.json (incremental saves after each system)
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
PHIS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
FCST = 500
TRAIN_HORIZON = 16


class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.K = nn.Linear(k, k, bias=False)

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))
    def step(self, z): return self.K(z)


def train_koopman(model, X_seqs, phi, horizon=TRAIN_HORIZON):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    T = X_seqs.shape[1]
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(X_seqs))
        for i in range(0, len(X_seqs), BS):
            batch = X_seqs[idx[i:i+BS]]
            z = model.encode(batch[:, 0])

            loss_rec = nn.functional.mse_loss(model.decode(z), batch[:, 0])
            loss_fcst = torch.tensor(0.0)

            if phi > 0:
                z_cur = z
                preds = []
                for t in range(1, min(horizon, T)):
                    z_cur = model.step(z_cur)
                    preds.append(model.decode(z_cur))
                if preds:
                    preds = torch.stack(preds, dim=1)
                    loss_fcst = nn.functional.mse_loss(preds, batch[:, 1:preds.shape[1]+1])

            loss = (1 - phi) * loss_rec + phi * loss_fcst
            opt.zero_grad(); loss.backward(); opt.step()


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)


def spectral_rollout(A, z0, steps):
    eigvals, V = np.linalg.eig(A)
    unstable = np.abs(eigvals) > 1.0
    eigvals[unstable] = eigvals[unstable] / np.abs(eigvals[unstable])
    V_inv = np.linalg.inv(V)
    z0_modal = V_inv @ z0
    out = np.empty((steps, len(z0)), dtype=np.complex128)
    for tau in range(steps):
        out[tau] = V @ (eigvals ** (tau + 1) * z0_modal)
    return out.real


def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)


def make_systems():
    systems = {}

    # Coupled Harmonic (6D)
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw*x1 - k12*(x1-x2), v2,
                -k12*(x2-x1) - k23*(x2-x3),
                v3, -k23*(x3-x2) - kw*x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    tr, te = raw[:4000], raw[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    systems["Coupled Harmonic"] = dict(
        tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=6, k=4)

    # Linear 5D
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5*0.05); raw5 = np.empty((10000,5)); raw5[0] = [1,0,0.5,0.5,1]
    for i in range(1, 10000): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    tr, te = raw[:4000], raw[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    systems["Linear 5D"] = dict(
        tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=5, k=4)

    # Brusselator (delay-embedded 10D)
    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s; return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    obs = delay(sol.y.T[200:], 5)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    systems["Brusselator"] = dict(
        tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=10, k=4)

    # Duffing (delay-embedded 10D)
    def duff(t, s):
        x, v = s; return [v, -0.05*v - x**3 + 8*np.cos(t)]
    sol = solve_ivp(duff, [0, 500], [1, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    obs = delay(sol.y.T[200:], 5)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    systems["Duffing"] = dict(
        tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=10, k=4)

    # Lorenz-96 (20D)
    N96, F96 = 20, 8.0
    def l96(t, x):
        d = np.empty(N96)
        for i in range(N96):
            d[i] = (x[(i+1)%N96] - x[(i-2)%N96]) * x[(i-1)%N96] - x[i] + F96
        return d
    x0 = F96 * np.ones(N96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 200], x0,
                    t_eval=np.arange(0, 200, 0.02), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[500:]
    tr, te = raw[:4000], raw[4000:4500]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    systems["Lorenz-96"] = dict(
        tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=20, k=8)

    return systems


def make_sequences(data, seq_len):
    T = len(data) - seq_len + 1
    seqs = np.array([data[i:i+seq_len] for i in range(T)])
    return torch.tensor(seqs, dtype=torch.float32)


def eval_spectral(model, te_n, fcst_steps):
    model.eval()
    Xte = torch.tensor(te_n, dtype=torch.float32)
    with torch.no_grad():
        rec = model(Xte).numpy()
    rmse_r = float(np.sqrt(np.mean((rec - te_n) ** 2)))

    with torch.no_grad():
        Z_te = model.encode(Xte).numpy()
    A = fit_dmd(Z_te)
    z0 = Z_te[0]
    pred_z = spectral_rollout(A, z0, fcst_steps)
    with torch.no_grad():
        pred_x = model.decode(torch.tensor(pred_z, dtype=torch.float32)).numpy()
    rmse_f = float(np.sqrt(np.mean((pred_x - te_n[:fcst_steps]) ** 2)))
    return rmse_r, rmse_f


if __name__ == "__main__":
    t0 = time.time()
    print("Generating systems...")
    systems = make_systems()
    all_results = {}

    for sname, cfg in systems.items():
        print(f"\n{'='*60}")
        print(f"  {sname}  (n={cfg['n']}, k={cfg['k']})")
        print(f"{'='*60}")

        n, k = cfg['n'], cfg['k']
        X_seqs = make_sequences(cfg['tr_n'], TRAIN_HORIZON + 1)
        phi_results = {}

        for phi in PHIS:
            rs, fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(n, k, 64)
                train_koopman(model, X_seqs, phi)
                r, f = eval_spectral(model, cfg['te_n'], FCST)
                rs.append(r); fs.append(f)

            phi_results[str(phi)] = {
                "rmse_r_mean": float(np.mean(rs)),
                "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)),
                "rmse_f_std": float(np.std(fs)),
            }
            print(f"  phi={phi:.2f}  R={np.mean(rs):.5f}+/-{np.std(rs):.5f}  "
                  f"F={np.mean(fs):.5f}+/-{np.std(fs):.5f}")

        all_results[sname] = {"phi_sweep": phi_results}

        # Save incrementally
        with open(OUT / "phi_sweep.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved to phi_sweep.json]")

    print(f"\n-> {OUT / 'phi_sweep.json'}")
    print(f"Total: {time.time()-t0:.0f}s")

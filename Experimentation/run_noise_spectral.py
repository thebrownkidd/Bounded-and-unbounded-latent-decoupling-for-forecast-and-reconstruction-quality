"""
Noise robustness with spectral DMD evaluation for fair comparison.

Both coupled (k-dim AE) and decoupled (j-dim AE) evaluated with spectral DMD
on their respective latent spaces.  Same evaluation method → fair comparison.

Spectral DMD: A = V Λ V⁻¹, predict z_{T+τ} = V Λ^τ V⁻¹ z_T.
Evaluated as 500-step spectral rollout (no autoregressive accumulation).

Outputs: Paper/noise_spectral.json
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
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2]
FCST = 500


class AE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))


def train_recon(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model(x), x)
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


def add_noise(obs, sigma, rng):
    if sigma == 0:
        return obs.copy()
    channel_std = obs.std(axis=0, keepdims=True)
    return obs + sigma * channel_std * rng.standard_normal(obs.shape)


def make_systems_noisy(sigma, noise_seed=999):
    rng = np.random.default_rng(noise_seed)
    systems = {}

    # Coupled Harmonic
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw*x1 - k12*(x1-x2), v2,
                -k12*(x2-x1) - k23*(x2-x3),
                v3, -k23*(x3-x2) - kw*x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    obs = add_noise(raw, sigma, rng)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Coupled Harmonic"] = dict(
        tr_n=(tr-mu)/sig_, te_n=(te-mu)/sig_, n=6, j=8, k=4)

    # Linear 5D
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5*0.05); raw5 = np.empty((10000,5)); raw5[0] = [1,0,0.5,0.5,1]
    for i in range(1, 10000): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    obs = add_noise(raw, sigma, rng)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Linear 5D"] = dict(
        tr_n=(tr-mu)/sig_, te_n=(te-mu)/sig_, n=5, j=8, k=4)

    # Brusselator
    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s; return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    raw_noisy = add_noise(raw_br, sigma, rng)
    obs = delay(raw_noisy, 5)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Brusselator"] = dict(
        tr_n=(tr-mu)/sig_, te_n=(te-mu)/sig_, n=10, j=8, k=4)

    # Duffing
    def duff(t, s):
        x, v = s; return [v, -0.05*v - x**3 + 8*np.cos(t)]
    sol = solve_ivp(duff, [0, 500], [1, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_df = sol.y.T[200:]
    raw_noisy = add_noise(raw_df, sigma, rng)
    obs = delay(raw_noisy, 5)
    tr, te = obs[:4000], obs[4000:4500]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Duffing"] = dict(
        tr_n=(tr-mu)/sig_, te_n=(te-mu)/sig_, n=10, j=8, k=4)

    return systems


def eval_spectral(model, A, te_n, fcst_steps):
    model.eval()
    Xte = torch.tensor(te_n, dtype=torch.float32)
    with torch.no_grad():
        rec = model(Xte).numpy()
    rmse_r = float(np.sqrt(np.mean((rec - te_n) ** 2)))

    with torch.no_grad():
        z0 = model.encode(Xte[0:1]).numpy().ravel()
    pred_z = spectral_rollout(A, z0, fcst_steps)
    with torch.no_grad():
        pred_x = model.decode(torch.tensor(pred_z, dtype=torch.float32)).numpy()
    rmse_f = float(np.sqrt(np.mean((pred_x - te_n[:fcst_steps]) ** 2)))
    return rmse_r, rmse_f


if __name__ == "__main__":
    t0 = time.time()
    all_results = {}

    for sname_ref in ["Coupled Harmonic", "Linear 5D", "Brusselator", "Duffing"]:
        all_results[sname_ref] = {}

    for sigma in SIGMAS:
        print(f"\n{'='*60}")
        print(f"  sigma = {sigma}")
        print(f"{'='*60}")
        systems = make_systems_noisy(sigma)

        for sname, cfg in systems.items():
            n, j, k = cfg['n'], cfg['j'], cfg['k']
            Xt = torch.tensor(cfg['tr_n'], dtype=torch.float32)

            coupled_rs, coupled_fs = [], []
            decoupled_rs, decoupled_fs = [], []

            for seed in SEEDS:
                # Coupled: k-dim AE + spectral DMD
                SeedAll(seed)
                ae_k = AE(n, k, 64)
                train_recon(ae_k, Xt); ae_k.eval()
                with torch.no_grad(): Z = ae_k.encode(Xt).numpy()
                A_k = fit_dmd(Z)
                r_c, f_c = eval_spectral(ae_k, A_k, cfg['te_n'], FCST)
                coupled_rs.append(r_c); coupled_fs.append(f_c)

                # Decoupled: j-dim AE + spectral DMD
                SeedAll(seed)
                ae_j = AE(n, j, 64)
                train_recon(ae_j, Xt); ae_j.eval()
                with torch.no_grad(): C = ae_j.encode(Xt).numpy()
                A_j = fit_dmd(C)
                r_d, f_d = eval_spectral(ae_j, A_j, cfg['te_n'], FCST)
                decoupled_rs.append(r_d); decoupled_fs.append(f_d)

            sig_key = str(sigma)
            all_results[sname][sig_key] = {
                "Coupled": {
                    "rmse_r_mean": float(np.mean(coupled_rs)),
                    "rmse_r_std": float(np.std(coupled_rs)),
                    "rmse_f_mean": float(np.mean(coupled_fs)),
                    "rmse_f_std": float(np.std(coupled_fs)),
                },
                "Decoupled": {
                    "rmse_r_mean": float(np.mean(decoupled_rs)),
                    "rmse_r_std": float(np.std(decoupled_rs)),
                    "rmse_f_mean": float(np.mean(decoupled_fs)),
                    "rmse_f_std": float(np.std(decoupled_fs)),
                }
            }
            print(f"  {sname:20s}  Coupled  R={np.mean(coupled_rs):.5f} F={np.mean(coupled_fs):.5f}")
            print(f"  {'':20s}  Decoup   R={np.mean(decoupled_rs):.5f} F={np.mean(decoupled_fs):.5f}")

        # Save after each sigma so partial results survive crashes
        with open(OUT / "noise_spectral.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [saved to noise_spectral.json]")

    print(f"\n-> {OUT / 'noise_spectral.json'}")
    print(f"Total: {time.time()-t0:.0f}s")

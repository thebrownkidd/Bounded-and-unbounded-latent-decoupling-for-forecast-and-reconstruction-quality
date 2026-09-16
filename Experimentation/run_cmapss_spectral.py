"""
C-MAPSS FD001 with spectral DMD evaluation on both coupled (k-dim) and
decoupled (j-dim) autoencoders.  Same evaluation method for fair comparison.

Spectral DMD: eigendecompose A = V Λ V⁻¹, predict z_{T+τ} = V Λ^τ V⁻¹ z_T.

Outputs: Paper/cmapss_spectral.json
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 256
SEEDS = [0, 1, 2, 42, 123]
N_OBS = 17; J = 16; K = 6; H = 64
HORIZONS = [8, 16, 32, 64]


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
    """Spectral DMD: z_{t+tau} = V Λ^tau V^{-1} z0.
    Eigenvalues with |λ| > 1 are projected onto the unit circle to prevent blowup."""
    eigvals, V = np.linalg.eig(A)
    # Stabilise: project unstable eigenvalues onto unit circle
    unstable = np.abs(eigvals) > 1.0
    eigvals[unstable] = eigvals[unstable] / np.abs(eigvals[unstable])
    V_inv = np.linalg.inv(V)
    z0_modal = V_inv @ z0
    out = np.empty((steps, len(z0)), dtype=np.complex128)
    for tau in range(steps):
        out[tau] = V @ (eigvals ** (tau + 1) * z0_modal)
    return out.real


def load_cmapss():
    data = np.load(ROOT / "Data" / "CmapssFD001.npz")
    train_series = data["train"]        # (60, 128, 17) — already normalised
    holdout_series = data["holdout_obs"]  # (30, 128, 17)
    # Data is pre-normalised; re-normalise from training stats for consistency
    all_train = train_series.reshape(-1, N_OBS)
    mu = all_train.mean(0)
    sig = all_train.std(0) + 1e-8
    train_flat = (all_train - mu) / sig
    train_n = (train_series - mu) / sig
    holdout_n = (holdout_series.reshape(-1, N_OBS) - mu) / sig
    holdout_n = holdout_n.reshape(holdout_series.shape[0], holdout_series.shape[1], N_OBS)
    return train_n, holdout_n, train_flat


def eval_cmapss_spectral(model, A, holdout_n, horizons):
    """Evaluate recon RMSE + multi-horizon spectral forecast RMSE."""
    rmse_rs = []
    rmse_fs = {h: [] for h in horizons}

    for eng in range(holdout_n.shape[0]):
        seq = holdout_n[eng]  # (128, 17)
        seq_t = torch.tensor(seq, dtype=torch.float32)
        with torch.no_grad():
            rec = model(seq_t).numpy()
        rmse_rs.append(np.sqrt(np.mean((rec - seq) ** 2)))

        with torch.no_grad():
            z0 = model.encode(seq_t[0:1]).numpy().ravel()

        for hz in horizons:
            if hz > len(seq):
                continue
            pred_z = spectral_rollout(A, z0, hz)
            with torch.no_grad():
                pred_x = model.decode(
                    torch.tensor(pred_z, dtype=torch.float32)).numpy()
            rmse_fs[hz].append(np.sqrt(np.mean((pred_x - seq[:hz]) ** 2)))

    rmse_r = float(np.mean(rmse_rs))
    rmse_f = {h: float(np.mean(rmse_fs[h])) for h in horizons}
    return rmse_r, rmse_f


def run_one(seed):
    train_n, holdout_n, train_flat = load_cmapss()
    Xt = torch.tensor(train_flat, dtype=torch.float32)
    res = {}

    # ── Coupled: AE with k-dim latent + spectral DMD ──
    SeedAll(seed)
    ae_k = AE(N_OBS, K, H)
    train_recon(ae_k, Xt); ae_k.eval()

    # Fit DMD on per-engine latents (pooled)
    all_Z = []
    for eng in range(train_n.shape[0]):
        eng_t = torch.tensor(train_n[eng], dtype=torch.float32)
        with torch.no_grad():
            all_Z.append(ae_k.encode(eng_t).numpy())
    Z_pool = np.concatenate(all_Z, axis=0)
    A_k = fit_dmd(Z_pool)

    rmse_r, rmse_f = eval_cmapss_spectral(ae_k, A_k, holdout_n, HORIZONS)
    res["Coupled (k-dim)"] = {"rmse_r": rmse_r, "rmse_f": rmse_f}

    # ── Decoupled: AE with j-dim carrier + spectral DMD ──
    SeedAll(seed)
    ae_j = AE(N_OBS, J, H)
    train_recon(ae_j, Xt); ae_j.eval()

    all_C = []
    for eng in range(train_n.shape[0]):
        eng_t = torch.tensor(train_n[eng], dtype=torch.float32)
        with torch.no_grad():
            all_C.append(ae_j.encode(eng_t).numpy())
    C_pool = np.concatenate(all_C, axis=0)
    A_j = fit_dmd(C_pool)

    rmse_r, rmse_f = eval_cmapss_spectral(ae_j, A_j, holdout_n, HORIZONS)
    res["Decoupled (j-dim)"] = {"rmse_r": rmse_r, "rmse_f": rmse_f}

    return res


if __name__ == "__main__":
    t0 = time.time()
    raw = {}

    for seed in SEEDS:
        print(f"\n  C-MAPSS FD001  seed={seed}")
        r = run_one(seed)
        for method, vals in r.items():
            raw.setdefault(method, []).append(vals)
            hz_str = "  ".join(f"h{h}={vals['rmse_f'][h]:.3f}" for h in HORIZONS)
            print(f"    {method:20s}  r={vals['rmse_r']:.4f}  {hz_str}")

    stats = {}
    for method in raw:
        rs = [x["rmse_r"] for x in raw[method]]
        stats[method] = {
            "rmse_r_mean": float(np.mean(rs)),
            "rmse_r_std": float(np.std(rs)),
        }
        for hz in HORIZONS:
            fs = [x["rmse_f"][hz] for x in raw[method]]
            stats[method][f"rmse_f{hz}_mean"] = float(np.mean(fs))
            stats[method][f"rmse_f{hz}_std"] = float(np.std(fs))

    with open(OUT / "cmapss_spectral.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n-> {OUT / 'cmapss_spectral.json'}")
    print(f"Total: {time.time()-t0:.0f}s")

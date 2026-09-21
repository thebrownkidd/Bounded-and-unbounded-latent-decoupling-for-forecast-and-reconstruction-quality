"""
Noise robustness sweep for Lorenz-96 (completes the 5-system × 5-noise grid).

Compares Coupled (k-dim AE, phi=0.5) vs Decoupled (carrier-residual) under
observation noise sigma ∈ {0.0, 0.01, 0.05, 0.1, 0.2}.

Both evaluated with spectral DMD 500-step forecast for fair comparison.
Outputs: /kaggle/working/noise_lorenz96.json (incremental saves after each sigma)
"""
import json, time, random
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp

OUT = "/kaggle/working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2]
FCST = 500
OUTFILE = f"{OUT}/noise_lorenz96.json"


def SeedAll(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

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


class ResidualModel(nn.Module):
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


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_coupled(model, Xt, phi=0.5):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl].to(DEVICE), Xt[sl + 1].to(DEVICE)
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_fcst = nn.functional.mse_loss(z_t, z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_fcst
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_teacher(model, Xt):
    model.to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]].to(DEVICE)
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
            dc = dC[idx[i:i + BS]].to(DEVICE)
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()


def train_gru(model, dC, Cc, Cn):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i + BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl].to(DEVICE)))
            loss = nn.functional.mse_loss(
                model.gru(dh, Cc[sl].to(DEVICE)), Cn[sl].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  DMD + spectral evaluation
# ═══════════════════════════════════════════════════════════════════

def spectral_forecast(Z, steps):
    X, Y = Z[:-1].T, Z[1:].T
    A = Y @ np.linalg.pinv(X)
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    c = V_inv @ Z[0]
    out = np.empty((steps, Z.shape[1]))
    for tau in range(steps):
        out[tau] = np.real(V @ (c * eigvals ** tau))
    return out


def eval_spectral(encode_fn, decode_fn, Xt_test, gt_dim):
    with torch.no_grad():
        Z = encode_fn(Xt_test.to(DEVICE)).cpu().numpy()
        recon = decode_fn(
            torch.tensor(Z, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    z_spectral = spectral_forecast(Z, FCST)
    with torch.no_grad():
        x_spectral = decode_fn(
            torch.tensor(z_spectral, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt_spectral = gt[:FCST]
    rmse_f = float(np.sqrt(np.mean(
        (x_spectral[:, :gt_dim] - gt_spectral[:, :gt_dim]) ** 2)))

    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Lorenz-96
# ═══════════════════════════════════════════════════════════════════

def make_lorenz96():
    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1) % N96] - x[(i-2) % N96]) * x[(i-1) % N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96) * 0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    return sol.y.T[200:]


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def save_incremental(results):
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved -> {OUTFILE}]")


if __name__ == "__main__":
    raw = make_lorenz96()
    n_obs, k, j, h, gt_dim = 20, 10, 12, 64, 20
    n_tr, n_te = 4000, 500

    t0 = time.time()
    results = {"Lorenz-96": {}}

    for sigma in SIGMAS:
        print(f"\n{'='*60}")
        print(f"  Lorenz-96 | sigma = {sigma}")
        print(f"{'='*60}")

        per_ch_std = raw[:n_tr].std(axis=0)

        coupled_r, coupled_f = [], []
        decoupled_r, decoupled_f = [], []

        for seed in SEEDS:
            SeedAll(seed)
            noise_tr = sigma * per_ch_std * np.random.randn(n_tr, n_obs)
            noise_te = sigma * per_ch_std * np.random.randn(n_te, n_obs)
            obs_tr = raw[:n_tr] + noise_tr
            obs_te = raw[n_tr:n_tr + n_te] + noise_te

            mu, sig = obs_tr.mean(0), obs_tr.std(0) + 1e-8
            tr_n = (obs_tr - mu) / sig
            te_n = (obs_te - mu) / sig

            Xt = torch.tensor(tr_n, dtype=torch.float32)
            Xt_test = torch.tensor(te_n, dtype=torch.float32)

            # Coupled
            SeedAll(seed)
            ae = AE(n_obs, k, h)
            train_coupled(ae, Xt, phi=0.5)
            ae.eval()
            rr, rf = eval_spectral(ae.encode, ae.decode, Xt_test, gt_dim)
            coupled_r.append(rr); coupled_f.append(rf)
            print(f"  Coupled seed={seed}: R={rr:.4f} F={rf:.4f}")

            # Decoupled
            SeedAll(seed)
            rm = ResidualModel(n_obs, j, k, h)
            train_teacher(rm, Xt); rm.eval()
            with torch.no_grad():
                C = rm.carrier(Xt.to(DEVICE)).cpu()
                dC = C[1:] - C[:-1]
            train_resid(rm, dC); rm.eval()
            with torch.no_grad():
                C = rm.carrier(Xt.to(DEVICE)).cpu()
                dC = C[1:] - C[:-1]
            train_gru(rm, dC, C[:-1], C[1:]); rm.eval()
            rr, rf = eval_spectral(rm.carrier, rm.dec, Xt_test, gt_dim)
            decoupled_r.append(rr); decoupled_f.append(rf)
            print(f"  Decoupled seed={seed}: R={rr:.4f} F={rf:.4f}")

        results["Lorenz-96"][str(sigma)] = {
            "Coupled": {
                "rmse_r_mean": float(np.mean(coupled_r)),
                "rmse_r_std": float(np.std(coupled_r)),
                "rmse_f_mean": float(np.mean(coupled_f)),
                "rmse_f_std": float(np.std(coupled_f)),
            },
            "Decoupled": {
                "rmse_r_mean": float(np.mean(decoupled_r)),
                "rmse_r_std": float(np.std(decoupled_r)),
                "rmse_f_mean": float(np.mean(decoupled_f)),
                "rmse_f_std": float(np.std(decoupled_f)),
            },
        }
        save_incremental(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'Sigma':>8s} {'Method':>12s} {'Recon':>10s} {'F500':>10s}")
    print("-" * 45)
    for sigma_s in results["Lorenz-96"]:
        for mname in ["Coupled", "Decoupled"]:
            r = results["Lorenz-96"][sigma_s][mname]
            print(f"{sigma_s:>8s} {mname:>12s} {r['rmse_r_mean']:.4f}    {r['rmse_f_mean']:.4f}")
        print()
    print("Done.")

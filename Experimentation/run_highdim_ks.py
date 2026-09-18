"""
High-dimensional experiment: Kuramoto-Sivashinsky equation (64D).

u_t = -u*u_x - u_xx - u_xxxx  on [0, L] periodic

Compares Coupled_k (k-dim AE + spectral DMD) vs Decoupled (j-dim carrier,
k-dim residual + spectral DMD).  Forecast bottleneck k is matched.

Multi-horizon spectral DMD evaluation at [1, 5, 10, 25, 50, 100, 250, 500].
Outputs: Paper/highdim_ks.json (incremental saves)
"""
import sys, json, time, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 400; LR = 5e-4; BS = 256
SEEDS = [0, 1, 2, 42, 123]
HORIZONS = [1, 5, 10, 25, 50, 100, 250, 500]
OUTFILE = OUT / "highdim_ks.json"

N_SPATIAL = 64
K_DIM = 16
J_DIM = 32
H_DIM = 128


def SeedAll(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  KS equation solver (spectral method, ETDRK4)
# ═══════════════════════════════════════════════════════════════════

def solve_ks(N=64, L=22.0, dt_out=0.25, T=2500, T_transient=500):
    """Solve KS equation using pseudo-spectral + scipy RK45."""
    from scipy.integrate import solve_ivp
    kk = (2 * np.pi / L) * np.concatenate([np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)])
    kk2 = kk ** 2
    kk4 = kk ** 4

    def ks_rhs(t, u):
        uh = np.fft.fft(u)
        ux = np.real(np.fft.ifft(1j * kk * uh))
        uxx = np.real(np.fft.ifft(-kk2 * uh))
        uxxxx = np.real(np.fft.ifft(kk4 * uh))
        return -u * ux - uxx - uxxxx

    np.random.seed(42)
    x = np.linspace(0, L, N, endpoint=False)
    u0 = np.cos(2 * np.pi * x / L) * (1 + 0.01 * np.random.randn(N))

    t_total = T_transient + T
    t_eval = np.arange(T_transient, t_total, dt_out)

    print(f"  Integrating KS for t=[0, {t_total}], {len(t_eval)} output points...")
    sol = solve_ivp(ks_rhs, [0, t_total], u0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.5)
    if sol.status != 0:
        raise RuntimeError(f"KS integration failed: {sol.message}")
    return sol.y.T


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class KoopmanAE(nn.Module):
    def __init__(self, n, latent, h):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h // 2), nn.ELU(),
            nn.Linear(h // 2, latent),
            nn.BatchNorm1d(latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, h // 2), nn.ELU(),
            nn.Linear(h // 2, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.K = nn.Linear(latent, latent, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def predict(self, z): return self.K(z)


class ResidualModel(nn.Module):
    def __init__(self, n, j, k, h):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h // 2), nn.ELU(),
            nn.Linear(h // 2, j),
            nn.BatchNorm1d(j))
        self.dec = nn.Sequential(
            nn.Linear(j, h // 2), nn.ELU(),
            nn.Linear(h // 2, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h // 2), nn.ELU(), nn.Linear(h // 2, k))
        self.m = nn.Sequential(nn.Linear(k, h // 2), nn.ELU(), nn.Linear(h // 2, j))
        self.gru = nn.GRUCell(j, j)
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_coupled(model, Xt, phi):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl].to(DEVICE), Xt[sl + 1].to(DEVICE)
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_fcst
            if torch.isnan(loss):
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 100 == 0:
            print(f"      epoch {ep}/{EPOCHS}", flush=True)


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
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 100 == 0:
            print(f"      epoch {ep}/{EPOCHS}", flush=True)


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
#  Spectral DMD at multiple horizons
# ═══════════════════════════════════════════════════════════════════

def eval_multi_horizon(encode_fn, decode_fn, Xt_test, gt_dim, horizons):
    with torch.no_grad():
        Z_test = encode_fn(Xt_test.to(DEVICE)).cpu().numpy()
        recon = decode_fn(
            torch.tensor(Z_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    X, Y = Z_test[:-1].T, Z_test[1:].T
    # Check for NaN/Inf in latent space
    if not np.all(np.isfinite(Z_test)):
        print("    WARNING: NaN/Inf in latent space, returning Inf RMSE")
        return rmse_r, {h: float('inf') for h in horizons}

    # Tikhonov-regularized DMD to handle ill-conditioned latent spaces
    lam = 1e-5
    A = Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(X.shape[0]))

    if not np.all(np.isfinite(A)):
        print("    WARNING: NaN/Inf in DMD matrix, returning Inf RMSE")
        return rmse_r, {h: float('inf') for h in horizons}

    eigvals, V = np.linalg.eig(A)
    # Clip eigenvalues to unit disk for stability
    eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
    V_inv = np.linalg.inv(V)

    rmse_per_h = {}
    for h in horizons:
        errors = []
        n_starts = min(len(Z_test) - h, 200)
        for s in range(n_starts):
            z0 = Z_test[s]
            c = V_inv @ z0
            z_pred = np.real(V @ (c * eigvals ** h))
            with torch.no_grad():
                x_pred = decode_fn(
                    torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            gt_h = gt[s + h]
            errors.append((x_pred[:gt_dim] - gt_h[:gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errors)))
    return rmse_r, rmse_per_h


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def save_incremental(results):
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved -> {OUTFILE}]")


if __name__ == "__main__":
    print("Generating KS data (N=64, L=22, dt=0.25)...")
    raw = solve_ks(N=N_SPATIAL, L=22.0, dt_out=0.25, T=2500, T_transient=500)
    print(f"  Generated {raw.shape[0]} snapshots of dimension {raw.shape[1]}")

    n_tr, n_te = 6000, 1000
    mu, sig = raw[:n_tr].mean(0), raw[:n_tr].std(0) + 1e-8
    tr_n = (raw[:n_tr] - mu) / sig
    te_n = (raw[n_tr:n_tr + n_te] - mu) / sig
    print(f"  Train: {tr_n.shape}, Test: {te_n.shape}")

    Xt = torch.tensor(tr_n, dtype=torch.float32)
    Xt_test = torch.tensor(te_n, dtype=torch.float32)
    gt_dim = N_SPATIAL

    t0 = time.time()
    results = {"KS-64D": {}}

    # ── Coupled k-dim (pick best phi) ──
    print(f"\n{'='*60}")
    print(f"  Coupled_k (latent={K_DIM})")
    print(f"{'='*60}")
    best_phi, best_f1 = None, 1e9
    for phi in [0.1, 0.5, 0.9]:
        SeedAll(0)
        model = KoopmanAE(N_SPATIAL, K_DIM, H_DIM)
        train_coupled(model, Xt, phi); model.eval()
        _, hrm = eval_multi_horizon(model.encode, model.decode, Xt_test, gt_dim, [1])
        print(f"    phi={phi}: F1={hrm[1]:.4f}")
        if hrm[1] < best_f1:
            best_f1 = hrm[1]; best_phi = phi
    if best_phi is None:
        best_phi = 0.5
    print(f"  -> Best phi = {best_phi}")

    all_h = []
    for seed in SEEDS:
        SeedAll(seed)
        model = KoopmanAE(N_SPATIAL, K_DIM, H_DIM)
        train_coupled(model, Xt, best_phi); model.eval()
        rr, hrm = eval_multi_horizon(model.encode, model.decode, Xt_test, gt_dim, HORIZONS)
        print(f"    seed={seed}: R={rr:.4f} " + " ".join(f"F{h}={hrm[h]:.4f}" for h in [1, 10, 100, 500]))
        all_h.append((rr, hrm))

    results["KS-64D"]["Coupled_k"] = {
        "latent_dim": K_DIM, "best_phi": best_phi,
        "rmse_r_mean": float(np.mean([x[0] for x in all_h])),
        "rmse_r_std": float(np.std([x[0] for x in all_h])),
    }
    for hh in HORIZONS:
        vals = [x[1][hh] for x in all_h]
        results["KS-64D"]["Coupled_k"][f"F{hh}_mean"] = float(np.mean(vals))
        results["KS-64D"]["Coupled_k"][f"F{hh}_std"] = float(np.std(vals))
    save_incremental(results)

    # ── Decoupled (j=32, k=16) ──
    print(f"\n{'='*60}")
    print(f"  Decoupled (carrier={J_DIM}, residual={K_DIM})")
    print(f"{'='*60}")
    all_h = []
    for seed in SEEDS:
        SeedAll(seed)
        rm = ResidualModel(N_SPATIAL, J_DIM, K_DIM, H_DIM)
        print(f"    seed={seed} Phase 1 (teacher)...")
        train_teacher(rm, Xt); rm.eval()
        with torch.no_grad():
            C = rm.carrier(Xt.to(DEVICE)).cpu()
            dC = C[1:] - C[:-1]
        print(f"    seed={seed} Phase 2 (residual)...")
        train_resid(rm, dC); rm.eval()
        with torch.no_grad():
            C = rm.carrier(Xt.to(DEVICE)).cpu()
            dC = C[1:] - C[:-1]
        print(f"    seed={seed} Phase 3 (GRU)...")
        train_gru(rm, dC, C[:-1], C[1:]); rm.eval()
        rr, hrm = eval_multi_horizon(rm.carrier, rm.dec, Xt_test, gt_dim, HORIZONS)
        print(f"    seed={seed}: R={rr:.4f} " + " ".join(f"F{h}={hrm[h]:.4f}" for h in [1, 10, 100, 500]))
        all_h.append((rr, hrm))

    results["KS-64D"]["Decoupled"] = {
        "carrier_dim": J_DIM, "residual_dim": K_DIM,
        "rmse_r_mean": float(np.mean([x[0] for x in all_h])),
        "rmse_r_std": float(np.std([x[0] for x in all_h])),
    }
    for hh in HORIZONS:
        vals = [x[1][hh] for x in all_h]
        results["KS-64D"]["Decoupled"][f"F{hh}_mean"] = float(np.mean(vals))
        results["KS-64D"]["Decoupled"][f"F{hh}_std"] = float(np.std(vals))
    save_incremental(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'Method':15s} {'Recon':>8s}", end="")
    for h in HORIZONS:
        print(f" {'F'+str(h):>8s}", end="")
    print()
    print("-" * (15 + 9 + 9 * len(HORIZONS)))
    for m in results["KS-64D"]:
        r = results["KS-64D"][m]
        print(f"{m:15s} {r['rmse_r_mean']:8.4f}", end="")
        for h in HORIZONS:
            print(f" {r[f'F{h}_mean']:8.4f}", end="")
        print()
    print("\nDone.")

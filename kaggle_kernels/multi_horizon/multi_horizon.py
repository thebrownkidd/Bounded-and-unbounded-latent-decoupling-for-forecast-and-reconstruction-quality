"""
Multi-horizon spectral DMD evaluation for ICLR 2027.

Trains 3 key methods (Coupled_k, Frozen_j, Decoupled) and evaluates
spectral DMD forecast RMSE at horizons [1, 5, 10, 25, 50, 100, 250, 500].

Outputs: /kaggle/working/multi_horizon.json (incremental saves)
"""
import json, time, random
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

OUT = "/kaggle/working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
HORIZONS = [1, 5, 10, 25, 50, 100, 250, 500]
OUTFILE = f"{OUT}/multi_horizon.json"


def SeedAll(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class SimpleAE(nn.Module):
    def __init__(self, n, latent, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))


class KoopmanAE(nn.Module):
    def __init__(self, n, latent, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, latent))
        self.dec = nn.Sequential(nn.Linear(latent, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.K = nn.Linear(latent, latent, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def predict(self, z): return self.K(z)


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

def train_ae_only(model, Xt):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]].to(DEVICE)
            loss = nn.functional.mse_loss(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()


def train_coupled(model, Xt, phi):
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
            L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
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
#  Spectral DMD at multiple horizons
# ═══════════════════════════════════════════════════════════════════

def eval_multi_horizon(encode_fn, decode_fn, Xt_test, gt_dim, horizons):
    with torch.no_grad():
        Z_test = encode_fn(Xt_test.to(DEVICE)).cpu().numpy()
        recon = decode_fn(
            torch.tensor(Z_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    # Fit DMD
    X, Y = Z_test[:-1].T, Z_test[1:].T
    A = Y @ np.linalg.pinv(X)
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)

    max_h = max(horizons)
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
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig

def make_systems():
    systems = {}

    # Coupled Harmonic (6D)
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw * x1 - k12 * (x1 - x2), v2, -k12 * (x2 - x1) - k23 * (x2 - x3),
                v3, -k23 * (x3 - x2) - kw * x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten = norm_split(raw, raw, 6, 4000, 1000)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    # Linear 5D
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    raw = raw5[200:]
    trn, ten = norm_split(raw, raw, 5, 4000, 1000)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    # Brusselator
    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2 * y - (B_br + 1) * x, B_br * x - x**2 * y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    trn, ten = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 1000)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    # Duffing
    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    trn, ten = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 1000)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    # Lorenz-96
    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1) % N96] - x[(i-2) % N96]) * x[(i-1) % N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96) * 0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_l96 = sol.y.T[200:]
    trn, ten = norm_split(raw_l96, raw_l96, 20, 4000, 1000)
    systems["Lorenz-96"] = dict(
        train_n=trn, test_n=ten, n_obs=20, k=10, j=12, h=64, gt_dim=20)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def save_incremental(results):
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved -> {OUTFILE}]")


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    results = {}

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        results[sname] = {}

        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]

        # ── Coupled k-dim (best phi from {0.1, 0.5, 0.9}) ──
        print("\n  [1] Coupled_k (best phi)")
        best_phi, best_f1 = None, 1e9
        for phi in [0.1, 0.5, 0.9]:
            SeedAll(0)
            model = KoopmanAE(n, k, h)
            train_coupled(model, Xt, phi); model.eval()
            _, horizon_rmse = eval_multi_horizon(model.encode, model.decode, Xt_test, gt_dim, [1])
            if horizon_rmse[1] < best_f1:
                best_f1 = horizon_rmse[1]; best_phi = phi
        print(f"    Best phi={best_phi}")

        all_r, all_h = {hh: [] for hh in HORIZONS}, []
        for seed in SEEDS:
            SeedAll(seed)
            model = KoopmanAE(n, k, h)
            train_coupled(model, Xt, best_phi); model.eval()
            rr, horizon_rmse = eval_multi_horizon(model.encode, model.decode, Xt_test, gt_dim, HORIZONS)
            all_r_seed = rr
            print(f"    seed={seed}: R={rr:.4f} " + " ".join(f"F{hh}={horizon_rmse[hh]:.4f}" for hh in HORIZONS))
            all_h.append((all_r_seed, horizon_rmse))

        results[sname]["Coupled_k"] = {
            "best_phi": best_phi,
            "rmse_r_mean": float(np.mean([x[0] for x in all_h])),
            "rmse_r_std": float(np.std([x[0] for x in all_h])),
        }
        for hh in HORIZONS:
            vals = [x[1][hh] for x in all_h]
            results[sname]["Coupled_k"][f"F{hh}_mean"] = float(np.mean(vals))
            results[sname]["Coupled_k"][f"F{hh}_std"] = float(np.std(vals))
        save_incremental(results)

        # ── Frozen j-dim (Met2Net/LatentTSF) ──
        print("\n  [2] Frozen_j (Met2Net/LatentTSF)")
        all_h = []
        for seed in SEEDS:
            SeedAll(seed)
            model = SimpleAE(n, j, h)
            train_ae_only(model, Xt); model.eval()
            rr, horizon_rmse = eval_multi_horizon(model.encode, model.decode, Xt_test, gt_dim, HORIZONS)
            print(f"    seed={seed}: R={rr:.4f} " + " ".join(f"F{hh}={horizon_rmse[hh]:.4f}" for hh in HORIZONS))
            all_h.append((rr, horizon_rmse))

        results[sname]["Frozen_j"] = {
            "rmse_r_mean": float(np.mean([x[0] for x in all_h])),
            "rmse_r_std": float(np.std([x[0] for x in all_h])),
        }
        for hh in HORIZONS:
            vals = [x[1][hh] for x in all_h]
            results[sname]["Frozen_j"][f"F{hh}_mean"] = float(np.mean(vals))
            results[sname]["Frozen_j"][f"F{hh}_std"] = float(np.std(vals))
        save_incremental(results)

        # ── Decoupled (ours) ──
        print("\n  [3] Decoupled (ours)")
        all_h = []
        for seed in SEEDS:
            SeedAll(seed)
            rm = ResidualModel(n, j, k, h)
            train_teacher(rm, Xt); rm.eval()
            with torch.no_grad():
                C = rm.carrier(Xt.to(DEVICE)).cpu()
                dC = C[1:] - C[:-1]
            train_resid(rm, dC); rm.eval()
            with torch.no_grad():
                C = rm.carrier(Xt.to(DEVICE)).cpu()
                dC = C[1:] - C[:-1]
            train_gru(rm, dC, C[:-1], C[1:]); rm.eval()
            rr, horizon_rmse = eval_multi_horizon(rm.carrier, rm.dec, Xt_test, gt_dim, HORIZONS)
            print(f"    seed={seed}: R={rr:.4f} " + " ".join(f"F{hh}={horizon_rmse[hh]:.4f}" for hh in HORIZONS))
            all_h.append((rr, horizon_rmse))

        results[sname]["Decoupled"] = {
            "rmse_r_mean": float(np.mean([x[0] for x in all_h])),
            "rmse_r_std": float(np.std([x[0] for x in all_h])),
        }
        for hh in HORIZONS:
            vals = [x[1][hh] for x in all_h]
            results[sname]["Decoupled"][f"F{hh}_mean"] = float(np.mean(vals))
            results[sname]["Decoupled"][f"F{hh}_std"] = float(np.std(vals))
        save_incremental(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")

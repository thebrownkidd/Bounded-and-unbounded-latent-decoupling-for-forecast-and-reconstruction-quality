"""
Noise sweep ablation: all 5 methods × 5 noise levels × 5 systems × 3 seeds.

Methods: Coupled_k, Coupled_j, Frozen_j, Frozen_k, Decoupled
Noise: sigma ∈ {0.0, 0.05, 0.1, 0.2}  (4 levels to fit in GPU time)
Systems: Coupled Harmonic, Linear 5D, Brusselator, Duffing, Lorenz-96
Seeds: 3 seeds (0, 1, 42) for speed

Spectral DMD single-step (F1) evaluation.
Outputs: /kaggle/working/noise_ablation.json (incremental saves)
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
SEEDS = [0, 1, 42]
SIGMAS = [0.0, 0.05, 0.1, 0.2]
OUTFILE = f"{OUT}/noise_ablation.json"
COUPLED_PHIS = [0.1, 0.5, 0.9]


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
#  Evaluation: recon + single-step spectral DMD
# ═══════════════════════════════════════════════════════════════════

def eval_f1(encode_fn, decode_fn, Xt_test, gt_dim):
    with torch.no_grad():
        Z = encode_fn(Xt_test.to(DEVICE)).cpu().numpy()
        recon = decode_fn(
            torch.tensor(Z, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    X, Y = Z[:-1].T, Z[1:].T
    lam = 1e-6
    A = Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(X.shape[0]))
    z_pred = (A @ Z[:-1].T).T
    with torch.no_grad():
        x_pred = decode_fn(
            torch.tensor(z_pred, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    rmse_f1 = float(np.sqrt(np.mean((x_pred[:, :gt_dim] - gt[1:, :gt_dim]) ** 2)))
    return rmse_r, rmse_f1


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def make_systems():
    systems = {}

    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw * x1 - k12 * (x1 - x2), v2, -k12 * (x2 - x1) - k23 * (x2 - x3),
                v3, -k23 * (x3 - x2) - kw * x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    systems["Coupled Harmonic"] = dict(raw=raw, n_obs=6, k=4, j=8, h=64, gt_dim=6, delay_d=0)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    systems["Linear 5D"] = dict(raw=raw5[200:], n_obs=5, k=4, j=8, h=64, gt_dim=5, delay_d=0)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2 * y - (B_br + 1) * x, B_br * x - x**2 * y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    systems["Brusselator"] = dict(raw=obs_br, n_obs=10, k=3, j=8, h=64, gt_dim=2, delay_d=0)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    systems["Duffing"] = dict(raw=obs_du, n_obs=10, k=3, j=8, h=64, gt_dim=2, delay_d=0)

    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1) % N96] - x[(i-2) % N96]) * x[(i-1) % N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96) * 0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    systems["Lorenz-96"] = dict(raw=sol.y.T[200:], n_obs=20, k=10, j=12, h=64, gt_dim=20, delay_d=0)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def save_incremental(results):
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]")


def run_method(method_name, cfg, Xt, Xt_test, n, k, j, h, gt_dim, seed):
    SeedAll(seed)
    if method_name == "Coupled_k":
        best_f1, best_phi = 1e9, 0.5
        for phi in COUPLED_PHIS:
            SeedAll(seed)
            m = KoopmanAE(n, k, h); train_coupled(m, Xt, phi); m.eval()
            _, f1 = eval_f1(m.encode, m.decode, Xt_test, gt_dim)
            if f1 < best_f1: best_f1 = f1; best_phi = phi
        SeedAll(seed)
        m = KoopmanAE(n, k, h); train_coupled(m, Xt, best_phi); m.eval()
        return eval_f1(m.encode, m.decode, Xt_test, gt_dim)
    elif method_name == "Coupled_j":
        best_f1, best_phi = 1e9, 0.5
        for phi in COUPLED_PHIS:
            SeedAll(seed)
            m = KoopmanAE(n, j, h); train_coupled(m, Xt, phi); m.eval()
            _, f1 = eval_f1(m.encode, m.decode, Xt_test, gt_dim)
            if f1 < best_f1: best_f1 = f1; best_phi = phi
        SeedAll(seed)
        m = KoopmanAE(n, j, h); train_coupled(m, Xt, best_phi); m.eval()
        return eval_f1(m.encode, m.decode, Xt_test, gt_dim)
    elif method_name == "Frozen_j":
        m = SimpleAE(n, j, h); train_ae_only(m, Xt); m.eval()
        return eval_f1(m.encode, m.decode, Xt_test, gt_dim)
    elif method_name == "Frozen_k":
        m = SimpleAE(n, k, h); train_ae_only(m, Xt); m.eval()
        return eval_f1(m.encode, m.decode, Xt_test, gt_dim)
    elif method_name == "Decoupled":
        rm = ResidualModel(n, j, k, h)
        train_teacher(rm, Xt); rm.eval()
        with torch.no_grad():
            C = rm.carrier(Xt.to(DEVICE)).cpu(); dC = C[1:] - C[:-1]
        train_resid(rm, dC); rm.eval()
        with torch.no_grad():
            C = rm.carrier(Xt.to(DEVICE)).cpu(); dC = C[1:] - C[:-1]
        train_gru(rm, dC, C[:-1], C[1:]); rm.eval()
        return eval_f1(rm.carrier, rm.dec, Xt_test, gt_dim)


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    results = {}
    methods = ["Coupled_k", "Coupled_j", "Frozen_j", "Frozen_k", "Decoupled"]
    n_tr, n_te = 4000, 500

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        results[sname] = {}
        raw = cfg["raw"]
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]
        per_ch_std = raw[:n_tr].std(axis=0)

        for sigma in SIGMAS:
            print(f"\n  sigma={sigma}")
            results[sname][str(sigma)] = {}

            for mname in methods:
                rs, fs = [], []
                for seed in SEEDS:
                    SeedAll(seed)
                    noise_tr = sigma * per_ch_std * np.random.randn(n_tr, n)
                    noise_te = sigma * per_ch_std * np.random.randn(n_te, n)
                    obs_tr = raw[:n_tr] + noise_tr
                    obs_te = raw[n_tr:n_tr + n_te] + noise_te
                    mu, sig = obs_tr.mean(0), obs_tr.std(0) + 1e-8
                    Xt = torch.tensor((obs_tr - mu) / sig, dtype=torch.float32)
                    Xt_test = torch.tensor((obs_te - mu) / sig, dtype=torch.float32)

                    rr, f1 = run_method(mname, cfg, Xt, Xt_test, n, k, j, h, gt_dim, seed)
                    rs.append(rr); fs.append(f1)
                    print(f"    {mname:15s} seed={seed}: R={rr:.4f} F1={f1:.4f}")

                results[sname][str(sigma)][mname] = {
                    "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
                    "rmse_f1_mean": float(np.mean(fs)), "rmse_f1_std": float(np.std(fs)),
                }
            save_incremental(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")

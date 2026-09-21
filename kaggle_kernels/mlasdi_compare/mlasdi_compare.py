"""
mLaSDI comparison for ICLR 2027.

Two new methods (Coupled_k and Decoupled already in ablation_baselines.json):
  1. mLaSDI_standalone — 2-level sequential AEs on x-space residuals,
     spectral DMD on stacked latent [z₁;z₂]
  2. Decoupled_mLaSDI — our carrier-residual pipeline + 2nd-level
     spectral DMD correction on carrier-space DMD residuals

5 systems × 2 methods × 5 seeds.
Multi-horizon spectral DMD at [1, 5, 10, 25, 50, 100, 250, 500].

Outputs: /kaggle/working/mlasdi_compare.json (incremental saves)
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
OUTFILE = f"{OUT}/mlasdi_compare.json"


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
#  Spectral DMD utilities
# ═══════════════════════════════════════════════════════════════════

def fit_spectral_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    lam = 1e-6
    A = Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(X.shape[0]))
    if not np.all(np.isfinite(A)):
        return None
    eigvals, V = np.linalg.eig(A)
    eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
    V_inv = np.linalg.inv(V)
    return V, eigvals, V_inv, A


def spectral_predict(V, eigvals, V_inv, z0, tau):
    c = V_inv @ z0
    return np.real(V @ (c * eigvals ** tau))


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_mlasdi(ae1, ae2, Xt_test, gt_dim, horizons):
    """mLaSDI 2-level: stack [z₁;z₂], spectral DMD, split & decode & sum."""
    ae1.eval(); ae2.eval()
    with torch.no_grad():
        Z1 = ae1.encode(Xt_test.to(DEVICE)).cpu().numpy()
        recon1 = ae1.decode(
            torch.tensor(Z1, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        residuals = Xt_test.numpy() - recon1
        Z2 = ae2.encode(
            torch.tensor(residuals, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        recon2 = ae2.decode(
            torch.tensor(Z2, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    recon_full = recon1 + recon2
    rmse_r = float(np.sqrt(np.mean((recon_full[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    k1 = Z1.shape[1]
    Z_stacked = np.concatenate([Z1, Z2], axis=1)

    dmd = fit_spectral_dmd(Z_stacked)
    if dmd is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V, eigvals, V_inv, _ = dmd

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(Z_stacked) - h, 200)
        for s in range(n_starts):
            zs_pred = spectral_predict(V, eigvals, V_inv, Z_stacked[s], h)
            z1_pred = zs_pred[:k1]
            z2_pred = zs_pred[k1:]
            with torch.no_grad():
                x1 = ae1.decode(
                    torch.tensor(z1_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
                x2 = ae2.decode(
                    torch.tensor(z2_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            x_pred = x1 + x2
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


def eval_decoupled_mlasdi(model, Xt_test, gt_dim, horizons):
    """Decoupled + mLaSDI: 2-level spectral DMD in carrier space.
    Level 1: DMD on carrier trajectory.
    Level 2: DMD on one-step carrier residuals from Level 1."""
    model.eval()
    with torch.no_grad():
        Z = model.carrier(Xt_test.to(DEVICE)).cpu().numpy()
        recon = model.dec(
            torch.tensor(Z, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    dmd1 = fit_spectral_dmd(Z)
    if dmd1 is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V1, eig1, Vi1, A1 = dmd1

    Z_pred1 = (A1 @ Z[:-1].T).T
    delta = Z[1:] - Z_pred1
    dmd2 = fit_spectral_dmd(delta)
    if dmd2 is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V2, eig2, Vi2, _ = dmd2

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(delta) - h, 200)
        for s in range(n_starts):
            z1_pred = spectral_predict(V1, eig1, Vi1, Z[s], h)
            d2_pred = spectral_predict(V2, eig2, Vi2, delta[s], h)
            z_combined = z1_pred + d2_pred
            with torch.no_grad():
                x_pred = model.dec(
                    torch.tensor(z_combined, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig, gt_dim, mu, sig

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
    trn, ten, gd, mu, sig = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    raw = raw5[200:]
    trn, ten, gd, mu, sig = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2 * y - (B_br + 1) * x, B_br * x - x**2 * y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    trn, ten, gd, mu, sig = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    trn, ten, gd, mu, sig = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 500)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

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
    trn, ten, gd, mu, sig = norm_split(raw_l96, raw_l96, 20, 4000, 500)
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


def aggregate_horizons(all_results, horizons):
    out = {
        "rmse_r_mean": float(np.mean([x[0] for x in all_results])),
        "rmse_r_std": float(np.std([x[0] for x in all_results])),
    }
    for h in horizons:
        vals = [x[1][h] for x in all_results]
        out[f"F{h}_mean"] = float(np.mean(vals))
        out[f"F{h}_std"] = float(np.std(vals))
    return out


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()

    try:
        with open(OUTFILE) as f:
            results = json.load(f)
        print(f"Loaded existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        if sname not in results:
            results[sname] = {}

        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]

        # ── mLaSDI standalone (2-level) ──
        if "mLaSDI" not in results[sname]:
            print("\n  [1] mLaSDI (2-level standalone)")
            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                ae1 = SimpleAE(n, k, h)
                train_ae_only(ae1, Xt); ae1.eval()

                with torch.no_grad():
                    z1 = ae1.encode(Xt.to(DEVICE))
                    recon1 = ae1.decode(z1).cpu()
                residuals = Xt - recon1

                SeedAll(seed + 10000)
                ae2 = SimpleAE(n, k, h)
                train_ae_only(ae2, residuals); ae2.eval()

                rr, hrm = eval_mlasdi(ae1, ae2, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F100={hrm[100]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["mLaSDI"] = aggregate_horizons(all_r, HORIZONS)
            results[sname]["mLaSDI"]["stacked_latent_dim"] = 2 * k
            save_incremental(results)
        else:
            print("  [1] mLaSDI — skipped (exists)")

        # ── Decoupled + mLaSDI correction ──
        if "Decoupled_mLaSDI" not in results[sname]:
            print("\n  [2] Decoupled + mLaSDI correction")
            all_r = []
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

                rr, hrm = eval_decoupled_mlasdi(rm, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F100={hrm[100]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["Decoupled_mLaSDI"] = aggregate_horizons(all_r, HORIZONS)
            save_incremental(results)
        else:
            print("  [2] Decoupled_mLaSDI — skipped (exists)")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    methods = ["mLaSDI", "Decoupled_mLaSDI"]
    print(f"\n{'System':20s} {'Method':22s} {'Recon':>8s}", end="")
    for hh in HORIZONS:
        print(f" {'F'+str(hh):>8s}", end="")
    print()
    print("-" * (20 + 22 + 9 + 9 * len(HORIZONS)))
    for sname in results:
        for mname in methods:
            if mname in results[sname]:
                r = results[sname][mname]
                print(f"{sname:20s} {mname:22s} {r['rmse_r_mean']:8.4f}", end="")
                for hh in HORIZONS:
                    print(f" {r[f'F{hh}_mean']:8.4f}", end="")
                print()
        print()
    print("Done.")

"""
Ablation baselines for ICLR 2027.

Tests whether the carrier-residual decomposition matters beyond just freezing:

1. Frozen-j AE + DMD (Met2Net/LatentTSF protocol):
   Train AE with j-dim latent (same as carrier), freeze, fit DMD post-hoc.
   This is EXACTLY the Met2Net/LatentTSF approach applied to our setting.
   If this matches our full pipeline, our contribution is just "freeze the AE."

2. Coupled j-dim AE + DMD:
   Train AE with j-dim latent (same capacity as carrier) jointly with DMD.
   Controls for capacity: if j-dim coupled matches j-dim frozen, freezing helps.
   If j-dim coupled beats k-dim coupled, it's capacity not decoupling.

3. Coupled k-dim AE + DMD (existing baseline):
   Standard coupled AE with k-dim latent. Already in the paper.

4. Decoupled carrier-residual (our method):
   Full pipeline: Phase 1 (j-dim AE) -> Phase 2 (k-dim residual) -> Phase 3 (GRU).

All evaluated with both single-step and spectral DMD (500-step).

Outputs:
    Paper/ablation_baselines.json — incremental saves after each system
"""
import sys, json, time, argparse
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
OUTFILE = OUT / "ablation_baselines.json"
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--system", nargs="+", default=None)
args = parser.parse_args()

if args.quick:
    EPOCHS = 5; SEEDS = [0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class SimpleAE(nn.Module):
    """AE with configurable latent dim. Used for both k-dim and j-dim."""
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
    """AE + linear Koopman operator for coupled training."""
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
    """Carrier-residual architecture."""
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
    """Train AE for reconstruction only (Phase 1 / frozen protocol)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]].to(device)
            loss = nn.functional.mse_loss(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()


def train_coupled(model, Xt, phi):
    """Joint (1-phi)*L_rec + phi*L_fcst training."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl].to(device), Xt[sl + 1].to(device)
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_fcst
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_teacher(model, Xt):
    model.to(device)
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]].to(device)
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
            dc = dC[idx[i:i + BS]].to(device)
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
            with torch.no_grad(): dh = model.m(model.f(dC[sl].to(device)))
            loss = nn.functional.mse_loss(
                model.gru(dh, Cc[sl].to(device)), Cn[sl].to(device))
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  DMD + spectral evaluation
# ═══════════════════════════════════════════════════════════════════

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)


def spectral_forecast(A, z0, steps):
    """Analytical forecast via eigendecomposition: z_{T+tau} = V Lambda^tau V^{-1} z_T."""
    eigvals, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    c = V_inv @ z0
    out = np.empty((steps, len(z0)))
    for tau in range(steps):
        out[tau] = np.real(V @ (c * eigvals ** tau))
    return out


def eval_all(model_encode, model_decode, Xt_test, gt_dim, fcst_steps):
    """Evaluate recon + single-step DMD + spectral 500-step."""
    with torch.no_grad():
        Z_test = model_encode(Xt_test.to(device)).cpu().numpy()
        recon = model_decode(
            torch.tensor(Z_test, dtype=torch.float32).to(device)).cpu().numpy()

    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    # Post-hoc DMD
    A = fit_dmd(Z_test)

    # Single-step forecast
    z_pred = (A @ Z_test[:-1].T).T
    with torch.no_grad():
        x_pred = model_decode(
            torch.tensor(z_pred, dtype=torch.float32).to(device)).cpu().numpy()
    rmse_f1 = float(np.sqrt(np.mean((x_pred[:, :gt_dim] - gt[1:, :gt_dim]) ** 2)))

    # Spectral 500-step forecast
    z0 = Z_test[0]
    z_spectral = spectral_forecast(A, z0, fcst_steps)
    with torch.no_grad():
        x_spectral = model_decode(
            torch.tensor(z_spectral, dtype=torch.float32).to(device)).cpu().numpy()
    gt_spectral = gt[:fcst_steps]
    rmse_f500 = float(np.sqrt(np.mean(
        (x_spectral[:, :gt_dim] - gt_spectral[:, :gt_dim]) ** 2)))

    return rmse_r, rmse_f1, rmse_f500


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw[n_tr:n_tr + n_te, :gt_dim]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig, gt, mu, sig

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
    trn, ten, gt, mu, sig = norm_split(raw, raw, 6, 4000, 500)
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
    trn, ten, gt, mu, sig = norm_split(raw, raw, 5, 4000, 500)
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
    trn, ten, gt, mu, sig = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 500)
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
    trn, ten, gt, mu, sig = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 500)
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
    trn, ten, gt, mu, sig = norm_split(raw_l96, raw_l96, 20, 4000, 500)
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
    if args.system:
        systems = {k: v for k, v in systems.items() if k in args.system}
    t0 = time.time()

    # Load existing results for incremental save
    results = {}
    if OUTFILE.exists():
        with open(OUTFILE) as f:
            results = json.load(f)

    FCST_STEPS = 500
    COUPLED_PHIS = [0.1, 0.5, 1.0]

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        if sname not in results:
            results[sname] = {}

        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]

        # ── Method 1: Coupled k-dim AE + DMD (existing baseline) ──
        print("\n  [1] Coupled k-dim AE + DMD")
        rs = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
        for phi in COUPLED_PHIS:
            rs_phi = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(n, k, h)
                train_coupled(model, Xt, phi); model.eval()
                rr, f1, f500 = eval_all(model.encode, model.decode, Xt_test, gt_dim, FCST_STEPS)
                rs_phi["rmse_r"].append(rr); rs_phi["rmse_f1"].append(f1); rs_phi["rmse_f500"].append(f500)
                print(f"    phi={phi} seed={seed}: R={rr:.4f} F1={f1:.4f} F500={f500:.4f}")
            # Track best phi by F500
            mean_f500 = np.mean(rs_phi["rmse_f500"])
            if not rs["rmse_r"] or mean_f500 < np.mean(rs["rmse_f500"]):
                rs = rs_phi
        results[sname]["Coupled_k"] = {
            "rmse_r_mean": float(np.mean(rs["rmse_r"])), "rmse_r_std": float(np.std(rs["rmse_r"])),
            "rmse_f1_mean": float(np.mean(rs["rmse_f1"])), "rmse_f1_std": float(np.std(rs["rmse_f1"])),
            "rmse_f500_mean": float(np.mean(rs["rmse_f500"])), "rmse_f500_std": float(np.std(rs["rmse_f500"])),
            "rmse_r_all": rs["rmse_r"], "rmse_f1_all": rs["rmse_f1"], "rmse_f500_all": rs["rmse_f500"],
        }
        save_incremental(results)

        # ── Method 2: Coupled j-dim AE + DMD (capacity control) ──
        print("\n  [2] Coupled j-dim AE + DMD")
        rs = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
        for phi in COUPLED_PHIS:
            rs_phi = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(n, j, h)  # j-dim latent!
                train_coupled(model, Xt, phi); model.eval()
                rr, f1, f500 = eval_all(model.encode, model.decode, Xt_test, gt_dim, FCST_STEPS)
                rs_phi["rmse_r"].append(rr); rs_phi["rmse_f1"].append(f1); rs_phi["rmse_f500"].append(f500)
                print(f"    phi={phi} seed={seed}: R={rr:.4f} F1={f1:.4f} F500={f500:.4f}")
            mean_f500 = np.mean(rs_phi["rmse_f500"])
            if not rs["rmse_r"] or mean_f500 < np.mean(rs["rmse_f500"]):
                rs = rs_phi
        results[sname]["Coupled_j"] = {
            "rmse_r_mean": float(np.mean(rs["rmse_r"])), "rmse_r_std": float(np.std(rs["rmse_r"])),
            "rmse_f1_mean": float(np.mean(rs["rmse_f1"])), "rmse_f1_std": float(np.std(rs["rmse_f1"])),
            "rmse_f500_mean": float(np.mean(rs["rmse_f500"])), "rmse_f500_std": float(np.std(rs["rmse_f500"])),
            "rmse_r_all": rs["rmse_r"], "rmse_f1_all": rs["rmse_f1"], "rmse_f500_all": rs["rmse_f500"],
        }
        save_incremental(results)

        # ── Method 3: Frozen j-dim AE + post-hoc DMD (Met2Net/LatentTSF protocol) ──
        print("\n  [3] Frozen j-dim AE + DMD (Met2Net/LatentTSF)")
        rs = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
        for seed in SEEDS:
            SeedAll(seed)
            model = SimpleAE(n, j, h)
            train_ae_only(model, Xt); model.eval()
            rr, f1, f500 = eval_all(model.encode, model.decode, Xt_test, gt_dim, FCST_STEPS)
            rs["rmse_r"].append(rr); rs["rmse_f1"].append(f1); rs["rmse_f500"].append(f500)
            print(f"    seed={seed}: R={rr:.4f} F1={f1:.4f} F500={f500:.4f}")
        results[sname]["Frozen_j"] = {
            "rmse_r_mean": float(np.mean(rs["rmse_r"])), "rmse_r_std": float(np.std(rs["rmse_r"])),
            "rmse_f1_mean": float(np.mean(rs["rmse_f1"])), "rmse_f1_std": float(np.std(rs["rmse_f1"])),
            "rmse_f500_mean": float(np.mean(rs["rmse_f500"])), "rmse_f500_std": float(np.std(rs["rmse_f500"])),
            "rmse_r_all": rs["rmse_r"], "rmse_f1_all": rs["rmse_f1"], "rmse_f500_all": rs["rmse_f500"],
        }
        save_incremental(results)

        # ── Method 4: Frozen k-dim AE + post-hoc DMD (freeze but small latent) ──
        print("\n  [4] Frozen k-dim AE + DMD")
        rs = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
        for seed in SEEDS:
            SeedAll(seed)
            model = SimpleAE(n, k, h)
            train_ae_only(model, Xt); model.eval()
            rr, f1, f500 = eval_all(model.encode, model.decode, Xt_test, gt_dim, FCST_STEPS)
            rs["rmse_r"].append(rr); rs["rmse_f1"].append(f1); rs["rmse_f500"].append(f500)
            print(f"    seed={seed}: R={rr:.4f} F1={f1:.4f} F500={f500:.4f}")
        results[sname]["Frozen_k"] = {
            "rmse_r_mean": float(np.mean(rs["rmse_r"])), "rmse_r_std": float(np.std(rs["rmse_r"])),
            "rmse_f1_mean": float(np.mean(rs["rmse_f1"])), "rmse_f1_std": float(np.std(rs["rmse_f1"])),
            "rmse_f500_mean": float(np.mean(rs["rmse_f500"])), "rmse_f500_std": float(np.std(rs["rmse_f500"])),
            "rmse_r_all": rs["rmse_r"], "rmse_f1_all": rs["rmse_f1"], "rmse_f500_all": rs["rmse_f500"],
        }
        save_incremental(results)

        # ── Method 5: Decoupled carrier-residual (ours) ──
        print("\n  [5] Decoupled carrier-residual (ours)")
        rs = {"rmse_r": [], "rmse_f1": [], "rmse_f500": []}
        for seed in SEEDS:
            SeedAll(seed)
            rm = ResidualModel(n, j, k, h)

            # Phase 1
            train_teacher(rm, Xt); rm.eval()
            with torch.no_grad():
                C = rm.carrier(Xt.to(device)).cpu()
                dC = C[1:] - C[:-1]

            # Phase 2
            train_resid(rm, dC); rm.eval()

            # Phase 3
            with torch.no_grad():
                C = rm.carrier(Xt.to(device)).cpu()
                dC = C[1:] - C[:-1]
            train_gru(rm, dC, C[:-1], C[1:]); rm.eval()

            # Eval: use carrier space for spectral DMD (same as paper)
            rr, f1, f500 = eval_all(rm.carrier, rm.dec, Xt_test, gt_dim, FCST_STEPS)
            rs["rmse_r"].append(rr); rs["rmse_f1"].append(f1); rs["rmse_f500"].append(f500)
            print(f"    seed={seed}: R={rr:.4f} F1={f1:.4f} F500={f500:.4f}")

        results[sname]["Decoupled"] = {
            "rmse_r_mean": float(np.mean(rs["rmse_r"])), "rmse_r_std": float(np.std(rs["rmse_r"])),
            "rmse_f1_mean": float(np.mean(rs["rmse_f1"])), "rmse_f1_std": float(np.std(rs["rmse_f1"])),
            "rmse_f500_mean": float(np.mean(rs["rmse_f500"])), "rmse_f500_std": float(np.std(rs["rmse_f500"])),
            "rmse_r_all": rs["rmse_r"], "rmse_f1_all": rs["rmse_f1"], "rmse_f500_all": rs["rmse_f500"],
        }
        save_incremental(results)

    # Final summary
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'System':20s} {'Method':25s} {'Recon':>10s} {'F1-step':>10s} {'F500-step':>10s}")
    print("-" * 80)
    for sname in results:
        for mname in results[sname]:
            r = results[sname][mname]
            print(f"{sname:20s} {mname:25s} "
                  f"{r['rmse_r_mean']:.4f}    "
                  f"{r['rmse_f1_mean']:.4f}    "
                  f"{r['rmse_f500_mean']:.4f}")
        print()
    print("Done.")

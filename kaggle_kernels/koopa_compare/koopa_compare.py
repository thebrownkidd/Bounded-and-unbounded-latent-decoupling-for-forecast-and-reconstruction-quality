"""
Koopa comparison for ICLR 2027.

Two methods (Coupled_k and Decoupled already in ablation_baselines.json):
  1. Koopa_AE — k-dim coupled AE with learned Koopman operator K,
     evaluated via spectral decomposition of K (vs post-hoc DMD in Coupled_k)
  2. Decoupled_Koopa — our carrier-residual + learned K on b-space
     (multi-step trained), evaluated by accumulating predicted carrier
     increments through m

5 systems × 2 methods × 5 seeds, multi-horizon spectral evaluation.

Outputs: Paper/koopa_compare.json (incremental saves)
"""
import sys, json, time, random
from pathlib import Path

ROOT = Path("/kaggle/working")


import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

OUT = Path("/kaggle/working")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 400; LR = 1e-3; BS = 512
K_EPOCHS = 1000; K_LR = 1e-2; K_MSTEPS = 10
SEEDS = [0, 1, 2, 42, 123]
COUPLED_PHIS = [0.1, 0.5, 1.0]
HORIZONS = [1, 5, 10, 25, 50, 100, 250, 500]
OUTFILE = Path("/kaggle/working/koopa_compare.json")


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

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


def train_koopa_K(B_train, k_dim):
    """Train a learned Koopman K on b-codes with multi-step loss.
    K^tau b_t should predict b_{t+tau} for tau=1..K_MSTEPS."""
    K = nn.Linear(k_dim, k_dim, bias=False).to(DEVICE)
    nn.init.eye_(K.weight)
    opt = torch.optim.Adam(K.parameters(), lr=K_LR)
    B = torch.tensor(B_train, dtype=torch.float32).to(DEVICE)
    max_start = len(B) - K_MSTEPS
    if max_start < 1:
        max_start = len(B) - 2
    for ep in range(1, K_EPOCHS + 1):
        idx = torch.randperm(max_start)[:min(512, max_start)]
        loss = 0.0
        for tau in range(1, min(K_MSTEPS + 1, len(B))):
            valid = idx[idx + tau < len(B)]
            if len(valid) == 0:
                break
            b_t = B[valid]
            b_target = B[valid + tau]
            b_pred = b_t
            for _ in range(tau):
                b_pred = K(b_pred)
            loss = loss + nn.functional.mse_loss(b_pred, b_target)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(K.parameters(), 1.0)
        opt.step()
    return K.weight.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
#  Spectral prediction utilities
# ═══════════════════════════════════════════════════════════════════

def spectral_decompose(A):
    """Eigendecompose A and clip eigenvalues to unit disk."""
    if not np.all(np.isfinite(A)):
        return None
    eigvals, V = np.linalg.eig(A)
    eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
    V_inv = np.linalg.inv(V)
    return V, eigvals, V_inv


def spectral_predict(V, eigvals, V_inv, z0, tau):
    c = V_inv @ z0
    return np.real(V @ (c * eigvals ** tau))


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_koopa_ae(model, Xt_test, gt_dim, horizons):
    """Koopa AE: use learned K (not post-hoc DMD) for spectral forecast."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt_test.to(DEVICE)).cpu().numpy()
        recon = model.decode(
            torch.tensor(Z, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    K_np = model.K.weight.detach().cpu().numpy()
    sd = spectral_decompose(K_np)
    if sd is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V, eigvals, V_inv = sd

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(Z) - h, 200)
        for s in range(n_starts):
            z_pred = spectral_predict(V, eigvals, V_inv, Z[s], h)
            with torch.no_grad():
                x_pred = model.decode(
                    torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


def eval_decoupled_koopa(model, K_np, Xt_test, gt_dim, horizons):
    """Decoupled + Koopa: learned K on b-space, accumulate carrier increments.
    Uses 2-frame init: (C[s], B[s]), predicts h steps ahead from C[s+1].
    Predicted C[s+1+h] = C[s+1] + sum_{tau=1}^{h} m(K^tau b_0)."""
    model.eval()
    with torch.no_grad():
        C_test = model.carrier(Xt_test.to(DEVICE)).cpu()
        recon = model.dec(C_test.to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    dC = C_test[1:] - C_test[:-1]
    with torch.no_grad():
        B_test = model.f(dC.to(DEVICE)).cpu().numpy()

    sd = spectral_decompose(K_np)
    if sd is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V, eigvals, V_inv = sd

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(B_test) - h, 200)
        for s in range(n_starts):
            b_0 = B_test[s]
            c = V_inv @ b_0

            taus = np.arange(1, h + 1)
            pow_mat = eigvals[None, :] ** taus[:, None]
            b_all = np.real((V @ (c[:, None] * pow_mat.T)).T)

            with torch.no_grad():
                dC_all = model.m(
                    torch.tensor(b_all, dtype=torch.float32).to(DEVICE))
                total_dC = dC_all.sum(dim=0)
                C_pred = C_test[s + 1].to(DEVICE) + total_dC
                x_pred = model.dec(C_pred.unsqueeze(0)).cpu().numpy()[0]

            errs.append((x_pred[:gt_dim] - gt[s + 1 + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig

def make_systems():
    systems = {}

    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw*x1 - k12*(x1-x2), v2, -k12*(x2-x1) - k23*(x2-x3),
                v3, -k23*(x3-x2) - kw*x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten = norm_split(raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000; raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    raw = raw5[200:]
    trn, ten = norm_split(raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s; return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]; obs_br = delay(raw_br, 5)
    trn, ten = norm_split(obs_br, 2, 4000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s; return [v, -delta_d*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]; obs_du = delay(raw_du, 5)
    trn, ten = norm_split(obs_du, 2, 4000, 500)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1)%N96] - x[(i-2)%N96]) * x[(i-1)%N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96)*0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_l96 = sol.y.T[200:]
    trn, ten = norm_split(raw_l96, 20, 4000, 500)
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


def aggregate(all_results, horizons):
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

        # ── Koopa AE (standalone) ──
        if "Koopa_AE" not in results[sname]:
            print("\n  [1] Koopa AE (learned K, spectral eval)")
            best_phi, best_f1 = None, 1e9
            for phi in COUPLED_PHIS:
                SeedAll(0)
                model = KoopmanAE(n, k, h)
                train_coupled(model, Xt, phi); model.eval()
                _, hrm = eval_koopa_ae(model, Xt_test, gt_dim, [1])
                print(f"    phi={phi}: F1={hrm[1]:.4f}")
                if hrm[1] < best_f1:
                    best_f1 = hrm[1]; best_phi = phi
            if best_phi is None:
                best_phi = 0.5
            print(f"    Best phi = {best_phi}")

            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(n, k, h)
                train_coupled(model, Xt, best_phi); model.eval()
                rr, hrm = eval_koopa_ae(model, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F100={hrm[100]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["Koopa_AE"] = aggregate(all_r, HORIZONS)
            results[sname]["Koopa_AE"]["best_phi"] = best_phi
            save_incremental(results)
        else:
            print("  [1] Koopa_AE — skipped (exists)")

        # ── Decoupled + Koopa head ──
        if "Decoupled_Koopa" not in results[sname]:
            print("\n  [2] Decoupled + Koopa head on b-space")
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

                # Phase 2.5: train Koopa K on b trajectory
                with torch.no_grad():
                    B_train = rm.f(dC.to(DEVICE)).cpu().numpy()
                print(f"    seed={seed}: training K (multi-step)...", end="", flush=True)
                K_np = train_koopa_K(B_train, k)
                print(" done")

                rr, hrm = eval_decoupled_koopa(rm, K_np, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F100={hrm[100]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["Decoupled_Koopa"] = aggregate(all_r, HORIZONS)
            save_incremental(results)
        else:
            print("  [2] Decoupled_Koopa — skipped (exists)")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    methods = ["Koopa_AE", "Decoupled_Koopa"]
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

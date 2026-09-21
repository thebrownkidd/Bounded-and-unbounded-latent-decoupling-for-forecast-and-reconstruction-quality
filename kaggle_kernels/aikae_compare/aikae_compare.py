"""
AIKAE comparison for ICLR 2027.

Two methods:
  1. AIKAE standalone — k-dim AE with learned Koopman K, trained with
     reconstruction + forward Koopman consistency + backward consistency losses.
     Evaluated via spectral decomposition of K.
  2. Decoupled_AIKAE — our carrier-residual + learned K on b-space with
     forward + backward Koopman consistency losses (Phase 2.5).

5 systems × 2 methods × 5 seeds, multi-horizon spectral evaluation.
Outputs: /kaggle/working/aikae_compare.json
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
K_EPOCHS = 1000; K_LR = 1e-2
SEEDS = [0, 1, 2, 42, 123]
HORIZONS = [1, 5, 10, 25, 50, 100, 250]
OUTFILE = f"{OUT}/aikae_compare.json"


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class AIKAE(nn.Module):
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


def train_aikae(model, Xt, phi_rec=0.4, phi_fwd=0.3, phi_bwd=0.3):
    """Train AIKAE with reconstruction + forward + backward Koopman losses."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t = Xt[sl].to(DEVICE)
            x_tp1 = Xt[sl + 1].to(DEVICE)
            z_t = model.encode(x_t)
            z_tp1 = model.encode(x_tp1)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            L_fwd = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            L_bwd = nn.functional.mse_loss(
                torch.linalg.solve(
                    model.K.weight.detach().T @ model.K.weight.detach()
                    + 1e-4 * torch.eye(z_t.shape[1], device=DEVICE),
                    model.K.weight.detach().T @ z_tp1.detach().T
                ).T,
                z_t.detach()
            )
            loss = phi_rec * L_rec + phi_fwd * L_fwd + phi_bwd * L_bwd
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_aikae_simple(model, Xt, phi_rec=0.4, phi_fwd=0.4, phi_bwd=0.2):
    """Simpler AIKAE: use K^T as approximate backward operator."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t = Xt[sl].to(DEVICE)
            x_tp1 = Xt[sl + 1].to(DEVICE)
            z_t = model.encode(x_t)
            z_tp1 = model.encode(x_tp1)
            L_rec = 0.5 * (nn.functional.mse_loss(model.decode(z_t), x_t)
                           + nn.functional.mse_loss(model.decode(z_tp1), x_tp1))
            L_fwd = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            z_bwd = z_tp1 @ model.K.weight
            L_bwd = nn.functional.mse_loss(z_bwd, z_t.detach())
            loss = phi_rec * L_rec + phi_fwd * L_fwd + phi_bwd * L_bwd
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
            with torch.no_grad():
                dh = model.m(model.f(dC[sl].to(DEVICE)))
            loss = nn.functional.mse_loss(
                model.gru(dh, Cc[sl].to(DEVICE)), Cn[sl].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()


def train_aikae_K(B_train, k_dim):
    """Train learned K on b-codes with forward + backward Koopman losses."""
    K = nn.Linear(k_dim, k_dim, bias=False).to(DEVICE)
    nn.init.eye_(K.weight)
    opt = torch.optim.Adam(K.parameters(), lr=K_LR)
    B = torch.tensor(B_train, dtype=torch.float32).to(DEVICE)
    for ep in range(1, K_EPOCHS + 1):
        idx = torch.randperm(len(B) - 1)[:min(512, len(B) - 1)]
        b_t = B[idx]
        b_tp1 = B[idx + 1]
        L_fwd = nn.functional.mse_loss(K(b_t), b_tp1)
        b_bwd = b_tp1 @ K.weight
        L_bwd = nn.functional.mse_loss(b_bwd, b_t)
        loss = 0.6 * L_fwd + 0.4 * L_bwd
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(K.parameters(), 1.0)
        opt.step()
    return K.weight.detach().cpu().numpy()


def spectral_decompose(A):
    if not np.all(np.isfinite(A)):
        return None
    eigvals, V = np.linalg.eig(A)
    eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
    V_inv = np.linalg.inv(V)
    return V, eigvals, V_inv


def spectral_predict(V, eigvals, V_inv, z0, tau):
    c = V_inv @ z0
    return np.real(V @ (c * eigvals ** tau))


def eval_aikae(model, Xt_test, gt_dim, horizons):
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
        if n_starts < 1:
            rmse_per_h[h] = float('nan')
            continue
        for s in range(n_starts):
            z_pred = spectral_predict(V, eigvals, V_inv, Z[s], h)
            with torch.no_grad():
                x_pred = model.decode(
                    torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


def eval_decoupled_aikae(model, K_np, Xt_test, gt_dim, horizons):
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
        if n_starts < 1:
            rmse_per_h[h] = float('nan')
            continue
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


def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

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
    systems["Coupled Harmonic"] = dict(raw=raw, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05); N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    systems["Linear 5D"] = dict(raw=raw5[200:], n_obs=5, k=4, j=8, h=64, gt_dim=5)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s; return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    systems["Brusselator"] = dict(raw=delay(sol.y.T[200:], 5), n_obs=10, k=3, j=8, h=64, gt_dim=2)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s; return [v, -delta_d*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    systems["Duffing"] = dict(raw=delay(sol.y.T[200:], 5), n_obs=10, k=3, j=8, h=64, gt_dim=2)

    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1)%N96] - x[(i-2)%N96]) * x[(i-1)%N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96)*0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    systems["Lorenz-96"] = dict(raw=sol.y.T[200:], n_obs=20, k=10, j=12, h=64, gt_dim=20)
    return systems


def save_incremental(results):
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved]")


def aggregate(all_results, horizons):
    out = {
        "rmse_r_mean": float(np.mean([x[0] for x in all_results])),
        "rmse_r_std": float(np.std([x[0] for x in all_results])),
    }
    for h in horizons:
        vals = [x[1][h] for x in all_results if not np.isnan(x[1].get(h, float('nan')))]
        if vals:
            out[f"F{h}_mean"] = float(np.mean(vals))
            out[f"F{h}_std"] = float(np.std(vals))
        else:
            out[f"F{h}_mean"] = float('nan')
            out[f"F{h}_std"] = float('nan')
    return out


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    n_tr, n_te = 4000, 500

    try:
        with open(OUTFILE) as f:
            results = json.load(f)
        print("Loaded existing results")
    except (FileNotFoundError, json.JSONDecodeError):
        results = {}

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        if sname not in results:
            results[sname] = {}
        raw = cfg["raw"]
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]
        mu, sig = raw[:n_tr].mean(0), raw[:n_tr].std(0) + 1e-8
        Xt = torch.tensor((raw[:n_tr] - mu) / sig, dtype=torch.float32)
        Xt_test = torch.tensor((raw[n_tr:n_tr + n_te] - mu) / sig, dtype=torch.float32)

        # ── AIKAE standalone ──
        if "AIKAE" not in results[sname]:
            print("\n  [1] AIKAE standalone")
            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                model = AIKAE(n, k, h)
                train_aikae_simple(model, Xt)
                model.eval()
                rr, hrm = eval_aikae(model, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F10={hrm[10]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["AIKAE"] = aggregate(all_r, HORIZONS)
            save_incremental(results)
        else:
            print("  [1] AIKAE — skipped (exists)")

        # ── Decoupled + AIKAE head ──
        if "Decoupled_AIKAE" not in results[sname]:
            print("\n  [2] Decoupled + AIKAE head on b-space")
            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                rm = ResidualModel(n, j, k, h)
                train_teacher(rm, Xt)
                rm.eval()
                with torch.no_grad():
                    C = rm.carrier(Xt.to(DEVICE)).cpu()
                    dC = C[1:] - C[:-1]
                train_resid(rm, dC)
                rm.eval()
                with torch.no_grad():
                    C = rm.carrier(Xt.to(DEVICE)).cpu()
                    dC = C[1:] - C[:-1]
                train_gru(rm, dC, C[:-1], C[1:])
                rm.eval()
                with torch.no_grad():
                    B_train = rm.f(dC.to(DEVICE)).cpu().numpy()
                print(f"    seed={seed}: training AIKAE K...", end="", flush=True)
                K_np = train_aikae_K(B_train, k)
                print(" done")
                rr, hrm = eval_decoupled_aikae(rm, K_np, Xt_test, gt_dim, HORIZONS)
                print(f"    seed={seed}: R={rr:.4f} F1={hrm[1]:.4f} F10={hrm[10]:.4f}")
                all_r.append((rr, hrm))
            results[sname]["Decoupled_AIKAE"] = aggregate(all_r, HORIZONS)
            save_incremental(results)
        else:
            print("  [2] Decoupled_AIKAE — skipped (exists)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")

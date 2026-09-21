"""
Comprehensive param-matched experiments for ICLR 2027.
Covers everything NOT in param_matched.json:
  A. Head comparison (Coupled MLP, Coupled NeuralODE)
  B. Ablation (Coupled_k, Coupled_j)
  C. Multi-horizon (Coupled spectral DMD at 6 horizons)
  D. KS-64D (Coupled)
  E. mLaSDI standalone
  F. Koopa standalone (spectral eval, R/F1/F10)
  G. AIKAE standalone spectral eval for F10

All coupled/standalone models get increased h to match Decoupled param count.
Decoupled stays at h=64. Latent dims (k, j) never change.

Outputs: Paper/param_matched_all.json (incremental saves)
"""
import sys, json, time, random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
OUTFILE = OUT / "param_matched_all.json"

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
HORIZONS = [1, 5, 10, 25, 50, 100]


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.K = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))
    def predict(self, z): return self.K(z)


class CoupledAE(nn.Module):
    def __init__(self, n, k, h, head_cls, head_kw=None):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.head = head_cls(k, **(head_kw or {}))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def predict(self, z): return self.head(z)


class MLPHead(nn.Module):
    def __init__(self, k, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, k))
    def forward(self, b): return self.net(b)
    def step_numpy(self, b):
        with torch.no_grad():
            return self.forward(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


class ODEFunc(nn.Module):
    def __init__(self, k, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, k))
    def forward(self, b): return self.net(b)


class NeuralODEHead(nn.Module):
    def __init__(self, k, hidden=64, n_steps=2):
        super().__init__()
        self.func = ODEFunc(k, hidden)
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
    def forward(self, b):
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self.func(b); k2 = self.func(b + 0.5*dt*k1)
            k3 = self.func(b + 0.5*dt*k2); k4 = self.func(b + dt*k3)
            b = b + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return b
    def step_numpy(self, b):
        with torch.no_grad():
            return self.forward(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_h_for_target(model_fn, target, h_start=64):
    for h in range(h_start, 300):
        if count_params(model_fn(h)) >= target:
            return h, count_params(model_fn(h))
    return h_start, count_params(model_fn(h_start))


# ═══════════════════════════════════════════════════════════════════
#  Spectral DMD
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
    return V, eigvals, V_inv


def spectral_predict(V, eigvals, V_inv, z0, tau):
    c = V_inv @ z0
    return np.real(V @ (c * eigvals ** tau))


# ═══════════════════════════════════════════════════════════════════
#  Training functions
# ═══════════════════════════════════════════════════════════════════

def train_coupled(model, Xt, phi=0.5):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_fcst
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_ae_only(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]]
            loss = nn.functional.mse_loss(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()


def train_koopa_K(model, Xt, K_EPOCHS=1000, K_LR=0.01, K_MSTEPS=10):
    """Freeze AE, train K with multi-step autoregressive loss (Koopa-style)."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt)
    K = nn.Linear(Z.shape[1], Z.shape[1], bias=False)
    opt = torch.optim.Adam(K.parameters(), lr=K_LR)
    for ep in range(K_EPOCHS):
        max_start = len(Z) - K_MSTEPS
        idx = torch.randperm(max_start)[:min(BS, max_start)]
        loss = torch.tensor(0.0)
        z = Z[idx]
        for s in range(1, K_MSTEPS + 1):
            z = K(z)
            loss = loss + nn.functional.mse_loss(z, Z[idx + s].detach())
        loss = loss / K_MSTEPS
        opt.zero_grad(); loss.backward(); opt.step()
    model.K.weight.data = K.weight.data.clone()
    return model


def train_aikae(model, Xt, phi_rec=0.4, phi_fwd=0.3, phi_bwd=0.3):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            z_t = model.encode(x_t)
            z_tp1 = model.encode(x_tp1)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            L_fwd = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            L_bwd = nn.functional.mse_loss(
                nn.functional.linear(z_tp1, model.K.weight.t()), z_t.detach())
            loss = phi_rec * L_rec + phi_fwd * L_fwd + phi_bwd * L_bwd
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_coupled_f1(model, Xt_test, gt_dim):
    model.eval()
    with torch.no_grad():
        z = model.encode(Xt_test)
        recon = model.decode(z).numpy()
        z_pred = model.predict(z[:-1])
        fcst = model.decode(z_pred).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))
    rmse_f = float(np.sqrt(np.mean((fcst[:, :gt_dim] - gt[1:, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


def eval_coupled_spectral(model, Xt_test, gt_dim, horizons, use_learned_K=False):
    """Spectral multi-horizon evaluation for coupled model."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt_test).numpy()
        recon = model.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    if use_learned_K and hasattr(model, 'K'):
        K_np = model.K.weight.detach().numpy()
        eigvals, V = np.linalg.eig(K_np)
        eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
        V_inv = np.linalg.inv(V)
        sd = (V, eigvals, V_inv)
    else:
        sd = fit_spectral_dmd(Z)

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
                    torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


def eval_coupled_head_rollout(model, Xt_test, gt_dim, fcst_steps=50):
    """Rollout evaluation for MLP/ODE heads."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt_test).numpy()
        recon = model.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    z0 = Z[0]
    out = np.empty((fcst_steps, Z.shape[1])); out[0] = z0
    for t in range(1, fcst_steps):
        out[t] = model.head.step_numpy(out[t-1])
    with torch.no_grad():
        x_pred = model.decode(torch.tensor(out, dtype=torch.float32)).numpy()
    rmse_f = float(np.sqrt(np.mean((x_pred[:, :gt_dim] - gt[:fcst_steps, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


def eval_mlasdi_standalone(ae1, ae2, Xt_test, gt_dim, horizons):
    ae1.eval(); ae2.eval()
    with torch.no_grad():
        Z1 = ae1.encode(Xt_test).numpy()
        recon1 = ae1.decode(torch.tensor(Z1, dtype=torch.float32)).numpy()
        residuals = Xt_test.numpy() - recon1
        Z2 = ae2.encode(torch.tensor(residuals, dtype=torch.float32)).numpy()
        recon2 = ae2.decode(torch.tensor(Z2, dtype=torch.float32)).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean(((recon1 + recon2)[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    k1 = Z1.shape[1]
    Z_stacked = np.concatenate([Z1, Z2], axis=1)
    sd = fit_spectral_dmd(Z_stacked)
    if sd is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V, eigvals, V_inv = sd

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(Z_stacked) - h, 200)
        for s in range(n_starts):
            zs = spectral_predict(V, eigvals, V_inv, Z_stacked[s], h)
            with torch.no_grad():
                x1 = ae1.decode(torch.tensor(zs[:k1], dtype=torch.float32).unsqueeze(0)).numpy()[0]
                x2 = ae2.decode(torch.tensor(zs[k1:], dtype=torch.float32).unsqueeze(0)).numpy()[0]
            errs.append(((x1 + x2)[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
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

def make_systems(include_ks=False):
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
    systems["Coupled Harmonic"] = dict(train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000; raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    trn, ten = norm_split(raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s; return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    obs = delay(sol.y.T[200:], 5)
    trn, ten = norm_split(obs, 2, 4000, 500)
    systems["Brusselator"] = dict(train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s; return [v, -delta_d*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    obs = delay(sol.y.T[200:], 5)
    trn, ten = norm_split(obs, 2, 4000, 500)
    systems["Duffing"] = dict(train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    N96, F96 = 20, 8.0
    def lorenz96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1)%N96] - x[(i-2)%N96]) * x[(i-1)%N96] - x[i] + F96
        return d
    x0 = np.random.RandomState(0).randn(N96) * 0.01; x0[0] = 1.0
    sol = solve_ivp(lorenz96, [0, 500], x0,
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_l96 = sol.y.T[200:]
    trn, ten = norm_split(raw_l96, 20, 4000, 500)
    systems["Lorenz-96"] = dict(train_n=trn, test_n=ten, n_obs=20, k=10, j=12, h=64, gt_dim=20)

    if include_ks:
        from scipy.fftpack import fft, ifft
        L, Nk = 22.0, 64
        dx = L / Nk
        x_grid = np.arange(Nk) * dx
        kk = np.fft.fftfreq(Nk, d=dx) * 2 * np.pi
        np.random.seed(42)
        u0 = np.cos(2 * np.pi * x_grid / L) + 0.1 * np.random.randn(Nk)
        dt_ks = 0.25; N_steps = 20000
        u = u0.copy()
        traj = [u.copy()]
        for _ in range(N_steps):
            u_hat = fft(u)
            lin = kk**2 - kk**4
            nl = -0.5 * 1j * kk * fft(u**2)
            u_hat1 = (u_hat + dt_ks * (lin * u_hat + nl)) / (1 - dt_ks * lin)
            u = np.real(ifft(u_hat1))
            traj.append(u.copy())
        raw_ks = np.array(traj[2000:])
        trn, ten = norm_split(raw_ks, 64, 12000, 2000)
        systems["KS-64D"] = dict(train_n=trn, test_n=ten, n_obs=64, k=16, j=32, h=64, gt_dim=64)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    systems = make_systems(include_ks=True)
    t0 = time.time()

    if OUTFILE.exists():
        with open(OUTFILE) as fp:
            results = json.load(fp)
    else:
        results = {}

    def save():
        with open(OUTFILE, "w") as fp:
            json.dump(results, fp, indent=2)

    # Compute all param targets
    print("=" * 70)
    print("  PARAM MATCHING")
    print("=" * 70)

    for sname, cfg in systems.items():
        n, k, j, h = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"]
        target = count_params(ResidualModel(n, j, k, h))

        # KoopmanAE matching (for coupled, Koopa, AIKAE, ablation)
        h_ae, p_ae = find_h_for_target(lambda hh: KoopmanAE(n, k, hh), target)
        cfg["h_ae"] = h_ae; cfg["p_ae"] = p_ae

        # CoupledAE+MLP matching (head adds params — match total)
        target_mlp = target + count_params(MLPHead(k))
        h_mlp, p_mlp = find_h_for_target(lambda hh: CoupledAE(n, k, hh, MLPHead), target_mlp)
        cfg["h_mlp"] = h_mlp; cfg["p_mlp"] = p_mlp

        # CoupledAE+ODE matching
        target_ode = target + count_params(NeuralODEHead(k))
        h_ode, p_ode = find_h_for_target(lambda hh: CoupledAE(n, k, hh, NeuralODEHead), target_ode)
        cfg["h_ode"] = h_ode; cfg["p_ode"] = p_ode

        # Coupled_j (AE with j-dim latent, no dynamics)
        h_j, p_j = find_h_for_target(lambda hh: SimpleAE(n, j, hh), target)
        cfg["h_j"] = h_j; cfg["p_j"] = p_j

        # mLaSDI: two SimpleAEs combined must match target
        def mlasdi_params(hh):
            return type('M', (), {'parameters': lambda self:
                list(SimpleAE(n, k, hh).parameters()) + list(SimpleAE(n, k, hh).parameters()),
                '__call__': lambda self, h: self})()
        # Manual search for mLaSDI
        for hh in range(30, 200):
            if 2 * count_params(SimpleAE(n, k, hh)) >= target:
                cfg["h_mlasdi"] = hh
                cfg["p_mlasdi"] = 2 * count_params(SimpleAE(n, k, hh))
                break

        print(f"  {sname:20s}  target={target:6d}  h_ae={h_ae}({p_ae})  "
              f"h_mlp={h_mlp}  h_ode={h_ode}  h_j={h_j}  h_mlasdi={cfg.get('h_mlasdi','?')}")

    # ═══════════════════════════════════════════════════════════════
    #  A. Head comparison: Coupled MLP + Coupled NeuralODE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  A. HEAD COMPARISON (MLP, NeuralODE)\n{'='*70}")

    five_sys = {k: v for k, v in systems.items() if k != "KS-64D"}
    for sname, cfg in five_sys.items():
        if sname not in results: results[sname] = {}
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)

        for head_name, head_cls, h_key in [("MLP", MLPHead, "h_mlp"), ("ODE", NeuralODEHead, "h_ode")]:
            key = f"Coupled_{head_name}_matched"
            if key in results[sname]:
                print(f"  Skipping {sname} {key}"); continue

            h_m = cfg[h_key]
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                model = CoupledAE(cfg["n_obs"], cfg["k"], h_m, head_cls)
                train_coupled(model, Xt, phi=0.5)
                r, f = eval_coupled_f1(model, Xt_test, cfg["gt_dim"])
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"    {sname:20s} {key:25s}  seed={seed}  R={r:.4f}  F1={f:.4f}", flush=True)
            results[sname][key] = {"rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "phi": 0.5, "h": h_m,
                "params": count_params(CoupledAE(cfg["n_obs"], cfg["k"], h_m, head_cls))}
            print(f"  {sname:20s} {key:30s}  R={results[sname][key]['rmse_r_mean']:.4f}  F1={results[sname][key]['rmse_f_mean']:.4f}", flush=True)
            save()

    # ═══════════════════════════════════════════════════════════════
    #  B. Ablation: Coupled_k, Coupled_j (param-matched)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  B. ABLATION (Coupled_k, Coupled_j)\n{'='*70}")

    for sname, cfg in five_sys.items():
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)

        # Coupled_k: AE with k-dim latent, trained phi=0.5
        key = "Coupled_k_matched"
        if key not in results[sname]:
            h_m = cfg["h_ae"]
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
                train_coupled(model, Xt, phi=0.5)
                r, f = eval_coupled_f1(model, Xt_test, cfg["gt_dim"])
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"  {sname:20s} {key:25s}  seed={seed}  R={r:.4f}  F1={f:.4f}")
            results[sname][key] = {"rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "h": h_m, "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m))}
            save()

        # Coupled_j: AE with j-dim latent, recon-only (phi=0)
        key = "Coupled_j_matched"
        if key not in results[sname]:
            h_m = cfg["h_j"]
            rmse_rs = []
            for seed in SEEDS:
                SeedAll(seed)
                model = SimpleAE(cfg["n_obs"], cfg["j"], h_m)
                train_ae_only(model, Xt)
                model.eval()
                with torch.no_grad():
                    recon = model(Xt_test).numpy()
                gt = Xt_test.numpy()
                r = float(np.sqrt(np.mean((recon[:, :cfg["gt_dim"]] - gt[:, :cfg["gt_dim"]]) ** 2)))
                rmse_rs.append(r)
                print(f"  {sname:20s} {key:25s}  seed={seed}  R={r:.4f}")
            results[sname][key] = {"rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "h": h_m, "params": count_params(SimpleAE(cfg["n_obs"], cfg["j"], h_m))}
            save()

    # ═══════════════════════════════════════════════════════════════
    #  C. Multi-horizon (Coupled spectral DMD)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  C. MULTI-HORIZON\n{'='*70}")

    for sname, cfg in five_sys.items():
        key = "Coupled_multihorizon_matched"
        if key in results[sname]:
            print(f"  Skipping {sname} {key}"); continue

        h_m = cfg["h_ae"]
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)

        all_horizons = {h: [] for h in HORIZONS}
        all_r = []
        for seed in SEEDS:
            SeedAll(seed)
            model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
            train_coupled(model, Xt, phi=0.5)
            rmse_r, rmse_per_h = eval_coupled_spectral(model, Xt_test, cfg["gt_dim"], HORIZONS)
            all_r.append(rmse_r)
            for h_val in HORIZONS:
                all_horizons[h_val].append(rmse_per_h[h_val])
            print(f"  {sname:20s} seed={seed}  R={rmse_r:.4f}  " +
                  "  ".join(f"F{h}={rmse_per_h[h]:.4f}" for h in HORIZONS[:4]))

        res = {"rmse_r_mean": float(np.mean(all_r)), "rmse_r_std": float(np.std(all_r)),
               "h": h_m, "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m))}
        for h_val in HORIZONS:
            res[f"F{h_val}_mean"] = float(np.mean(all_horizons[h_val]))
            res[f"F{h_val}_std"] = float(np.std(all_horizons[h_val]))
        results[sname][key] = res
        save()

    # ═══════════════════════════════════════════════════════════════
    #  D. KS-64D
    # ═══════════════════════════════════════════════════════════════
    if "KS-64D" in systems:
        print(f"\n{'='*70}\n  D. KS-64D\n{'='*70}")
        cfg = systems["KS-64D"]
        if "KS-64D" not in results: results["KS-64D"] = {}
        key = "Coupled_matched"
        if key not in results["KS-64D"]:
            h_m = cfg["h_ae"]
            Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
            Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
            ks_horizons = [1, 10, 50, 100]

            all_horizons = {h: [] for h in ks_horizons}
            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
                train_coupled(model, Xt, phi=0.5)
                rmse_r, rmse_per_h = eval_coupled_spectral(model, Xt_test, cfg["gt_dim"], ks_horizons)
                all_r.append(rmse_r)
                for hv in ks_horizons: all_horizons[hv].append(rmse_per_h[hv])
                print(f"  KS-64D  seed={seed}  R={rmse_r:.4f}  " +
                      "  ".join(f"F{hv}={rmse_per_h[hv]:.4f}" for hv in ks_horizons))

            res = {"rmse_r_mean": float(np.mean(all_r)), "rmse_r_std": float(np.std(all_r)),
                   "h": h_m, "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m))}
            for hv in ks_horizons:
                res[f"F{hv}_mean"] = float(np.mean(all_horizons[hv]))
                res[f"F{hv}_std"] = float(np.std(all_horizons[hv]))
            results["KS-64D"][key] = res
            save()

    # ═══════════════════════════════════════════════════════════════
    #  E. mLaSDI standalone
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  E. mLaSDI STANDALONE\n{'='*70}")

    for sname, cfg in five_sys.items():
        key = "mLaSDI_matched"
        if key in results[sname]:
            print(f"  Skipping {sname} {key}"); continue

        h_m = cfg["h_mlasdi"]
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        mlasdi_horizons = [1, 100]

        all_horizons = {h: [] for h in mlasdi_horizons}
        all_r = []
        for seed in SEEDS:
            SeedAll(seed)
            ae1 = SimpleAE(cfg["n_obs"], cfg["k"], h_m)
            train_ae_only(ae1, Xt)
            ae1.eval()
            with torch.no_grad():
                recon1 = ae1(Xt).numpy()
            residuals = Xt.numpy() - recon1
            ae2 = SimpleAE(cfg["n_obs"], cfg["k"], h_m)
            train_ae_only(ae2, torch.tensor(residuals, dtype=torch.float32))
            rmse_r, rmse_per_h = eval_mlasdi_standalone(ae1, ae2, Xt_test, cfg["gt_dim"], mlasdi_horizons)
            all_r.append(rmse_r)
            for hv in mlasdi_horizons: all_horizons[hv].append(rmse_per_h[hv])
            print(f"  {sname:20s} seed={seed}  R={rmse_r:.4f}  F1={rmse_per_h[1]:.4f}  F100={rmse_per_h[100]:.4f}")

        res = {"rmse_r_mean": float(np.mean(all_r)), "rmse_r_std": float(np.std(all_r)),
               "h": h_m, "params": 2 * count_params(SimpleAE(cfg["n_obs"], cfg["k"], h_m))}
        for hv in mlasdi_horizons:
            res[f"F{hv}_mean"] = float(np.mean(all_horizons[hv]))
            res[f"F{hv}_std"] = float(np.std(all_horizons[hv]))
        results[sname][key] = res
        save()

    # ═══════════════════════════════════════════════════════════════
    #  F. Koopa standalone (spectral, R/F1/F10)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  F. KOOPA STANDALONE\n{'='*70}")

    for sname, cfg in five_sys.items():
        key = "Koopa_matched"
        if key in results[sname]:
            print(f"  Skipping {sname} {key}"); continue

        h_m = cfg["h_ae"]
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        koopa_horizons = [1, 10]

        all_horizons = {h: [] for h in koopa_horizons}
        all_r = []
        for seed in SEEDS:
            SeedAll(seed)
            model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
            train_ae_only(model, Xt)
            train_koopa_K(model, Xt)
            rmse_r, rmse_per_h = eval_coupled_spectral(model, Xt_test, cfg["gt_dim"],
                                                        koopa_horizons, use_learned_K=True)
            all_r.append(rmse_r)
            for hv in koopa_horizons: all_horizons[hv].append(rmse_per_h[hv])
            print(f"  {sname:20s} seed={seed}  R={rmse_r:.4f}  F1={rmse_per_h[1]:.4f}  F10={rmse_per_h[10]:.4f}")

        res = {"rmse_r_mean": float(np.mean(all_r)), "rmse_r_std": float(np.std(all_r)),
               "h": h_m, "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m))}
        for hv in koopa_horizons:
            res[f"F{hv}_mean"] = float(np.mean(all_horizons[hv]))
            res[f"F{hv}_std"] = float(np.std(all_horizons[hv]))
        results[sname][key] = res
        save()

    # ═══════════════════════════════════════════════════════════════
    #  G. AIKAE standalone spectral (for F10)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}\n  G. AIKAE STANDALONE (spectral F10)\n{'='*70}")

    for sname, cfg in five_sys.items():
        key = "AIKAE_spectral_matched"
        if key in results[sname]:
            print(f"  Skipping {sname} {key}"); continue

        h_m = cfg["h_ae"]
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
        Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
        aikae_horizons = [1, 10]

        all_horizons = {h: [] for h in aikae_horizons}
        all_r = []
        for seed in SEEDS:
            SeedAll(seed)
            model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
            train_aikae(model, Xt)
            rmse_r, rmse_per_h = eval_coupled_spectral(model, Xt_test, cfg["gt_dim"],
                                                        aikae_horizons, use_learned_K=True)
            all_r.append(rmse_r)
            for hv in aikae_horizons: all_horizons[hv].append(rmse_per_h[hv])
            print(f"  {sname:20s} seed={seed}  R={rmse_r:.4f}  F1={rmse_per_h[1]:.4f}  F10={rmse_per_h[10]:.4f}")

        res = {"rmse_r_mean": float(np.mean(all_r)), "rmse_r_std": float(np.std(all_r)),
               "h": h_m, "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m))}
        for hv in aikae_horizons:
            res[f"F{hv}_mean"] = float(np.mean(all_horizons[hv]))
            res[f"F{hv}_std"] = float(np.std(all_horizons[hv]))
        results[sname][key] = res
        save()

    # ═══════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\nResults: {OUTFILE}")
    for sname in results:
        print(f"\n  {sname}:")
        for mname in sorted(results[sname]):
            r = results[sname][mname]
            parts = f"  R={r.get('rmse_r_mean', 0):.4f}"
            if 'rmse_f_mean' in r: parts += f"  F1={r['rmse_f_mean']:.4f}"
            if 'F1_mean' in r: parts += f"  F1={r['F1_mean']:.4f}"
            if 'F10_mean' in r: parts += f"  F10={r['F10_mean']:.4f}"
            if 'F100_mean' in r: parts += f"  F100={r['F100_mean']:.4f}"
            print(f"    {mname:35s} {parts}")

"""
Noise robustness experiment for ICLR 2027 §5.4.

Tests AE+DMD (coupled) vs Resid+GRU+DMD (decoupled) across observation noise
levels σ ∈ {0, 0.01, 0.05, 0.1, 0.2} on 4 systems, 5 seeds each.

Noise is additive Gaussian: x_noisy = x + σ · std(x_channel) · ε.
Applied AFTER the ODE solve, BEFORE train/test split and normalisation.

Outputs:
    Paper/noise_sweep.json — {system: {sigma: {method: {rmse_r_mean, ...}}}}
"""
import sys, json, time
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
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
SIGMAS = [0.0, 0.01, 0.05, 0.1, 0.2]


# ═══════════════════════════════════════════════════════════════════
#  Models (identical to run_multiseed.py)
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


class ResidualGRU(nn.Module):
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


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)


# ═══════════════════════════════════════════════════════════════════
#  Training (identical to run_multiseed.py)
# ═══════════════════════════════════════════════════════════════════

def train_ae(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            loss = nn.functional.mse_loss(model(Xt[idx[i:i + BS]]), Xt[idx[i:i + BS]])
            opt.zero_grad(); loss.backward(); opt.step()

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]]
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
            dc = dC[idx[i:i + BS]]
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
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_metrics(recon_fn, fcst_fn, test_n, gt_test, mu, sig, gt_dim, fcst_steps):
    pred_r = recon_fn(test_n)
    gt_r = (gt_test - mu[:gt_dim]) / sig[:gt_dim] if gt_dim < test_n.shape[1] \
        else test_n
    rmse_r = float(np.sqrt(np.mean((pred_r[:, :gt_dim] - gt_r) ** 2)))

    pred_f = fcst_fn()
    gt_f = (gt_test[:fcst_steps] - mu[:gt_dim]) / sig[:gt_dim] \
        if gt_dim < test_n.shape[1] else test_n[:fcst_steps]
    rmse_f = float(np.sqrt(np.mean((pred_f[:, :gt_dim] - gt_f) ** 2)))
    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)


def add_noise(obs, sigma, rng):
    """Add Gaussian observation noise scaled by per-channel std."""
    if sigma == 0:
        return obs.copy()
    channel_std = obs.std(axis=0, keepdims=True)
    return obs + sigma * channel_std * rng.standard_normal(obs.shape)


def make_systems_noisy(sigma, noise_seed=999):
    """Generate all 4 systems with observation noise at level sigma."""
    rng = np.random.default_rng(noise_seed)
    systems = {}

    # 1. Coupled Harmonic
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw * x1 - k12 * (x1 - x2), v2,
                -k12 * (x2 - x1) - k23 * (x2 - x3),
                v3, -k23 * (x3 - x2) - kw * x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    obs = add_noise(raw, sigma, rng)
    n_tr, n_te = 4000, 500
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw[n_tr:n_tr + n_te]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Coupled Harmonic"] = dict(
        train_n=(tr - mu) / sig_, test_n=(te - mu) / sig_,
        gt_test=gt, mu=mu, sig=sig_,
        n_obs=6, gt_dim=6, fcst_steps=500, j=8, k=4, h=64)

    # 2. Linear 5D
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    raw = raw5[200:]
    obs = add_noise(raw, sigma, rng)
    n_tr, n_te = 4000, 500
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw[n_tr:n_tr + n_te]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Linear 5D"] = dict(
        train_n=(tr - mu) / sig_, test_n=(te - mu) / sig_,
        gt_test=gt, mu=mu, sig=sig_,
        n_obs=5, gt_dim=5, fcst_steps=500, j=8, k=4, h=64)

    # 3. Brusselator (delay-embedded)
    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2 * y - (B_br + 1) * x, B_br * x - x**2 * y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    # Add noise BEFORE delay embedding (noise is on observations)
    raw_noisy = add_noise(raw_br, sigma, rng)
    obs = delay(raw_noisy, 5)
    raw_clean_trimmed = raw_br[:len(obs)]
    n_tr, n_te = 4000, 500
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw_clean_trimmed[n_tr:n_tr + n_te, :2]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Brusselator"] = dict(
        train_n=(tr - mu) / sig_, test_n=(te - mu) / sig_,
        gt_test=gt, mu=mu, sig=sig_,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

    # 4. Duffing (delay-embedded)
    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    raw_noisy = add_noise(raw_du, sigma, rng)
    obs = delay(raw_noisy, 5)
    raw_clean_trimmed = raw_du[:len(obs)]
    n_tr, n_te = 4000, 500
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw_clean_trimmed[n_tr:n_tr + n_te, :2]
    mu, sig_ = tr.mean(0), tr.std(0) + 1e-8
    systems["Duffing"] = dict(
        train_n=(tr - mu) / sig_, test_n=(te - mu) / sig_,
        gt_test=gt, mu=mu, sig=sig_,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Run one system × one sigma × one seed
# ═══════════════════════════════════════════════════════════════════

def run_one(cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)
    last_train = train_n[-1]
    res = {}

    # ── AE+DMD ──
    SeedAll(seed)
    ae = AE(n_obs, k, h); train_ae(ae, Xt); ae.eval()
    with torch.no_grad(): Z = ae.encode(Xt).numpy()
    A = fit_dmd(Z)
    def _r(tn, m=ae):
        with torch.no_grad():
            return m(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _f(m=ae, A_=A):
        with torch.no_grad():
            z0 = m.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
        out = np.empty((fcst_steps, k)); out[0] = z0
        for t in range(1, fcst_steps): out[t] = A_ @ out[t - 1]
        with torch.no_grad():
            return m.decode(torch.tensor(out, dtype=torch.float32)).numpy()
    rr, rf = eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["AE+DMD"] = {"rmse_r": rr, "rmse_f": rf}

    # ── Resid+GRU ──
    SeedAll(seed)
    r2 = ResidualGRU(n_obs, j, k, h=h)
    train_teacher(r2, Xt); r2.eval()
    with torch.no_grad(): C2 = r2.carrier(Xt)
    C2c, C2n = C2[:-1], C2[1:]
    dC2 = C2n - C2c
    train_resid(r2, dC2)
    train_gru(r2, dC2, C2c, C2n); r2.eval()
    with torch.no_grad(): B2 = r2.f(dC2).numpy()
    A2 = fit_dmd(B2)
    def _r2(tn, m=r2):
        with torch.no_grad():
            return m.recon(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _f2(m=r2, A_=A2):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None], dtype=torch.float32))
            Cc = m.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
            b0 = m.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dh = m.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc
    rr, rf = eval_metrics(_r2, _f2, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    return res


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    results = {}

    for sigma in SIGMAS:
        print(f"\n{'='*60}")
        print(f"  Noise σ = {sigma}")
        print(f"{'='*60}")
        systems = make_systems_noisy(sigma)

        for sname, cfg in systems.items():
            key = f"{sname}"
            results.setdefault(key, {})
            results[key].setdefault(str(sigma), {"AE+DMD": [], "Resid+GRU": []})

            for seed in SEEDS:
                print(f"  {sname} σ={sigma} seed={seed} ... ", end="", flush=True)
                r = run_one(cfg, seed)
                for method in r:
                    results[key][str(sigma)][method].append(r[method])
                print(f"AE r={r['AE+DMD']['rmse_r']:.4f}/f={r['AE+DMD']['rmse_f']:.3f}  "
                      f"GRU r={r['Resid+GRU']['rmse_r']:.4f}/f={r['Resid+GRU']['rmse_f']:.3f}")

    # ── Compute mean ± std ──
    stats = {}
    for sname in results:
        stats[sname] = {}
        for sigma in results[sname]:
            stats[sname][sigma] = {}
            for method in results[sname][sigma]:
                rs = [x["rmse_r"] for x in results[sname][sigma][method]]
                fs = [x["rmse_f"] for x in results[sname][sigma][method]]
                stats[sname][sigma][method] = {
                    "rmse_r_mean": float(np.mean(rs)),
                    "rmse_r_std": float(np.std(rs)),
                    "rmse_f_mean": float(np.mean(fs)),
                    "rmse_f_std": float(np.std(fs)),
                }

    # ── Save ──
    with open(OUT / "noise_sweep.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ {OUT / 'noise_sweep.json'}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("\nDone.")

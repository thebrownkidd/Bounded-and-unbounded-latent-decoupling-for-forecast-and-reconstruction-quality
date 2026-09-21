"""
Param-matched rerun: increase Coupled hidden dim h so param count >= Decoupled.
Latent dims (k, j) stay fixed. Only h changes for coupled baselines.

Methods: Naive(phi=0.5), PCGrad, CAGrad, Nash-MTL, FAMO, AIKAE_standalone, Decoupled
5 systems x 5 seeds, F1 + recon RMSE.
"""
import sys, json, time, random, math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
OUTFILE = OUT / "param_matched.json"

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2, 42, 123]
PHI = 0.5


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.A = nn.Linear(k, k, bias=False)

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def predict(self, z): return self.A(z)


class AIKAE_Model(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.K = nn.Linear(k, k, bias=False)

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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def find_matched_h(n, k, target_params, h_start=64):
    """Find smallest h such that KoopmanAE(n, k, h) has >= target_params."""
    for h in range(h_start, 300):
        m = KoopmanAE(n, k, h)
        if count_params(m) >= target_params:
            return h, count_params(m)
    return h_start, count_params(KoopmanAE(n, k, h_start))


# ═══════════════════════════════════════════════════════════════════
#  Gradient utilities
# ═══════════════════════════════════════════════════════════════════

def get_task_grads(model, x_t, x_tp1):
    params = list(model.parameters())

    model.zero_grad()
    z_t = model.encode(x_t)
    L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
    L_rec.backward(retain_graph=True)
    g_rec = torch.cat([p.grad.clone().flatten() if p.grad is not None
                       else torch.zeros(p.numel()) for p in params])

    model.zero_grad()
    z_t = model.encode(x_t)
    z_tp1 = model.encode(x_tp1)
    L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
    L_fcst.backward()
    g_fcst = torch.cat([p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel()) for p in params])

    return g_rec, g_fcst, L_rec.item(), L_fcst.item()


def set_grads_from_flat(model, flat_grad):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat_grad[offset:offset + numel].view_as(p).clone()
        offset += numel


# ═══════════════════════════════════════════════════════════════════
#  PCGrad
# ═══════════════════════════════════════════════════════════════════

def pcgrad_surgery(g1, g2):
    g1_out, g2_out = g1.clone(), g2.clone()
    d12 = torch.dot(g1_out, g2_out)
    if d12 < 0:
        g1_out = g1_out - (d12 / (torch.dot(g2_out, g2_out) + 1e-12)) * g2_out
    d21 = torch.dot(g2_out, g1_out)
    if d21 < 0:
        g2_out = g2_out - (d21 / (torch.dot(g1_out, g1_out) + 1e-12)) * g1_out
    return g1_out, g2_out


# ═══════════════════════════════════════════════════════════════════
#  CAGrad
# ═══════════════════════════════════════════════════════════════════

def cagrad_direction(g1, g2, c=0.5):
    g_avg = 0.5 * (g1 + g2)
    g_avg_norm = g_avg.norm()
    if g_avg_norm < 1e-12:
        return g_avg
    g_diff = g1 - g2
    g_diff_norm = g_diff.norm()
    if g_diff_norm < 1e-12:
        return g_avg
    proj_coeff = torch.dot(g_diff, g_avg) / (g_avg_norm ** 2 + 1e-12)
    g_diff_orth = g_diff - proj_coeff * g_avg
    g_diff_orth_norm = g_diff_orth.norm()
    if g_diff_orth_norm < 1e-12:
        return g_avg
    delta_dir = g_diff_orth / g_diff_orth_norm
    ip1 = torch.dot(g_avg, g1)
    ip2 = torch.dot(g_avg, g2)
    if ip1 < ip2:
        sign = torch.sign(torch.dot(delta_dir, g1))
    else:
        sign = torch.sign(torch.dot(delta_dir, g2))
    alpha = c * g_avg_norm
    return g_avg + sign * alpha * delta_dir


# ═══════════════════════════════════════════════════════════════════
#  Nash-MTL
# ═══════════════════════════════════════════════════════════════════

def nash_direction(g1, g2, n_grid=50):
    a = torch.dot(g1, g1)
    b = torch.dot(g1, g2)
    c = torch.dot(g2, g2)
    best_alpha, best_val = 0.5, -float('inf')
    for alpha in torch.linspace(0, 1, n_grid):
        u1 = alpha * a + (1 - alpha) * b
        u2 = alpha * b + (1 - alpha) * c
        val = u1 * u2
        if val > best_val:
            best_val = val
            best_alpha = alpha
    return best_alpha * g1 + (1 - best_alpha) * g2


# ═══════════════════════════════════════════════════════════════════
#  Training functions
# ═══════════════════════════════════════════════════════════════════

def train_naive(model, Xt, phi=PHI):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
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


def train_pcgrad(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            g_rec, g_fcst, _, _ = get_task_grads(model, x_t, x_tp1)
            g_rec_pc, g_fcst_pc = pcgrad_surgery(g_rec, g_fcst)
            combined = 0.5 * (g_rec_pc + g_fcst_pc)
            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_cagrad(model, Xt, c=0.5):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            g_rec, g_fcst, _, _ = get_task_grads(model, x_t, x_tp1)
            combined = cagrad_direction(g_rec, g_fcst, c=c)
            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_nash(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            g_rec, g_fcst, _, _ = get_task_grads(model, x_t, x_tp1)
            combined = nash_direction(g_rec, g_fcst)
            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_famo(model, Xt, eta_w=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log_w = torch.zeros(2)
    prev_losses = torch.ones(2)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]
            g_rec, g_fcst, l_rec, l_fcst = get_task_grads(model, x_t, x_tp1)
            curr_losses = torch.tensor([l_rec, l_fcst])
            delta = curr_losses / (prev_losses + 1e-8)
            log_w = log_w + eta_w * delta
            w = torch.softmax(log_w, dim=0)
            prev_losses = curr_losses.detach()
            combined = w[0] * g_rec + w[1] * g_fcst
            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


def train_aikae(model, Xt, phi_rec=0.4, phi_fwd=0.3, phi_bwd=0.3):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
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


# Decoupled training
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

def eval_coupled(model, Xt_test, gt_dim):
    model.eval()
    with torch.no_grad():
        z_test = model.encode(Xt_test)
        recon = model.decode(z_test).numpy()
        z_pred = model.predict(z_test[:-1])
        fcst = model.decode(z_pred).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))
    rmse_f = float(np.sqrt(np.mean((fcst[:, :gt_dim] - gt[1:, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


def eval_decoupled(model, Xt_test, gt_dim):
    model.eval()
    with torch.no_grad():
        C_test = model.carrier(Xt_test)
        recon = model.dec(C_test).numpy()
        dC_test = C_test[1:] - C_test[:-1]
        B_test = model.f(dC_test).numpy()
    X, Y = B_test[:-1].T, B_test[1:].T
    A_dmd = Y @ np.linalg.pinv(X)
    b_pred = (A_dmd @ B_test[:-1].T).T
    with torch.no_grad():
        dC_hat = model.m(torch.tensor(b_pred, dtype=torch.float32)).numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))
    C_np = C_test.numpy()
    C_next = C_np[1:-1] + dC_hat
    with torch.no_grad():
        x_hat = model.dec(torch.tensor(C_next, dtype=torch.float32)).numpy()
    rmse_f = float(np.sqrt(np.mean((x_hat[:, :gt_dim] - gt[2:len(C_next)+2, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig, gt_dim

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
    trn, ten, gd = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    trn, ten, gd = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    trn, ten, gd = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    trn, ten, gd = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 500)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

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
    trn, ten, gd = norm_split(raw_l96, raw_l96, 20, 4000, 500)
    systems["Lorenz-96"] = dict(
        train_n=trn, test_n=ten, n_obs=20, k=10, j=12, h=64, gt_dim=20)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()

    # Step 1: compute param-matched hidden dims
    print("=" * 70)
    print("  PARAM MATCHING")
    print("=" * 70)
    matched_h = {}
    for sname, cfg in systems.items():
        decoupled = ResidualModel(cfg["n_obs"], cfg["j"], cfg["k"], cfg["h"])
        target = count_params(decoupled)
        orig = count_params(KoopmanAE(cfg["n_obs"], cfg["k"], cfg["h"]))
        h_m, actual = find_matched_h(cfg["n_obs"], cfg["k"], target)
        matched_h[sname] = h_m
        print(f"  {sname:20s}  Decoupled={target:6d}  Coupled@h=64={orig:6d}  "
              f"Matched h={h_m:3d} -> {actual:6d} params")

    # Step 2: define methods
    coupled_methods = {
        "Naive": lambda model, Xt: train_naive(model, Xt, phi=0.5),
        "PCGrad": train_pcgrad,
        "CAGrad": lambda model, Xt: train_cagrad(model, Xt, c=0.5),
        "Nash-MTL": train_nash,
        "FAMO": train_famo,
    }

    # Load existing results for resume
    if OUTFILE.exists():
        with open(OUTFILE) as fp:
            results = json.load(fp)
        print(f"\n  Loaded existing results")
    else:
        results = {}

    def save():
        with open(OUTFILE, "w") as fp:
            json.dump(results, fp, indent=2)

    # Step 3: run coupled methods with matched params
    for sname, cfg in systems.items():
        if sname not in results:
            results[sname] = {}

        h_m = matched_h[sname]

        print(f"\n{'=' * 70}")
        print(f"  {sname}  (h_matched={h_m})")
        print(f"{'=' * 70}")

        for mname, train_fn in coupled_methods.items():
            key = f"{mname}_matched"
            if key in results[sname]:
                print(f"  Skipping {key} (done)")
                continue

            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
                Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
                model = KoopmanAE(cfg["n_obs"], cfg["k"], h_m)
                train_fn(model, Xt)
                r, f = eval_coupled(model, Xt_test, cfg["gt_dim"])
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"  {key:25s}  seed={seed}  R={r:.4f}  F1={f:.4f}")

            results[sname][key] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "h": h_m,
                "params": count_params(KoopmanAE(cfg["n_obs"], cfg["k"], h_m)),
            }
            save()

        # AIKAE standalone with matched params
        key = "AIKAE_matched"
        if key not in results[sname]:
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
                Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
                model = AIKAE_Model(cfg["n_obs"], cfg["k"], h_m)
                train_aikae(model, Xt)
                r, f = eval_coupled(model, Xt_test, cfg["gt_dim"])
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"  {key:25s}  seed={seed}  R={r:.4f}  F1={f:.4f}")

            results[sname][key] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "h": h_m,
                "params": count_params(AIKAE_Model(cfg["n_obs"], cfg["k"], h_m)),
            }
            save()
        else:
            print(f"  Skipping {key} (done)")

        # Decoupled (always h=64, unchanged)
        key = "Decoupled"
        if key not in results[sname]:
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                SeedAll(seed)
                Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
                Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
                model = ResidualModel(cfg["n_obs"], cfg["j"], cfg["k"], cfg["h"])
                train_teacher(model, Xt)
                with torch.no_grad():
                    C = model.carrier(Xt)
                    dC = C[1:] - C[:-1]
                train_resid(model, dC)
                with torch.no_grad():
                    C = model.carrier(Xt)
                    dC = C[1:] - C[:-1]
                train_gru(model, dC, C[:-1], C[1:])
                r, f = eval_decoupled(model, Xt_test, cfg["gt_dim"])
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"  {key:25s}  seed={seed}  R={r:.4f}  F1={f:.4f}")

            results[sname][key] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "h": 64,
                "params": count_params(ResidualModel(cfg["n_obs"], cfg["j"], cfg["k"], cfg["h"])),
            }
            save()
        else:
            print(f"  Skipping {key} (done)")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY  ({elapsed:.0f}s total)")
    print(f"{'=' * 70}")
    print(f"\n{'System':20s} {'Method':25s} {'Params':>7s} {'R':>8s} {'F1':>8s}  {'Flipped?':>8s}")
    print("-" * 82)
    for sname in results:
        dec = results[sname].get("Decoupled", {})
        dec_f = dec.get("rmse_f_mean", 999)
        dec_r = dec.get("rmse_r_mean", 999)
        for mname in sorted(results[sname]):
            r = results[sname][mname]
            flipped = ""
            if mname != "Decoupled":
                f_flip = r["rmse_f_mean"] < dec_f
                r_flip = r["rmse_r_mean"] < dec_r
                if f_flip or r_flip:
                    flipped = "R!" if r_flip else ""
                    flipped += "F!" if f_flip else ""
                else:
                    flipped = "no"
            print(f"{sname:20s} {mname:25s} {r.get('params',''):>7} "
                  f"{r['rmse_r_mean']:8.4f} {r['rmse_f_mean']:8.4f}  {flipped:>8s}")
        print()

    print(f"\nResults saved to {OUTFILE}")

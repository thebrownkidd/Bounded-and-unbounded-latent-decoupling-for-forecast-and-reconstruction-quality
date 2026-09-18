"""
MTL gradient surgery baselines for ICLR 2027.

Compares the coupled Koopman AE trained with:
    1. Naive weighting:  (1-φ)L_rec + φ L_fcst  (existing baseline)
    2. PCGrad (Yu et al., NeurIPS 2020): project out conflicting gradient components
    3. CAGrad (Liu et al., NeurIPS 2021): find common descent direction within a ball

Plus the decoupled carrier-residual architecture (no gradient conflict by construction).

Reports single-step forecast RMSE and reconstruction RMSE, same protocol as
run_head_comparison.py and run_gradient_conflict.py.

Outputs:
    Paper/mtl_baselines.json
"""
import sys, json, time, copy, argparse
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
PHI = 0.5  # fixed weighting for fair comparison

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true",
                    help="Smoke test: 5 epochs, 1 seed")
parser.add_argument("--system", nargs="+", default=None,
                    help="Run only these systems")
args = parser.parse_args()

if args.quick:
    EPOCHS = 5; SEEDS = [0]


# ═══════════════════════════════════════════════════════════════════
#  Model
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

    def encoder_params(self):
        return list(self.enc.parameters())


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
#  PCGrad (Yu et al., NeurIPS 2020)
# ═══════════════════════════════════════════════════════════════════

def pcgrad_surgery(g1, g2):
    """Project out conflicting components between two gradient vectors.
    Returns modified (g1', g2') as flat tensors."""
    dot = torch.dot(g1, g2)
    if dot < 0:
        g1 = g1 - (dot / (torch.dot(g2, g2) + 1e-12)) * g2
        # recompute dot for g2 projection with original g1
        # PCGrad projects each against the other independently
    dot2 = torch.dot(g2, g1)  # use original g1 before modification? No — PCGrad uses random order.
    # Standard PCGrad: random order, project the second against the first's modified version.
    # For 2 tasks, project g2 against g1 (original) as well.
    g2_orig = g2.clone()
    dot_orig = torch.dot(g1, g2_orig)  # already computed, but let's be clean
    # Actually, canonical PCGrad for 2 tasks:
    # For each task i, for each other task j (random order):
    #   if <g_i, g_j> < 0: g_i = g_i - (<g_i, g_j> / ||g_j||^2) * g_j
    # With 2 tasks, each task projects against the other.
    g1_out = g1.clone()
    g2_out = g2.clone()
    d12 = torch.dot(g1_out, g2_out)
    if d12 < 0:
        g1_out = g1_out - (d12 / (torch.dot(g2_out, g2_out) + 1e-12)) * g2_out
    d21 = torch.dot(g2_out, g1_out)
    if d21 < 0:
        g2_out = g2_out - (d21 / (torch.dot(g1_out, g1_out) + 1e-12)) * g1_out
    return g1_out, g2_out


# ═══════════════════════════════════════════════════════════════════
#  CAGrad (Liu et al., NeurIPS 2021)
# ═══════════════════════════════════════════════════════════════════

def cagrad_direction(g1, g2, c=0.5):
    """Compute the CAGrad common descent direction for 2 tasks.

    Finds d = g_avg + alpha * delta such that:
      min_{||delta|| <= c * ||g_avg||} min_i <g_avg + delta, g_i>
    is maximised. For 2 tasks this has a closed-form solution.

    c: the constraint radius as fraction of ||g_avg||.
    """
    g_avg = 0.5 * (g1 + g2)
    g_avg_norm = g_avg.norm()
    if g_avg_norm < 1e-12:
        return g_avg

    g_diff = g1 - g2  # direction along which the two gradients disagree
    g_diff_norm = g_diff.norm()

    if g_diff_norm < 1e-12:
        return g_avg

    # For 2 tasks, the worst-case inner product is with whichever g_i
    # has smaller projection onto d. The optimal delta points along
    # the component of g_diff that is orthogonal to g_avg (if any),
    # scaled to the constraint radius.

    # Project g_diff onto g_avg direction and orthogonal complement
    proj_coeff = torch.dot(g_diff, g_avg) / (g_avg_norm ** 2 + 1e-12)
    g_diff_orth = g_diff - proj_coeff * g_avg
    g_diff_orth_norm = g_diff_orth.norm()

    if g_diff_orth_norm < 1e-12:
        return g_avg

    # The optimal delta is along g_diff_orth, scaled to c * ||g_avg||
    # but only if it improves the minimum inner product
    delta_dir = g_diff_orth / g_diff_orth_norm

    # Check which task benefits: <g_avg + alpha*delta_dir, g1> vs <..., g2>
    # We want to increase the minimum. The task with smaller <g_avg, g_i>
    # needs help. Move delta toward it.
    ip1 = torch.dot(g_avg, g1)
    ip2 = torch.dot(g_avg, g2)

    if ip1 < ip2:
        # g1 is the worse-off task, move toward it
        sign = torch.sign(torch.dot(delta_dir, g1))
    else:
        sign = torch.sign(torch.dot(delta_dir, g2))

    alpha = c * g_avg_norm
    d = g_avg + sign * alpha * delta_dir

    return d


# ═══════════════════════════════════════════════════════════════════
#  Training functions
# ═══════════════════════════════════════════════════════════════════

def get_task_grads(model, x_t, x_tp1):
    """Compute per-task gradients on ALL model parameters (not just encoder).
    Returns flat gradient vectors and scalar losses."""
    params = list(model.parameters())

    # L_rec gradient
    model.zero_grad()
    z_t = model.encode(x_t)
    L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
    L_rec.backward(retain_graph=True)
    g_rec = torch.cat([p.grad.clone().flatten() if p.grad is not None
                       else torch.zeros(p.numel()) for p in params])

    # L_fcst gradient
    model.zero_grad()
    z_t = model.encode(x_t)
    z_tp1 = model.encode(x_tp1)
    L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
    L_fcst.backward()
    g_fcst = torch.cat([p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel()) for p in params])

    return g_rec, g_fcst, L_rec.item(), L_fcst.item()


def set_grads_from_flat(model, flat_grad):
    """Write a flat gradient vector back into model parameter .grad fields."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset + n].view_as(p).clone()
        offset += n


def train_naive(model, Xt, phi=PHI):
    """Standard (1-phi)*L_rec + phi*L_fcst training."""
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
    """PCGrad training: project out conflicting gradient components."""
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
    """CAGrad training: find common descent direction."""
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

def eval_coupled(model, Xt_test, gt_dim, mu, sig):
    """Evaluate a coupled model: recon + single-step forecast RMSE."""
    model.eval()
    with torch.no_grad():
        z_test = model.encode(Xt_test)
        recon = model.decode(z_test).numpy()
        z_pred = model.predict(z_test[:-1])
        fcst = model.decode(z_pred).numpy()

    gt_test = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt_test[:, :gt_dim]) ** 2)))
    rmse_f = float(np.sqrt(np.mean((fcst[:, :gt_dim] - gt_test[1:, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


def eval_decoupled(model, Xt_test, gt_dim):
    """Evaluate the decoupled model: recon (Phase 1) + single-step (DMD on b)."""
    model.eval()
    with torch.no_grad():
        C_test = model.carrier(Xt_test)
        recon = model.dec(C_test).numpy()
        dC_test = C_test[1:] - C_test[:-1]
        B_test = model.f(dC_test).numpy()

    # Post-hoc DMD on b
    X, Y = B_test[:-1].T, B_test[1:].T
    A_dmd = Y @ np.linalg.pinv(X)

    # Single-step forecast in b-space
    b_pred = (A_dmd @ B_test[:-1].T).T
    # Decode: b -> dC_hat -> C_hat -> x_hat
    with torch.no_grad():
        dC_hat = model.m(torch.tensor(b_pred, dtype=torch.float32)).numpy()

    gt_test = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt_test[:, :gt_dim]) ** 2)))

    # For single-step forecast: C_{t+1} = C_t + dC_hat, then decode
    C_np = C_test.numpy()
    C_next = C_np[1:-1] + dC_hat  # align: dC_hat[i] predicts C_{i+2} from b_{i} (which is dC_{i})
    with torch.no_grad():
        x_hat = model.dec(torch.tensor(C_next, dtype=torch.float32)).numpy()
    rmse_f = float(np.sqrt(np.mean((x_hat[:, :gt_dim] - gt_test[2:len(C_next)+2, :gt_dim]) ** 2)))

    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Data generators (same as other scripts)
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
        train_n=trn, test_n=ten, n_obs=6, k=4, j=8, h=64, gt_dim=6, mu=mu, sig=sig)

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
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5, mu=mu, sig=sig)

    # Brusselator (delay-embedded)
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
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2, mu=mu, sig=sig)

    # Duffing (delay-embedded)
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
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2, mu=mu, sig=sig)

    # Lorenz-96 (20D)
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
        train_n=trn, test_n=ten, n_obs=20, k=10, j=12, h=64, gt_dim=20, mu=mu, sig=sig)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def run_one_coupled(method_name, train_fn, cfg, seed):
    """Run one coupled experiment and return (rmse_r, rmse_f)."""
    SeedAll(seed)
    Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
    Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)

    model = KoopmanAE(cfg["n_obs"], cfg["k"], cfg["h"])
    train_fn(model, Xt)
    return eval_coupled(model, Xt_test, cfg["gt_dim"], cfg["mu"], cfg["sig"])


def run_decoupled(cfg, seed):
    """Run the decoupled carrier-residual model."""
    SeedAll(seed)
    Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
    Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)

    model = ResidualModel(cfg["n_obs"], cfg["j"], cfg["k"], cfg["h"])

    # Phase 1: teacher AE
    train_teacher(model, Xt)

    # Phase 2: residual compressor
    with torch.no_grad():
        C = model.carrier(Xt)
        dC = C[1:] - C[:-1]
    train_resid(model, dC)

    # Phase 3: GRU accumulation
    with torch.no_grad():
        C = model.carrier(Xt)
        dC = C[1:] - C[:-1]
    train_gru(model, dC, C[:-1], C[1:])

    return eval_decoupled(model, Xt_test, cfg["gt_dim"])


if __name__ == "__main__":
    systems = make_systems()
    if args.system:
        systems = {k: v for k, v in systems.items() if k in args.system}
    t0 = time.time()

    methods = {
        "Naive (phi=0.5)": lambda model, Xt: train_naive(model, Xt, phi=0.5),
        "Naive (phi=0.1)": lambda model, Xt: train_naive(model, Xt, phi=0.1),
        "Naive (phi=0.9)": lambda model, Xt: train_naive(model, Xt, phi=0.9),
        "PCGrad": train_pcgrad,
        "CAGrad": train_cagrad,
    }

    # Load existing results for incremental resume
    outpath = OUT / "mtl_baselines.json"
    if outpath.exists():
        with open(outpath) as fp:
            results = json.load(fp)
        print(f"  Loaded existing results: {list(results.keys())}")
    else:
        results = {}

    def save_incremental():
        with open(outpath, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"  [saved -> {outpath}]")

    for sname, cfg in systems.items():
        if sname in results and len(results[sname]) >= 6:
            print(f"\n  Skipping {sname} (already complete)")
            continue

        print(f"\n{'='*60}")
        print(f"  {sname}")
        print(f"{'='*60}")
        if sname not in results:
            results[sname] = {}

        for mname, train_fn in methods.items():
            if mname in results.get(sname, {}):
                print(f"  Skipping {mname} (already done)")
                continue
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                print(f"  {mname:20s}  seed={seed} ... ", end="", flush=True)
                r, f = run_one_coupled(mname, train_fn, cfg, seed)
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"R={r:.4f}  F={f:.4f}")

            results[sname][mname] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "rmse_r_all": rmse_rs,
                "rmse_f_all": rmse_fs,
            }
            save_incremental()

        # Decoupled baseline
        if "Decoupled" not in results.get(sname, {}):
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                print(f"  {'Decoupled':20s}  seed={seed} ... ", end="", flush=True)
                r, f = run_decoupled(cfg, seed)
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"R={r:.4f}  F={f:.4f}")

            results[sname]["Decoupled"] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
                "rmse_r_all": rmse_rs,
                "rmse_f_all": rmse_fs,
            }
            save_incremental()
        else:
            print(f"  Skipping Decoupled (already done)")

    # Final save
    save_incremental()
    print(f"\n-> {outpath}")

    # Summary table
    print(f"\nTotal time: {time.time() - t0:.0f}s\n")
    print(f"{'System':20s} {'Method':20s} {'Recon RMSE':>14s} {'Fcst RMSE':>14s}")
    print("-" * 72)
    for sname in results:
        for mname in results[sname]:
            r = results[sname][mname]
            print(f"{sname:20s} {mname:20s} "
                  f"{r['rmse_r_mean']:.4f}+-{r['rmse_r_std']:.4f} "
                  f"{r['rmse_f_mean']:.4f}+-{r['rmse_f_std']:.4f}")
        print()
    print("Done.")

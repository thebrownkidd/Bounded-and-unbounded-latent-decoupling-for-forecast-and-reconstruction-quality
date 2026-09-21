"""
Nash-MTL + FAMO baselines for ICLR 2027.

Adds two more MTL gradient methods to the existing PCGrad/CAGrad comparison:
  1. Nash-MTL (Navon et al., ICML 2022): Nash bargaining on task gradients
  2. FAMO (Liu et al., NeurIPS 2024): fast adaptive task weight optimization

Also runs Naive (phi=0.5) and Decoupled for cross-validation with existing results.

5 systems × 4 methods × 5 seeds, single-step F1 evaluation.
Outputs: /kaggle/working/mtl_nash_famo.json
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
PHI = 0.5
OUTFILE = f"{OUT}/mtl_nash_famo.json"


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


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
#  Gradient utilities
# ═══════════════════════════════════════════════════════════════════

def get_task_grads(model, x_t, x_tp1):
    params = list(model.parameters())

    model.zero_grad()
    z_t = model.encode(x_t)
    L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
    L_rec.backward(retain_graph=True)
    g_rec = torch.cat([p.grad.clone().flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=x_t.device) for p in params])

    model.zero_grad()
    z_t = model.encode(x_t)
    z_tp1 = model.encode(x_tp1)
    L_fcst = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
    L_fcst.backward()
    g_fcst = torch.cat([p.grad.clone().flatten() if p.grad is not None
                        else torch.zeros(p.numel(), device=x_t.device) for p in params])

    return g_rec, g_fcst, L_rec.item(), L_fcst.item()


def set_grads_from_flat(model, flat_grad):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat_grad[offset:offset + n].view_as(p).clone()
        offset += n


# ═══════════════════════════════════════════════════════════════════
#  Nash-MTL (Navon et al., ICML 2022)
# ═══════════════════════════════════════════════════════════════════

def nash_direction(g1, g2, n_grid=50):
    """Nash bargaining solution for 2-task gradients.

    Finds alpha in [0,1] maximizing the product of task utilities:
      u1(alpha) * u2(alpha)
    where u_i(alpha) = <alpha*g1 + (1-alpha)*g2, g_i>.
    """
    a = torch.dot(g1, g1)
    b = torch.dot(g1, g2)
    c = torch.dot(g2, g2)

    best_alpha, best_val = 0.5, -float('inf')
    for alpha in torch.linspace(0, 1, n_grid, device=g1.device):
        u1 = alpha * a + (1 - alpha) * b
        u2 = alpha * b + (1 - alpha) * c
        val = u1 * u2
        if val > best_val:
            best_val = val
            best_alpha = alpha

    return best_alpha * g1 + (1 - best_alpha) * g2


def train_nash(model, Xt):
    """Nash-MTL training."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t = Xt[sl].to(DEVICE)
            x_tp1 = Xt[sl + 1].to(DEVICE)

            g_rec, g_fcst, _, _ = get_task_grads(model, x_t, x_tp1)
            combined = nash_direction(g_rec, g_fcst)

            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ═══════════════════════════════════════════════════════════════════
#  FAMO (Liu et al., NeurIPS 2024)
# ═══════════════════════════════════════════════════════════════════

def train_famo(model, Xt, eta_w=0.1):
    """FAMO training: fast adaptive multitask optimization.

    Maintains log-weights updated via exponentiated gradient ascent
    on per-task losses. Tasks with higher loss get higher weight.
    """
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    log_w = torch.zeros(2, device=DEVICE)  # 2 tasks: rec, fcst
    prev_losses = torch.ones(2, device=DEVICE)

    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t = Xt[sl].to(DEVICE)
            x_tp1 = Xt[sl + 1].to(DEVICE)

            g_rec, g_fcst, l_rec, l_fcst = get_task_grads(model, x_t, x_tp1)

            curr_losses = torch.tensor([l_rec, l_fcst], device=DEVICE)
            delta = curr_losses / (prev_losses + 1e-8)
            log_w = log_w + eta_w * delta
            w = torch.softmax(log_w, dim=0)
            prev_losses = curr_losses.detach()

            combined = w[0] * g_rec + w[1] * g_fcst

            model.zero_grad()
            set_grads_from_flat(model, combined)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Naive and Decoupled training (reference)
# ═══════════════════════════════════════════════════════════════════

def train_naive(model, Xt, phi=PHI):
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
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl].to(DEVICE)), Cn[sl].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_coupled(model, Xt_test, gt_dim):
    model.eval()
    with torch.no_grad():
        Xt_dev = Xt_test.to(DEVICE)
        z_test = model.encode(Xt_dev)
        recon = model.decode(z_test).cpu().numpy()
        z_pred = model.predict(z_test[:-1])
        fcst = model.decode(z_pred).cpu().numpy()

    gt_test = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt_test[:, :gt_dim]) ** 2)))
    rmse_f = float(np.sqrt(np.mean((fcst[:, :gt_dim] - gt_test[1:, :gt_dim]) ** 2)))
    return rmse_r, rmse_f


def eval_decoupled(model, Xt_test, gt_dim):
    model.eval()
    with torch.no_grad():
        Xt_dev = Xt_test.to(DEVICE)
        C_test = model.carrier(Xt_dev)
        recon = model.dec(C_test).cpu().numpy()
        dC_test = C_test[1:] - C_test[:-1]
        B_test = model.f(dC_test).cpu().numpy()

    X, Y = B_test[:-1].T, B_test[1:].T
    A_dmd = Y @ np.linalg.pinv(X)

    b_pred = (A_dmd @ B_test[:-1].T).T
    with torch.no_grad():
        dC_hat = model.m(torch.tensor(b_pred, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    gt_test = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt_test[:, :gt_dim]) ** 2)))

    C_np = C_test.cpu().numpy()
    C_next = C_np[1:-1] + dC_hat
    with torch.no_grad():
        x_hat = model.dec(torch.tensor(C_next, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    rmse_f = float(np.sqrt(np.mean((x_hat[:, :gt_dim] - gt_test[2:len(C_next)+2, :gt_dim]) ** 2)))

    return rmse_r, rmse_f


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

    # Coupled Harmonic (6D)
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw*x1 - k12*(x1-x2), v2, -k12*(x2-x1) - k23*(x2-x3),
                v3, -k23*(x3-x2) - kw*x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten, gd, mu, sig = norm_split(raw, raw, 6, 4000, 500)
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
    trn, ten, gd, mu, sig = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, n_obs=5, k=4, j=8, h=64, gt_dim=5)

    # Brusselator (delay-embedded)
    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2*y - (B_br+1)*x, B_br*x - x**2*y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    trn, ten, gd, mu, sig = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    # Duffing (delay-embedded)
    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    trn, ten, gd, mu, sig = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 500)
    systems["Duffing"] = dict(
        train_n=trn, test_n=ten, n_obs=10, k=3, j=8, h=64, gt_dim=2)

    # Lorenz-96 (20D)
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
    trn, ten, gd, mu, sig = norm_split(raw_l96, raw_l96, 20, 4000, 500)
    systems["Lorenz-96"] = dict(
        train_n=trn, test_n=ten, n_obs=20, k=10, j=12, h=64, gt_dim=20)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def run_one_coupled(train_fn, cfg, seed):
    SeedAll(seed)
    Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
    Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
    model = KoopmanAE(cfg["n_obs"], cfg["k"], cfg["h"])
    train_fn(model, Xt)
    return eval_coupled(model, Xt_test, cfg["gt_dim"])


def run_decoupled(cfg, seed):
    SeedAll(seed)
    Xt = torch.tensor(cfg["train_n"], dtype=torch.float32)
    Xt_test = torch.tensor(cfg["test_n"], dtype=torch.float32)
    model = ResidualModel(cfg["n_obs"], cfg["j"], cfg["k"], cfg["h"])

    train_teacher(model, Xt)
    with torch.no_grad():
        C = model.carrier(Xt.to(DEVICE)).cpu()
        dC = C[1:] - C[:-1]
    train_resid(model, dC)
    with torch.no_grad():
        C = model.carrier(Xt.to(DEVICE)).cpu()
        dC = C[1:] - C[:-1]
    train_gru(model, dC, C[:-1], C[1:])
    return eval_decoupled(model, Xt_test, cfg["gt_dim"])


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()

    methods = {
        "Naive (phi=0.5)": lambda model, Xt: train_naive(model, Xt, phi=0.5),
        "Nash-MTL": train_nash,
        "FAMO": train_famo,
    }

    results = {}

    for sname, cfg in systems.items():
        print(f"\n{'='*60}")
        print(f"  {sname}")
        print(f"{'='*60}")
        results[sname] = {}

        for mname, train_fn in methods.items():
            rmse_rs, rmse_fs = [], []
            for seed in SEEDS:
                print(f"  {mname:20s}  seed={seed} ... ", end="", flush=True)
                r, f = run_one_coupled(train_fn, cfg, seed)
                rmse_rs.append(r); rmse_fs.append(f)
                print(f"R={r:.4f}  F={f:.4f}")

            results[sname][mname] = {
                "rmse_r_mean": float(np.mean(rmse_rs)),
                "rmse_r_std": float(np.std(rmse_rs)),
                "rmse_f_mean": float(np.mean(rmse_fs)),
                "rmse_f_std": float(np.std(rmse_fs)),
            }
            with open(OUTFILE, "w") as fp:
                json.dump(results, fp, indent=2)
            print(f"  [saved]")

        # Decoupled
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
        }
        with open(OUTFILE, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"  [saved]")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"\n{'System':20s} {'Method':20s} {'F1 RMSE':>14s}")
    print("-" * 60)
    for sname in results:
        for mname in results[sname]:
            r = results[sname][mname]
            print(f"{sname:20s} {mname:20s} {r['rmse_f_mean']:.4f}+-{r['rmse_f_std']:.4f}")
        print()
    print("Done.")

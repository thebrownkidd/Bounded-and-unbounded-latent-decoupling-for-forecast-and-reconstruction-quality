"""
Head comparison experiment for ICLR 2027 §5.1.
Tests decoupling principle across DMD, MLP, and Neural ODE forecast heads.

Outputs: /kaggle/working/head_comparison.json
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
COUPLED_PHIS = [0.1, 0.5, 1.0]


def SeedAll(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════
#  Dynamics heads
# ═══════════════════════════════════════════════════════════════════

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)


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
            return self.forward(
                torch.tensor(b, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            ).squeeze(0).cpu().numpy()


class ODEFunc(nn.Module):
    def __init__(self, k, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, k))
    def forward(self, b): return self.net(b)


class NeuralODEHead(nn.Module):
    def __init__(self, k, hidden=64, n_steps=4):
        super().__init__()
        self.func = ODEFunc(k, hidden)
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

    def forward(self, b):
        dt = self.dt
        for _ in range(self.n_steps):
            k1 = self.func(b)
            k2 = self.func(b + 0.5 * dt * k1)
            k3 = self.func(b + 0.5 * dt * k2)
            k4 = self.func(b + dt * k3)
            b = b + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return b

    def step_numpy(self, b):
        with torch.no_grad():
            return self.forward(
                torch.tensor(b, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            ).squeeze(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════
#  Model architectures
# ═══════════════════════════════════════════════════════════════════

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
#  Training functions
# ═══════════════════════════════════════════════════════════════════

def train_coupled(model, Xt, phi):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1, device=DEVICE)
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


def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt), device=DEVICE)
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
        model.train(); idx = torch.randperm(len(dC), device=DEVICE)
        for i in range(0, len(dC), BS):
            dc = dC[idx[i:i + BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()


def train_gru(model, dC, Cc, Cn):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Cc), device=DEVICE)
        for i in range(0, len(Cc), BS):
            sl = idx[i:i + BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()


def train_head_on_b(head, B_train):
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    bt, btp1 = B_train[:-1], B_train[1:]
    for ep in range(1, EPOCHS + 1):
        head.train(); idx = torch.randperm(len(bt), device=DEVICE)
        for i in range(0, len(bt), BS):
            sl = idx[i:i + BS]
            loss = nn.functional.mse_loss(head(bt[sl]), btp1[sl])
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
    gt_f_raw = gt_test[:fcst_steps]
    gt_f = (gt_f_raw - mu[:gt_dim]) / sig[:gt_dim] if gt_dim < test_n.shape[1] \
        else test_n[:fcst_steps]
    rmse_f = float(np.sqrt(np.mean((pred_f[:, :gt_dim] - gt_f) ** 2)))
    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Run one system × one seed
# ═══════════════════════════════════════════════════════════════════

def run_one(sname, cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32).to(DEVICE)
    last_train = train_n[-1]
    res = {}

    # Coupled baselines: AE + {MLP, ODE}
    for head_name, head_cls in [("MLP", MLPHead), ("ODE", NeuralODEHead)]:
        best_phi_res = None
        for phi in COUPLED_PHIS:
            SeedAll(seed)
            model = CoupledAE(n_obs, k, h, head_cls).to(DEVICE)
            train_coupled(model, Xt, phi); model.eval()

            def _r(tn, m=model):
                with torch.no_grad():
                    return m.decode(m.encode(
                        torch.tensor(tn, dtype=torch.float32).to(DEVICE))).cpu().numpy()

            def _f(m=model):
                with torch.no_grad():
                    z = m.encode(torch.tensor(test_n[:1], dtype=torch.float32).to(DEVICE)
                                ).squeeze(0).cpu().numpy()
                out = np.empty((fcst_steps, k)); out[0] = z
                for t in range(1, fcst_steps):
                    out[t] = m.head.step_numpy(out[t - 1])
                with torch.no_grad():
                    return m.decode(torch.tensor(out, dtype=torch.float32).to(DEVICE)).cpu().numpy()

            rr, rf = eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
            if best_phi_res is None or rf < best_phi_res["rmse_f"]:
                best_phi_res = {"rmse_r": rr, "rmse_f": rf, "phi": phi}
        res[f"AE+{head_name}"] = best_phi_res

    # AE+DMD
    SeedAll(seed)
    ae_enc = nn.Sequential(nn.Linear(n_obs, h), nn.ELU(),
                           nn.Linear(h, h), nn.ELU(), nn.Linear(h, k)).to(DEVICE)
    ae_dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                           nn.Linear(h, h), nn.ELU(), nn.Linear(h, n_obs)).to(DEVICE)
    opt = torch.optim.Adam(list(ae_enc.parameters()) + list(ae_dec.parameters()), lr=LR)
    for ep in range(1, EPOCHS + 1):
        ae_enc.train(); ae_dec.train()
        idx = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i + BS]]
            loss = nn.functional.mse_loss(ae_dec(ae_enc(x)), x)
            opt.zero_grad(); loss.backward(); opt.step()
    ae_enc.eval(); ae_dec.eval()
    with torch.no_grad(): Z = ae_enc(Xt).cpu().numpy()
    A_dmd = fit_dmd(Z)

    def _r_dmd(tn):
        with torch.no_grad():
            return ae_dec(ae_enc(torch.tensor(tn, dtype=torch.float32).to(DEVICE))).cpu().numpy()
    def _f_dmd():
        with torch.no_grad():
            z0 = ae_enc(torch.tensor(test_n[:1], dtype=torch.float32).to(DEVICE)).cpu().numpy().ravel()
        out = np.empty((fcst_steps, k)); out[0] = z0
        for t in range(1, fcst_steps): out[t] = A_dmd @ out[t - 1]
        with torch.no_grad():
            return ae_dec(torch.tensor(out, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    rr, rf = eval_metrics(_r_dmd, _f_dmd, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["AE+DMD"] = {"rmse_r": rr, "rmse_f": rf}

    # Decoupled: Resid + {DMD, MLP, ODE} + GRU
    SeedAll(seed)
    rm = ResidualModel(n_obs, j, k, h=h).to(DEVICE)
    train_teacher(rm, Xt); rm.eval()
    with torch.no_grad(): C = rm.carrier(Xt)
    Cc, Cn = C[:-1], C[1:]
    dC = Cn - Cc
    train_resid(rm, dC)
    train_gru(rm, dC, Cc, Cn); rm.eval()
    with torch.no_grad(): B = rm.f(dC).cpu().numpy()
    B_tensor = rm.f(dC).detach()

    def _r_resid(tn, m=rm):
        with torch.no_grad():
            return m.recon(torch.tensor(tn, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    def make_fcst_fn(m, step_fn):
        def _f():
            with torch.no_grad():
                Cp = m.carrier(torch.tensor(last_train[None], dtype=torch.float32).to(DEVICE))
                Cc_init = m.carrier(torch.tensor(test_n[:1], dtype=torch.float32).to(DEVICE))
                b0 = m.f(Cc_init - Cp).cpu().numpy().ravel()
            fc = np.empty((fcst_steps, n_obs))
            C_ = Cc_init.squeeze(0); b = b0.copy()
            with torch.no_grad():
                fc[0] = m.dec(C_.unsqueeze(0)).cpu().numpy().ravel()
            for t in range(1, fcst_steps):
                b = step_fn(b)
                with torch.no_grad():
                    dh = m.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                    C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                    fc[t] = m.dec(C_.unsqueeze(0)).cpu().numpy().ravel()
            return fc
        return _f

    # Resid+DMD
    A_resid = fit_dmd(B)
    rr, rf = eval_metrics(
        _r_resid, make_fcst_fn(rm, lambda b, A_=A_resid: A_ @ b),
        test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+DMD+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    # Resid+MLP
    SeedAll(seed + 1000)
    mlp_head = MLPHead(k).to(DEVICE)
    train_head_on_b(mlp_head, B_tensor); mlp_head.eval()
    rr, rf = eval_metrics(
        _r_resid, make_fcst_fn(rm, mlp_head.step_numpy),
        test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+MLP+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    # Resid+ODE
    SeedAll(seed + 2000)
    ode_head = NeuralODEHead(k).to(DEVICE)
    train_head_on_b(ode_head, B_tensor); ode_head.eval()
    rr, rf = eval_metrics(
        _r_resid, make_fcst_fn(rm, ode_head.step_numpy),
        test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+ODE+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    return res


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

    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw * x1 - k12 * (x1 - x2), v2,
                -k12 * (x2 - x1) - k23 * (x2 - x3),
                v3, -k23 * (x3 - x2) - kw * x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=6, gt_dim=6, fcst_steps=500, j=8, k=4, h=64)

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
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=5, gt_dim=5, fcst_steps=500, j=8, k=4, h=64)

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
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

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
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, k=3, h=64)

    N96 = 20; F96 = 8.0
    def l96(t, x):
        d = np.empty_like(x)
        for i in range(N96):
            d[i] = (x[(i+1) % N96] - x[(i-2) % N96]) * x[(i-1) % N96] - x[i] + F96
        return d
    x0 = F96 * np.ones(N96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 200], x0,
                    t_eval=np.arange(0, 200, 0.05), rtol=1e-8, atol=1e-8)
    raw = sol.y.T[400:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, N96, 2500, 500)
    systems["Lorenz-96"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=N96, gt_dim=N96, fcst_steps=500, j=12, k=10, h=64)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    raw_results = {}

    for sname, cfg in systems.items():
        raw_results[sname] = {}
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"  {sname}  seed={seed}")
            print(f"{'='*60}")
            r = run_one(sname, cfg, seed)
            for method, vals in r.items():
                raw_results[sname].setdefault(method, []).append(vals)
                rr = vals["rmse_r"]; rf = vals["rmse_f"]
                phi_str = f" (φ={vals['phi']})" if "phi" in vals else ""
                print(f"  {method:20s}  r={rr:.4f}  f={rf:.3f}{phi_str}")

    stats = {}
    for sname in raw_results:
        stats[sname] = {}
        for method in raw_results[sname]:
            rs = [x["rmse_r"] for x in raw_results[sname][method]]
            fs = [x["rmse_f"] for x in raw_results[sname][method]]
            stats[sname][method] = {
                "rmse_r_mean": float(np.mean(rs)),
                "rmse_r_std": float(np.std(rs)),
                "rmse_f_mean": float(np.mean(fs)),
                "rmse_f_std": float(np.std(fs)),
                "rmse_r_all": rs,
                "rmse_f_all": fs,
            }

    with open(f"{OUT}/head_comparison.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ {OUT}/head_comparison.json")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for sname in stats:
        print(f"\n  {sname}:")
        print(f"    {'Method':20s}  {'Recon':>16s}  {'Forecast':>16s}")
        print(f"    {'-'*20}  {'-'*16}  {'-'*16}")
        for method in stats[sname]:
            s = stats[sname][method]
            print(f"    {method:20s}  {s['rmse_r_mean']:.4f}±{s['rmse_r_std']:.4f}"
                  f"  {s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")
    print("\nDone.")

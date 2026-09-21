"""
Course Correcting Koopman comparison for ICLR 2027.

Course correction = at inference, periodically re-encode the decoded
prediction to correct latent drift.

Four methods compared:
  1. Coupled — standard AE + spectral DMD (no correction)
  2. Coupled_CC — same + course correction every cc_interval steps
  3. Decoupled — our carrier-residual + spectral DMD (no correction)
  4. Decoupled_CC — same + course correction on carrier

5 systems × 4 methods × 5 seeds, multi-horizon spectral evaluation.
Outputs: /kaggle/working/course_correct.json
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
HORIZONS = [1, 5, 10, 25, 50, 100, 250]
CC_INTERVAL = 5
OUTFILE = f"{OUT}/course_correct.json"


def SeedAll(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


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


def train_coupled(model, Xt, phi=0.5):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt) - 1)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t = Xt[sl].to(DEVICE)
            x_tp1 = Xt[sl + 1].to(DEVICE)
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1_pred = z_t
            L_fcst = nn.functional.mse_loss(z_tp1_pred, model.encode(x_tp1).detach())
            loss = (1 - phi) * L_rec + phi * L_fcst
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


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
            with torch.no_grad():
                dh = model.m(model.f(dC[sl].to(DEVICE)))
            loss = nn.functional.mse_loss(
                model.gru(dh, Cc[sl].to(DEVICE)), Cn[sl].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()


def spectral_decompose(A):
    if not np.all(np.isfinite(A)):
        return None
    eigvals, V = np.linalg.eig(A)
    eigvals = np.where(np.abs(eigvals) > 1.0, eigvals / np.abs(eigvals), eigvals)
    V_inv = np.linalg.inv(V)
    return V, eigvals, V_inv


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    lam = 1e-6
    A = Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(X.shape[0]))
    return A


def eval_coupled_spectral(model, Xt_test, gt_dim, horizons, course_correct=False):
    """Evaluate coupled AE with spectral DMD, optionally with course correction."""
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt_test.to(DEVICE)).cpu().numpy()
        recon = model.decode(
            torch.tensor(Z, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    A = fit_dmd(Z)
    sd = spectral_decompose(A)
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
            if not course_correct:
                c = V_inv @ Z[s]
                z_pred = np.real(V @ (c * eigvals ** h))
            else:
                z_cur = Z[s].copy()
                steps_done = 0
                while steps_done < h:
                    steps_to_go = min(CC_INTERVAL, h - steps_done)
                    c = V_inv @ z_cur
                    z_cur = np.real(V @ (c * eigvals ** steps_to_go))
                    steps_done += steps_to_go
                    if steps_done < h:
                        with torch.no_grad():
                            x_dec = model.decode(
                                torch.tensor(z_cur, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                            z_cur = model.encode(x_dec).cpu().numpy()[0]
                z_pred = z_cur
            with torch.no_grad():
                x_pred = model.decode(
                    torch.tensor(z_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
        rmse_per_h[h] = float(np.sqrt(np.mean(errs)))
    return rmse_r, rmse_per_h


def eval_decoupled_spectral(model, Xt_test, gt_dim, horizons, course_correct=False):
    """Evaluate decoupled model with spectral DMD on carrier, optionally CC."""
    model.eval()
    with torch.no_grad():
        C_test = model.carrier(Xt_test.to(DEVICE)).cpu().numpy()
        recon = model.dec(
            torch.tensor(C_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    gt = Xt_test.numpy()
    rmse_r = float(np.sqrt(np.mean((recon[:, :gt_dim] - gt[:, :gt_dim]) ** 2)))

    A = fit_dmd(C_test)
    sd = spectral_decompose(A)
    if sd is None:
        return rmse_r, {h: float('inf') for h in horizons}
    V, eigvals, V_inv = sd

    rmse_per_h = {}
    for h in horizons:
        errs = []
        n_starts = min(len(C_test) - h, 200)
        if n_starts < 1:
            rmse_per_h[h] = float('nan')
            continue
        for s in range(n_starts):
            if not course_correct:
                c = V_inv @ C_test[s]
                c_pred = np.real(V @ (c * eigvals ** h))
            else:
                c_cur = C_test[s].copy()
                steps_done = 0
                while steps_done < h:
                    steps_to_go = min(CC_INTERVAL, h - steps_done)
                    c_coeff = V_inv @ c_cur
                    c_cur = np.real(V @ (c_coeff * eigvals ** steps_to_go))
                    steps_done += steps_to_go
                    if steps_done < h:
                        with torch.no_grad():
                            x_dec = model.dec(
                                torch.tensor(c_cur, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                            c_cur = model.enc(x_dec).cpu().numpy()[0]
                c_pred = c_cur
            with torch.no_grad():
                x_pred = model.dec(
                    torch.tensor(c_pred, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            errs.append((x_pred[:gt_dim] - gt[s + h, :gt_dim]) ** 2)
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
    systems["Coupled Harmonic"] = dict(raw=sol.y.T[200:], n_obs=6, k=4, j=8, h=64, gt_dim=6)

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

    BEST_PHIS = {
        "Coupled Harmonic": 0.1, "Linear 5D": 0.1, "Brusselator": 0.1,
        "Duffing": 0.1, "Lorenz-96": 0.1
    }

    methods = [
        ("Coupled", False, "coupled"),
        ("Coupled_CC", True, "coupled"),
        ("Decoupled", False, "decoupled"),
        ("Decoupled_CC", True, "decoupled"),
    ]

    for sname, cfg in systems.items():
        print(f"\n{'='*60}\n  {sname}\n{'='*60}")
        if sname not in results:
            results[sname] = {}
        raw = cfg["raw"]
        n, k, j, h, gt_dim = cfg["n_obs"], cfg["k"], cfg["j"], cfg["h"], cfg["gt_dim"]
        mu, sig_ = raw[:n_tr].mean(0), raw[:n_tr].std(0) + 1e-8
        Xt = torch.tensor((raw[:n_tr] - mu) / sig_, dtype=torch.float32)
        Xt_test = torch.tensor((raw[n_tr:n_tr + n_te] - mu) / sig_, dtype=torch.float32)

        for mname, cc, arch in methods:
            if mname in results[sname]:
                print(f"  {mname} — skipped (exists)")
                continue

            print(f"\n  [{mname}]")
            all_r = []
            for seed in SEEDS:
                SeedAll(seed)
                if arch == "coupled":
                    model = SimpleAE(n, k, h)
                    train_ae_only(model, Xt)
                    model.eval()
                    rr, hrm = eval_coupled_spectral(model, Xt_test, gt_dim,
                                                     HORIZONS, course_correct=cc)
                else:
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
                    rr, hrm = eval_decoupled_spectral(rm, Xt_test, gt_dim,
                                                       HORIZONS, course_correct=cc)
                f1 = hrm.get(1, float('nan'))
                f100 = hrm.get(100, float('nan'))
                print(f"    seed={seed}: R={rr:.4f} F1={f1:.4f} F100={f100:.4f}")
                all_r.append((rr, hrm))

            results[sname][mname] = aggregate(all_r, HORIZONS)
            save_incremental(results)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")

"""
Quick architecture comparison on Linear 5D: 4 methods, 3 seeds.

Methods:
  1. AE+DMD (coupled baseline)
  2. Resid+GRU+DMD (current decoupled)
  3. DualDec+DMD (Option A: separate forecast decoder, no carrier accumulation)
  4. CarrierCond+DMD (Option B: forecast conditioned on frozen initial carrier + b-trajectory)
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import expm
from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2]
FCST = 500

# ── Data ──────────────────────────────────────────────────────────
def make_linear5d():
    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N = 10000; raw = np.empty((N, 5)); raw[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N): raw[i] = Ad @ raw[i - 1]
    raw = raw[200:]
    n_tr, n_te = 4000, 500
    tr, te = raw[:n_tr], raw[n_tr:n_tr + n_te]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig

N_OBS = 5; J = 8; K = 4; H = 64

# ── Shared modules ────────────────────────────────────────────────
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

class CarrierResidBase(nn.Module):
    """Shared teacher AE + residual compressor for all decoupled variants."""
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j, self.k = j, k
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(nn.Linear(j, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        self.m = nn.Sequential(nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))

    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

# ── Training helpers ──────────────────────────────────────────────
def train_ae(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            loss = nn.functional.mse_loss(model(Xt[idx[i:i+BS]]), Xt[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i+BS]]
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
            dc = dC[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()

def train_gru(gru, model, dC, Cc, Cn):
    opt = torch.optim.Adam(gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        gru.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i+BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()

# ── Phase 1+2 shared setup ────────────────────────────────────────
def setup_decoupled(seed, Xt, train_n):
    SeedAll(seed)
    model = CarrierResidBase(N_OBS, J, K, H)
    train_teacher(model, Xt)
    model.eval()
    with torch.no_grad():
        C = model.carrier(Xt)
    dC = C[1:] - C[:-1]
    Cc, Cn = C[:-1], C[1:]
    train_resid(model, dC)
    model.eval()
    with torch.no_grad():
        B = model.f(dC).numpy()
    A = fit_dmd(B)
    return model, C, dC, Cc, Cn, B, A

# ── Evaluation ────────────────────────────────────────────────────
def evaluate(recon_fn, forecast_fn, test_n):
    Xte = torch.tensor(test_n, dtype=torch.float32)
    with torch.no_grad():
        pred_r = recon_fn(Xte).numpy()
    rmse_r = float(np.sqrt(np.mean((pred_r - test_n) ** 2)))

    pred_f = forecast_fn(test_n[0], FCST)
    rmse_f = float(np.sqrt(np.mean((pred_f - test_n[:FCST]) ** 2)))
    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Method 1: AE+DMD (coupled baseline)
# ═══════════════════════════════════════════════════════════════════
def run_ae_dmd(seed, train_n, test_n):
    SeedAll(seed)
    Xt = torch.tensor(train_n, dtype=torch.float32)
    ae = AE(N_OBS, K, H)
    train_ae(ae, Xt); ae.eval()
    with torch.no_grad(): Z = ae.encode(Xt).numpy()
    A = fit_dmd(Z)

    def recon(x): return ae(x)
    def forecast(x0, steps):
        with torch.no_grad():
            z = ae.encode(torch.tensor(x0[None], dtype=torch.float32)).numpy().ravel()
        out = np.empty((steps, K)); out[0] = z
        for t in range(1, steps): out[t] = A @ out[t-1]
        with torch.no_grad():
            return ae.decode(torch.tensor(out, dtype=torch.float32)).numpy()

    return evaluate(recon, forecast, test_n)


# ═══════════════════════════════════════════════════════════════════
#  Method 2: Resid+GRU+DMD (current decoupled)
# ═══════════════════════════════════════════════════════════════════
def run_resid_gru(seed, train_n, test_n):
    Xt = torch.tensor(train_n, dtype=torch.float32)
    model, C, dC, Cc, Cn, B, A = setup_decoupled(seed, Xt, train_n)

    gru = nn.GRUCell(J, J)
    train_gru(gru, model, dC, Cc, Cn)
    gru.eval()

    def recon(x): return model.recon(x)
    def forecast(x0, steps):
        with torch.no_grad():
            c = model.carrier(torch.tensor(x0[None], dtype=torch.float32)).squeeze(0)
            dc_last = dC[-1:]
            b = model.f(dc_last).numpy().ravel()
        out = np.empty((steps, N_OBS))
        with torch.no_grad():
            out[0] = model.dec(c.unsqueeze(0)).numpy().ravel()
        for t in range(1, steps):
            b = A @ b
            with torch.no_grad():
                dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                c = gru(dh, c.unsqueeze(0)).squeeze(0)
                out[t] = model.dec(c.unsqueeze(0)).numpy().ravel()
        return out

    return evaluate(recon, forecast, test_n)


# ═══════════════════════════════════════════════════════════════════
#  Method 3: DualDec+DMD (Option A — separate forecast decoder)
# ═══════════════════════════════════════════════════════════════════
class ForecastDecoder(nn.Module):
    """Maps accumulated b-space state to observations."""
    def __init__(self, k, n, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
    def forward(self, z): return self.net(z)

def run_dual_dec(seed, train_n, test_n):
    Xt = torch.tensor(train_n, dtype=torch.float32)
    model, C, dC, Cc, Cn, B, A = setup_decoupled(seed, Xt, train_n)

    # Train forecast decoder: map b cumsum state -> observations
    # The "forecast state" is a running sum of b in k-space
    B_t = torch.tensor(B, dtype=torch.float32)  # (T-1, k)
    b_cumsum = torch.cumsum(B_t, dim=0)  # running state in b-space
    # Target: the actual observations at those timesteps (offset by 1)
    X_targets = Xt[1:]  # aligned with dC/B

    SeedAll(seed + 1000)
    fdec = ForecastDecoder(K, N_OBS, H)
    opt = torch.optim.Adam(fdec.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        fdec.train(); idx = torch.randperm(len(b_cumsum))
        for i in range(0, len(b_cumsum), BS):
            sl = idx[i:i+BS]
            loss = nn.functional.mse_loss(fdec(b_cumsum[sl]), X_targets[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    fdec.eval()

    def recon(x): return model.recon(x)
    def forecast(x0, steps):
        with torch.no_grad():
            dc_last = dC[-1:]
            b = model.f(dc_last).numpy().ravel()
        # Get initial b_cumsum from training
        b_sum = B.sum(axis=0).copy()  # last training cumsum
        out = np.empty((steps, N_OBS))
        for t in range(steps):
            b = A @ b
            b_sum = b_sum + b
            with torch.no_grad():
                out[t] = fdec(torch.tensor(b_sum, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return out

    return evaluate(recon, forecast, test_n)


# ═══════════════════════════════════════════════════════════════════
#  Method 4: CarrierCond+DMD (Option B — initial carrier + b-trajectory)
# ═══════════════════════════════════════════════════════════════════
class ConditionedDecoder(nn.Module):
    """Maps (initial_carrier, cumulative_residual, step_encoding) -> observation."""
    def __init__(self, j, k, n, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(j + k + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
    def forward(self, c0, b_cum, tau):
        inp = torch.cat([c0, b_cum, tau], dim=-1)
        return self.net(inp)

def run_carrier_cond(seed, train_n, test_n):
    Xt = torch.tensor(train_n, dtype=torch.float32)
    model, C, dC, Cc, Cn, B, A = setup_decoupled(seed, Xt, train_n)

    B_t = torch.tensor(B, dtype=torch.float32)
    b_cumsum = torch.cumsum(B_t, dim=0)
    # For each timestep t, the "initial carrier" is C[0] of some window.
    # Use a sliding approach: for offset tau from some anchor,
    # predict x_{anchor+tau} from (C_anchor, cumsum(b_{anchor:anchor+tau}), tau/T)
    # Simple version: anchor = 0, predict all timesteps
    C0 = C[0:1].expand(len(b_cumsum), -1)  # (T-1, j) — broadcast initial carrier
    taus = torch.arange(1, len(b_cumsum) + 1, dtype=torch.float32).unsqueeze(1) / len(b_cumsum)
    X_targets = Xt[1:]

    SeedAll(seed + 2000)
    cdec = ConditionedDecoder(J, K, N_OBS, H)
    opt = torch.optim.Adam(cdec.parameters(), lr=LR)
    for ep in range(1, EPOCHS + 1):
        cdec.train(); idx = torch.randperm(len(b_cumsum))
        for i in range(0, len(b_cumsum), BS):
            sl = idx[i:i+BS]
            loss = nn.functional.mse_loss(
                cdec(C0[sl], b_cumsum[sl], taus[sl]), X_targets[sl])
            opt.zero_grad(); loss.backward(); opt.step()
    cdec.eval()

    def recon(x): return model.recon(x)
    def forecast(x0, steps):
        with torch.no_grad():
            c0 = model.carrier(torch.tensor(x0[None], dtype=torch.float32))
            dc_last = dC[-1:]
            b = model.f(dc_last).numpy().ravel()
        b_sum = np.zeros(K)
        out = np.empty((steps, N_OBS))
        for t in range(steps):
            b = A @ b
            b_sum = b_sum + b
            tau_enc = np.array([(t + 1) / steps])
            with torch.no_grad():
                out[t] = cdec(
                    c0,
                    torch.tensor(b_sum, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(tau_enc, dtype=torch.float32).unsqueeze(0)
                ).numpy().ravel()
        return out

    return evaluate(recon, forecast, test_n)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_n, test_n = make_linear5d()
    print(f"Linear 5D: train {train_n.shape}, test {test_n.shape}")

    methods = {
        "AE+DMD":        run_ae_dmd,
        "Resid+GRU":     run_resid_gru,
        "DualDec":       run_dual_dec,
        "CarrierCond":   run_carrier_cond,
    }

    results = {m: [] for m in methods}
    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"  Seed {seed}")
        print(f"{'='*50}")
        for name, fn in methods.items():
            t0 = time.time()
            rmse_r, rmse_f = fn(seed, train_n, test_n)
            dt = time.time() - t0
            results[name].append({"rmse_r": rmse_r, "rmse_f": rmse_f})
            print(f"  {name:15s}  R={rmse_r:.5f}  F={rmse_f:.5f}  ({dt:.0f}s)")

    print(f"\n{'='*50}")
    print(f"  SUMMARY (mean +/- std over {len(SEEDS)} seeds)")
    print(f"{'='*50}")
    summary = {}
    for name in methods:
        rs = [x["rmse_r"] for x in results[name]]
        fs = [x["rmse_f"] for x in results[name]]
        summary[name] = {
            "rmse_r_mean": float(np.mean(rs)), "rmse_r_std": float(np.std(rs)),
            "rmse_f_mean": float(np.mean(fs)), "rmse_f_std": float(np.std(fs)),
        }
        print(f"  {name:15s}  R={np.mean(rs):.5f}±{np.std(rs):.5f}  "
              f"F={np.mean(fs):.5f}±{np.std(fs):.5f}")

    with open(OUT / "arch_compare.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> {OUT / 'arch_compare.json'}")

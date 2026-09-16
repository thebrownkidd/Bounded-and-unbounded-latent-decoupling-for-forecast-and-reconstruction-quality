"""
Non-autoregressive forecast experiment for ICLR 2027 §5.3.

Instead of stepping b autoregressively and accumulating into the carrier via
GRU at each step, we:

    1. Compute the full b trajectory analytically via DMD:
       b_{T+τ} = A^τ b_T  (eigendecomposition, no iteration)

    2. Decode the full residual trajectory in one batch:
       {m(b_{T+τ})} for τ = 1..H

    3. Apply a learned correction network that maps the sequence of
       decoded residuals + initial carrier to the full carrier sequence
       in a single forward pass. No autoregressive loop.

Correction network variants:
    cumsum   — C_{t+τ} = C_0 + Σ_{s=1}^{τ} α_s · m(b_{T+s})
               where α_s are learned per-step damping coefficients
    conv1d   — 1D causal convolution over the residual sequence
    mlp      — per-step MLP(m(b_{T+τ}), C_0, τ/H) → C_{T+τ}

Compared against:
    Resid+DMD+add  — naive addition (existing ablation)
    Resid+DMD+GRU  — GRU accumulation (current best)

Outputs:
    Paper/nonautoregressive.json
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

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--quick", action="store_true")
_args = _parser.parse_args()

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
LR = 1e-3; BS = 512
if _args.quick:
    EPOCHS = 5; SEEDS = [0]; UNROLL = 10
else:
    EPOCHS = 400; SEEDS = [0, 1, 2, 42, 123]; UNROLL = 50


# ═══════════════════════════════════════════════════════════════════
#  Correction networks
# ═══════════════════════════════════════════════════════════════════

class CumsumCorrection(nn.Module):
    """Damped cumulative-sum accumulation.

    C_{T+τ} = C_0 + Σ_{s=1}^{τ} gate_s ⊙ δ_s

    where gate_s = sigmoid(α) is a learned per-dim damping factor (shared
    across steps) and δ_s = m(b_{T+s}).  The gate starts near 1 (so the
    model starts life as plain addition) and can learn to shrink.

    This is fully parallel: the cumulative sum is O(H) with torch.cumsum.
    """
    def __init__(self, j):
        super().__init__()
        # Initialize alpha so that sigmoid(alpha) ≈ 0.95
        self.alpha = nn.Parameter(torch.full((j,), 3.0))

    def forward(self, deltas, C0):
        """deltas: (B, H, j), C0: (B, j) -> carriers: (B, H, j)."""
        gate = torch.sigmoid(self.alpha)  # (j,)
        gated = deltas * gate.unsqueeze(0).unsqueeze(0)  # (B, H, j)
        cumulative = torch.cumsum(gated, dim=1)  # (B, H, j)
        return C0.unsqueeze(1) + cumulative


class Conv1dCorrection(nn.Module):
    """Causal 1D convolution correction.

    Takes the residual sequence (B, H, j) and produces carrier updates
    via causal convolution (no future information leaks).
    """
    def __init__(self, j, kernel_size=16, hidden_channels=32):
        super().__init__()
        self.j = j
        self.pad = kernel_size - 1  # causal padding
        self.net = nn.Sequential(
            nn.Conv1d(j, hidden_channels, kernel_size, padding=0),
            nn.ELU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.ELU(),
            nn.Conv1d(hidden_channels, j, 1),
        )
        # Zero-initialize last layer so it starts as identity (addition)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, deltas, C0):
        """deltas: (B, H, j), C0: (B, j) -> carriers: (B, H, j)."""
        # Causal: pad on the left
        x = deltas.transpose(1, 2)  # (B, j, H)
        x = torch.nn.functional.pad(x, (self.pad, 0))  # (B, j, H + pad)
        correction = self.net(x).transpose(1, 2)  # (B, H, j)
        # Base: cumulative sum + learned correction
        cumulative = torch.cumsum(deltas, dim=1)
        return C0.unsqueeze(1) + cumulative + correction


class MLPCorrection(nn.Module):
    """Per-step MLP: takes (delta, C0, τ/H) and produces the carrier directly.

    Non-autoregressive: each step is independent, conditioned on the initial
    carrier and the step index.
    """
    def __init__(self, j, hidden=128):
        super().__init__()
        # Input: delta (j) + C0 (j) + position (1) = 2j + 1
        self.net = nn.Sequential(
            nn.Linear(2 * j + 1, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, j),
        )

    def forward(self, deltas, C0):
        """deltas: (B, H, j), C0: (B, j) -> carriers: (B, H, j)."""
        B, H, j = deltas.shape
        # Position encoding: τ/H ∈ [0, 1]
        pos = torch.arange(1, H + 1, dtype=deltas.dtype,
                          device=deltas.device).unsqueeze(0) / H  # (1, H)
        pos = pos.unsqueeze(-1).expand(B, H, 1)  # (B, H, 1)
        # Cumulative delta up to each step
        cum_delta = torch.cumsum(deltas, dim=1)  # (B, H, j)
        C0_exp = C0.unsqueeze(1).expand(B, H, j)  # (B, H, j)
        inp = torch.cat([cum_delta, C0_exp, pos], dim=-1)  # (B, H, 2j+1)
        correction = self.net(inp.reshape(B * H, -1)).reshape(B, H, j)
        return C0.unsqueeze(1) + cum_delta + correction


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

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


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)


def dmd_rollout_batch(A, b0, horizon):
    """Compute b_{T+1}, ..., b_{T+H} = A^1 b_0, ..., A^H b_0.

    Uses eigendecomposition for stability: A = V Λ V^{-1},
    so A^τ b_0 = V Λ^τ V^{-1} b_0.

    Returns: (H, k) array.
    """
    eigvals, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    c = Vinv @ b0  # coefficients in eigenbasis
    taus = np.arange(1, horizon + 1)
    # Λ^τ c for each τ: (H, k) complex, take real part
    trajectory = np.real(V @ (c[:, None] * eigvals[:, None] ** taus[None, :]))
    return trajectory.T  # (H, k)


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

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


def train_correction(correction_net, model, C_all, B_all, A_dmd, unroll=UNROLL):
    """Train a correction network on multi-step windows.

    For each training window of length `unroll`:
        1. Take initial carrier C_0 and initial residual code b_0
        2. Roll out b via DMD: b_{0+τ} = A^τ b_0
        3. Decode residuals: δ_τ = m(b_{0+τ})
        4. Predict carriers via correction network
        5. Loss = MSE(predicted carriers, true carriers)
    """
    # Prepare windows: (N, unroll, j) carriers, (N, unroll, k) residual codes
    N = len(C_all) - unroll
    if N < BS:
        print(f"    WARNING: only {N} windows for unroll={unroll}")

    opt = torch.optim.Adam(correction_net.parameters(), lr=LR)
    k = B_all.shape[1]

    for ep in range(1, EPOCHS + 1):
        correction_net.train()
        idx = torch.randperm(N)
        for i in range(0, N, BS):
            sl = idx[i:i + BS]
            starts = sl.numpy()

            # Initial carriers and residual codes
            C0 = C_all[starts]  # (B, j)
            # Target carrier sequence
            C_target = torch.stack([C_all[s + 1:s + 1 + unroll] for s in starts])  # (B, U, j)

            # DMD rollout in b-space (batched)
            b0s = B_all[starts].numpy()  # (B, k)
            deltas_list = []
            for bi in range(len(b0s)):
                b_traj = dmd_rollout_batch(A_dmd, b0s[bi], unroll)  # (U, k)
                with torch.no_grad():
                    delta = model.m(torch.tensor(b_traj, dtype=torch.float32))  # (U, j)
                deltas_list.append(delta)
            deltas = torch.stack(deltas_list)  # (B, U, j)

            # Correction network predicts carrier sequence
            C_pred = correction_net(deltas, C0)

            loss = nn.functional.mse_loss(C_pred, C_target)
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
#  Data generators (same as run_multiseed.py)
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

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Run one system × one seed
# ═══════════════════════════════════════════════════════════════════

def run_one(cfg, seed):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps, k, j, h = cfg["fcst_steps"], cfg["k"], cfg["j"], cfg["h"]
    Xt = torch.tensor(train_n, dtype=torch.float32)
    last_train = train_n[-1]
    res = {}

    # ── Shared: Phase 1 + 2 (teacher AE + residual compressor) ──
    SeedAll(seed)
    rm = ResidualModel(n_obs, j, k, h=h)
    train_teacher(rm, Xt); rm.eval()
    with torch.no_grad(): C = rm.carrier(Xt)
    Cc, Cn = C[:-1], C[1:]
    dC = Cn - Cc
    train_resid(rm, dC); rm.eval()
    with torch.no_grad(): B = rm.f(dC).numpy()
    B_tensor = rm.f(dC).detach()  # (T-1, k)
    A_dmd = fit_dmd(B)

    def _r(tn, m=rm):
        with torch.no_grad():
            return m.recon(torch.tensor(tn, dtype=torch.float32)).numpy()

    # ── Resid+DMD+add (baseline, no Phase 3) ──
    def _f_add(m=rm, A_=A_dmd):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None], dtype=torch.float32))
            Cc_init = m.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
            b0 = m.f(Cc_init - Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc_init.squeeze(0).numpy(); b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(torch.tensor(C_, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dc = m.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C_ = C_ + dc
            with torch.no_grad():
                fc[t] = m.dec(torch.tensor(C_, dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    rr, rf = eval_metrics(_r, _f_add, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+add"] = {"rmse_r": rr, "rmse_f": rf}

    # ── Resid+DMD+GRU (Phase 3: train GRU) ──
    train_gru(rm, dC, Cc, Cn); rm.eval()

    def _f_gru(m=rm, A_=A_dmd):
        with torch.no_grad():
            Cp = m.carrier(torch.tensor(last_train[None], dtype=torch.float32))
            Cc_init = m.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
            b0 = m.f(Cc_init - Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc_init.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_ @ b
            with torch.no_grad():
                dh = m.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc
    rr, rf = eval_metrics(_r, _f_gru, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    res["Resid+GRU"] = {"rmse_r": rr, "rmse_f": rf}

    # ── Non-autoregressive correction variants ──
    for corr_name, corr_cls in [("cumsum", CumsumCorrection),
                                 ("conv1d", Conv1dCorrection),
                                 ("mlp", MLPCorrection)]:
        SeedAll(seed + 3000)
        corr = corr_cls(j)
        train_correction(corr, rm, C, B_tensor, A_dmd); corr.eval()

        def _f_corr(m=rm, A_=A_dmd, c=corr):
            with torch.no_grad():
                Cp = m.carrier(torch.tensor(last_train[None], dtype=torch.float32))
                Cc_init = m.carrier(torch.tensor(test_n[:1], dtype=torch.float32))
                b0 = m.f(Cc_init - Cp).numpy().ravel()

            # Full DMD rollout (no iteration)
            b_traj = dmd_rollout_batch(A_, b0, fcst_steps)  # (H, k)

            with torch.no_grad():
                deltas = m.m(torch.tensor(b_traj, dtype=torch.float32))  # (H, j)
                C0 = Cc_init.squeeze(0)  # (j,)
                carriers = c(deltas.unsqueeze(0), C0.unsqueeze(0))  # (1, H, j)
                fc = m.dec(carriers.squeeze(0)).numpy()  # (H, n_obs)
            return fc

        rr, rf = eval_metrics(_r, _f_corr, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
        res[f"Resid+{corr_name}"] = {"rmse_r": rr, "rmse_f": rf}

    return res


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
            print(f"\n  {sname}  seed={seed}")
            r = run_one(cfg, seed)
            for method, vals in r.items():
                raw_results[sname].setdefault(method, []).append(vals)
                print(f"    {method:16s}  r={vals['rmse_r']:.4f}  f={vals['rmse_f']:.3f}")

    # ── Compute mean ± std ──
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

    with open(OUT / "nonautoregressive.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ {OUT / 'nonautoregressive.json'}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print("\nSummary:")
    for sname in stats:
        print(f"\n  {sname}:")
        for method in stats[sname]:
            s = stats[sname][method]
            print(f"    {method:16s}  r={s['rmse_r_mean']:.4f}±{s['rmse_r_std']:.4f}"
                  f"  f={s['rmse_f_mean']:.3f}±{s['rmse_f_std']:.3f}")
    print("\nDone.")

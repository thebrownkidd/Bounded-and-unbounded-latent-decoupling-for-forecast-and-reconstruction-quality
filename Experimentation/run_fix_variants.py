"""
Fix variants: MLP Transition (no phase 3) + Delay-embedded DMD
===============================================================
① MLP Transition — NO phase 3 (keep teacher dec intact for recon)
② Residual + delay-embedded DMD + naked add
③ Residual + delay-embedded DMD + GRU gate
④ Std AE + DMD (baseline)

All use DMD as the forecast engine (fair comparison).
Delay embedding: stack [b_t, b_{t-1}, ..., b_{t-d}] before fitting DMD.

Run on: Lorenz-63, Rössler, Lorenz-96.
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
DEVICE = "cpu"
SEED = 0
EPOCHS = 400
LR = 1e-3
BS = 512
DELAY_D = 3   # number of delays to embed in latent space

# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

class AE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))

class ResidualBase(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j = j; self.k = k
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        self.m = nn.Sequential(nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

class ResidualGRU(ResidualBase):
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.gru = nn.GRUCell(j, j)

class TransitionModel(nn.Module):
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j = j; self.k = k
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.f = nn.Sequential(
            nn.Linear(2*j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.M = nn.Sequential(
            nn.Linear(k+j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def delay_embed(B, d):
    """Stack [b_t, b_{t-1}, ..., b_{t-d+1}] for t = d-1 .. N-1.
    Returns (N-d+1, k*d) array."""
    N, k = B.shape
    out = np.empty((N - d + 1, k * d))
    for i in range(d):
        out[:, i*k:(i+1)*k] = B[d-1-i:N-i]
    return out

def rollout_delay_dmd(A_del, z_aug, k, d, n_steps):
    """Rollout delay-embedded DMD. z_aug is (k*d,).
    Returns (n_steps, k) — only the current b (first k dims)."""
    out = np.empty((n_steps, k))
    z = z_aug.copy()
    out[0] = z[:k]
    for t in range(1, n_steps):
        z = A_del @ z
        out[t] = z[:k]
    return out


# ═══════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════

def train_teacher(model, Xt_all):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        p1 ep {ep}")

def train_residual_ae(model, delta_C):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(delta_C))
        for i in range(0, len(delta_C), BS):
            dc = delta_C[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        p2a ep {ep}")

def train_gru(model, dC, C_cur, C_nxt):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(C_cur))
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            with torch.no_grad():
                delta_hat = model.m(model.f(dC[sl]))
            c_next_hat = model.gru(delta_hat, C_cur[sl])
            loss = nn.functional.mse_loss(c_next_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        gru ep {ep}")


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(name, train_n, test_n, last_train, gt_test, mu, sig,
               n_obs, gt_dim, lt_steps, fcst_len,
               k_dim, carrier_dim, h_base, h_tp):

    Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
    Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
    Xt_all = torch.tensor(train_n, dtype=torch.float32)
    j = carrier_dim; k = k_dim; d = DELAY_D

    def evaluate(tag, recon_fn, fcst_fn):
        rec = recon_fn(test_n)
        rec_p = (rec * sig + mu)[:, :gt_dim]
        fc = fcst_fn()
        fc_p = (fc * sig + mu)[:, :gt_dim]
        N = min(lt_steps, len(fc_p))
        rmse_r = np.sqrt(np.mean((rec_p - gt_test)**2))
        rmse_f = np.sqrt(np.mean((fc_p[:N] - gt_test[:N])**2))
        mx = np.max(np.abs(fc_p))
        div = mx > 500
        print(f"  {tag:45s} recon={rmse_r:.4f}  fcst={rmse_f:.4f}"
              f"{'  DIVERGED' if div else ''}")
        return dict(rmse_r=float(rmse_r), rmse_f=float(rmse_f),
                    mx=float(mx), div=bool(div))

    results = {}

    # ── Std AE + DMD ──────────────────────────────────────────────
    print(f"\n  --- Std AE + DMD ---")
    torch.manual_seed(SEED)
    ae = AE(n_obs, k, h_base)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        ae.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            loss = nn.functional.mse_loss(ae(Xt_all[idx[i:i+BS]]), Xt_all[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): Z_s = ae.encode(Xt_all).numpy()
    A_s = fit_dmd(Z_s)

    def _r_std(tn):
        with torch.no_grad(): return ae(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _f_std():
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(test_n[:1], dtype=torch.float32)).numpy().ravel()
        out = np.empty((fcst_len, k)); out[0] = z0
        for t in range(1, fcst_len): out[t] = A_s @ out[t-1]
        with torch.no_grad(): return ae.decode(torch.tensor(out, dtype=torch.float32)).numpy()
    results["Std AE+DMD"] = evaluate("Std AE + DMD", _r_std, _f_std)

    # ── Std AE + delay-embedded DMD ───────────────────────────────
    print(f"\n  --- Std AE + delay DMD (d={d}) ---")
    Z_del = delay_embed(Z_s, d)
    A_sd = fit_dmd(Z_del)
    eig_sd = np.sort(np.abs(np.linalg.eigvals(A_sd)))[::-1]
    print(f"      delay DMD |λ| top-5: {eig_sd[:5]}")

    def _f_std_del():
        with torch.no_grad():
            # need d consecutive latents ending at test[0]
            # use last d-1 training + first test
            x_init = torch.tensor(
                np.vstack([train_n[-(d-1):], test_n[:1]]), dtype=torch.float32)
            z_init = ae.encode(x_init).numpy()  # (d, k)
        z_aug = np.concatenate([z_init[d-1-i] for i in range(d)])  # (k*d,)
        B_roll = rollout_delay_dmd(A_sd, z_aug, k, d, fcst_len)
        with torch.no_grad():
            return ae.decode(torch.tensor(B_roll, dtype=torch.float32)).numpy()
    results["Std AE+delDMD"] = evaluate("Std AE + delay DMD", _r_std, _f_std_del)

    # ════════════════════════════════════════════════════════════════
    #  Helpers for residual models
    # ════════════════════════════════════════════════════════════════
    def get_carriers(model):
        model.eval()
        with torch.no_grad(): C = model.carrier(Xt_all)
        return C, C[:-1], C[1:]

    # ── Resid + DMD + add ─────────────────────────────────────────
    print(f"\n  --- Resid + DMD + add ---")
    SeedAll(SEED)
    r1 = ResidualBase(n_obs, j, k, h=h_tp)
    train_teacher(r1, Xt_all)
    C_all, C_cur, C_nxt = get_carriers(r1)
    dC = C_nxt - C_cur
    train_residual_ae(r1, dC)
    r1.eval()
    with torch.no_grad(): B1 = r1.f(dC).numpy()
    A1 = fit_dmd(B1)

    def _r_r1(tn):
        with torch.no_grad(): return r1.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_r1():
        with torch.no_grad():
            Cp = r1.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = r1.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = r1.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs)); C=Cc.numpy().ravel(); b=b0.copy()
        with torch.no_grad():
            fc[0] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A1 @ b
            with torch.no_grad():
                dc = r1.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+DMD+add"] = evaluate("Resid + DMD + add", _r_r1, _f_r1)

    # ── Resid + delay DMD + add ───────────────────────────────────
    print(f"\n  --- Resid + delay DMD + add (d={d}) ---")
    B1_del = delay_embed(B1, d)
    A1d = fit_dmd(B1_del)
    eig_1d = np.sort(np.abs(np.linalg.eigvals(A1d)))[::-1]
    print(f"      delay DMD |λ| top-5: {eig_1d[:5]}")

    def _f_r1d():
        with torch.no_grad():
            # Need d consecutive carrier residuals ending at last_train→test[0]
            # Use last d training carriers + test[0] carrier
            x_block = torch.tensor(
                np.vstack([train_n[-(d):], test_n[:1]]), dtype=torch.float32)
            C_block = r1.carrier(x_block)
            dC_block = C_block[1:] - C_block[:-1]  # (d, j)
            B_block = r1.f(dC_block).numpy()        # (d, k)
        z_aug = np.concatenate([B_block[d-1-i] for i in range(d)])
        B_roll = rollout_delay_dmd(A1d, z_aug, k, d, fcst_len)
        # Accumulate carriers
        fc = np.empty((fcst_len, n_obs))
        with torch.no_grad():
            C_start = r1.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
        C = C_start.copy()
        with torch.no_grad():
            fc[0] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            with torch.no_grad():
                dc = r1.m(torch.tensor(B_roll[t],dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = r1.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+delDMD+add"] = evaluate("Resid + delay DMD + add", _r_r1, _f_r1d)

    # ── Resid + delay DMD + GRU ───────────────────────────────────
    print(f"\n  --- Resid + delay DMD + GRU (d={d}) ---")
    SeedAll(SEED)
    r2 = ResidualGRU(n_obs, j, k, h=h_tp)
    train_teacher(r2, Xt_all)
    C_all2, C_cur2, C_nxt2 = get_carriers(r2)
    dC2 = C_nxt2 - C_cur2
    train_residual_ae(r2, dC2)
    train_gru(r2, dC2, C_cur2, C_nxt2)
    r2.eval()
    with torch.no_grad(): B2 = r2.f(dC2).numpy()
    B2_del = delay_embed(B2, d)
    A2d = fit_dmd(B2_del)

    def _r_r2(tn):
        with torch.no_grad(): return r2.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_r2d():
        with torch.no_grad():
            x_block = torch.tensor(
                np.vstack([train_n[-(d):], test_n[:1]]), dtype=torch.float32)
            C_block = r2.carrier(x_block)
            dC_block = C_block[1:] - C_block[:-1]
            B_block = r2.f(dC_block).numpy()
        z_aug = np.concatenate([B_block[d-1-i] for i in range(d)])
        B_roll = rollout_delay_dmd(A2d, z_aug, k, d, fcst_len)
        fc = np.empty((fcst_len, n_obs))
        with torch.no_grad():
            C = r2.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).squeeze(0)
        with torch.no_grad():
            fc[0] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            with torch.no_grad():
                delta_hat = r2.m(torch.tensor(B_roll[t],dtype=torch.float32).unsqueeze(0))
                C = r2.gru(delta_hat, C.unsqueeze(0)).squeeze(0)
                fc[t] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+delDMD+GRU"] = evaluate("Resid + delay DMD + GRU", _r_r2, _f_r2d)

    # ── MLP Transition NO phase 3 ────────────────────────────────
    print(f"\n  --- MLP Transition (no phase 3) ---")
    SeedAll(SEED)
    tm = TransitionModel(n_obs, j, k, h=h_tp)
    train_teacher(tm, Xt_all)
    C_allT, C_curT, C_nxtT = get_carriers(tm)

    # Phase 2 only: f + M
    print("      Phase 2: f + M")
    for p in tm.parameters(): p.requires_grad_(False)
    for mod in [tm.f, tm.M]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(C_curT))
        for i in range(0, len(C_curT), BS):
            sl = idx[i:i+BS]
            b = tm.f(torch.cat([C_curT[sl], C_nxtT[sl]], dim=-1))
            c_hat = tm.M(torch.cat([b, C_curT[sl]], dim=-1))
            loss = nn.functional.mse_loss(c_hat, C_nxtT[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        ep {ep}")
    # NO phase 3 — decoder stays as trained in phase 1

    tm.eval()
    with torch.no_grad():
        B_tm = tm.f(torch.cat([C_curT, C_nxtT], dim=-1)).numpy()
    A_tm = fit_dmd(B_tm)

    def _r_tm(tn):
        with torch.no_grad(): return tm.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_tm():
        with torch.no_grad():
            Cp = tm.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = tm.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = tm.f(torch.cat([Cp, Cc], dim=-1)).numpy().ravel()
        fc = np.empty((fcst_len, n_obs)); C = Cc.numpy().ravel(); b = b0.copy()
        with torch.no_grad():
            fc[0] = tm.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A_tm @ b
            with torch.no_grad():
                inp = torch.tensor(np.concatenate([b,C]),dtype=torch.float32).unsqueeze(0)
                C = tm.M(inp).numpy().ravel()
                fc[t] = tm.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["MLPTrans(noP3)"] = evaluate("MLP Transition (no phase 3)", _r_tm, _f_tm)

    return results


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def gen_lorenz63():
    dt = 0.02
    def ode(t, s):
        x, y, z = s
        return [10*(y-x), x*(28-z)-y, x*y - 8/3*z]
    sol = solve_ivp(ode, [0, 120], [1,1,1],
                    t_eval=np.arange(0, 120, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[1000:]
    DELAYS = 5
    obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
    N_TRAIN, N_TEST = 3000, 500
    train_obs = obs[:N_TRAIN]; test_obs = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt_test = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    return dict(name="Lorenz-63", train_n=(train_obs-mu)/sig, test_n=(test_obs-mu)/sig,
                last_train=((train_obs-mu)/sig)[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=55, fcst_len=400,
                k_dim=4, carrier_dim=16, h_base=96, h_tp=64)

def gen_rossler():
    a_, b_, c_ = 0.2, 0.2, 5.7; dt = 0.05
    def ode(t, s):
        x, y, z = s
        return [-(y+z), x + a_*y, b_ + z*(x - c_)]
    sol = solve_ivp(ode, [0, 800], [1, 1, 0],
                    t_eval=np.arange(0, 800, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[4000:]
    DELAYS = 5
    obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
    N_TRAIN, N_TEST = 3000, 500
    train_obs = obs[:N_TRAIN]; test_obs = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt_test = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    return dict(name="Rössler", train_n=(train_obs-mu)/sig, test_n=(test_obs-mu)/sig,
                last_train=((train_obs-mu)/sig)[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=290, fcst_len=500,
                k_dim=4, carrier_dim=16, h_base=96, h_tp=64)

def gen_lorenz96():
    N_L96, F_L96 = 20, 8.0; dt = 0.01
    def l96(t, x):
        d = np.empty_like(x)
        for i in range(len(x)):
            d[i] = (x[(i+1)%N_L96] - x[(i-2)%N_L96]) * x[(i-1)%N_L96] - x[i] + F_L96
        return d
    x0 = F_L96 * np.ones(N_L96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 200], x0,
                    t_eval=np.arange(0, 200, dt),
                    method="RK45", rtol=1e-9, atol=1e-9)
    raw = sol.y.T[2000:]
    N_TRAIN, N_TEST = 4000, 500
    train_obs = raw[:N_TRAIN]; test_obs = raw[N_TRAIN:N_TRAIN+N_TEST]
    gt_test = raw[N_TRAIN:N_TRAIN+N_TEST]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    return dict(name="Lorenz-96", train_n=(train_obs-mu)/sig, test_n=(test_obs-mu)/sig,
                last_train=((train_obs-mu)/sig)[-1], gt_test=gt_test, mu=mu, sig=sig,
                n_obs=20, gt_dim=20, lt_steps=67, fcst_len=200,
                k_dim=10, carrier_dim=18, h_base=96, h_tp=64)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

all_results = {}
t0 = time.time()

for gen_fn in [gen_lorenz63, gen_rossler, gen_lorenz96]:
    cfg = gen_fn()
    name = cfg.pop("name")
    print(f"\n{'#'*65}")
    print(f"  {name}")
    print(f"{'#'*65}")

    results = run_system(name, **cfg)
    all_results[name] = results

    print(f"\n  {'Method':<25s} {'Recon':>8s} {'Fcst':>8s}")
    print(f"  {'-'*45}")
    for tag, r in results.items():
        d_ = " DIV" if r["div"] else ""
        print(f"  {tag:<25s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d_}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Cross-system Summary")
print(f"{'='*70}")
for sname, results in all_results.items():
    print(f"\n  {sname}:")
    print(f"    {'Method':<25s} {'Recon':>8s} {'Fcst':>8s}")
    print(f"    {'-'*45}")
    for tag, r in results.items():
        d_ = " DIV" if r["div"] else ""
        print(f"    {tag:<25s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d_}")

# ── Pareto plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {
    "Std AE+DMD": "#888888", "Std AE+delDMD": "#aaaaaa",
    "Resid+DMD+add": "#1f77b4", "Resid+delDMD+add": "#17becf",
    "Resid+delDMD+GRU": "#2ca02c", "MLPTrans(noP3)": "#ff7f0e",
}
markers = {
    "Std AE+DMD": "s", "Std AE+delDMD": "D",
    "Resid+DMD+add": "o", "Resid+delDMD+add": "^",
    "Resid+delDMD+GRU": "*", "MLPTrans(noP3)": "P",
}

for ax, (sname, results) in zip(axes, all_results.items()):
    for tag, r in results.items():
        if r["div"] or r["rmse_f"] > 20: continue
        ax.plot(r["rmse_r"], r["rmse_f"],
                markers.get(tag,"o"), color=colors.get(tag,"#000"),
                ms=11, zorder=4, label=tag)
    ax.set_xlabel("Reconstruction RMSE")
    ax.set_ylabel("Forecast RMSE (1 LT)")
    ax.set_title(sname)
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.25)

fig.suptitle("Fix Variants: Delay DMD + MLP Transition (no P3)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fix_variants.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'fix_variants.png'}")

with open(OUT / "fix_variants.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'fix_variants.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

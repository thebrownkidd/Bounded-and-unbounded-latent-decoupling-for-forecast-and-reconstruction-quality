"""
Full-rank residual: k = carrier_dim (no compression)
=====================================================
f: ΔC → b ∈ ℝ^j  (same dim, learned change-of-basis)
m: b → ΔC ∈ ℝ^j
DMD operates in the full carrier-dim space.

Also test: raw DMD on ΔC directly (no f/m at all).

Methods:
  ① Std AE + DMD
  ② Std AE + delay DMD (d=3)
  ③ Raw carrier-resid DMD + add  (DMD on ΔC, no f/m)
  ④ Raw carrier-resid DMD + GRU
  ⑤ Full-rank resid + DMD + add  (k=j, learned f/m)
  ⑥ Full-rank resid + DMD + GRU
  ⑦ Full-rank resid + delay DMD + GRU

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
SEED = 0
EPOCHS = 400
LR = 1e-3
BS = 512
DELAY_D = 3

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

class TeacherAE(nn.Module):
    """Teacher AE only — carrier dim j."""
    def __init__(self, n, j, h=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

class FullRankResidual(nn.Module):
    """Residual model with k = j (no compression)."""
    def __init__(self, n, j, h=64):
        super().__init__()
        self.j = j
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        # f: ΔC ∈ ℝ^j → b ∈ ℝ^j  (same dim, change of basis)
        self.f = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, j))
        # m: b ∈ ℝ^j → ΔC ∈ ℝ^j
        self.m = nn.Sequential(nn.Linear(j, h), nn.ELU(), nn.Linear(h, j))
    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)

class FullRankResidualGRU(FullRankResidual):
    def __init__(self, n, j, h=64):
        super().__init__(n, j, h)
        self.gru = nn.GRUCell(j, j)


def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def delay_embed(B, d):
    N, k = B.shape
    out = np.empty((N - d + 1, k * d))
    for i in range(d):
        out[:, i*k:(i+1)*k] = B[d-1-i:N-i]
    return out

def rollout_delay_dmd(A_del, z_aug, k, d, n_steps):
    out = np.empty((n_steps, k))
    z = z_aug.copy(); out[0] = z[:k]
    for t in range(1, n_steps):
        z = A_del @ z; out[t] = z[:k]
    return out


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_teacher_ae(model, Xt_all):
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

def train_resid_ae(model, dC):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(dC))
        for i in range(0, len(dC), BS):
            dc = dC[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        p2 ep {ep}")

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
            c_hat = model.gru(delta_hat, C_cur[sl])
            loss = nn.functional.mse_loss(c_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        gru ep {ep}")

def train_gru_raw(gru, dC_raw, C_cur, C_nxt):
    """Train GRU with raw carrier residuals (no f/m)."""
    opt = torch.optim.Adam(gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        gru.train(); idx = torch.randperm(len(C_cur))
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            c_hat = gru(dC_raw[sl], C_cur[sl])
            loss = nn.functional.mse_loss(c_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        gru ep {ep}")


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(name, train_n, test_n, last_train, gt_test, mu, sig,
               n_obs, gt_dim, lt_steps, fcst_len,
               carrier_dim, h_base, h_tp):

    Xt_all = torch.tensor(train_n, dtype=torch.float32)
    j = carrier_dim; d = DELAY_D

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
        print(f"  {tag:45s} r={rmse_r:.4f}  f={rmse_f:.4f}"
              f"{'  DIV' if div else ''}")
        return dict(rmse_r=float(rmse_r), rmse_f=float(rmse_f),
                    mx=float(mx), div=bool(div))

    results = {}

    # ── ① Std AE + DMD ───────────────────────────────────────────
    # Use k = carrier_dim for fair comparison (same latent size)
    print(f"\n  --- Std AE + DMD (k={j}) ---")
    torch.manual_seed(SEED)
    ae = AE(n_obs, j, h_base)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        ae.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            loss = nn.functional.mse_loss(ae(Xt_all[idx[i:i+BS]]), Xt_all[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): Z_ae = ae.encode(Xt_all).numpy()
    A_ae = fit_dmd(Z_ae)

    def _r_ae(tn):
        with torch.no_grad(): return ae(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_ae():
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
        out = np.empty((fcst_len, j)); out[0] = z0
        for t in range(1, fcst_len): out[t] = A_ae @ out[t-1]
        with torch.no_grad(): return ae.decode(torch.tensor(out,dtype=torch.float32)).numpy()
    results["Std AE+DMD"] = evaluate("Std AE + DMD", _r_ae, _f_ae)

    # ── ② Std AE + delay DMD ─────────────────────────────────────
    print(f"\n  --- Std AE + delay DMD ---")
    Z_del = delay_embed(Z_ae, d)
    A_aed = fit_dmd(Z_del)

    def _f_aed():
        with torch.no_grad():
            x_init = torch.tensor(np.vstack([train_n[-(d-1):], test_n[:1]]),dtype=torch.float32)
            z_init = ae.encode(x_init).numpy()
        z_aug = np.concatenate([z_init[d-1-i] for i in range(d)])
        B_roll = rollout_delay_dmd(A_aed, z_aug, j, d, fcst_len)
        with torch.no_grad(): return ae.decode(torch.tensor(B_roll,dtype=torch.float32)).numpy()
    results["Std AE+delDMD"] = evaluate("Std AE + delay DMD", _r_ae, _f_aed)

    # ════════════════════════════════════════════════════════════════
    #  Train shared teacher AE for raw-residual methods
    # ════════════════════════════════════════════════════════════════
    print(f"\n  --- Training teacher AE (j={j}) ---")
    SeedAll(SEED)
    teacher = TeacherAE(n_obs, j, h=h_tp)
    for p in teacher.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(teacher.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        teacher.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(teacher.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        teacher ep {ep}")
    teacher.eval()
    with torch.no_grad():
        C_all = teacher.carrier(Xt_all)
    C_cur, C_nxt = C_all[:-1], C_all[1:]
    dC = C_nxt - C_cur

    def _r_teacher(tn):
        with torch.no_grad(): return teacher.recon(torch.tensor(tn,dtype=torch.float32)).numpy()

    # ── ③ Raw carrier-resid DMD + add ─────────────────────────────
    print(f"\n  --- Raw resid DMD + add ---")
    dC_np = dC.numpy()
    A_raw = fit_dmd(dC_np)
    eig_raw = np.sort(np.abs(np.linalg.eigvals(A_raw)))[::-1]
    print(f"      |λ| top-5: {eig_raw[:5]}")

    def _f_raw_add():
        with torch.no_grad():
            Cp = teacher.carrier(torch.tensor(last_train[None],dtype=torch.float32)).numpy().ravel()
            Cc = teacher.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
        dc0 = Cc - Cp
        fc = np.empty((fcst_len, n_obs)); C = Cc.copy(); dc = dc0.copy()
        with torch.no_grad():
            fc[0] = teacher.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            dc = A_raw @ dc
            C = C + dc
            with torch.no_grad():
                fc[t] = teacher.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["RawResid+add"] = evaluate("Raw resid DMD + add", _r_teacher, _f_raw_add)

    # ── ④ Raw carrier-resid DMD + GRU ─────────────────────────────
    print(f"\n  --- Raw resid DMD + GRU ---")
    gru_raw = nn.GRUCell(j, j)
    train_gru_raw(gru_raw, dC, C_cur, C_nxt)
    gru_raw.eval()

    def _f_raw_gru():
        with torch.no_grad():
            Cp = teacher.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = teacher.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
        dc0 = (Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs)); C = Cc.squeeze(0); dc = dc0.copy()
        with torch.no_grad():
            fc[0] = teacher.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            dc = A_raw @ dc
            with torch.no_grad():
                C = gru_raw(torch.tensor(dc,dtype=torch.float32).unsqueeze(0),
                            C.unsqueeze(0)).squeeze(0)
                fc[t] = teacher.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["RawResid+GRU"] = evaluate("Raw resid DMD + GRU", _r_teacher, _f_raw_gru)

    # ── ⑤ Full-rank resid + DMD + add ────────────────────────────
    print(f"\n  --- Full-rank resid + DMD + add ---")
    SeedAll(SEED)
    fr = FullRankResidual(n_obs, j, h=h_tp)
    train_teacher_ae(fr, Xt_all)
    fr.eval()
    with torch.no_grad():
        C_fr = fr.carrier(Xt_all)
    dC_fr = C_fr[1:] - C_fr[:-1]
    train_resid_ae(fr, dC_fr)
    fr.eval()
    with torch.no_grad(): B_fr = fr.f(dC_fr).numpy()
    A_fr = fit_dmd(B_fr)

    def _r_fr(tn):
        with torch.no_grad(): return fr.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_fr_add():
        with torch.no_grad():
            Cp = fr.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = fr.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = fr.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs)); C=Cc.numpy().ravel(); b=b0.copy()
        with torch.no_grad():
            fc[0] = fr.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A_fr @ b
            with torch.no_grad():
                dc = fr.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = fr.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["FullRank+add"] = evaluate("Full-rank resid + DMD + add", _r_fr, _f_fr_add)

    # ── ⑥ Full-rank resid + DMD + GRU ────────────────────────────
    print(f"\n  --- Full-rank resid + DMD + GRU ---")
    SeedAll(SEED)
    frg = FullRankResidualGRU(n_obs, j, h=h_tp)
    train_teacher_ae(frg, Xt_all)
    frg.eval()
    with torch.no_grad():
        C_frg = frg.carrier(Xt_all)
    C_cur_g, C_nxt_g = C_frg[:-1], C_frg[1:]
    dC_frg = C_nxt_g - C_cur_g
    train_resid_ae(frg, dC_frg)
    train_gru(frg, dC_frg, C_cur_g, C_nxt_g)
    frg.eval()
    with torch.no_grad(): B_frg = frg.f(dC_frg).numpy()
    A_frg = fit_dmd(B_frg)

    def _r_frg(tn):
        with torch.no_grad(): return frg.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_frg():
        with torch.no_grad():
            Cp = frg.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = frg.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = frg.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs))
        C = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = frg.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A_frg @ b
            with torch.no_grad():
                dh = frg.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C = frg.gru(dh, C.unsqueeze(0)).squeeze(0)
                fc[t] = frg.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["FullRank+GRU"] = evaluate("Full-rank resid + DMD + GRU", _r_frg, _f_frg)

    # ── ⑦ Full-rank resid + delay DMD + GRU ───────────────────────
    print(f"\n  --- Full-rank resid + delay DMD + GRU ---")
    B_frg_del = delay_embed(B_frg, d)
    A_frg_d = fit_dmd(B_frg_del)

    def _f_frg_del():
        with torch.no_grad():
            x_block = torch.tensor(np.vstack([train_n[-(d):], test_n[:1]]),dtype=torch.float32)
            C_block = frg.carrier(x_block)
            dC_block = C_block[1:] - C_block[:-1]
            B_block = frg.f(dC_block).numpy()
        z_aug = np.concatenate([B_block[d-1-i] for i in range(d)])
        B_roll = rollout_delay_dmd(A_frg_d, z_aug, j, d, fcst_len)
        fc = np.empty((fcst_len, n_obs))
        with torch.no_grad():
            C = frg.carrier(torch.tensor(test_n[:1],dtype=torch.float32)).squeeze(0)
            fc[0] = frg.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            with torch.no_grad():
                dh = frg.m(torch.tensor(B_roll[t],dtype=torch.float32).unsqueeze(0))
                C = frg.gru(dh, C.unsqueeze(0)).squeeze(0)
                fc[t] = frg.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["FullRank+delGRU"] = evaluate("Full-rank resid + delDMD + GRU", _r_frg, _f_frg_del)

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
    tr = obs[:N_TRAIN]; te = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="Lorenz-63", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=55, fcst_len=400,
                carrier_dim=16, h_base=96, h_tp=64)

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
    tr = obs[:N_TRAIN]; te = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="Rössler", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=obs.shape[1], gt_dim=3, lt_steps=290, fcst_len=500,
                carrier_dim=16, h_base=96, h_tp=64)

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
    tr = raw[:N_TRAIN]; te = raw[N_TRAIN:N_TRAIN+N_TEST]
    gt = raw[N_TRAIN:N_TRAIN+N_TEST]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return dict(name="Lorenz-96", train_n=(tr-mu)/sig, test_n=(te-mu)/sig,
                last_train=((tr-mu)/sig)[-1], gt_test=gt, mu=mu, sig=sig,
                n_obs=20, gt_dim=20, lt_steps=67, fcst_len=200,
                carrier_dim=18, h_base=96, h_tp=64)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

all_results = {}
t0 = time.time()

for gen_fn in [gen_lorenz63, gen_rossler, gen_lorenz96]:
    cfg = gen_fn()
    name = cfg.pop("name")
    print(f"\n{'#'*65}")
    print(f"  {name}  (carrier_dim={cfg['carrier_dim']})")
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
print(f"  Cross-system Summary (full-rank, k=j)")
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
    "RawResid+add": "#1f77b4", "RawResid+GRU": "#17becf",
    "FullRank+add": "#d62728", "FullRank+GRU": "#2ca02c",
    "FullRank+delGRU": "#ff7f0e",
}
markers = {
    "Std AE+DMD": "s", "Std AE+delDMD": "D",
    "RawResid+add": "v", "RawResid+GRU": "^",
    "FullRank+add": "o", "FullRank+GRU": "*",
    "FullRank+delGRU": "P",
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

fig.suptitle("Full-rank Residual (k=j) — No Compression", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "fullrank_residual.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'fullrank_residual.png'}")

with open(OUT / "fullrank_residual.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'fullrank_residual.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

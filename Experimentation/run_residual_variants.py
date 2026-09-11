"""
Residual architecture variants — head-to-head comparison
=========================================================
All share the same teacher AE and residual compressor f.
Differ in how the forecast latent is rolled out / accumulated:

  ① Residual + DMD + naked add   (baseline from run_residual_comparison)
  ② Residual + DMD + GRU gate    (gated accumulation)
  ③ Residual + MLP forecast head (nonlinear forecast, naked add)
  ④ Residual + MLP forecast + GRU gate (both fixes)

Plus baselines:
  ⑤ Standard AE + DMD
  ⑥ MLP Transition f(C_t,C_{t+1})→b, M(b,C_t)→C_{t+1}

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
    """Shared teacher AE + residual compressor/decoder.
    Subclasses add forecast + accumulation logic."""
    def __init__(self, n, j, k, h=64):
        super().__init__()
        self.j = j; self.k = k
        self.enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, j))
        self.dec = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        # f: ΔC → b
        self.f = nn.Sequential(
            nn.Linear(j, h), nn.ELU(), nn.Linear(h, k))
        # m: b → ΔC
        self.m = nn.Sequential(
            nn.Linear(k, h), nn.ELU(), nn.Linear(h, j))

    def recon(self, x): return self.dec(self.enc(x))
    def carrier(self, x): return self.enc(x)


class ResidualGRU(ResidualBase):
    """Residual + GRU-gated accumulation."""
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.gru = nn.GRUCell(j, j)   # input=delta, hidden=carrier


class ResidualMLPForecast(ResidualBase):
    """Residual + MLP forecast head (instead of DMD)."""
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.forecast_mlp = nn.Sequential(
            nn.Linear(k, h), nn.ELU(), nn.Linear(h, k))


class ResidualMLPGRU(ResidualBase):
    """Residual + MLP forecast + GRU gate. Both fixes combined."""
    def __init__(self, n, j, k, h=64):
        super().__init__(n, j, k, h)
        self.gru = nn.GRUCell(j, j)
        self.forecast_mlp = nn.Sequential(
            nn.Linear(k, h), nn.ELU(), nn.Linear(h, k))


class TransitionModel(nn.Module):
    """f(C_t, C_{t+1}) → b, M(b, C_t) → C_{t+1}."""
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

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out


# ═══════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════

def train_teacher(model, Xt_all, epochs=EPOCHS):
    """Phase 1: train enc + dec as autoencoder."""
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, epochs+1):
        model.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"        p1 ep {ep}")


def train_residual_ae(model, delta_C, epochs=EPOCHS):
    """Phase 2a: train f + m as residual autoencoder."""
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f, model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, epochs+1):
        model.train(); idx = torch.randperm(len(delta_C)); losses=[]
        for i in range(0, len(delta_C), BS):
            dc = delta_C[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        p2a ep {ep}  resid_ae={np.mean(losses):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  Run one system
# ═══════════════════════════════════════════════════════════════════

def run_system(name, train_n, test_n, last_train, gt_test, mu, sig,
               n_obs, gt_dim, lt_steps, fcst_len,
               k_dim, carrier_dim, h_base, h_tp,
               koopman_alpha=0.5):

    Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
    Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
    Xt_all = torch.tensor(train_n, dtype=torch.float32)
    j = carrier_dim; k = k_dim

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

    # ── ⑤ Standard AE + DMD ──────────────────────────────────────
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
        Z = rollout_np(z0, lambda z: A_s @ z, fcst_len)
        with torch.no_grad(): return ae.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    results["Std AE+DMD"] = evaluate("Std AE + DMD", _r_std, _f_std)

    # ════════════════════════════════════════════════════════════
    #  Helper: get carrier trajectory from a trained teacher
    # ════════════════════════════════════════════════════════════
    def get_carriers(model):
        model.eval()
        with torch.no_grad():
            C = model.carrier(Xt_all)
        return C, C[:-1], C[1:]

    # ── ① Residual + DMD + naked add ─────────────────────────────
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

    # ── ② Residual + DMD + GRU gate ──────────────────────────────
    print(f"\n  --- Resid + DMD + GRU ---")
    SeedAll(SEED)
    r2 = ResidualGRU(n_obs, j, k, h=h_tp)
    train_teacher(r2, Xt_all)
    C_all, C_cur, C_nxt = get_carriers(r2)
    dC = C_nxt - C_cur
    train_residual_ae(r2, dC)

    # Phase 2b: train GRU on carrier transitions using m(f(ΔC)) as input
    print("      Phase 2b: GRU gate")
    for p in r2.parameters(): p.requires_grad_(False)
    for p in r2.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(r2.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        r2.train(); idx = torch.randperm(len(C_cur)); losses=[]
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            with torch.no_grad():
                delta_hat = r2.m(r2.f(dC[sl]))
            c_next_hat = r2.gru(delta_hat, C_cur[sl])
            loss = nn.functional.mse_loss(c_next_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep}  gru={np.mean(losses):.6f}")

    r2.eval()
    with torch.no_grad(): B2 = r2.f(dC).numpy()
    A2 = fit_dmd(B2)

    def _r_r2(tn):
        with torch.no_grad(): return r2.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_r2():
        with torch.no_grad():
            Cp = r2.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = r2.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = r2.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs))
        C = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            b = A2 @ b
            with torch.no_grad():
                delta_hat = r2.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C = r2.gru(delta_hat, C.unsqueeze(0)).squeeze(0)
                fc[t] = r2.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+DMD+GRU"] = evaluate("Resid + DMD + GRU gate", _r_r2, _f_r2)

    # ── ③ Residual + MLP forecast + add ──────────────────────────
    print(f"\n  --- Resid + MLP fcst + add ---")
    SeedAll(SEED)
    r3 = ResidualMLPForecast(n_obs, j, k, h=h_tp)
    train_teacher(r3, Xt_all)
    C_all, C_cur, C_nxt = get_carriers(r3)
    dC = C_nxt - C_cur
    train_residual_ae(r3, dC)

    # Phase 2b: train MLP forecast head on b pairs
    print("      Phase 2b: MLP forecast head")
    r3.eval()
    with torch.no_grad():
        B_cur = r3.f(dC[:-1])   # b_t, t=0..N-3
        B_nxt = r3.f(dC[1:])    # b_{t+1}

    for p in r3.parameters(): p.requires_grad_(False)
    for p in r3.forecast_mlp.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(r3.forecast_mlp.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        r3.train(); idx = torch.randperm(len(B_cur)); losses=[]
        for i in range(0, len(B_cur), BS):
            sl = idx[i:i+BS]
            loss = nn.functional.mse_loss(r3.forecast_mlp(B_cur[sl]), B_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep}  mlp_fcst={np.mean(losses):.6f}")

    r3.eval()

    def _r_r3(tn):
        with torch.no_grad(): return r3.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_r3():
        with torch.no_grad():
            Cp = r3.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = r3.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = r3.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs)); C=Cc.numpy().ravel(); b=b0.copy()
        with torch.no_grad():
            fc[0] = r3.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            with torch.no_grad():
                bt = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
                b = r3.forecast_mlp(bt).numpy().ravel()
                dc = r3.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
            C = C + dc
            with torch.no_grad():
                fc[t] = r3.dec(torch.tensor(C,dtype=torch.float32).unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+MLP+add"] = evaluate("Resid + MLP fcst + add", _r_r3, _f_r3)

    # ── ④ Residual + MLP forecast + GRU gate ─────────────────────
    print(f"\n  --- Resid + MLP fcst + GRU ---")
    SeedAll(SEED)
    r4 = ResidualMLPGRU(n_obs, j, k, h=h_tp)
    train_teacher(r4, Xt_all)
    C_all, C_cur, C_nxt = get_carriers(r4)
    dC = C_nxt - C_cur
    train_residual_ae(r4, dC)

    # Phase 2b: train MLP forecast on b pairs
    print("      Phase 2b: MLP forecast head")
    r4.eval()
    with torch.no_grad():
        B_cur = r4.f(dC[:-1])
        B_nxt = r4.f(dC[1:])
    for p in r4.parameters(): p.requires_grad_(False)
    for p in r4.forecast_mlp.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(r4.forecast_mlp.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        r4.train(); idx = torch.randperm(len(B_cur))
        for i in range(0, len(B_cur), BS):
            sl = idx[i:i+BS]
            loss = nn.functional.mse_loss(r4.forecast_mlp(B_cur[sl]), B_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 2c: train GRU gate
    print("      Phase 2c: GRU gate")
    for p in r4.parameters(): p.requires_grad_(False)
    for p in r4.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(r4.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        r4.train(); idx = torch.randperm(len(C_cur)); losses=[]
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            with torch.no_grad():
                delta_hat = r4.m(r4.f(dC[sl]))
            c_next_hat = r4.gru(delta_hat, C_cur[sl])
            loss = nn.functional.mse_loss(c_next_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep}  gru={np.mean(losses):.6f}")

    r4.eval()

    def _r_r4(tn):
        with torch.no_grad(): return r4.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_r4():
        with torch.no_grad():
            Cp = r4.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc = r4.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = r4.f(Cc - Cp).numpy().ravel()
        fc = np.empty((fcst_len, n_obs))
        C = Cc.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = r4.dec(C.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_len):
            with torch.no_grad():
                bt = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
                b = r4.forecast_mlp(bt).numpy().ravel()
                delta_hat = r4.m(torch.tensor(b,dtype=torch.float32).unsqueeze(0))
                C = r4.gru(delta_hat, C.unsqueeze(0)).squeeze(0)
                fc[t] = r4.dec(C.unsqueeze(0)).numpy().ravel()
        return fc
    results["Resid+MLP+GRU"] = evaluate("Resid + MLP fcst + GRU gate", _r_r4, _f_r4)

    # ── ⑥ MLP Transition ─────────────────────────────────────────
    print(f"\n  --- MLP Transition ---")
    SeedAll(SEED)
    tm = TransitionModel(n_obs, j, k, h=h_tp)

    # Phase 1
    train_teacher(tm, Xt_all)
    C_all, C_cur, C_nxt = get_carriers(tm)

    # Phase 2: f + M jointly
    print("      Phase 2: f + M")
    for p in tm.parameters(): p.requires_grad_(False)
    for mod in [tm.f, tm.M]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(C_cur)); losses=[]
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            b = tm.f(torch.cat([C_cur[sl], C_nxt[sl]], dim=-1))
            c_hat = tm.M(torch.cat([b, C_cur[sl]], dim=-1))
            loss = nn.functional.mse_loss(c_hat, C_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep}  transition={np.mean(losses):.6f}")

    # Phase 3: fine-tune M + dec
    print("      Phase 3: M + dec")
    for p in tm.parameters(): p.requires_grad_(False)
    for mod in [tm.M, tm.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tm.train(); idx = torch.randperm(len(C_cur))
        for i in range(0, len(C_cur), BS):
            sl = idx[i:i+BS]
            b = tm.f(torch.cat([C_cur[sl], C_nxt[sl]], dim=-1))
            c_hat = tm.M(torch.cat([b, C_cur[sl]], dim=-1))
            loss = nn.functional.mse_loss(tm.dec(c_hat), Xt_nxt[sl])
            opt.zero_grad(); loss.backward(); opt.step()

    tm.eval()
    with torch.no_grad():
        B_tm = tm.f(torch.cat([C_cur, C_nxt], dim=-1)).numpy()
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
    results["MLP Transition"] = evaluate("MLP Transition + DMD", _r_tm, _f_tm)

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
        d = " DIV" if r["div"] else ""
        print(f"  {tag:<25s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d}")

# ═══════════════════════════════════════════════════════════════════
#  Summary + plots
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  Cross-system Summary")
print(f"{'='*70}")
for sname, results in all_results.items():
    print(f"\n  {sname}:")
    print(f"    {'Method':<25s} {'Recon':>8s} {'Fcst':>8s}")
    print(f"    {'-'*45}")
    for tag, r in results.items():
        d = " DIV" if r["div"] else ""
        print(f"    {tag:<25s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f}{d}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = {
    "Std AE+DMD": "#888888",
    "Resid+DMD+add": "#1f77b4",
    "Resid+DMD+GRU": "#2ca02c",
    "Resid+MLP+add": "#d62728",
    "Resid+MLP+GRU": "#9467bd",
    "MLP Transition": "#ff7f0e",
}
markers = {
    "Std AE+DMD": "s",
    "Resid+DMD+add": "o",
    "Resid+DMD+GRU": "D",
    "Resid+MLP+add": "^",
    "Resid+MLP+GRU": "*",
    "MLP Transition": "P",
}

for ax, (sname, results) in zip(axes, all_results.items()):
    for tag, r in results.items():
        if r["div"]: continue
        if r["rmse_f"] > 20: continue   # skip off-scale
        c = colors.get(tag, "#000")
        mk = markers.get(tag, "o")
        ms = 12 if tag in ("Resid+MLP+GRU","MLP Transition") else 9
        ax.plot(r["rmse_r"], r["rmse_f"], mk, color=c, ms=ms, zorder=4, label=tag)
    ax.set_xlabel("Reconstruction RMSE")
    ax.set_ylabel("Forecast RMSE (1 LT)")
    ax.set_title(sname)
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.25)

fig.suptitle("Residual Variants — Head to Head", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "residual_variants.png", dpi=150, bbox_inches="tight")
print(f"\n→ {OUT / 'residual_variants.png'}")

with open(OUT / "residual_variants.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"→ {OUT / 'residual_variants.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

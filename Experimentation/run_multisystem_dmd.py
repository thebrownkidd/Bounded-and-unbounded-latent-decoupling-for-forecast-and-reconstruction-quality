"""
Multi-system DMD comparison: Three-phase vs Koopman AE vs Standard AE
=====================================================================
Same experiment structure as run_lorenz63_comparison.py, replicated on:

  ① Rössler attractor   (3D, delay-embedded → 15D, k=4)
  ② Lorenz-96           (20D, no delay, k=6)

Each system runs:
  - Koopman AE (joint, sweep α)
  - Standard AE + post-hoc DMD
  - Three-phase + learned K (β=0.1)
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

# ═══════════════════════════════════════════════════════════════════
#  Shared infrastructure
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

class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,k))
        self.dec = nn.Sequential(nn.Linear(k,h), nn.ELU(),
                                 nn.Linear(h,h), nn.ELU(), nn.Linear(h,n))
        self.K = nn.Linear(k, k, bias=False)
    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def advance(self, z): return self.K(z)
    def forward(self, x): return self.decode(self.encode(x))

class ThreePhaseModel(nn.Module):
    def __init__(self, n, k, carrier_dim, h=64):
        super().__init__()
        self.teacher_enc = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        self.decoder = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        self.mapping = nn.Sequential(
            nn.Linear(k, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
    def teacher_recon(self, x):
        return self.decoder(self.teacher_enc(x))
    def encode(self, x):
        return self.compressor(x)
    def decode_latent(self, z):
        return self.decoder(self.mapping(z))
    def student_recon(self, x):
        return self.decode_latent(self.encode(x))

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out


def run_system(name, obs_train, obs_test, gt_test, dt,
               k_dim, carrier_dim, h_base, h_tp,
               lt_steps, fcst_len, epochs=400, lr=1e-3, bs=512, beta=0.1):
    """Run the full comparison on one system. Returns dict of results."""

    n_obs = obs_train.shape[1]
    mu, sig = obs_train.mean(0), obs_train.std(0) + 1e-8
    train_n = (obs_train - mu) / sig
    test_n  = (obs_test  - mu) / sig

    Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
    Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
    Xt_all = torch.tensor(train_n, dtype=torch.float32)

    # How many gt columns to compare (first state_dim columns of obs)
    gt_dim = gt_test.shape[1]

    def evaluate(tag, recon_fn, fcst_fn):
        rec = recon_fn(test_n)
        rec_phys = (rec * sig + mu)[:, :gt_dim]
        fc = fcst_fn(test_n)
        fc_phys = (fc * sig + mu)[:, :gt_dim]
        rmse_r = np.sqrt(np.mean((rec_phys - gt_test)**2))
        rmse_f = np.sqrt(np.mean((fc_phys[:lt_steps] - gt_test[:lt_steps])**2))
        mx = np.max(np.abs(fc_phys))
        div = mx > 500
        print(f"  {tag:45s} recon={rmse_r:.4f}  fcst={rmse_f:.4f}  max={mx:.0f}"
              f"{'  DIVERGED' if div else ''}")
        return dict(rmse_r=rmse_r, rmse_f=rmse_f, mx=mx, div=div,
                    rec_phys=rec_phys, fc_phys=fc_phys)

    results = {}
    alphas = [0.01, 0.1, 0.5, 1.0, 5.0]

    # ── ① Koopman AE sweep ───────────────────────────────────────
    koop = {}
    for alpha in alphas:
        print(f"\n  --- Koopman AE α={alpha} ---")
        torch.manual_seed(SEED)
        m = KoopmanAE(n_obs, k_dim, h_base)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for ep in range(1, epochs+1):
            m.train(); idx = torch.randperm(len(Xt_cur))
            for i in range(0, len(Xt_cur), bs):
                sl = idx[i:i+bs]; xc, xn = Xt_cur[sl], Xt_nxt[sl]
                z = m.encode(xc)
                Lr = nn.functional.mse_loss(m.decode(z), xc)
                Lf = nn.functional.mse_loss(m.decode(m.advance(z)), xn)
                loss = Lr + alpha * Lf
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 100 == 0: print(f"      ep {ep}")
        m.eval(); K_np = m.K.weight.detach().numpy()
        def _recon(tn, _m=m):
            with torch.no_grad(): return _m(torch.tensor(tn, dtype=torch.float32)).numpy()
        def _fcst(tn, _m=m, _K=K_np):
            with torch.no_grad():
                z0 = _m.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
            Z = rollout_np(z0, lambda z: _K @ z, fcst_len)
            with torch.no_grad(): return _m.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
        koop[alpha] = evaluate(f"Koopman α={alpha}", _recon, _fcst)
    results["koopman"] = koop

    # ── ② Standard AE + DMD ──────────────────────────────────────
    print(f"\n  --- Standard AE + DMD ---")
    torch.manual_seed(SEED)
    ae_s = AE(n_obs, k_dim, h_base)
    opt = torch.optim.Adam(ae_s.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        ae_s.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), bs):
            loss = nn.functional.mse_loss(ae_s(Xt_all[idx[i:i+bs]]), Xt_all[idx[i:i+bs]])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0: print(f"      ep {ep}")
    ae_s.eval()
    with torch.no_grad(): Z_s = ae_s.encode(Xt_all).numpy()
    A_s = fit_dmd(Z_s)
    def _recon_s(tn):
        with torch.no_grad(): return ae_s(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fcst_s(tn):
        with torch.no_grad():
            z0 = ae_s.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: A_s @ z, fcst_len)
        with torch.no_grad(): return ae_s.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    results["std_ae"] = evaluate("Standard AE + DMD", _recon_s, _fcst_s)

    # ── ③ Three-phase + DMD ──────────────────────────────────────
    print(f"\n  --- Three-phase + DMD (β={beta}) ---")
    SeedAll(SEED)
    tpm = ThreePhaseModel(n_obs, k_dim, carrier_dim, h=h_tp).to(DEVICE)
    K_learn = nn.Linear(k_dim, k_dim, bias=False).to(DEVICE)

    tp = sum(p.numel() for p in tpm.parameters()) + sum(p.numel() for p in K_learn.parameters())
    teacher_p = sum(p.numel() for p in tpm.teacher_enc.parameters())
    print(f"      Params: {tp:,} total, {tp-teacher_p:,} inference")

    # Phase 1
    print("      Phase 1: teacher + decoder")
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.teacher_enc, tpm.decoder]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=lr)
    for ep in range(1, epochs+1):
        tpm.train(); idx = torch.randperm(len(Xt_all)); losses = []
        for i in range(0, len(Xt_all), bs):
            x = Xt_all[idx[i:i+bs]]
            loss = nn.functional.mse_loss(tpm.teacher_recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  recon={np.mean(losses):.6f}")

    # Phase 2
    print(f"      Phase 2: compressor + mapping + K (β={beta})")
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.compressor, tpm.mapping]:
        for p in mod.parameters(): p.requires_grad_(True)
    for p in K_learn.parameters(): p.requires_grad_(True)
    all_p2 = [p for p in tpm.parameters() if p.requires_grad] + list(K_learn.parameters())
    opt = torch.optim.Adam(all_p2, lr=lr)
    with torch.no_grad():
        tc_cur = tpm.teacher_enc(Xt_cur)
    for ep in range(1, epochs+1):
        tpm.train(); K_learn.train()
        idx = torch.randperm(len(Xt_cur))
        for i in range(0, len(Xt_cur), bs):
            sl = idx[i:i+bs]
            z_c = tpm.compressor(Xt_cur[sl])
            z_n = tpm.compressor(Xt_nxt[sl])
            L_c = nn.functional.mse_loss(tpm.mapping(z_c), tc_cur[sl])
            L_d = nn.functional.mse_loss(K_learn(z_c), z_n)
            loss = L_c + beta * L_d
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            print(f"        ep {ep:3d}")
    K_learn.eval(); tpm.eval()
    K_np = K_learn.weight.detach().numpy()
    eig = np.sort(np.abs(np.linalg.eigvals(K_np)))[::-1]
    print(f"      K |λ|: {eig}")

    # Phase 3
    print("      Phase 3: fine-tune mapping + decoder")
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.mapping, tpm.decoder]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=lr*0.3)
    for ep in range(1, epochs+1):
        tpm.train(); idx = torch.randperm(len(Xt_all)); losses = []
        for i in range(0, len(Xt_all), bs):
            x = Xt_all[idx[i:i+bs]]
            loss = nn.functional.mse_loss(tpm.student_recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % 100 == 0:
            print(f"        ep {ep:3d}  recon={np.mean(losses):.6f}")
    tpm.eval()

    def _recon_tp(tn):
        with torch.no_grad():
            return tpm.student_recon(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fcst_tp(tn):
        with torch.no_grad():
            z0 = tpm.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: K_np @ z, fcst_len)
        with torch.no_grad():
            return tpm.decode_latent(torch.tensor(Z, dtype=torch.float32)).numpy()
    results["three_phase"] = evaluate("Three-phase + DMD (Ours)", _recon_tp, _fcst_tp)

    return results, alphas, gt_test, dt


# ═══════════════════════════════════════════════════════════════════
#  System 1: Rössler
# ═══════════════════════════════════════════════════════════════════
def make_rossler():
    print(f"\n{'#'*65}")
    print(f"  RÖSSLER ATTRACTOR")
    print(f"{'#'*65}")

    a, b, c = 0.2, 0.2, 5.7
    dt = 0.05
    def rossler(t, s):
        x, y, z = s
        return [-(y+z), x + a*y, b + z*(x - c)]

    sol = solve_ivp(rossler, [0, 800], [1, 1, 0],
                    t_eval=np.arange(0, 800, dt),
                    method="RK45", rtol=1e-10, atol=1e-10)
    raw = sol.y.T[4000:]  # drop 200 time-unit transient

    # Delay embed
    DELAYS = 5
    obs = np.concatenate([raw[i:len(raw)-DELAYS+1+i] for i in range(DELAYS)], axis=1)
    N_TRAIN, N_TEST = 3000, 500
    train_obs = obs[:N_TRAIN]
    test_obs  = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt_test   = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    print(f"  obs={obs.shape[1]}D  delay={DELAYS}  train={N_TRAIN}  test={N_TEST}")

    # λ_max ≈ 0.069 → 1 LT ≈ 14.5 s → 290 steps at dt=0.05
    LT = 290
    FCST_LEN = 500
    return run_system("Rössler", train_obs, test_obs, gt_test, dt,
                      k_dim=4, carrier_dim=16, h_base=96, h_tp=64,
                      lt_steps=LT, fcst_len=FCST_LEN)


# ═══════════════════════════════════════════════════════════════════
#  System 2: Lorenz-96
# ═══════════════════════════════════════════════════════════════════
def make_lorenz96():
    print(f"\n{'#'*65}")
    print(f"  LORENZ-96  (N=20, F=8)")
    print(f"{'#'*65}")

    N_L96, F_L96 = 20, 8.0
    dt = 0.01

    def l96(t, x):
        d = np.empty_like(x)
        for i in range(len(x)):
            d[i] = (x[(i+1) % N_L96] - x[(i-2) % N_L96]) * x[(i-1) % N_L96] - x[i] + F_L96
        return d

    x0 = F_L96 * np.ones(N_L96); x0[0] += 0.01
    sol = solve_ivp(l96, [0, 200], x0,
                    t_eval=np.arange(0, 200, dt),
                    method="RK45", rtol=1e-9, atol=1e-9)
    raw = sol.y.T[2000:]  # drop 20 time-unit transient

    # No delay embedding — 20D directly
    N_TRAIN, N_TEST = 4000, 500
    train_obs = raw[:N_TRAIN]
    test_obs  = raw[N_TRAIN:N_TRAIN+N_TEST]
    gt_test   = raw[N_TRAIN:N_TRAIN+N_TEST]  # all 20 dims are "ground truth"
    print(f"  obs={raw.shape[1]}D  no delay  train={N_TRAIN}  test={N_TEST}")

    # λ_max ≈ 1.5 → 1 LT ≈ 0.67 s → 67 steps at dt=0.01
    LT = 67
    FCST_LEN = 200
    return run_system("Lorenz-96", train_obs, test_obs, gt_test, dt,
                      k_dim=6, carrier_dim=20, h_base=96, h_tp=64,
                      lt_steps=LT, fcst_len=FCST_LEN)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
all_results = {}

t0 = time.time()
res_r, alphas_r, gt_r, dt_r = make_rossler()
t_ross = time.time() - t0
print(f"\n  Rössler finished in {t_ross:.0f}s")
all_results["Rössler"] = res_r

t1 = time.time()
res_l, alphas_l, gt_l, dt_l = make_lorenz96()
t_l96 = time.time() - t1
print(f"\n  Lorenz-96 finished in {t_l96:.0f}s")
all_results["Lorenz-96"] = res_l

# ═══════════════════════════════════════════════════════════════════
#  Summary + plots
# ═══════════════════════════════════════════════════════════════════
def print_table(name, res, alphas):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"{'Method':<35s} {'Recon':>8s} {'Fcst':>8s} {'Max|f|':>8s}")
    print("-" * 65)
    for a in alphas:
        r = res["koopman"][a]
        print(f"{'Koopman α='+str(a):<35s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f} {r['mx']:>8.0f}"
              f"{'  DIV' if r.get('div') else ''}")
    r = res["std_ae"]
    print(f"{'Standard AE + DMD':<35s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f} {r['mx']:>8.0f}")
    r = res["three_phase"]
    print(f"{'Three-phase + DMD (Ours)':<35s} {r['rmse_r']:>8.4f} {r['rmse_f']:>8.4f} {r['mx']:>8.0f}")
    print("-" * 65)

print_table("Rössler", res_r, alphas_r)
print_table("Lorenz-96", res_l, alphas_l)

def make_pareto(name, res, alphas, gt, dt, slug):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    kr = [res["koopman"][a]["rmse_r"] for a in alphas]
    kf = [res["koopman"][a]["rmse_f"] for a in alphas]
    ax.plot(kr, kf, "o-", color="#d62728", lw=1.5, ms=7, zorder=3,
            label="Koopman AE (joint, sweep α)")
    for a, xr, yf in zip(alphas, kr, kf):
        ax.annotate(f"α={a}", (xr, yf), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color="#d62728")
    r_s = res["std_ae"]
    ax.plot(r_s["rmse_r"], r_s["rmse_f"], "s", color="#1f77b4",
            ms=11, zorder=4, label="Standard AE + DMD")
    r_o = res["three_phase"]
    ax.plot(r_o["rmse_r"], r_o["rmse_f"], "*", color="#2ca02c",
            ms=16, zorder=4, label="Three-phase + DMD (Ours)")
    ax.set_xlabel("Reconstruction RMSE", fontsize=11)
    ax.set_ylabel("Forecast RMSE (1 Lyapunov time)", fontsize=11)
    ax.set_title(f"Reconstruction–Forecasting Tradeoff  ·  {name}", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_pareto.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUT / f'{slug}_pareto.png'}")
    plt.close(fig)

make_pareto("Rössler", res_r, alphas_r, gt_r, dt_r, "rossler")
make_pareto("Lorenz-96 (N=20)", res_l, alphas_l, gt_l, dt_l, "lorenz96")

# Save JSON
def jsonable(res, alphas):
    out = {}
    for a in alphas:
        r = res["koopman"][a]
        out[f"Koopman α={a}"] = {k: float(v) for k, v in r.items() if k in ("rmse_r","rmse_f","mx","div")}
    for key, label in [("std_ae", "Std AE + DMD"), ("three_phase", "Three-phase + DMD")]:
        r = res[key]
        out[label] = {k: float(v) for k, v in r.items() if k in ("rmse_r","rmse_f","mx","div")}
    return out

nums = {"Rössler": jsonable(res_r, alphas_r),
        "Lorenz-96": jsonable(res_l, alphas_l)}
with open(OUT / "multisystem_dmd.json", "w") as f:
    json.dump(nums, f, indent=2)
print(f"→ {OUT / 'multisystem_dmd.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

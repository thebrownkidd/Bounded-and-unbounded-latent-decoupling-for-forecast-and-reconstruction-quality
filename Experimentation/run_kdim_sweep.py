"""
Latent dimension sweep: find the best k for each system × method
================================================================
For each system, sweeps k_dim and runs all three methods.
Carrier dim scales as 2*k.  Two Koopman α values (0.1, 1.0) per k.

Systems:
  Rössler   — k ∈ {2, 3, 4, 6}        (attractor dim ≈ 2)
  Lorenz-96 — k ∈ {6, 10, 14, 18}     (KY dim ≈ 13 for N=20, F=8)
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
BETA = 0.1

# ═══════════════════════════════════════════════════════════════════
#  Model definitions
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
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, carrier_dim))
        self.decoder = nn.Sequential(
            nn.Linear(carrier_dim, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, n))
        self.compressor = nn.Sequential(
            nn.Linear(n, h), nn.ELU(), nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, k))
        self.mapping = nn.Sequential(
            nn.Linear(k, h), nn.ELU(), nn.Linear(h, carrier_dim))
    def teacher_recon(self, x): return self.decoder(self.teacher_enc(x))
    def encode(self, x): return self.compressor(x)
    def decode_latent(self, z): return self.decoder(self.mapping(z))
    def student_recon(self, x): return self.decode_latent(self.encode(x))

def fit_dmd(Z):
    X, Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

def rollout_np(z0, step, n):
    out = np.empty((n, len(z0))); out[0] = z0
    for t in range(1, n): out[t] = step(out[t-1])
    return out


# ═══════════════════════════════════════════════════════════════════
#  Core runner: one (system, k) point
# ═══════════════════════════════════════════════════════════════════

def run_one_k(train_n, test_n, gt_test, mu, sig, n_obs, gt_dim, lt_steps, fcst_len,
              k_dim, h_base, h_tp, koopman_alphas=(0.1, 1.0)):
    """Run all three methods at a single k.  Returns dict of results."""

    carrier_dim = 2 * k_dim
    Xt_cur = torch.tensor(train_n[:-1], dtype=torch.float32)
    Xt_nxt = torch.tensor(train_n[1:],  dtype=torch.float32)
    Xt_all = torch.tensor(train_n, dtype=torch.float32)

    def evaluate(tag, recon_fn, fcst_fn):
        rec = recon_fn(test_n)
        rec_p = (rec * sig + mu)[:, :gt_dim]   # denormalize → physical
        fc = fcst_fn(test_n)
        fc_p = (fc * sig + mu)[:, :gt_dim]
        rmse_r = np.sqrt(np.mean((rec_p - gt_test)**2))
        rmse_f = np.sqrt(np.mean((fc_p[:lt_steps] - gt_test[:lt_steps])**2))
        mx = np.max(np.abs(fc_p))
        div = mx > 500
        print(f"    {tag:42s} recon={rmse_r:.4f}  fcst={rmse_f:.4f}"
              f"{'  DIV' if div else ''}")
        return dict(rmse_r=float(rmse_r), rmse_f=float(rmse_f),
                    mx=float(mx), div=bool(div))

    out = {}

    # ── Koopman AE ────────────────────────────────────────────────
    for alpha in koopman_alphas:
        torch.manual_seed(SEED)
        m = KoopmanAE(n_obs, k_dim, h_base)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        for ep in range(1, EPOCHS+1):
            m.train(); idx = torch.randperm(len(Xt_cur))
            for i in range(0, len(Xt_cur), BS):
                sl = idx[i:i+BS]; xc, xn = Xt_cur[sl], Xt_nxt[sl]
                z = m.encode(xc)
                Lr = nn.functional.mse_loss(m.decode(z), xc)
                Lf = nn.functional.mse_loss(m.decode(m.advance(z)), xn)
                loss = Lr + alpha * Lf
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); K_np = m.K.weight.detach().numpy()
        def _r(tn, _m=m):
            with torch.no_grad(): return _m(torch.tensor(tn, dtype=torch.float32)).numpy()
        def _f(tn, _m=m, _K=K_np):
            with torch.no_grad():
                z0 = _m.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
            Z = rollout_np(z0, lambda z: _K @ z, fcst_len)
            with torch.no_grad(): return _m.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
        out[f"Koopman α={alpha}"] = evaluate(f"Koopman α={alpha}", _r, _f)

    # ── Standard AE + DMD ────────────────────────────────────────
    torch.manual_seed(SEED)
    ae = AE(n_obs, k_dim, h_base)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        ae.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            loss = nn.functional.mse_loss(ae(Xt_all[idx[i:i+BS]]), Xt_all[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad(): Z_s = ae.encode(Xt_all).numpy()
    A_s = fit_dmd(Z_s)
    def _rs(tn):
        with torch.no_grad(): return ae(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _fs(tn):
        with torch.no_grad():
            z0 = ae.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: A_s @ z, fcst_len)
        with torch.no_grad(): return ae.decode(torch.tensor(Z, dtype=torch.float32)).numpy()
    out["Std AE + DMD"] = evaluate("Std AE + DMD", _rs, _fs)

    # ── Three-phase + DMD ────────────────────────────────────────
    SeedAll(SEED)
    tpm = ThreePhaseModel(n_obs, k_dim, carrier_dim, h=h_tp)
    K_learn = nn.Linear(k_dim, k_dim, bias=False)

    # Phase 1
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.teacher_enc, tpm.decoder]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
        tpm.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(tpm.teacher_recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()

    # Phase 2
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.compressor, tpm.mapping]:
        for p in mod.parameters(): p.requires_grad_(True)
    for p in K_learn.parameters(): p.requires_grad_(True)
    all_p2 = [p for p in tpm.parameters() if p.requires_grad] + list(K_learn.parameters())
    opt = torch.optim.Adam(all_p2, lr=LR)
    with torch.no_grad(): tc_cur = tpm.teacher_enc(Xt_cur)
    for ep in range(1, EPOCHS+1):
        tpm.train(); K_learn.train(); idx = torch.randperm(len(Xt_cur))
        for i in range(0, len(Xt_cur), BS):
            sl = idx[i:i+BS]
            z_c = tpm.compressor(Xt_cur[sl])
            z_n = tpm.compressor(Xt_nxt[sl])
            L_c = nn.functional.mse_loss(tpm.mapping(z_c), tc_cur[sl])
            L_d = nn.functional.mse_loss(K_learn(z_c), z_n)
            loss = L_c + BETA * L_d
            opt.zero_grad(); loss.backward(); opt.step()
    K_learn.eval(); tpm.eval()
    K_np = K_learn.weight.detach().numpy()

    # Phase 3
    for p in tpm.parameters(): p.requires_grad_(False)
    for mod in [tpm.mapping, tpm.decoder]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in tpm.parameters() if p.requires_grad], lr=LR*0.3)
    for ep in range(1, EPOCHS+1):
        tpm.train(); idx = torch.randperm(len(Xt_all))
        for i in range(0, len(Xt_all), BS):
            x = Xt_all[idx[i:i+BS]]
            loss = nn.functional.mse_loss(tpm.student_recon(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
    tpm.eval()

    def _rt(tn):
        with torch.no_grad():
            return tpm.student_recon(torch.tensor(tn, dtype=torch.float32)).numpy()
    def _ft(tn):
        with torch.no_grad():
            z0 = tpm.encode(torch.tensor(tn[:1], dtype=torch.float32)).numpy().ravel()
        Z = rollout_np(z0, lambda z: K_np @ z, fcst_len)
        with torch.no_grad():
            return tpm.decode_latent(torch.tensor(Z, dtype=torch.float32)).numpy()
    out["Three-phase"] = evaluate("Three-phase + DMD", _rt, _ft)

    return out


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def gen_rossler():
    a_, b_, c_ = 0.2, 0.2, 5.7
    dt = 0.05
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
    train_obs = obs[:N_TRAIN]
    test_obs  = obs[N_TRAIN:N_TRAIN+N_TEST]
    gt_test   = raw[N_TRAIN:N_TRAIN+N_TEST, :3]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    return ((train_obs-mu)/sig, (test_obs-mu)/sig,
            gt_test, mu, sig, obs.shape[1], 3, 290, 500)  # lt=290, fcst=500

def gen_lorenz96():
    N_L96, F_L96 = 20, 8.0
    dt = 0.01
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
    train_obs = raw[:N_TRAIN]
    test_obs  = raw[N_TRAIN:N_TRAIN+N_TEST]
    gt_test   = raw[N_TRAIN:N_TRAIN+N_TEST]
    mu, sig = train_obs.mean(0), train_obs.std(0)+1e-8
    return ((train_obs-mu)/sig, (test_obs-mu)/sig,
            gt_test, mu, sig, 20, 20, 67, 200)  # lt=67, fcst=200


# ═══════════════════════════════════════════════════════════════════
#  Run sweeps
# ═══════════════════════════════════════════════════════════════════

all_results = {}
t0 = time.time()

# ── Rössler ───────────────────────────────────────────────────────
print(f"\n{'#'*65}")
print(f"  RÖSSLER — latent dimension sweep")
print(f"{'#'*65}")
ross_data = gen_rossler()
ross_ks = [2, 3, 4, 6]
ross_results = {}
for k in ross_ks:
    print(f"\n  ──── k = {k} ────")
    ross_results[k] = run_one_k(*ross_data, k_dim=k, h_base=96, h_tp=64)
all_results["Rössler"] = ross_results

# ── Lorenz-96 ─────────────────────────────────────────────────────
print(f"\n{'#'*65}")
print(f"  LORENZ-96 (N=20) — latent dimension sweep")
print(f"{'#'*65}")
l96_data = gen_lorenz96()
l96_ks = [6, 10, 14, 18]
l96_results = {}
for k in l96_ks:
    print(f"\n  ──── k = {k} ────")
    l96_results[k] = run_one_k(*l96_data, k_dim=k, h_base=96, h_tp=64)
all_results["Lorenz-96"] = l96_results


# ═══════════════════════════════════════════════════════════════════
#  Summary tables
# ═══════════════════════════════════════════════════════════════════

def print_sweep(name, results, ks):
    print(f"\n{'='*75}")
    print(f"  {name} — k sweep summary")
    print(f"{'='*75}")
    methods = ["Koopman α=0.1", "Koopman α=1.0", "Std AE + DMD", "Three-phase"]
    print(f"{'k':>3s}  ", end="")
    for m in methods:
        print(f"  {m:>20s}", end="")
    print()
    print(f"{'':>3s}  ", end="")
    for _ in methods:
        print(f"  {'recon / fcst':>20s}", end="")
    print()
    print("-" * 75)
    for k in ks:
        print(f"{k:>3d}  ", end="")
        for m in methods:
            r = results[k].get(m)
            if r:
                d = " D" if r["div"] else "  "
                print(f"  {r['rmse_r']:.3f}/{r['rmse_f']:.3f}{d}", end="")
            else:
                print(f"  {'—':>20s}", end="")
        print()
    print("-" * 75)

    # Best per method
    print(f"\n  Best k per method:")
    for m in methods:
        # Best = lowest sum of recon + fcst among non-diverged
        valid = [(k, results[k][m]) for k in ks
                 if m in results[k] and not results[k][m]["div"]]
        if valid:
            # Best recon
            br = min(valid, key=lambda x: x[1]["rmse_r"])
            # Best forecast
            bf = min(valid, key=lambda x: x[1]["rmse_f"])
            print(f"    {m:25s}  best recon: k={br[0]} ({br[1]['rmse_r']:.4f})"
                  f"   best fcst: k={bf[0]} ({bf[1]['rmse_f']:.4f})")

print_sweep("Rössler", ross_results, ross_ks)
print_sweep("Lorenz-96", l96_results, l96_ks)


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_sweep(name, results, ks, slug):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{name} — Latent Dimension Sweep", fontsize=13)

    colors = {"Koopman α=0.1": "#d62728", "Koopman α=1.0": "#ff7f0e",
              "Std AE + DMD": "#1f77b4", "Three-phase": "#2ca02c"}
    markers = {"Koopman α=0.1": "o", "Koopman α=1.0": "^",
               "Std AE + DMD": "s", "Three-phase": "*"}
    ms_map = {"Koopman α=0.1": 7, "Koopman α=1.0": 7,
              "Std AE + DMD": 8, "Three-phase": 12}

    for meth in colors:
        rr = [results[k][meth]["rmse_r"] for k in ks if not results[k][meth]["div"]]
        rf = [results[k][meth]["rmse_f"] for k in ks if not results[k][meth]["div"]]
        kk = [k for k in ks if not results[k][meth]["div"]]
        if not kk: continue
        ax1.plot(kk, rr, f"{markers[meth]}-", color=colors[meth], ms=ms_map[meth],
                 lw=1.5, label=meth)
        ax2.plot(kk, rf, f"{markers[meth]}-", color=colors[meth], ms=ms_map[meth],
                 lw=1.5, label=meth)

    ax1.set_xlabel("Latent dimension k"); ax1.set_ylabel("Reconstruction RMSE")
    ax1.set_title("Reconstruction vs k"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)
    ax2.set_xlabel("Latent dimension k"); ax2.set_ylabel("Forecast RMSE (1 LT)")
    ax2.set_title("Forecast vs k"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / f"{slug}_ksweep.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUT / f'{slug}_ksweep.png'}")
    plt.close(fig)

    # Pareto (recon vs fcst, all k pooled)
    fig2, ax = plt.subplots(figsize=(7.5, 5.5))
    for meth in colors:
        rr = [results[k][meth]["rmse_r"] for k in ks if not results[k][meth]["div"]]
        rf = [results[k][meth]["rmse_f"] for k in ks if not results[k][meth]["div"]]
        kk = [k for k in ks if not results[k][meth]["div"]]
        if not kk: continue
        ax.plot(rr, rf, f"{markers[meth]}-", color=colors[meth], ms=ms_map[meth],
                lw=1.2, label=meth, zorder=3)
        for ki, xr, yf in zip(kk, rr, rf):
            ax.annotate(f"k={ki}", (xr, yf), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=colors[meth])
    ax.set_xlabel("Reconstruction RMSE"); ax.set_ylabel("Forecast RMSE (1 LT)")
    ax.set_title(f"Recon–Forecast Tradeoff  ·  {name}  (k sweep)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(OUT / f"{slug}_ksweep_pareto.png", dpi=150, bbox_inches="tight")
    print(f"→ {OUT / f'{slug}_ksweep_pareto.png'}")
    plt.close(fig2)

plot_sweep("Rössler", ross_results, ross_ks, "rossler")
plot_sweep("Lorenz-96 (N=20)", l96_results, l96_ks, "lorenz96")

# Save JSON
with open(OUT / "kdim_sweep.json", "w") as f:
    json.dump({sys: {str(k): v for k, v in res.items()}
               for sys, res in all_results.items()}, f, indent=2)
print(f"→ {OUT / 'kdim_sweep.json'}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Done.")

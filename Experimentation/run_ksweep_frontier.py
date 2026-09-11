"""
k-sweep on standard AE+DMD to demonstrate the recon-forecast tradeoff.

For each system, train AE+DMD at multiple latent dims k.
Plot recon vs forecast RMSE → traces out the frontier.
Overlay our Resid+GRU point (j fixed, k = k_ae) → below the frontier.

Output: Paper/ksweep_frontier.json, Paper/ksweep_frontier.png
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
SEED = 0; EPOCHS = 400; LR = 1e-3; BS = 512


# ═══════════════════════════════════════════════════════════════════
#  Models (copied from run_big_sweep.py)
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

class ResidualGRU(nn.Module):
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


# ═══════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════

def train_ae(model, Xt):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            loss = nn.functional.mse_loss(model(Xt[idx[i:i+BS]]), Xt[idx[i:i+BS]])
            opt.zero_grad(); loss.backward(); opt.step()

def train_teacher(model, Xt):
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc, model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for ep in range(1, EPOCHS+1):
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
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(dC))
        for i in range(0, len(dC), BS):
            dc = dC[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model.m(model.f(dc)), dc)
            opt.zero_grad(); loss.backward(); opt.step()

def train_gru(model, dC, Cc, Cn):
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gru.parameters(): p.requires_grad_(True)
    opt = torch.optim.Adam(model.gru.parameters(), lr=LR)
    for ep in range(1, EPOCHS+1):
        model.train(); idx = torch.randperm(len(Cc))
        for i in range(0, len(Cc), BS):
            sl = idx[i:i+BS]
            with torch.no_grad(): dh = model.m(model.f(dC[sl]))
            loss = nn.functional.mse_loss(model.gru(dh, Cc[sl]), Cn[sl])
            opt.zero_grad(); loss.backward(); opt.step()


# ═══════════════════════════════════════════════════════════════════
#  Data generators
# ═══════════════════════════════════════════════════════════════════

def delay(raw, nd):
    return np.concatenate([raw[i:len(raw)-nd+1+i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr+n_te]
    gt = raw[n_tr:n_tr+n_te, :gt_dim]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, gt, mu, sig


def make_systems():
    systems = {}

    # Coupled Harmonic (6D, no delays)
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1,v1,x2,v2,x3,v3 = s
        return [v1, -kw*x1-k12*(x1-x2), v2, -k12*(x2-x1)-k23*(x2-x3),
                v3, -k23*(x3-x2)-kw*x3]
    sol = solve_ivp(osc, [0,500], [1,0,0,0.5,-0.5,0],
                    t_eval=np.arange(0,500,0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=6, gt_dim=6, fcst_steps=500, j=8, h=64,
        last_train=trn[-1],
        ks_sweep=[2, 3, 4, 5, 6, 8],
        k_ours=4)

    # Brusselator (2D + 5 delays = 10D)
    A_br, B_br = 1.0, 3.0
    sol = solve_ivp(lambda t,s: [A_br-(B_br+1)*s[0]+s[0]**2*s[1],
                                  B_br*s[0]-s[0]**2*s[1]],
                    [0,200], [1.0,1.0], t_eval=np.arange(0,200,0.01),
                    rtol=1e-10, atol=1e-10)
    raw = sol.y.T[2000:]
    obs = delay(raw, 5)
    trn, ten, gt, mu, sig = norm_split(obs, raw, 2, 3000, 500)
    systems["Brusselator"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=10, gt_dim=2, fcst_steps=500, j=8, h=64,
        last_train=trn[-1],
        ks_sweep=[2, 3, 4, 5, 6, 8],
        k_ours=3)

    # Linear 5D
    w1, w2, w3 = 1.0, np.sqrt(2), np.pi
    A5 = np.array([[0,w1,0,0,0],[-w1,0,0,0,0],
                    [0,0,0,w2,0],[0,0,-w2,0,0],
                    [0,0,0,0,-0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1,0,0.5,0.5,1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i-1]
    raw = raw5[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(
        train_n=trn, test_n=ten, gt_test=gt, mu=mu, sig=sig,
        n_obs=5, gt_dim=5, fcst_steps=500, j=8, h=64,
        last_train=trn[-1],
        ks_sweep=[2, 3, 4, 5, 6, 8],
        k_ours=4)

    return systems


# ═══════════════════════════════════════════════════════════════════
#  Evaluate one model
# ═══════════════════════════════════════════════════════════════════

def eval_metrics(rfn, ffn, test_n, gt_test, mu, sig, gt_dim, fcst_steps):
    rec = rfn(test_n)
    rp = (rec * sig + mu)[:, :gt_dim]
    fc = ffn()
    fp = (fc * sig + mu)[:, :gt_dim]
    N = min(fcst_steps, len(fp), len(gt_test))
    rr = float(np.sqrt(np.mean((rp[:len(gt_test)] - gt_test)**2)))
    rf = float(np.sqrt(np.mean((fp[:N] - gt_test[:N])**2)))
    return rr, rf


# ═══════════════════════════════════════════════════════════════════
#  Run sweep
# ═══════════════════════════════════════════════════════════════════

def run_sweep(sname, cfg):
    train_n, test_n = cfg["train_n"], cfg["test_n"]
    last_train = cfg["last_train"]
    gt_test, mu, sig = cfg["gt_test"], cfg["mu"], cfg["sig"]
    n_obs, gt_dim = cfg["n_obs"], cfg["gt_dim"]
    fcst_steps = cfg["fcst_steps"]
    j, h = cfg["j"], cfg["h"]
    ks_sweep = cfg["ks_sweep"]
    k_ours = cfg["k_ours"]

    Xt = torch.tensor(train_n, dtype=torch.float32)
    results = {"ae_sweep": {}, "ours": {}}

    # ── k-sweep on standard AE+DMD ──
    for k in ks_sweep:
        print(f"  AE+DMD k={k} ... ", end="", flush=True)
        torch.manual_seed(SEED)
        ae = AE(n_obs, k, h); train_ae(ae, Xt); ae.eval()
        with torch.no_grad(): Z = ae.encode(Xt).numpy()
        A = fit_dmd(Z)

        def _r(tn, m=ae):
            with torch.no_grad(): return m(torch.tensor(tn,dtype=torch.float32)).numpy()
        def _f(m=ae, A_=A, k_=k):
            with torch.no_grad():
                z0 = m.encode(torch.tensor(test_n[:1],dtype=torch.float32)).numpy().ravel()
            out = np.empty((fcst_steps, k_)); out[0]=z0
            for t in range(1, fcst_steps): out[t]=A_@out[t-1]
            with torch.no_grad(): return m.decode(torch.tensor(out,dtype=torch.float32)).numpy()

        rr, rf = eval_metrics(_r, _f, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
        results["ae_sweep"][str(k)] = {"rmse_r": rr, "rmse_f": rf}
        print(f"r={rr:.6f}  f={rf:.6f}")

    # ── Our Resid+GRU (fixed j, k = k_ours) ──
    print(f"  Resid+GRU j={j} k={k_ours} ... ", end="", flush=True)
    SeedAll(SEED)
    model = ResidualGRU(n_obs, j, k_ours, h=h)
    train_teacher(model, Xt); model.eval()
    with torch.no_grad(): C = model.carrier(Xt)
    Cc, Cn = C[:-1], C[1:]
    dC = Cn - Cc
    train_resid(model, dC)
    train_gru(model, dC, Cc, Cn); model.eval()
    with torch.no_grad(): B = model.f(dC).numpy()
    A_ours = fit_dmd(B)

    def _r_ours(tn):
        with torch.no_grad(): return model.recon(torch.tensor(tn,dtype=torch.float32)).numpy()
    def _f_ours():
        with torch.no_grad():
            Cp = model.carrier(torch.tensor(last_train[None],dtype=torch.float32))
            Cc_ = model.carrier(torch.tensor(test_n[:1],dtype=torch.float32))
            b0 = model.f(Cc_ - Cp).numpy().ravel()
        fc = np.empty((fcst_steps, n_obs))
        C_ = Cc_.squeeze(0); b = b0.copy()
        with torch.no_grad():
            fc[0] = model.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, fcst_steps):
            b = A_ours @ b
            with torch.no_grad():
                dh = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                C_ = model.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = model.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc

    rr, rf = eval_metrics(_r_ours, _f_ours, test_n, gt_test, mu, sig, gt_dim, fcst_steps)
    results["ours"] = {"rmse_r": rr, "rmse_f": rf, "j": j, "k": k_ours}
    print(f"r={rr:.6f}  f={rf:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    systems = make_systems()
    all_results = {}
    t0 = time.time()

    for sname, cfg in systems.items():
        print(f"\n{'='*60}")
        print(f"  {sname}  (obs={cfg['n_obs']}D)")
        print(f"{'='*60}")
        all_results[sname] = run_sweep(sname, cfg)

    # Save
    with open(OUT / "ksweep_frontier.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n→ {OUT / 'ksweep_frontier.json'}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (sname, res) in enumerate(all_results.items()):
        ax = axes[idx]
        sweep = res["ae_sweep"]
        ks = sorted(sweep.keys(), key=int)
        rs = [sweep[k]["rmse_r"] for k in ks]
        fs = [sweep[k]["rmse_f"] for k in ks]

        # AE+DMD frontier
        ax.plot(rs, fs, 'o-', color="#666", ms=8, lw=1.5, zorder=3,
                label="AE+DMD (k-sweep)")
        for i, k in enumerate(ks):
            ax.annotate(f"k={k}", (rs[i], fs[i]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color="#444")

        # Our method
        o = res["ours"]
        ax.plot(o["rmse_r"], o["rmse_f"], '*', color="#d62728", ms=16, zorder=5,
                markeredgecolor="k", markeredgewidth=0.5,
                label=f"Ours (j={o['j']}, k={o['k']})")

        ax.set_xlabel("Reconstruction RMSE", fontsize=10)
        ax.set_ylabel("Forecast RMSE", fontsize=10)
        ax.set_title(sname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("The Reconstruction–Forecasting Tradeoff", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "ksweep_frontier.png", dpi=200, bbox_inches="tight")
    print(f"→ {OUT / 'ksweep_frontier.png'}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("Done.")

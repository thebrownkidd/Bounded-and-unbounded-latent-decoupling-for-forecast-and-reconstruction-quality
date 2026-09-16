"""
C-MAPSS FD001 experiment using the carrier-residual architecture.
Replaces the old DecoupledModel-based run_cmapss.py for ICLR 2027.

C-MAPSS is multi-series: 60 training engines, each 128 timesteps × 17 sensors.
The carrier-residual protocol is adapted:
    Phase 1: Teacher AE on pooled (all engines flattened)
    Phase 2: Residual compressor on per-engine ΔC
    Phase 3: GRU on per-engine carrier sequences
    DMD: fit on pooled b trajectory

Evaluation on 30 holdout engines: recon RMSE + multi-horizon forecast RMSE.

Outputs: Paper/cmapss_resid.json
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from Utils.Benchmark import SeedAll

OUT = ROOT / "Paper"; OUT.mkdir(exist_ok=True)
EPOCHS = 400; LR = 1e-3; BS = 256
SEEDS = [0, 1, 2, 42, 123]

# C-MAPSS config
N_OBS = 17; J = 16; K = 6; H = 64
HORIZONS = [8, 16, 32, 64]


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

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
    for ep in range(1, EPOCHS + 1):
        model.train(); idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            loss = nn.functional.mse_loss(model(Xt[idx[i:i + BS]]), Xt[idx[i:i + BS]])
            opt.zero_grad(); loss.backward(); opt.step()


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


# ═══════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════

def load_cmapss():
    D = np.load(ROOT / "Data" / "CmapssFD001.npz", allow_pickle=True)
    train_series = D["train"]     # (60, 128, 17)
    holdout = D["holdout_obs"]    # (30, 128, 17)

    # Global normalisation from training data
    all_train = train_series.reshape(-1, N_OBS)
    mu = all_train.mean(0)
    sig = all_train.std(0) + 1e-8

    train_n = (all_train - mu) / sig
    holdout_n = (holdout.reshape(-1, N_OBS) - mu) / sig
    holdout_n = holdout_n.reshape(holdout.shape[0], holdout.shape[1], N_OBS)

    # Per-engine training data for sequential operations
    train_series_n = (train_series - mu) / sig

    return train_n, train_series_n, holdout_n, holdout, mu, sig


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def eval_cmapss(recon_fn, fcst_fn, holdout_n, horizons):
    """Evaluate on holdout engines.
    recon_fn: (T, 17) normalised -> (T, 17) reconstruction
    fcst_fn: (init_obs_n,) -> forecasts dict {horizon: (horizon, 17)}
    """
    # Reconstruction: across all holdout timesteps
    all_holdout = holdout_n.reshape(-1, N_OBS)
    pred_r = recon_fn(all_holdout)
    rmse_r = float(np.sqrt(np.mean((pred_r - all_holdout) ** 2)))

    # Forecast: for each holdout engine, use first half as warm-up,
    # forecast the second half
    rmse_f = {}
    for hz in horizons:
        errs = []
        for eng in range(holdout_n.shape[0]):
            eng_data = holdout_n[eng]  # (128, 17)
            mid = len(eng_data) // 2
            if mid + hz > len(eng_data):
                continue
            pred = fcst_fn(eng_data[:mid], hz)  # (hz, 17)
            gt = eng_data[mid:mid + hz]
            errs.append(np.mean((pred - gt) ** 2))
        rmse_f[hz] = float(np.sqrt(np.mean(errs))) if errs else float('nan')

    return rmse_r, rmse_f


# ═══════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════

def run_one(seed):
    train_n, train_series_n, holdout_n, holdout_raw, mu, sig = load_cmapss()
    Xt = torch.tensor(train_n, dtype=torch.float32)
    res = {}

    # ── AE+DMD ──
    SeedAll(seed)
    ae = AE(N_OBS, K, H)
    train_ae(ae, Xt); ae.eval()
    with torch.no_grad(): Z = ae.encode(Xt).numpy()
    A = fit_dmd(Z)

    def _r_ae(tn, m=ae):
        with torch.no_grad():
            return m(torch.tensor(tn, dtype=torch.float32)).numpy()

    def _f_ae(warm_n, hz, m=ae, A_=A):
        with torch.no_grad():
            z0 = m.encode(torch.tensor(warm_n[-1:], dtype=torch.float32)).numpy().ravel()
        out = np.empty((hz, K)); out[0] = z0
        for t in range(1, hz): out[t] = A_ @ out[t - 1]
        with torch.no_grad():
            return m.decode(torch.tensor(out, dtype=torch.float32)).numpy()

    rmse_r, rmse_f = eval_cmapss(_r_ae, _f_ae, holdout_n, HORIZONS)
    res["AE+DMD"] = {"rmse_r": rmse_r, "rmse_f": rmse_f}

    # ── Resid+GRU+DMD ──
    SeedAll(seed)
    rm = ResidualGRU(N_OBS, J, K, h=H)
    train_teacher(rm, Xt); rm.eval()

    # Compute per-engine carriers and residuals
    all_dC = []
    all_Cc = []
    all_Cn = []
    for eng in range(train_series_n.shape[0]):
        eng_t = torch.tensor(train_series_n[eng], dtype=torch.float32)
        with torch.no_grad():
            C = rm.carrier(eng_t)
        Cc, Cn = C[:-1], C[1:]
        dC = Cn - Cc
        all_dC.append(dC)
        all_Cc.append(Cc)
        all_Cn.append(Cn)

    dC_all = torch.cat(all_dC, dim=0)
    Cc_all = torch.cat(all_Cc, dim=0)
    Cn_all = torch.cat(all_Cn, dim=0)

    train_resid(rm, dC_all)
    train_gru(rm, dC_all, Cc_all, Cn_all); rm.eval()

    # DMD on pooled b
    with torch.no_grad(): B = rm.f(dC_all).numpy()
    A2 = fit_dmd(B)

    def _r_resid(tn, m=rm):
        with torch.no_grad():
            return m.recon(torch.tensor(tn, dtype=torch.float32)).numpy()

    def _f_resid(warm_n, hz, m=rm, A_=A2):
        warm_t = torch.tensor(warm_n, dtype=torch.float32)
        with torch.no_grad():
            C_warm = m.carrier(warm_t)
        # Last carrier and residual code
        C_last = C_warm[-1]
        if len(C_warm) > 1:
            dC_last = C_warm[-1] - C_warm[-2]
            with torch.no_grad():
                b0 = m.f(dC_last.unsqueeze(0)).numpy().ravel()
        else:
            b0 = np.zeros(K)

        fc = np.empty((hz, N_OBS))
        C_ = C_last; b = b0.copy()
        with torch.no_grad():
            fc[0] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        for t in range(1, hz):
            b = A_ @ b
            with torch.no_grad():
                dh = m.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
                C_ = m.gru(dh, C_.unsqueeze(0)).squeeze(0)
                fc[t] = m.dec(C_.unsqueeze(0)).numpy().ravel()
        return fc

    rmse_r, rmse_f = eval_cmapss(_r_resid, _f_resid, holdout_n, HORIZONS)
    res["Resid+GRU"] = {"rmse_r": rmse_r, "rmse_f": rmse_f}

    return res


if __name__ == "__main__":
    t0 = time.time()
    raw_results = {}

    for seed in SEEDS:
        print(f"\n  C-MAPSS FD001  seed={seed}")
        r = run_one(seed)
        for method, vals in r.items():
            raw_results.setdefault(method, []).append(vals)
            hz_str = "  ".join(f"h{h}={vals['rmse_f'][h]:.3f}" for h in HORIZONS)
            print(f"    {method:12s}  r={vals['rmse_r']:.4f}  {hz_str}")

    # Aggregate
    stats = {}
    for method in raw_results:
        rs = [x["rmse_r"] for x in raw_results[method]]
        stats[method] = {
            "rmse_r_mean": float(np.mean(rs)),
            "rmse_r_std": float(np.std(rs)),
        }
        for hz in HORIZONS:
            fs = [x["rmse_f"][hz] for x in raw_results[method]]
            stats[method][f"rmse_f{hz}_mean"] = float(np.mean(fs))
            stats[method][f"rmse_f{hz}_std"] = float(np.std(fs))

    with open(OUT / "cmapss_resid.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n→ {OUT / 'cmapss_resid.json'}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("\nDone.")

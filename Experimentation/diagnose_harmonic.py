"""
Diagnose Coupled Harmonic forecast degradation.
Where does the 20% forecast loss come from?
  (a) DMD in b-space vs z-space quality
  (b) f/m roundtrip loss
  (c) GRU accumulation error
  (d) Oracle test: forecast with TRUE b through GRU vs DMD b through GRU
"""
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.integrate import solve_ivp

from Utils.Benchmark import SeedAll

EPOCHS = 400; LR = 1e-3; BS = 512; SEED = 0

# ── Models ──

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
    A = Y @ np.linalg.pinv(X)
    return A

def dmd_fit_error(Z, A):
    """One-step prediction error of DMD on training data."""
    pred = (A @ Z[:-1].T).T
    actual = Z[1:]
    return float(np.sqrt(np.mean((pred - actual)**2)))

# ── Training (same as always) ──

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

# ── Data ──

def make_coupled_harmonic():
    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1,v1,x2,v2,x3,v3 = s
        return [v1, -kw*x1-k12*(x1-x2), v2, -k12*(x2-x1)-k23*(x2-x3),
                v3, -k23*(x3-x2)-kw*x3]
    sol = solve_ivp(osc, [0,500], [1,0,0,0.5,-0.5,0],
                    t_eval=np.arange(0,500,0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    n_tr, n_te = 4000, 500
    tr, te = raw[:n_tr], raw[n_tr:n_tr+n_te]
    mu, sig = tr.mean(0), tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig, te, mu, sig

# ── Main ──

if __name__ == "__main__":
    train_n, test_n, gt_test, mu, sig = make_coupled_harmonic()
    n_obs, k, j, h = 6, 4, 8, 64
    fcst_steps = 500
    Xt = torch.tensor(train_n, dtype=torch.float32)
    Xt_test = torch.tensor(test_n, dtype=torch.float32)
    last_train = train_n[-1]

    print("="*60)
    print("  PART 1: Train both models")
    print("="*60)

    # Train AE+DMD baseline
    print("  Training AE+DMD ... ", end="", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    ae = AE(n_obs, k, h); train_ae(ae, Xt); ae.eval()
    with torch.no_grad():
        Z_train = ae.encode(Xt).numpy()
        Z_test = ae.encode(Xt_test).numpy()
    A_ae = fit_dmd(Z_train)
    print("done")

    # Train Resid+GRU
    print("  Training Resid+GRU ... ", end="", flush=True)
    SeedAll(SEED)
    rg = ResidualGRU(n_obs, j, k, h=h)
    train_teacher(rg, Xt); rg.eval()
    with torch.no_grad():
        C_train = rg.carrier(Xt)
        C_test = rg.carrier(Xt_test)
    C_train_np = C_train.numpy()
    C_test_np = C_test.numpy()
    dC_train = C_train[1:] - C_train[:-1]
    dC_test = C_test[1:] - C_test[:-1]
    train_resid(rg, dC_train)
    rg.eval()
    with torch.no_grad():
        B_train = rg.f(dC_train).numpy()
        B_test = rg.f(dC_test).numpy()
    A_b = fit_dmd(B_train)

    # GRU training
    Cc, Cn = C_train[:-1], C_train[1:]
    train_gru(rg, dC_train, Cc, Cn); rg.eval()
    print("done")

    print("\n" + "="*60)
    print("  PART 2: DMD quality comparison")
    print("="*60)

    # DMD fit on z (AE latent)
    ae_train_err = dmd_fit_error(Z_train, A_ae)
    ae_test_err = dmd_fit_error(Z_test, A_ae)
    ae_eigs = np.abs(np.linalg.eigvals(A_ae))
    print(f"\n  AE z-space DMD (k={k}):")
    print(f"    Train 1-step RMSE: {ae_train_err:.6f}")
    print(f"    Test  1-step RMSE: {ae_test_err:.6f}")
    print(f"    Eigenvalue magnitudes: {np.sort(ae_eigs)[::-1]}")

    # DMD fit on b (residual codes)
    b_train_err = dmd_fit_error(B_train, A_b)
    b_test_err = dmd_fit_error(B_test, A_b)
    b_eigs = np.abs(np.linalg.eigvals(A_b))
    print(f"\n  Resid b-space DMD (k={k}):")
    print(f"    Train 1-step RMSE: {b_train_err:.6f}")
    print(f"    Test  1-step RMSE: {b_test_err:.6f}")
    print(f"    Eigenvalue magnitudes: {np.sort(b_eigs)[::-1]}")

    print("\n" + "="*60)
    print("  PART 3: f/m roundtrip quality")
    print("="*60)
    with torch.no_grad():
        dC_recon = rg.m(rg.f(dC_train))
        fm_train = float(nn.functional.mse_loss(dC_recon, dC_train).sqrt())
        dC_recon_te = rg.m(rg.f(dC_test))
        fm_test = float(nn.functional.mse_loss(dC_recon_te, dC_test).sqrt())
    dC_norm_train = float(dC_train.norm() / len(dC_train)**0.5)
    dC_norm_test = float(dC_test.norm() / len(dC_test)**0.5)
    print(f"  f→m roundtrip RMSE (train): {fm_train:.6f}  (signal norm: {dC_norm_train:.6f}, ratio: {fm_train/dC_norm_train:.4f})")
    print(f"  f→m roundtrip RMSE (test):  {fm_test:.6f}  (signal norm: {dC_norm_test:.6f}, ratio: {fm_test/dC_norm_test:.4f})")

    print("\n" + "="*60)
    print("  PART 4: GRU one-step quality")
    print("="*60)
    with torch.no_grad():
        # One-step GRU with true deltas
        dh_true = rg.m(rg.f(dC_test))
        C_pred = rg.gru(dh_true, C_test[:-1])
        gru_1step = float(nn.functional.mse_loss(C_pred, C_test[1:]).sqrt())
        # Compare to just adding
        C_add = C_test[:-1] + dC_test
        add_1step = float(nn.functional.mse_loss(C_add, C_test[1:]).sqrt())
        # Compare to adding m(f(delta))
        C_add_fm = C_test[:-1] + dC_recon_te
        add_fm_1step = float(nn.functional.mse_loss(C_add_fm, C_test[1:]).sqrt())
    print(f"  True add (C + ΔC) 1-step RMSE:   {add_1step:.8f}  (should be ~0)")
    print(f"  f/m add (C + m(f(ΔC))) 1-step:    {add_fm_1step:.6f}")
    print(f"  GRU(m(f(ΔC)), C) 1-step:          {gru_1step:.6f}")

    print("\n" + "="*60)
    print("  PART 5: Multi-step rollout forecasts")
    print("="*60)

    def forecast_rmse(fc_norm, gt, mu, sig):
        fp = fc_norm * sig + mu
        N = min(len(fp), len(gt))
        return float(np.sqrt(np.mean((fp[:N] - gt[:N])**2)))

    # (a) AE+DMD forecast (baseline)
    with torch.no_grad():
        z0 = ae.encode(Xt_test[:1]).numpy().ravel()
    fc_ae = np.empty((fcst_steps, k)); fc_ae[0] = z0
    for t in range(1, fcst_steps): fc_ae[t] = A_ae @ fc_ae[t-1]
    with torch.no_grad():
        fc_ae_obs = ae.decode(torch.tensor(fc_ae, dtype=torch.float32)).numpy()
    print(f"\n  (a) AE+DMD forecast:              {forecast_rmse(fc_ae_obs, gt_test, mu, sig):.4f}")

    # (b) Resid+GRU with DMD b (our method)
    with torch.no_grad():
        Cp = rg.carrier(torch.tensor(last_train[None], dtype=torch.float32))
        Cc0 = rg.carrier(Xt_test[:1])
        b0 = rg.f(Cc0 - Cp).numpy().ravel()
    fc_gru_dmd = np.empty((fcst_steps, n_obs))
    C_ = Cc0.squeeze(0); b = b0.copy()
    with torch.no_grad():
        fc_gru_dmd[0] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    for t in range(1, fcst_steps):
        b = A_b @ b
        with torch.no_grad():
            dh = rg.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
            C_ = rg.gru(dh, C_.unsqueeze(0)).squeeze(0)
            fc_gru_dmd[t] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    print(f"  (b) Resid+GRU (DMD b):            {forecast_rmse(fc_gru_dmd, gt_test, mu, sig):.4f}")

    # (c) Oracle: Resid+GRU with TRUE b from test data
    fc_gru_oracle = np.empty((fcst_steps, n_obs))
    C_ = Cc0.squeeze(0)
    with torch.no_grad():
        fc_gru_oracle[0] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    for t in range(1, min(fcst_steps, len(B_test))):
        with torch.no_grad():
            dh = rg.m(torch.tensor(B_test[t-1], dtype=torch.float32).unsqueeze(0))
            C_ = rg.gru(dh, C_.unsqueeze(0)).squeeze(0)
            fc_gru_oracle[t] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    # Fill rest if B_test shorter
    for t in range(len(B_test), fcst_steps):
        fc_gru_oracle[t] = fc_gru_oracle[t-1]
    N_oracle = min(fcst_steps, len(B_test))
    fp_oracle = fc_gru_oracle[:N_oracle] * sig + mu
    oracle_rmse = float(np.sqrt(np.mean((fp_oracle - gt_test[:N_oracle])**2)))
    print(f"  (c) Resid+GRU (TRUE b, oracle):   {oracle_rmse:.4f}  (over {N_oracle} steps)")

    # (d) Oracle: true carrier differences through GRU (no f/m at all)
    fc_gru_true_dc = np.empty((fcst_steps, n_obs))
    C_ = Cc0.squeeze(0)
    with torch.no_grad():
        fc_gru_true_dc[0] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    for t in range(1, min(fcst_steps, len(dC_test))):
        with torch.no_grad():
            C_ = rg.gru(dC_test[t-1:t], C_.unsqueeze(0)).squeeze(0)
            fc_gru_true_dc[t] = rg.dec(C_.unsqueeze(0)).numpy().ravel()
    N_dc = min(fcst_steps, len(dC_test))
    fp_dc = fc_gru_true_dc[:N_dc] * sig + mu
    dc_rmse = float(np.sqrt(np.mean((fp_dc - gt_test[:N_dc])**2)))
    print(f"  (d) GRU with TRUE ΔC (no f/m):    {dc_rmse:.4f}  (over {N_dc} steps)")

    # (e) Multi-step DMD divergence: compare b_pred vs b_true over time
    print(f"\n  (e) DMD rollout divergence in b-space:")
    b_rolled = np.empty((min(fcst_steps, len(B_test)), k))
    b_rolled[0] = B_test[0]
    for t in range(1, len(b_rolled)):
        b_rolled[t] = A_b @ b_rolled[t-1]
    for horizon in [10, 50, 100, 200, 499]:
        if horizon < len(b_rolled):
            err = np.sqrt(np.mean((b_rolled[:horizon] - B_test[:horizon])**2))
            print(f"    t=1..{horizon:3d}: RMSE(DMD_b vs true_b) = {err:.6f}")

    # Same for AE z-space
    print(f"\n  (f) DMD rollout divergence in z-space:")
    z_rolled = np.empty((min(fcst_steps, len(Z_test)), k))
    z_rolled[0] = Z_test[0]
    for t in range(1, len(z_rolled)):
        z_rolled[t] = A_ae @ z_rolled[t-1]
    for horizon in [10, 50, 100, 200, 499]:
        if horizon < len(z_rolled):
            err = np.sqrt(np.mean((z_rolled[:horizon] - Z_test[:horizon])**2))
            print(f"    t=1..{horizon:3d}: RMSE(DMD_z vs true_z) = {err:.6f}")

    print("\nDone.")

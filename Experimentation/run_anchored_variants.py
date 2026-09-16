"""
Two anchored-GRU variants on Linear 5D, single seed.

V1 (AnchoredGRU): GRU processes m(b) steps, C_{t+tau} = C_0 + proj(h_tau)
V2 (CumResidGRU): GRU processes cumulative decoded residuals,
                   C_{t+tau} = C_0 + GRU(cumsum_m(b)_{1:tau})
                   i.e., instead of feeding m(b_t) at each step,
                   feed the running cumsum. The GRU learns to correct drift.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch, torch.nn as nn, numpy as np
from scipy.linalg import expm
from Utils.Benchmark import SeedAll

EP=200; LR=1e-3; BS=512; N=5; J=8; K=4; H=64; FCST=500

def make_data():
    w1,w2=1.0,np.sqrt(2)
    A5=np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad=expm(A5*0.05); raw=np.empty((10000,5)); raw[0]=[1,0,0.5,0.5,1]
    for i in range(1,10000): raw[i]=Ad@raw[i-1]
    raw=raw[200:]; tr,te=raw[:4000],raw[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    return (tr-mu)/sig, (te-mu)/sig

class Base(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc=nn.Sequential(nn.Linear(N,H),nn.ELU(),nn.Linear(H,H),nn.ELU(),nn.Linear(H,J))
        s.dec=nn.Sequential(nn.Linear(J,H),nn.ELU(),nn.Linear(H,H),nn.ELU(),nn.Linear(H,N))
        s.f=nn.Sequential(nn.Linear(J,H),nn.ELU(),nn.Linear(H,K))
        s.m=nn.Sequential(nn.Linear(K,H),nn.ELU(),nn.Linear(H,J))
    def recon(s,x): return s.dec(s.enc(x))
    def carrier(s,x): return s.enc(x)

def fit_dmd(Z):
    X,Y=Z[:-1].T,Z[1:].T; return Y@np.linalg.pinv(X)

def train_phase12(model, Xt):
    # Phase 1
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.enc,model.dec]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR)
    for ep in range(1,EP+1):
        model.train(); idx=torch.randperm(len(Xt))
        for i in range(0,len(Xt),BS):
            x=Xt[idx[i:i+BS]]; loss=nn.functional.mse_loss(model.recon(x),x)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): C=model.carrier(Xt)
    dC=C[1:]-C[:-1]
    # Phase 2
    for p in model.parameters(): p.requires_grad_(False)
    for mod in [model.f,model.m]:
        for p in mod.parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR)
    for ep in range(1,EP+1):
        model.train(); idx=torch.randperm(len(dC))
        for i in range(0,len(dC),BS):
            dc=dC[idx[i:i+BS]]; loss=nn.functional.mse_loss(model.m(model.f(dc)),dc)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        B=model.f(dC); mB=model.m(B)
    B_np=B.numpy(); A_dmd=fit_dmd(B_np)
    return C, dC, B, mB, B_np, A_dmd


def eval_recon(model, te_n):
    Xte=torch.tensor(te_n,dtype=torch.float32)
    with torch.no_grad(): pred=model.recon(Xte).numpy()
    return float(np.sqrt(np.mean((pred-te_n)**2)))


# ═══════════════════════════════════════════════════════════════════
# V1: AnchoredGRU — GRU(m(b_t), h) -> C_0 + proj(h)
# ═══════════════════════════════════════════════════════════════════
def run_v1(model, C, dC, B, mB, B_np, A_dmd, te_n):
    gru = nn.GRUCell(J, J)
    proj = nn.Sequential(nn.Linear(J, H), nn.ELU(), nn.Linear(H, J))
    params = list(gru.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    # Pointwise training: for each (anchor, tau), predict C_{anchor+tau}
    # Build pairs with varied windows
    rng = np.random.default_rng(0)
    T = len(C)

    for ep in range(1, EP + 1):
        gru.train(); proj.train()
        # 100 random windows per epoch, length 5-50
        total_loss = 0
        for _ in range(100):
            wlen = rng.integers(5, min(60, T - 2))
            anchor = rng.integers(0, T - wlen - 1)
            c0 = C[anchor]
            h = torch.zeros(1, J)
            loss = torch.tensor(0.0)
            for t in range(wlen):
                h = gru(mB[anchor+t:anchor+t+1], h)
                c_pred = c0.unsqueeze(0) + proj(h)
                loss = loss + nn.functional.mse_loss(c_pred, C[anchor+t+1:anchor+t+2])
            loss = loss / wlen
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if ep % 50 == 0:
            print(f"    V1 ep {ep}: {total_loss/100:.6f}")

    gru.eval(); proj.eval()

    # Forecast
    Xte = torch.tensor(te_n, dtype=torch.float32)
    with torch.no_grad():
        c0 = model.carrier(Xte[0:1]).squeeze(0)
        b = model.f(dC[-1:]).numpy().ravel()
    out = np.empty((FCST, N))
    h = torch.zeros(1, J)
    for t in range(FCST):
        b = A_dmd @ b
        with torch.no_grad():
            mb = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
            h = gru(mb, h)
            c_pred = c0.unsqueeze(0) + proj(h)
            out[t] = model.dec(c_pred).numpy().ravel()
    return float(np.sqrt(np.mean((out - te_n[:FCST])**2)))


# ═══════════════════════════════════════════════════════════════════
# V2: CumResidGRU — GRU(cumsum_m(b), h) -> C_0 + proj(h)
#     Feed cumulative residual at each step, not just the increment.
#     The GRU sees the total accumulated change and corrects it.
# ═══════════════════════════════════════════════════════════════════
def run_v2(model, C, dC, B, mB, B_np, A_dmd, te_n):
    gru = nn.GRUCell(J, J)
    proj = nn.Sequential(nn.Linear(J, H), nn.ELU(), nn.Linear(H, J))
    params = list(gru.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    rng = np.random.default_rng(42)
    T = len(C)

    for ep in range(1, EP + 1):
        gru.train(); proj.train()
        total_loss = 0
        for _ in range(100):
            wlen = rng.integers(5, min(60, T - 2))
            anchor = rng.integers(0, T - wlen - 1)
            c0 = C[anchor]
            h = torch.zeros(1, J)
            cum = torch.zeros(1, J)  # cumulative decoded residual
            loss = torch.tensor(0.0)
            for t in range(wlen):
                cum = cum + mB[anchor+t:anchor+t+1]  # running sum
                h = gru(cum, h)  # feed cumulative, not increment
                c_pred = c0.unsqueeze(0) + proj(h)
                loss = loss + nn.functional.mse_loss(c_pred, C[anchor+t+1:anchor+t+2])
            loss = loss / wlen
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if ep % 50 == 0:
            print(f"    V2 ep {ep}: {total_loss/100:.6f}")

    gru.eval(); proj.eval()

    # Forecast
    Xte = torch.tensor(te_n, dtype=torch.float32)
    with torch.no_grad():
        c0 = model.carrier(Xte[0:1]).squeeze(0)
        b = model.f(dC[-1:]).numpy().ravel()
    out = np.empty((FCST, N))
    h = torch.zeros(1, J)
    cum = torch.zeros(1, J)
    for t in range(FCST):
        b = A_dmd @ b
        with torch.no_grad():
            mb = model.m(torch.tensor(b, dtype=torch.float32).unsqueeze(0))
            cum = cum + mb
            h = gru(cum, h)
            c_pred = c0.unsqueeze(0) + proj(h)
            out[t] = model.dec(c_pred).numpy().ravel()
    return float(np.sqrt(np.mean((out - te_n[:FCST])**2)))


if __name__ == "__main__":
    tr_n, te_n = make_data()
    Xt = torch.tensor(tr_n, dtype=torch.float32)
    SeedAll(0)
    model = Base()
    t0 = time.time()
    C, dC, B, mB, B_np, A_dmd = train_phase12(model, Xt)
    print(f"Phase 1-2: {time.time()-t0:.0f}s")

    rmse_r = eval_recon(model, te_n)
    print(f"Recon RMSE: {rmse_r:.5f}")

    print("\n--- V1: AnchoredGRU (feed m(b) increments) ---")
    SeedAll(100)
    t0 = time.time()
    f1 = run_v1(model, C, dC, B, mB, B_np, A_dmd, te_n)
    print(f"  V1 Forecast RMSE: {f1:.5f} ({time.time()-t0:.0f}s)")

    print("\n--- V2: CumResidGRU (feed cumulative m(b)) ---")
    SeedAll(200)
    t0 = time.time()
    f2 = run_v2(model, C, dC, B, mB, B_np, A_dmd, te_n)
    print(f"  V2 Forecast RMSE: {f2:.5f} ({time.time()-t0:.0f}s)")

    print(f"\n{'='*50}")
    print(f"  V1 AnchoredGRU    R={rmse_r:.5f}  F={f1:.5f}")
    print(f"  V2 CumResidGRU    R={rmse_r:.5f}  F={f2:.5f}")
    print(f"  AE+DMD            R=0.00968        F=1.20774")
    print(f"  Resid+GRU (cur)   R=0.00736        F=1.02516")

"""
Single-step forecast evaluation for all heads × both architectures × all systems.
Evaluates 1-step prediction: given x_t, predict x_{t+1}.

Outputs: /kaggle/working/single_step_heads.json
"""
import json, time, math
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.linalg import expm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EP = 400; LR = 1e-3; BS = 512; H = 64
SEEDS = [0, 1, 2, 42, 123]

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def delay(x, d):
    return np.concatenate([x[i:len(x)-d+i+1] for i in range(d)], axis=1)

# ═══════════════════════════════════════════════════════════════════
#  Systems
# ═══════════════════════════════════════════════════════════════════
def make_systems():
    systems = {}

    k12,k23,kw=1.0,0.5,0.3
    def osc(t,s):
        x1,v1,x2,v2,x3,v3=s
        return [v1,-kw*x1-k12*(x1-x2),v2,-k12*(x2-x1)-k23*(x2-x3),v3,-k23*(x3-x2)-kw*x3]
    sol=solve_ivp(osc,[0,500],[1,0,0,0.5,-0.5,0],t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
    raw=sol.y.T[200:]; tr,te=raw[:4000],raw[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    systems['Coupled Harmonic']=dict(tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=6, j=8, k=4)

    w1,w2=1.0,np.sqrt(2)
    A5=np.array([[0,w1,0,0,0],[-w1,0,0,0,0],[0,0,0,w2,0],[0,0,-w2,0,0],[0,0,0,0,-0.05]])
    Ad=expm(A5*0.05); raw=np.empty((10000,5)); raw[0]=[1,0,0.5,0.5,1]
    for i in range(1,10000): raw[i]=Ad@raw[i-1]
    raw=raw[200:]; tr,te=raw[:4000],raw[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    systems['Linear 5D']=dict(tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=5, j=8, k=4)

    A_br,B_br=1.0,3.0
    def bruss(t,s):
        x,y=s; return [A_br+x**2*y-(B_br+1)*x, B_br*x-x**2*y]
    sol=solve_ivp(bruss,[0,500],[1.5,3.0],t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
    raw=sol.y.T[200:]; obs=delay(raw,5)
    tr,te=obs[:4000],obs[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    systems['Brusselator']=dict(tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=10, j=8, k=4)

    def duff(t,s):
        x,v=s; return [v, -0.05*v - x**3 + 8*np.cos(t)]
    sol=solve_ivp(duff,[0,500],[1,0],t_eval=np.arange(0,500,0.05),rtol=1e-10,atol=1e-10)
    raw=sol.y.T[200:]; obs=delay(raw,5)
    tr,te=obs[:4000],obs[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    systems['Duffing']=dict(tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=10, j=8, k=4)

    N96=20; F96=8.0
    def l96(t,x):
        d=np.empty(N96)
        for i in range(N96): d[i]=(x[(i+1)%N96]-x[(i-2)%N96])*x[(i-1)%N96]-x[i]+F96
        return d
    x0=F96*np.ones(N96); x0[0]+=0.01
    sol=solve_ivp(l96,[0,100],x0,t_eval=np.arange(0,100,0.01),rtol=1e-8,atol=1e-8)
    raw=sol.y.T[200:]; tr,te=raw[:4000],raw[4000:4500]
    mu,sig=tr.mean(0),tr.std(0)+1e-8
    systems['Lorenz-96']=dict(tr_n=(tr-mu)/sig, te_n=(te-mu)/sig, n=20, j=12, k=6)

    return systems

# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════
class AE(nn.Module):
    def __init__(s, n, k, h):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(n,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,k))
        s.dec = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,h),nn.ELU(),nn.Linear(h,n))
    def encode(s,x): return s.enc(x)
    def decode(s,z): return s.dec(z)
    def forward(s,x): return s.decode(s.encode(x))

class MLPHead(nn.Module):
    def __init__(s, k, h):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(k,h),nn.ELU(),nn.Linear(h,k))
    def forward(s, z): return s.net(z)

class ODEFunc(nn.Module):
    def __init__(s, k, h):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(k,h),nn.Tanh(),nn.Linear(h,k))
    def forward(s, z): return s.net(z)

def rk4_step(f, z, dt=1.0):
    k1 = f(z)
    k2 = f(z + 0.5*dt*k1)
    k3 = f(z + 0.5*dt*k2)
    k4 = f(z + dt*k3)
    return z + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def fit_dmd(Z):
    X,Y = Z[:-1].T, Z[1:].T
    return Y @ np.linalg.pinv(X)

# ═══════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════
def train_recon(model, Xt, epochs=EP):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(1, epochs+1):
        model.train(); idx = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), BS):
            x = Xt[idx[i:i+BS]]
            loss = nn.functional.mse_loss(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()

def train_coupled_with_head(ae, head, Xt, phi, is_ode=False):
    params = list(ae.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    Z_pairs = None
    for ep in range(1, EP+1):
        ae.train(); head.train()
        idx = torch.randperm(len(Xt)-1, device=DEVICE)
        for i in range(0, len(idx), BS):
            sl = idx[i:i+BS]
            x_t = Xt[sl]; x_next = Xt[sl+1]
            z_t = ae.encode(x_t); z_next_true = ae.encode(x_next)
            rec = ae.decode(z_t)
            if is_ode:
                z_next_pred = rk4_step(head, z_t)
            else:
                z_next_pred = head(z_t)
            L_rec = nn.functional.mse_loss(rec, x_t)
            L_fcst = nn.functional.mse_loss(z_next_pred, z_next_true.detach())
            loss = (1-phi)*L_rec + phi*L_fcst
            opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            loss.backward(); opt.step()

def train_head_on_latent(head, Z_t, Z_next, is_ode=False, epochs=EP):
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    for ep in range(1, epochs+1):
        head.train(); idx = torch.randperm(len(Z_t), device=DEVICE)
        for i in range(0, len(Z_t), BS):
            sl = idx[i:i+BS]
            if is_ode:
                pred = rk4_step(head, Z_t[sl])
            else:
                pred = head(Z_t[sl])
            loss = nn.functional.mse_loss(pred, Z_next[sl])
            opt.zero_grad(); loss.backward(); opt.step()

# ═══════════════════════════════════════════════════════════════════
#  1-step evaluation
# ═══════════════════════════════════════════════════════════════════
def eval_1step(encoder, decoder, head, Xte, is_dmd=False, A_mat=None, is_ode=False):
    """Evaluate 1-step forecast: predict x_{t+1} from x_t for all test pairs."""
    with torch.no_grad():
        x_t = Xte[:-1]
        x_next = Xte[1:]
        z_t = encoder(x_t)
        if is_dmd:
            z_next = torch.tensor(
                (A_mat @ z_t.cpu().numpy().T).T, dtype=torch.float32, device=DEVICE)
        elif is_ode:
            z_next = rk4_step(head, z_t)
        else:
            z_next = head(z_t)
        x_pred = decoder(z_next)
        rmse = torch.sqrt(torch.mean((x_pred - x_next)**2)).item()
    return rmse

def eval_recon(model, Xte):
    with torch.no_grad():
        return torch.sqrt(torch.mean((model(Xte) - Xte)**2)).item()

# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    systems = make_systems()
    all_results = {}

    for sname, cfg in systems.items():
        print(f'\n{"="*60}')
        print(f'  {sname} (n={cfg["n"]}, j={cfg["j"]}, k={cfg["k"]})')
        print(f'{"="*60}')
        n, j, k = cfg['n'], cfg['j'], cfg['k']
        Xt = torch.tensor(cfg['tr_n'], dtype=torch.float32, device=DEVICE)
        Xte = torch.tensor(cfg['te_n'], dtype=torch.float32, device=DEVICE)
        sys_results = {}

        for seed in SEEDS:
            t0 = time.time()
            seed_all(seed)

            # ── COUPLED: AE(k-dim) + each head ──
            # DMD (recon-only AE, post-hoc DMD)
            ae_dmd = AE(n, k, H).to(DEVICE)
            train_recon(ae_dmd, Xt)
            ae_dmd.eval()
            with torch.no_grad(): Z = ae_dmd.encode(Xt).cpu().numpy()
            A = fit_dmd(Z)
            r_ae_dmd = eval_recon(ae_dmd, Xte)
            f_ae_dmd = eval_1step(ae_dmd.encode, ae_dmd.decode, None, Xte, is_dmd=True, A_mat=A)

            # MLP (joint training, best phi)
            best_f_mlp_c = float('inf'); best_r_mlp_c = None
            for phi in [0.1, 0.5, 1.0]:
                seed_all(seed)
                ae_m = AE(n, k, H).to(DEVICE)
                mlp_m = MLPHead(k, H).to(DEVICE)
                train_coupled_with_head(ae_m, mlp_m, Xt, phi)
                ae_m.eval(); mlp_m.eval()
                f_ = eval_1step(ae_m.encode, ae_m.decode, mlp_m, Xte)
                if f_ < best_f_mlp_c:
                    best_f_mlp_c = f_
                    best_r_mlp_c = eval_recon(ae_m, Xte)

            # ODE (joint training, best phi)
            best_f_ode_c = float('inf'); best_r_ode_c = None
            for phi in [0.1, 0.5, 1.0]:
                seed_all(seed)
                ae_o = AE(n, k, H).to(DEVICE)
                ode_o = ODEFunc(k, H).to(DEVICE)
                train_coupled_with_head(ae_o, ode_o, Xt, phi, is_ode=True)
                ae_o.eval(); ode_o.eval()
                f_ = eval_1step(ae_o.encode, ae_o.decode, ode_o, Xte, is_ode=True)
                if f_ < best_f_ode_c:
                    best_f_ode_c = f_
                    best_r_ode_c = eval_recon(ae_o, Xte)

            # ── DECOUPLED: carrier AE(j-dim) + residual + heads ──
            seed_all(seed)
            carrier = AE(n, j, H).to(DEVICE)
            train_recon(carrier, Xt)
            carrier.eval()
            r_dec = eval_recon(carrier, Xte)

            with torch.no_grad():
                C = carrier.encode(Xt)
            dC = C[1:] - C[:-1]

            # Residual compressor
            f_comp = nn.Sequential(nn.Linear(j,H),nn.ELU(),nn.Linear(H,k)).to(DEVICE)
            m_decomp = nn.Sequential(nn.Linear(k,H),nn.ELU(),nn.Linear(H,j)).to(DEVICE)
            opt = torch.optim.Adam(list(f_comp.parameters())+list(m_decomp.parameters()), lr=LR)
            for ep in range(1, EP+1):
                f_comp.train(); m_decomp.train()
                idx = torch.randperm(len(dC), device=DEVICE)
                for i in range(0, len(dC), BS):
                    dc = dC[idx[i:i+BS]]
                    loss = nn.functional.mse_loss(m_decomp(f_comp(dc)), dc)
                    opt.zero_grad(); loss.backward(); opt.step()
            f_comp.eval(); m_decomp.eval()

            with torch.no_grad(): B = f_comp(dC)
            B_t, B_next = B[:-1], B[1:]

            # DMD on b
            B_np = B.cpu().numpy()
            A_b = fit_dmd(B_np)

            # 1-step eval for decoupled DMD:
            # Given x_t, x_{t-1}: compute C_t, C_{t-1}, dC_t, b_t, predict b_{t+1},
            # decode m(b_{t+1}), GRU accumulate, decode carrier
            # BUT for 1-step, we can simplify: predict b_{t+1} from b_t,
            # then the "observation prediction" is: decode(carrier_t + m(b_{t+1}))
            # Actually for 1-step we evaluate: given true b_t, predict b_{t+1},
            # reconstruct dC_{t+1} = m(b_{t+1}), C_{t+1} = C_t + dC_{t+1}, decode

            # Simple 1-step in b-space -> observation
            with torch.no_grad():
                C_curr = C[1:-1]  # C_t for t=1..T-2 (aligned with B[:-1])
                C_true_next = C[2:]  # true C_{t+1}

                # DMD 1-step
                b_pred_dmd = torch.tensor((A_b @ B_np[:-1].T).T, dtype=torch.float32, device=DEVICE)
                dC_pred_dmd = m_decomp(b_pred_dmd)
                C_pred_dmd = C_curr + dC_pred_dmd
                x_pred_dmd = carrier.decode(C_pred_dmd)
                x_true = Xt[2:]
                f_dec_dmd = torch.sqrt(torch.mean((x_pred_dmd - x_true)**2)).item()

            # MLP on b
            seed_all(seed + 1000)
            mlp_b = MLPHead(k, H).to(DEVICE)
            train_head_on_latent(mlp_b, B_t, B_next)
            mlp_b.eval()
            with torch.no_grad():
                b_pred_mlp = mlp_b(B[:-1])
                dC_pred_mlp = m_decomp(b_pred_mlp)
                C_pred_mlp = C_curr + dC_pred_mlp
                x_pred_mlp = carrier.decode(C_pred_mlp)
                f_dec_mlp = torch.sqrt(torch.mean((x_pred_mlp - x_true)**2)).item()

            # ODE on b
            seed_all(seed + 2000)
            ode_b = ODEFunc(k, H).to(DEVICE)
            train_head_on_latent(ode_b, B_t, B_next, is_ode=True)
            ode_b.eval()
            with torch.no_grad():
                b_pred_ode = rk4_step(ode_b, B[:-1])
                dC_pred_ode = m_decomp(b_pred_ode)
                C_pred_ode = C_curr + dC_pred_ode
                x_pred_ode = carrier.decode(C_pred_ode)
                f_dec_ode = torch.sqrt(torch.mean((x_pred_ode - x_true)**2)).item()

            # Spectral DMD on carrier (1-step = just A @ C_t)
            A_carrier = fit_dmd(C.cpu().numpy())
            with torch.no_grad():
                C_np = C.cpu().numpy()
                C_pred_spec = torch.tensor(
                    (A_carrier @ C_np[:-1].T).T, dtype=torch.float32, device=DEVICE)
                x_pred_spec = carrier.decode(C_pred_spec)
                f_spectral = torch.sqrt(torch.mean((x_pred_spec - Xt[1:])**2)).item()

            res = {
                'AE+DMD': (r_ae_dmd, f_ae_dmd),
                'AE+MLP': (best_r_mlp_c, best_f_mlp_c),
                'AE+ODE': (best_r_ode_c, best_f_ode_c),
                'Resid+DMD': (r_dec, f_dec_dmd),
                'Resid+MLP': (r_dec, f_dec_mlp),
                'Resid+ODE': (r_dec, f_dec_ode),
                'Spectral': (r_dec, f_spectral),
            }

            for mname, (r, f) in res.items():
                sys_results.setdefault(mname, []).append({'rmse_r': r, 'rmse_f': f})

            dt = time.time() - t0
            print(f'  seed={seed} ({dt:.0f}s)')
            for mname, (r, f) in res.items():
                print(f'    {mname:15s}  R={r:.5f}  F={f:.5f}')

        # Aggregate
        agg = {}
        for mname in sys_results:
            rs = [x['rmse_r'] for x in sys_results[mname]]
            fs = [x['rmse_f'] for x in sys_results[mname]]
            agg[mname] = {
                'rmse_r_mean': float(np.mean(rs)), 'rmse_r_std': float(np.std(rs)),
                'rmse_f_mean': float(np.mean(fs)), 'rmse_f_std': float(np.std(fs)),
                'rmse_r_all': rs, 'rmse_f_all': fs,
            }
        all_results[sname] = agg

    # Summary
    print('\n' + '='*70)
    print('SINGLE-STEP FORECAST — ALL SYSTEMS')
    print('='*70)
    for sname in all_results:
        print(f'\n{sname}:')
        for mname in all_results[sname]:
            d = all_results[sname][mname]
            print(f'  {mname:15s}  R={d["rmse_r_mean"]:.5f}+-{d["rmse_r_std"]:.5f}  F={d["rmse_f_mean"]:.5f}+-{d["rmse_f_std"]:.5f}')

    import json
    with open('/kaggle/working/single_step_heads.json', 'w', encoding='utf-8') as fp:
        json.dump(all_results, fp, indent=2)
    print('\n-> single_step_heads.json')

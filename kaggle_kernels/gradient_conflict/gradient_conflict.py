"""
Gradient conflict experiment for ICLR 2027 Theory §4B.

Trains a Koopman AE (coupled baseline) at several φ values and logs the
cosine similarity between ∇_{θ_enc} L_rec and ∇_{θ_enc} L_fcst at each
training batch.

Outputs: /kaggle/working/gradient_conflict.json
"""
import json, time, random
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

OUT = "/kaggle/working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 400; LR = 1e-3; BS = 512
SEEDS = [0, 1, 2]
PHIS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
LOG_EVERY = 10


def SeedAll(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KoopmanAE(nn.Module):
    def __init__(self, n, k, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, k))
        self.dec = nn.Sequential(nn.Linear(k, h), nn.ELU(),
                                 nn.Linear(h, h), nn.ELU(), nn.Linear(h, n))
        self.A = nn.Linear(k, k, bias=False)

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x): return self.decode(self.encode(x))
    def predict(self, z): return self.A(z)

    def encoder_params(self):
        return list(self.enc.parameters())


def grad_cosine(model, x_t, x_tp1):
    enc_params = model.encoder_params()

    model.zero_grad()
    z_t = model.encode(x_t)
    L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
    L_rec.backward(retain_graph=True)
    g_rec = torch.cat([p.grad.flatten() for p in enc_params if p.grad is not None])

    model.zero_grad()
    z_t = model.encode(x_t)
    z_tp1 = model.encode(x_tp1)
    L_lin = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
    L_lin.backward()
    g_fcst = torch.cat([p.grad.flatten() for p in enc_params if p.grad is not None])

    cos = torch.dot(g_rec, g_fcst) / (g_rec.norm() * g_fcst.norm() + 1e-8)
    return cos.item(), L_rec.item(), L_lin.item()


def train_koopman_with_logging(model, Xt, phi, log_every=LOG_EVERY):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    trace = []
    batch_count = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xt) - 1, device=DEVICE)
        for i in range(0, len(idx), BS):
            sl = idx[i:i + BS]
            x_t, x_tp1 = Xt[sl], Xt[sl + 1]

            if batch_count % log_every == 0:
                cos, lr_, lf = grad_cosine(model, x_t, x_tp1)
                trace.append({"epoch": ep, "batch": batch_count,
                              "cosine": cos, "L_rec": lr_, "L_fcst": lf})

            model.zero_grad()
            z_t = model.encode(x_t)
            L_rec = nn.functional.mse_loss(model.decode(z_t), x_t)
            z_tp1 = model.encode(x_tp1)
            L_lin = nn.functional.mse_loss(model.predict(z_t), z_tp1.detach())
            loss = (1 - phi) * L_rec + phi * L_lin
            opt.zero_grad(); loss.backward(); opt.step()
            batch_count += 1
    return trace


def delay(raw, nd):
    return np.concatenate([raw[i:len(raw) - nd + 1 + i] for i in range(nd)], axis=1)

def norm_split(obs, raw, gt_dim, n_tr, n_te):
    tr, te = obs[:n_tr], obs[n_tr:n_tr + n_te]
    gt = raw[n_tr:n_tr + n_te, :gt_dim]
    mu, sig = tr.mean(0), tr.std(0) + 1e-8
    return (tr - mu) / sig, (te - mu) / sig, gt, mu, sig


def make_systems():
    systems = {}

    k12, k23, kw = 1.0, 0.5, 0.3
    def osc(t, s):
        x1, v1, x2, v2, x3, v3 = s
        return [v1, -kw * x1 - k12 * (x1 - x2), v2, -k12 * (x2 - x1) - k23 * (x2 - x3),
                v3, -k23 * (x3 - x2) - kw * x3]
    sol = solve_ivp(osc, [0, 500], [1, 0, 0, 0.5, -0.5, 0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw = sol.y.T[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 6, 4000, 500)
    systems["Coupled Harmonic"] = dict(train_n=trn, test_n=ten, n_obs=6, k=4, h=64)

    w1, w2 = 1.0, np.sqrt(2)
    A5 = np.array([[0, w1, 0, 0, 0], [-w1, 0, 0, 0, 0],
                    [0, 0, 0, w2, 0], [0, 0, -w2, 0, 0],
                    [0, 0, 0, 0, -0.05]])
    Ad = expm(A5 * 0.05)
    N5 = 10000
    raw5 = np.empty((N5, 5)); raw5[0] = [1, 0, 0.5, 0.5, 1]
    for i in range(1, N5): raw5[i] = Ad @ raw5[i - 1]
    raw = raw5[200:]
    trn, ten, gt, mu, sig = norm_split(raw, raw, 5, 4000, 500)
    systems["Linear 5D"] = dict(train_n=trn, test_n=ten, n_obs=5, k=4, h=64)

    A_br, B_br = 1.0, 3.0
    def bruss(t, s):
        x, y = s
        return [A_br + x**2 * y - (B_br + 1) * x, B_br * x - x**2 * y]
    sol = solve_ivp(bruss, [0, 500], [1.5, 3.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_br = sol.y.T[200:]
    obs_br = delay(raw_br, 5)
    trn, ten, gt, mu, sig = norm_split(obs_br, raw_br[:len(obs_br)], 2, 4000, 500)
    systems["Brusselator"] = dict(train_n=trn, test_n=ten, n_obs=10, k=3, h=64)

    alpha, beta, delta_d, gamma, omega = -1.0, 1.0, 0.3, 0.5, 1.2
    def duffing(t, s):
        x, v = s
        return [v, -delta_d * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)]
    sol = solve_ivp(duffing, [0, 500], [0.5, 0.0],
                    t_eval=np.arange(0, 500, 0.05), rtol=1e-10, atol=1e-10)
    raw_du = sol.y.T[200:]
    obs_du = delay(raw_du, 5)
    trn, ten, gt, mu, sig = norm_split(obs_du, raw_du[:len(obs_du)], 2, 4000, 500)
    systems["Duffing"] = dict(train_n=trn, test_n=ten, n_obs=10, k=3, h=64)

    return systems


if __name__ == "__main__":
    systems = make_systems()
    t0 = time.time()
    results = {}

    for sname, cfg in systems.items():
        results[sname] = {}
        Xt = torch.tensor(cfg["train_n"], dtype=torch.float32).to(DEVICE)

        for phi in PHIS:
            results[sname][str(phi)] = {}
            for seed in SEEDS:
                print(f"  {sname}  φ={phi}  seed={seed} ... ", end="", flush=True)
                SeedAll(seed)
                model = KoopmanAE(cfg["n_obs"], cfg["k"], cfg["h"]).to(DEVICE)
                trace = train_koopman_with_logging(model, Xt, phi)
                results[sname][str(phi)][str(seed)] = trace
                cosines = [t["cosine"] for t in trace]
                mean_cos = np.mean(cosines)
                frac_neg = np.mean([c < 0 for c in cosines])
                print(f"mean_cos={mean_cos:.3f}  frac_neg={frac_neg:.2f}")

    with open(f"{OUT}/gradient_conflict.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\n→ {OUT}/gradient_conflict.json")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print("\nSummary (mean cosine similarity, fraction negative):")
    for sname in results:
        print(f"\n  {sname}:")
        for phi in results[sname]:
            all_cos = []
            for seed in results[sname][phi]:
                all_cos.extend([t["cosine"] for t in results[sname][phi][seed]])
            mc = np.mean(all_cos)
            fn = np.mean([c < 0 for c in all_cos])
            print(f"    φ={phi:>5s}  mean_cos={mc:+.3f}  frac_neg={fn:.2f}")
    print("\nDone.")

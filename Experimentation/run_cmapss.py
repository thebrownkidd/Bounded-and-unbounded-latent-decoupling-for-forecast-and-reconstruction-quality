"""Three-phase pipeline on C-MAPSS FD001, plus two reference baselines.

Same schedule as the Lorenz experiment:
  phase 1  train E, r, D            L = MSE(D(r(E(x))), x)
  phase 2  train f, g, m            L = MSE(C_{t+1}, m(g(f(h_t))))   D unused
  phase 3  fine tune m and D        L = MSE(Chat, C) + MSE(xhat, x)  g unused

Two phi baselines are included, not as a frontier but so the numbers mean
something. Without any shared-latent reference there is nothing to decouple
against.

Metric suite is adapted for degradation data. See METRICS.md notes at the
bottom of this file for what transfers and what does not.
"""
import sys, json, copy, io
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from Src import DecoupledModel
from Src.LatentMapping import LatentMapping
from Comp import JointAEGRU
from Utils.Checkpoints import TrainOrLoad
from Utils.Rollout import MakeSeriesWindows, EvalWindowsFromTrajectories, PersistenceRollout
from Utils.Benchmark import CountParams, MatchWidth, EpochTrain, PinThreads, SeedAll

torch.set_num_threads(8)
DEV = "cpu"

# ---------------------------------------------------------------- data
D = np.load(ROOT / "Data" / "CmapssFD001.npz", allow_pickle=True)
TrainSeries, ValSeries, Holdout = D["train"], D["val"], D["holdout_obs"]
FLOOR_EST = float(D["noise_floor_mse_est"])
train = TrainSeries.reshape(-1, TrainSeries.shape[-1])

N = TrainSeries.shape[-1]
# k is the forecast carrier width. On Lorenz it was set to the known intrinsic
# dimension 3. C-MAPSS has no known intrinsic dimension, so it must be swept.
# Overridable so the sweep does not need a second copy of this file.
import os as _os
K = int(_os.environ.get("CMAPSS_K", 3))
LATENT = int(_os.environ.get("CMAPSS_LATENT", 16))
SUFFIX = _os.environ.get("CMAPSS_SUFFIX", "")
A, B = 8, 8
WARM, UNROLL, HORIZON = 16, 8, 32          # cycles, not Lyapunov times
# FD001 is 7,680 rows against Lorenz's 112,000. Reusing Lorenz's 60 epochs
# gave phase 1 only 420 gradient steps and val_recon was still falling at the
# end, so the first run was undertrained rather than converged. Budget is now
# set in GRADIENT STEPS, which is what actually matters, and is identical for
# every model including the baselines.
EPOCHS = int(_os.environ.get("CMAPSS_EPOCHS", 400))
SEEDS = [0, 1, 2]
MARKS = tuple(int(EPOCHS * f) for f in (0.05, 0.1, 0.25, 0.5, 1.0))
BATCH_P1, BATCH_P2, BATCH_P3, BATCH_SEQ = 256, 128, 128, 128
SEQ_STRIDE, LR = 2, 1e-3
THRESH = 0.4
LEADS = [1, 2, 4, 8, 16, 32]               # cycles
ACC_T = 0.6

PinThreads(8)
CK = ROOT / "Checkpoints" / "cmapss"
CK.mkdir(parents=True, exist_ok=True)

SCALE = float(Holdout.std())
BOUNDS = (float(train.min()), float(train.max()))
CLIM_MEAN = train.mean(0)

WarmNp, FutNp = EvalWindowsFromTrajectories(Holdout, WARM, HORIZON, per_traj=2, seed=7)
Warm = torch.tensor(WarmNp, dtype=torch.float32)
Fut = torch.tensor(FutNp, dtype=torch.float32)

XFullNp = MakeSeriesWindows(TrainSeries, WARM + UNROLL, stride=SEQ_STRIDE)
XFull = torch.tensor(XFullNp, dtype=torch.float32)
XValFullNp = MakeSeriesWindows(ValSeries, WARM + UNROLL, stride=8)
XValFull = torch.tensor(XValFullNp, dtype=torch.float32)
XValWarm, XValFut = XValFull[:, :WARM], XValFull[:, WARM:]
XTrain = torch.tensor(train, dtype=torch.float32)

MseT = nn.MSELoss()
print(f"channels {N}  train {TrainSeries.shape}  val {ValSeries.shape}  holdout {Holdout.shape}")
print(f"windows: train {XFull.shape}  val {XValFull.shape}  eval {Warm.shape} -> {Fut.shape}")
print(f"noise floor ESTIMATE {FLOOR_EST:.4f}   scale {SCALE:.4f}   bounds {BOUNDS}")


def SetTrainable(Model, Mods):
    for p in Model.parameters():
        p.requires_grad_(False)
    for m in Mods:
        for p in m.parameters():
            p.requires_grad_(True)
    return [p for p in Model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- ours, 3 phases
def Phase1Epoch(M, Opt):
    perm = torch.randperm(XTrain.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_P1):
        x = XTrain[perm[i:i + BATCH_P1]]
        L = MseT(M.D(M.R(M.E(x))), x)
        Opt.zero_grad(); L.backward(); Opt.step()
        s += L.item(); nb += 1
    return {"recon": s / max(nb, 1)}


@torch.no_grad()
def PrecomputeP2(M, Seq):
    Bn, T, n = Seq.shape
    h = M.E(Seq.reshape(Bn * T, n))
    return (h.view(Bn, T, n)[:, :WARM].contiguous(),
            M.R(h).view(Bn, T, A, B)[:, WARM:].contiguous())


def Phase2Epoch(M, Opt, Hw, Cf):
    perm = torch.randperm(Hw.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_P2):
        j = perm[i:i + BATCH_P2]
        h, Ct = Hw[j], Cf[j]
        Bn, T, n = h.shape
        b = M.F(h.reshape(Bn * T, n)).view(Bn, T, K)
        _, st = M.G(b[:, :-1], None)
        cur, out = b[:, -1], []
        for _ in range(UNROLL):
            cur, st = M.G.Step(cur, st); out.append(cur)
        Ch = M.M(torch.stack(out, 1).reshape(-1, K)).view(Bn, UNROLL, A, B)
        L = MseT(Ch, Ct)
        Opt.zero_grad(); L.backward(); Opt.step()
        s += L.item(); nb += 1
    return {"c_fcst": s / max(nb, 1)}


@torch.no_grad()
def PrecomputeP3(M, Seq):
    Bn, T, n = Seq.shape
    h = M.E(Seq.reshape(Bn * T, n))
    return M.F(h).view(Bn, T, K), M.R(h).view(Bn, T, A, B)


def Phase3Epoch(M, Opt, bT, CT, X):
    perm = torch.randperm(bT.shape[0])
    s, nb = [0.0, 0.0], 0
    for i in range(0, len(perm), BATCH_P3):
        j = perm[i:i + BATCH_P3]
        b, Ct, x = bT[j], CT[j], X[j]
        Bn, T = b.shape[:2]
        Ch = M.M(b.reshape(Bn * T, K)).view(Bn, T, A, B)
        xh = M.D(Ch.reshape(Bn * T, A, B)).view(Bn, T, N)
        Lc, Lx = MseT(Ch, Ct), MseT(xh, x)
        L = Lc + Lx
        Opt.zero_grad(); L.backward(); Opt.step()
        s = [s[0] + Lc.item(), s[1] + Lx.item()]; nb += 1
    nb = max(nb, 1)
    return {"c_mse": s[0] / nb, "recon": s[1] / nb}


@torch.no_grad()
def Val(M):
    was = M.training; M.eval()
    o = {"val_recon": MseT(M.Reconstruct(XValFull), XValFull).item(),
         "val_fcst": MseT(M.Rollout(XValWarm, UNROLL), XValFut).item()}
    if was:
        M.train()
    return o


def TrainOurs(seed, verbose=True):
    def p1():
        def Build():
            SeedAll(seed); return DecoupledModel(N, K, A, B)

        def Train(M):
            P = SetTrainable(M, [M.E, M.R, M.D])
            Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: Phase1Epoch(M, Opt),
                              EvalFn=lambda: {"val_recon": MseT(M.D(M.R(M.E(XValFull.reshape(-1, N)))),
                                                                XValFull.reshape(-1, N)).item()},
                              epochs=EPOCHS, marks=MARKS, name=f"cmapss p1 s{seed}",
                              verbose=verbose)
        return TrainOrLoad(CK / f"p1_k{K}_seed{seed}.pt", Build, Train, DEV)

    M1, _ = p1()

    def p2():
        def Build():
            SeedAll(seed + 500); return copy.deepcopy(M1)

        def Train(M):
            P = SetTrainable(M, [M.F, M.G, M.M])
            Hw, Cf = PrecomputeP2(M, XFull)
            Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: Phase2Epoch(M, Opt, Hw, Cf),
                              epochs=EPOCHS, marks=MARKS, name=f"cmapss p2 s{seed}",
                              verbose=verbose)
        return TrainOrLoad(CK / f"p2_k{K}_seed{seed}.pt", Build, Train, DEV)

    M2, _ = p2()

    def p3():
        def Build():
            SeedAll(seed + 1000); return copy.deepcopy(M2)

        def Train(M):
            bT, CT = PrecomputeP3(M, XFull)
            P = SetTrainable(M, [M.M, M.D])
            Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: Phase3Epoch(M, Opt, bT, CT, XFull),
                              EvalFn=lambda: Val(M),
                              epochs=EPOCHS, marks=MARKS, name=f"cmapss p3 s{seed}",
                              verbose=verbose)
        return TrainOrLoad(CK / f"p3_k{K}_seed{seed}.pt", Build, Train, DEV)

    return p3()


# ---------------------------------------------------------------- phi reference
TARGET = CountParams(DecoupledModel(N, K, A, B))
WJ = MatchWidth(lambda w: JointAEGRU(N, latent=LATENT, width=w), TARGET)


def JointEpoch(M, Opt, phi):
    perm = torch.randperm(XFull.shape[0])
    s, nb = [0.0, 0.0], 0
    for i in range(0, len(perm), BATCH_SEQ):
        rec, fc = M.Losses(XFull[perm[i:i + BATCH_SEQ]], WARM)
        L = (1 - phi) * rec + phi * fc
        Opt.zero_grad(); L.backward(); Opt.step()
        s = [s[0] + rec.item(), s[1] + fc.item()]; nb += 1
    nb = max(nb, 1)
    return {"recon": s[0] / nb, "fcst": s[1] / nb}


def TrainPhi(phi, seed, verbose=False):
    tag = f"{phi:g}".replace(".", "_")

    def Build():
        SeedAll(seed); return JointAEGRU(N, latent=LATENT, width=WJ)

    def Train(M):
        Opt = torch.optim.Adam(M.parameters(), lr=LR)
        return EpochTrain(lambda: JointEpoch(M, Opt, phi), EvalFn=lambda: Val(M),
                          epochs=EPOCHS, marks=MARKS, name=f"cmapss phi{phi:g} s{seed}",
                          verbose=verbose)
    return TrainOrLoad(CK / f"joint_l{LATENT}_phi{tag}_seed{seed}.pt", Build, Train, DEV)


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def Evaluate(models, label):
    """Degradation-appropriate suite. Lead times in CYCLES, never Lyapunov times."""
    per = []
    for M in models:
        M.eval()
        rec = M.Reconstruct(Warm).cpu().numpy()
        pred = M.Rollout(Warm, HORIZON).cpu().numpy()
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        truth, warm = FutNp, WarmNp
        persist = np.asarray(PersistenceRollout(warm, HORIZON))

        r = {}
        r["recon_mse"] = float(((rec - warm) ** 2).mean())
        r["recon_over_floor_est"] = r["recon_mse"] / FLOOR_EST
        ss = ((warm - warm.mean(0)) ** 2).sum(); r2 = 1 - ((warm - rec) ** 2).sum() / ss
        r["recon_r2"] = float(r2)
        per_ch = 1 - ((warm - rec) ** 2).sum((0, 1)) / ((warm - warm.mean((0, 1))) ** 2).sum((0, 1))
        r["recon_r2_worst"] = float(per_ch.min())

        lo, hi = BOUNDS
        inr = ((pred >= lo) & (pred <= hi)).all(axis=(1, 2))
        r["divergence"] = float(1.0 - (inr & np.isfinite(pred).all(axis=(1, 2))).mean())

        # per-lead profile, in cycles
        err = np.sqrt(((pred - truth) ** 2).mean(axis=(0, 2))) / SCALE     # NRMSE curve
        r["curve"] = err.tolist()
        for h in LEADS:
            i = h - 1
            p, t, q = pred[:, i, :], truth[:, i, :], persist[:, i, :]
            r[f"mae@{h}"] = float(np.abs(p - t).mean())
            r[f"rmse@{h}"] = float(np.sqrt(((p - t) ** 2).mean()))
            ap, at = p - CLIM_MEAN, t - CLIM_MEAN
            den = np.sqrt((ap ** 2).sum() * (at ** 2).sum())
            r[f"acc@{h}"] = float((ap * at).sum() / den) if den > 0 else 0.0
            mm, mp = ((p - t) ** 2).mean(), ((q - t) ** 2).mean()
            r[f"ss_persist@{h}"] = float(1 - mm / mp) if mp > 0 else 0.0
            r[f"var_ratio@{h}"] = float(p.std() / t.std()) if t.std() > 0 else 0.0

        # lead time at which the forecast stops being useful, IN CYCLES
        cross = np.nonzero(err > THRESH)[0]
        r["nrmse_horizon_cycles"] = float(cross[0] + 1) if len(cross) else float(HORIZON)
        accs = []
        for i in range(HORIZON):
            ap, at = pred[:, i, :] - CLIM_MEAN, truth[:, i, :] - CLIM_MEAN
            den = np.sqrt((ap ** 2).sum() * (at ** 2).sum())
            accs.append((ap * at).sum() / den if den > 0 else 0.0)
        below = np.nonzero(np.array(accs) < ACC_T)[0]
        r["acc_horizon_cycles"] = float(below[0] + 1) if len(below) else float(HORIZON)

        # free run growth: does the rollout inflate relative to the truth
        r["freerun_growth"] = float(np.linalg.norm(pred[:, -1, :], axis=1).mean()
                                    / max(np.linalg.norm(truth[:, -1, :], axis=1).mean(), 1e-9))
        # degradation is near monotone; count sign flips of the mean channel trend
        dt = np.diff(pred.mean(2), axis=1)
        r["mono_violation"] = float((np.sign(dt[:, 1:]) != np.sign(dt[:, :-1])).mean())
        per.append(r)

    out = {"model": label, "n_seeds": len(per)}
    for k in per[0]:
        if k == "curve":
            out["curve"] = np.mean([p["curve"] for p in per], axis=0).tolist()
        else:
            v = [p[k] for p in per]
            out[k] = float(np.mean(v)); out[k + "_std"] = float(np.std(v))
    return out


if __name__ == "__main__":
    RES = {}
    print(f"\nparam target {TARGET:,}  joint width {WJ} "
          f"({CountParams(JointAEGRU(N, LATENT, WJ)):,})\n")

    ours = []
    for s in SEEDS:
        M, _ = TrainOurs(s, verbose=(s == SEEDS[0]))
        ours.append(M)
    RES[f"Ours k={K}"] = Evaluate(ours, f"Ours k={K}")

    for phi in (0.1, 0.5):
        ms = [TrainPhi(phi, s)[0] for s in SEEDS]
        RES[f"phi={phi:g} lat={LATENT}"] = Evaluate(ms, f"phi={phi:g} lat={LATENT}")

    RES["persistence"] = {"model": "persistence", "n_seeds": 0}
    pz = np.asarray(PersistenceRollout(WarmNp, HORIZON))
    for h in LEADS:
        i = h - 1
        RES["persistence"][f"mae@{h}"] = float(np.abs(pz[:, i] - FutNp[:, i]).mean())

    out = ROOT / "Paper" / f"cmapss_numbers{SUFFIX}.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"config": {"dataset": "FD001", "n_obs": N, "k": K, "grid": [A, B],
                    "warm": WARM, "unroll": UNROLL, "horizon": HORIZON,
                    "epochs": EPOCHS, "seeds": SEEDS, "leads_cycles": LEADS,
                    "noise_floor_estimate": FLOOR_EST, "scale": SCALE,
                    "target_params": TARGET},
         "results": RES}, indent=1))
    print(f"\nwrote {out}")

    print(f"\n{'model':22s} {'recon':>8s} {'xflr*':>7s} {'R2':>7s} "
          f"{'MAE@8':>7s} {'ACC@8':>7s} {'ACChor':>7s} {'div':>7s}")
    for k, r in RES.items():
        if "recon_mse" not in r:
            continue
        print(f"{k:22s} {r['recon_mse']:8.4f} {r['recon_over_floor_est']:7.3f} "
              f"{r['recon_r2']:7.3f} {r['mae@8']:7.3f} {r['acc@8']:7.3f} "
              f"{r['acc_horizon_cycles']:7.1f} {100*r['divergence']:6.1f}%")

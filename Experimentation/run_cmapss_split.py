"""True two-carrier split on C-MAPSS: narrow forecast carrier, wider recon carrier.

The model in the paper is not actually split at the bottleneck. Reconstruction
runs x -> E -> h -> f -> b(k) -> m -> C -> D -> xhat, so C sits strictly
downstream of b and carries at most k degrees of freedom. Calling that a
"3-d forecast carrier and a 64-d reconstruction carrier" overstates it.

This variant makes the split real by using the two paths that already exist:

    reconstruction   x -> E -> h -> r -> C_r (a x b, bounded) -> D -> xhat
    forecast         x -> E -> h -> f -> b(k) -> g -> b' -> m -> C -> D -> xhat

r maps the full h to the bounded code directly, so reconstruction never passes
through b. The two carriers now have genuinely different widths.

Consequence for the schedule: D is shared by both paths, so phase 3 trains m
ONLY. Fine tuning D on the forecast path would pull it away from the code r
produces and undo phase 1, which is exactly the failure diagnosed on the
non-split version (phase 1 reached 0.0014 and phase 3 landed at 0.3136 because
m(b) could not reproduce r(h)).
"""
import sys, json, copy, io, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from Src import DecoupledModel
from Comp import JointAEGRU
from Utils.Checkpoints import TrainOrLoad
from Utils.Rollout import MakeSeriesWindows, EvalWindowsFromTrajectories, PersistenceRollout
from Utils.Benchmark import CountParams, MatchWidth, EpochTrain, PinThreads, SeedAll

torch.set_num_threads(8)
DEV = "cpu"

D_ = np.load(ROOT / "Data" / "CmapssFD001.npz", allow_pickle=True)
TrainSeries, ValSeries, Holdout = D_["train"], D_["val"], D_["holdout_obs"]
FLOOR_EST = float(D_["noise_floor_mse_est"])
train = TrainSeries.reshape(-1, TrainSeries.shape[-1])
N = TrainSeries.shape[-1]

K = int(os.environ.get("CMAPSS_K", 2))               # forecast carrier
A = int(os.environ.get("CMAPSS_A", 4))               # recon carrier is A x B
B = int(os.environ.get("CMAPSS_B", 4))
LATENT = int(os.environ.get("CMAPSS_LATENT", A * B))  # baseline, matched to recon width
SUFFIX = os.environ.get("CMAPSS_SUFFIX", "_split")
EPOCHS = int(os.environ.get("CMAPSS_EPOCHS", 400))

WARM, UNROLL, HORIZON = 16, 8, 32
SEEDS = [0, 1, 2]
MARKS = tuple(int(EPOCHS * f) for f in (0.05, 0.25, 1.0))
BATCH_P1, BATCH_P2, BATCH_P3, BATCH_SEQ = 256, 128, 128, 128
SEQ_STRIDE, LR, THRESH, ACC_T = 2, 1e-3, 0.4, 0.6
LEADS = [1, 2, 4, 8, 16, 32]

PinThreads(8)
CK = ROOT / "Checkpoints" / "cmapss_split"
CK.mkdir(parents=True, exist_ok=True)

SCALE = float(Holdout.std())
BOUNDS = (float(train.min()), float(train.max()))
CLIM_MEAN = train.mean(0)
WarmNp, FutNp = EvalWindowsFromTrajectories(Holdout, WARM, HORIZON, per_traj=2, seed=7)
Warm = torch.tensor(WarmNp, dtype=torch.float32)
XFull = torch.tensor(MakeSeriesWindows(TrainSeries, WARM + UNROLL, stride=SEQ_STRIDE),
                     dtype=torch.float32)
XValFull = torch.tensor(MakeSeriesWindows(ValSeries, WARM + UNROLL, stride=8),
                        dtype=torch.float32)
XValWarm, XValFut = XValFull[:, :WARM], XValFull[:, WARM:]
XTrain = torch.tensor(train, dtype=torch.float32)
MseT = nn.MSELoss()


class SplitModel(nn.Module):
    """Reconstruction through r, forecast through b. D is shared."""

    def __init__(self, base):
        super().__init__()
        self.Base = base
        self.n = base.n

    def Reconstruct(self, window):
        Bn, T, n = window.shape
        h = self.Base.E(window.reshape(Bn * T, n))
        return self.Base.D(self.Base.R(h)).view(Bn, T, n)

    def Rollout(self, window, horizon):
        Bn, T, n = window.shape
        h = self.Base.E(window.reshape(Bn * T, n))
        b = self.Base.F(h).view(Bn, T, K)
        state = None
        if T > 1:
            _, state = self.Base.G(b[:, :-1], None)
        cur, out = b[:, -1], []
        for _ in range(horizon):
            cur, state = self.Base.G.Step(cur, state)
            out.append(cur)
        Bs = torch.stack(out, 1)
        C = self.Base.M(Bs.reshape(-1, K))
        return self.Base.D(C).view(Bn, horizon, self.n)


def SetTrainable(M, mods):
    for p in M.parameters():
        p.requires_grad_(False)
    for m in mods:
        for p in m.parameters():
            p.requires_grad_(True)
    return [p for p in M.parameters() if p.requires_grad]


def P1Epoch(M, Opt):
    perm = torch.randperm(XTrain.shape[0]); s, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_P1):
        x = XTrain[perm[i:i + BATCH_P1]]
        L = MseT(M.D(M.R(M.E(x))), x)
        Opt.zero_grad(); L.backward(); Opt.step(); s += L.item(); nb += 1
    return {"recon": s / max(nb, 1)}


@torch.no_grad()
def PreP2(M, Seq):
    Bn, T, n = Seq.shape
    h = M.E(Seq.reshape(Bn * T, n))
    return (h.view(Bn, T, n)[:, :WARM].contiguous(),
            M.R(h).view(Bn, T, A, B)[:, WARM:].contiguous())


def P2Epoch(M, Opt, Hw, Cf):
    perm = torch.randperm(Hw.shape[0]); s, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_P2):
        j = perm[i:i + BATCH_P2]; h, Ct = Hw[j], Cf[j]
        Bn, T, n = h.shape
        b = M.F(h.reshape(Bn * T, n)).view(Bn, T, K)
        _, st = M.G(b[:, :-1], None)
        cur, out = b[:, -1], []
        for _ in range(UNROLL):
            cur, st = M.G.Step(cur, st); out.append(cur)
        Ch = M.M(torch.stack(out, 1).reshape(-1, K)).view(Bn, UNROLL, A, B)
        L = MseT(Ch, Ct)
        Opt.zero_grad(); L.backward(); Opt.step(); s += L.item(); nb += 1
    return {"c_fcst": s / max(nb, 1)}


@torch.no_grad()
def PreP3(M, Seq):
    Bn, T, n = Seq.shape
    h = M.E(Seq.reshape(Bn * T, n))
    return M.F(h).view(Bn, T, K), M.R(h).view(Bn, T, A, B)


def P3Epoch(M, Opt, bT, CT):
    """m only. D is shared with the reconstruction path and must not move."""
    perm = torch.randperm(bT.shape[0]); s, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_P3):
        j = perm[i:i + BATCH_P3]; b, Ct = bT[j], CT[j]
        Bn, T = b.shape[:2]
        Ch = M.M(b.reshape(Bn * T, K)).view(Bn, T, A, B)
        L = MseT(Ch, Ct)
        Opt.zero_grad(); L.backward(); Opt.step(); s += L.item(); nb += 1
    return {"c_mse": s / max(nb, 1)}


def TrainOurs(seed, verbose=False):
    def b1():
        def Build():
            SeedAll(seed); return DecoupledModel(N, K, A, B)

        def Train(M):
            P = SetTrainable(M, [M.E, M.R, M.D]); Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: P1Epoch(M, Opt), epochs=EPOCHS, marks=MARKS,
                              name=f"split p1 s{seed}", verbose=verbose)
        return TrainOrLoad(CK / f"p1_k{K}_{A}x{B}_s{seed}.pt", Build, Train, DEV)
    M1, _ = b1()

    def b2():
        def Build():
            SeedAll(seed + 500); return copy.deepcopy(M1)

        def Train(M):
            P = SetTrainable(M, [M.F, M.G, M.M]); Hw, Cf = PreP2(M, XFull)
            Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: P2Epoch(M, Opt, Hw, Cf), epochs=EPOCHS,
                              marks=MARKS, name=f"split p2 s{seed}", verbose=verbose)
        return TrainOrLoad(CK / f"p2_k{K}_{A}x{B}_s{seed}.pt", Build, Train, DEV)
    M2, _ = b2()

    def b3():
        def Build():
            SeedAll(seed + 1000); return copy.deepcopy(M2)

        def Train(M):
            bT, CT = PreP3(M, XFull); P = SetTrainable(M, [M.M])
            Opt = torch.optim.Adam(P, lr=LR)
            return EpochTrain(lambda: P3Epoch(M, Opt, bT, CT), epochs=EPOCHS,
                              marks=MARKS, name=f"split p3 s{seed}", verbose=verbose)
        return TrainOrLoad(CK / f"p3_k{K}_{A}x{B}_s{seed}.pt", Build, Train, DEV)
    M3, _ = b3()
    return SplitModel(M3)


TARGET = CountParams(DecoupledModel(N, K, A, B))
WJ = MatchWidth(lambda w: JointAEGRU(N, latent=LATENT, width=w), TARGET)


def JointEpoch(M, Opt, phi):
    perm = torch.randperm(XFull.shape[0]); s, nb = [0.0, 0.0], 0
    for i in range(0, len(perm), BATCH_SEQ):
        rec, fc = M.Losses(XFull[perm[i:i + BATCH_SEQ]], WARM)
        L = (1 - phi) * rec + phi * fc
        Opt.zero_grad(); L.backward(); Opt.step()
        s = [s[0] + rec.item(), s[1] + fc.item()]; nb += 1
    return {"recon": s[0] / max(nb, 1), "fcst": s[1] / max(nb, 1)}


def TrainPhi(phi, seed):
    tag = f"{phi:g}".replace(".", "_")

    def Build():
        SeedAll(seed); return JointAEGRU(N, latent=LATENT, width=WJ)

    def Train(M):
        Opt = torch.optim.Adam(M.parameters(), lr=LR)
        return EpochTrain(lambda: JointEpoch(M, Opt, phi), epochs=EPOCHS,
                          marks=MARKS, name=f"split phi{phi:g} s{seed}", verbose=False)
    return TrainOrLoad(CK / f"joint_l{LATENT}_phi{tag}_s{seed}.pt", Build, Train, DEV)


@torch.no_grad()
def Evaluate(models, label):
    per = []
    persist = np.asarray(PersistenceRollout(WarmNp, HORIZON))
    for M in models:
        M.eval()
        rec = M.Reconstruct(Warm).cpu().numpy()
        pred = np.nan_to_num(M.Rollout(Warm, HORIZON).cpu().numpy(),
                             nan=0.0, posinf=1e6, neginf=-1e6)
        r = {"recon_mse": float(((rec - WarmNp) ** 2).mean())}
        r["recon_over_floor_est"] = r["recon_mse"] / FLOOR_EST
        ss = ((WarmNp - WarmNp.mean(0)) ** 2).sum()
        r["recon_r2"] = float(1 - ((WarmNp - rec) ** 2).sum() / ss)
        lo, hi = BOUNDS
        ok = ((pred >= lo) & (pred <= hi)).all(axis=(1, 2)) & np.isfinite(pred).all(axis=(1, 2))
        r["divergence"] = float(1.0 - ok.mean())
        err = np.sqrt(((pred - FutNp) ** 2).mean(axis=(0, 2))) / SCALE
        accs = []
        for i in range(HORIZON):
            ap, at = pred[:, i] - CLIM_MEAN, FutNp[:, i] - CLIM_MEAN
            den = np.sqrt((ap ** 2).sum() * (at ** 2).sum())
            accs.append(float((ap * at).sum() / den) if den > 0 else 0.0)
        for h in LEADS:
            i = h - 1
            r[f"mae@{h}"] = float(np.abs(pred[:, i] - FutNp[:, i]).mean())
            r[f"acc@{h}"] = accs[i]
            mp = ((persist[:, i] - FutNp[:, i]) ** 2).mean()
            r[f"ss_persist@{h}"] = float(1 - ((pred[:, i] - FutNp[:, i]) ** 2).mean() / mp)
        below = np.nonzero(np.array(accs) < ACC_T)[0]
        r["acc_horizon_cycles"] = float(below[0] + 1) if len(below) else float(HORIZON)
        r["freerun_growth"] = float(np.linalg.norm(pred[:, -1], axis=1).mean()
                                    / max(np.linalg.norm(FutNp[:, -1], axis=1).mean(), 1e-9))
        per.append(r)
    out = {"model": label, "n_seeds": len(per)}
    for k in per[0]:
        v = [p[k] for p in per]
        out[k] = float(np.mean(v)); out[k + "_std"] = float(np.std(v))
    return out


if __name__ == "__main__":
    print(f"n_obs {N}   forecast carrier k={K}   recon carrier {A}x{B}={A*B}   "
          f"baseline latent {LATENT}")
    print(f"bottleneck ratios: forecast {K/N:.3f}   recon {A*B/N:.3f}   "
          f"baseline {LATENT/N:.3f}")
    print(f"params target {TARGET:,}  baseline width {WJ} "
          f"({CountParams(JointAEGRU(N, LATENT, WJ)):,})\n")

    RES = {}
    RES[f"Ours split k={K} C={A}x{B}"] = Evaluate(
        [TrainOurs(s, verbose=(s == 0)) for s in SEEDS], f"Ours split k={K} C={A}x{B}")
    for phi in (0.1, 0.5):
        RES[f"phi={phi:g} lat={LATENT}"] = Evaluate(
            [TrainPhi(phi, s)[0] for s in SEEDS], f"phi={phi:g} lat={LATENT}")

    pz = np.asarray(PersistenceRollout(WarmNp, HORIZON))
    RES["persistence"] = {"model": "persistence", "n_seeds": 0,
                          **{f"mae@{h}": float(np.abs(pz[:, h-1] - FutNp[:, h-1]).mean())
                             for h in LEADS}}

    out = ROOT / "Paper" / f"cmapss_numbers{SUFFIX}.json"
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"config": {"k": K, "recon_grid": [A, B], "latent": LATENT, "n_obs": N,
                    "epochs": EPOCHS, "leads_cycles": LEADS}, "results": RES}, indent=1))
    print(f"\nwrote {out}\n")
    print(f"{'model':24s} {'recon':>8s} {'R2':>7s} {'MAE@1':>7s} {'MAE@8':>7s} "
          f"{'MAE@32':>7s} {'ACChor':>7s} {'div':>7s}")
    for k, r in RES.items():
        if "recon_mse" in r:
            print(f"{k:24s} {r['recon_mse']:8.4f} {r['recon_r2']:7.3f} {r['mae@1']:7.3f} "
                  f"{r['mae@8']:7.3f} {r['mae@32']:7.3f} {r['acc_horizon_cycles']:7.1f} "
                  f"{100*r['divergence']:6.1f}%")
        else:
            print(f"{k:24s} {'--':>8s} {'--':>7s} {r['mae@1']:7.3f} {r['mae@8']:7.3f} "
                  f"{r['mae@32']:7.3f}")

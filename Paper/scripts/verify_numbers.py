"""Regenerate every number the paper quotes, straight from the checkpoints.

Writes scratchpad/paper_numbers.json. Nothing in the paper should be typed by
hand; if it is not in here it does not go in.
"""
import sys, json, io
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from Src import DecoupledModel
from Comp import JointAEGRU, JointAEGRUSigmoid, PhysicsLatentAE
from Utils import LoadLorenzMultiSeries
from Utils.Metrics import (StepsPerLyapunov, LAMBDA_MAX, AttractorStats,
                           BoundViolation, Saturation)
from Utils.Rollout import (EvaluateModel, ForecastMetrics, PersistenceRollout,
                           MeanRollout, EvalWindowsFromTrajectories, MakeSeriesWindows)
from Utils.Benchmark import CountParams, CountInferenceParams, MatchWidth

torch.set_num_threads(8)
DEV = "cpu"
OUT = {}

# ---------------------------------------------------------------- data
M = LoadLorenzMultiSeries(ROOT / "Data" / "LorenzLiftMulti.npz")
TrainSeries, ValSeries, TestSeries = M["train"], M["val"], M["test"]
Ho = M["holdout_obs"]
train = TrainSeries.reshape(-1, TrainSeries.shape[-1])
N = train.shape[-1]
K, A, B = 3, 8, 8
DT, WARM, UNROLL, HORIZON, CLIMATE, THRESH = 0.01, 64, 20, 550, 5500, 0.4
FLOOR = float(M["noise_floor_mse"])
SPL = StepsPerLyapunov(DT)

SCALE = float(Ho.std())
BOUNDS = (float(train.min()), float(train.max()))
WarmNp, FutNp = EvalWindowsFromTrajectories(Ho, WARM, HORIZON, per_traj=2, seed=7)
Warm = torch.tensor(WarmNp, dtype=torch.float32)
Fut = torch.tensor(FutNp, dtype=torch.float32)
# per_traj=2 so the long-horizon pass uses the same number of windows as the
# 5 LT pass. The start indices still differ, because a 5,500 step future needs
# a much earlier start than a 550 step one.
WarmCNp, FutCNp = EvalWindowsFromTrajectories(Ho, WARM, CLIMATE, per_traj=2, seed=11)
WarmC = torch.tensor(WarmCNp, dtype=torch.float32)

EvalKw = dict(dt=DT, threshold=THRESH, scale=SCALE, noise_floor=FLOOR, bounds=BOUNDS)

OUT["data"] = {
    "n_obs": N, "k": K, "grid": [A, B], "dt": DT,
    "n_train_series": int(TrainSeries.shape[0]),
    "train_shape": list(TrainSeries.shape), "val_shape": list(ValSeries.shape),
    "test_shape": list(TestSeries.shape), "holdout_shape": list(Ho.shape),
    "pooled_train_steps": int(train.shape[0]),
    "noise_floor_mse": FLOOR, "scale": SCALE,
    "bounds": [BOUNDS[0], BOUNDS[1]],
    "steps_per_lyapunov": float(SPL),
    "horizon_steps": HORIZON, "horizon_LT": HORIZON / SPL,
    "climate_steps": CLIMATE, "climate_LT": CLIMATE / SPL,
    "warm": WARM, "unroll": UNROLL, "threshold": THRESH,
    "eval_windows": list(WarmNp.shape), "climate_windows": list(WarmCNp.shape),
}

# ---------------------------------------------------------------- params
Ref = DecoupledModel(N, K, A, B)
TARGET = CountParams(Ref)
WJ = MatchWidth(lambda w: JointAEGRU(N, latent=16, width=w), TARGET)
WS = MatchWidth(lambda w: JointAEGRUSigmoid(N, latent=16, width=w), TARGET)
WP = MatchWidth(lambda w: PhysicsLatentAE(N, width=w), TARGET)
WJ3 = MatchWidth(lambda w: JointAEGRU(N, latent=3, width=w), TARGET)

OUT["params"] = {
    "target": TARGET,
    "ours_total": CountParams(Ref),
    "ours_inference": CountInferenceParams(Ref),
    "per_module": {n: CountParams(m, trainable=False) for n, m in
                   [("E", Ref.E), ("f", Ref.F), ("m", Ref.M),
                    ("r", Ref.R), ("D", Ref.D), ("g", Ref.G)]},
    "joint_width": WJ, "joint_params": CountParams(JointAEGRU(N, 16, WJ)),
    "sigmoid_width": WS, "sigmoid_params": CountParams(JointAEGRUSigmoid(N, 16, WS)),
    "pinn_width": WP, "pinn_params": CountParams(PhysicsLatentAE(N, width=WP)),
    "joint3_width": WJ3,
}

# ---------------------------------------------------------------- loaders
CK_M = ROOT / "Checkpoints" / "multi_series"
CK_3 = ROOT / "Checkpoints" / "three_phase"


def load(build, path):
    mdl = build()
    mdl.load_state_dict(torch.load(path, weights_only=False, map_location=DEV)["state_dict"])
    return mdl.eval()


def agg(models, label):
    runs = [EvaluateModel(m, Warm, Fut, **EvalKw) for m in models]
    o = {"model": label, "n_seeds": len(runs)}
    for k in runs[0]:
        if k == "curve":
            o["curve"] = np.mean([r["curve"] for r in runs], axis=0).tolist()
        elif isinstance(runs[0][k], (int, float, bool)):
            v = [float(r[k]) for r in runs]
            o[k] = float(np.mean(v)); o[k + "_std"] = float(np.std(v))
    o["recon_over_floor"] = o["recon_mse"] / FLOOR
    return o


RES = {}
MODELS = {}          # keep the objects so the 50 LT pass can reuse them

# ours = three-phase, variant A
oursA = [load(lambda: DecoupledModel(N, K, A, B), CK_3 / f"phase3A_seed{s}.pt")
         for s in (0, 1, 2)]
RES["Ours (three-phase)"] = agg(oursA, "Ours (three-phase)")
MODELS["Ours (three-phase)"] = oursA

# phi sweep
PHIS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
MULTI = {0.1, 0.25, 0.5}
for phi in PHIS:
    tag = f"{phi:g}".replace(".", "_")
    seeds = (0, 1, 2) if phi in MULTI else (0,)
    ms = [load(lambda: JointAEGRU(N, 16, WJ), CK_M / f"joint_phi{tag}_seed{s}.pt")
          for s in seeds]
    RES[f"phi={phi:g}"] = agg(ms, f"phi={phi:g}")
    MODELS[f"phi={phi:g}"] = ms

# sigmoid control
sig = [load(lambda: JointAEGRUSigmoid(N, 16, WS), CK_M / f"sigmoid_phi0_5_seed{s}.pt")
       for s in (0, 1, 2)]
RES["AEGRU+sigmoid"] = agg(sig, "AEGRU+sigmoid")
MODELS["AEGRU+sigmoid"] = sig

# pinn
def build_pinn(rho):
    def f():
        p = PhysicsLatentAE(N, width=WP, dt=DT, rho=rho)
        return p
    return f


pinn = [load(build_pinn(28.0), CK_M / f"pinn_rho28_seed{s}.pt") for s in (0, 1, 2)]
RES["PINN (true physics)"] = agg(pinn, "PINN (true physics)")
MODELS["PINN (true physics)"] = pinn
pinn26 = [load(build_pinn(26.0), CK_M / "pinn_rho26_seed0.pt")]
RES["PINN (rho=26)"] = agg(pinn26, "PINN (rho=26)")
MODELS["PINN (rho=26)"] = pinn26

# schedule ablation: the same six modules trained end to end in one stage
# instead of the three phase schedule. Isolates the schedule from the
# architecture, since the wiring and the parameter count are identical.
oursjoint = [load(lambda: DecoupledModel(N, K, A, B), CK_M / "oursjoint_seed0.pt")]
RES["Ours (one-stage)"] = agg(oursjoint, "Ours (one-stage)")
MODELS["Ours (one-stage)"] = oursjoint

# capacity-matched diagnostic
diag3 = [load(lambda: JointAEGRU(N, 3, WJ3), CK_M / "diag_latent3_phi0_seed0.pt")]
RES["phi=0, latent=3"] = agg(diag3, "phi=0, latent=3")
MODELS["phi=0, latent=3"] = diag3

# trivial references
for name, pred in [("persistence", PersistenceRollout(WarmNp, HORIZON)),
                   ("climatology", MeanRollout(WarmNp, HORIZON, train.mean(0)))]:
    fm = ForecastMetrics(FutNp, pred, DT, THRESH, SCALE, BOUNDS)
    RES[name] = {"model": name, "n_seeds": 0,
                 **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in fm.items()}}

OUT["results"] = RES

# ---------------------------------------------------------------- climate / attractor
# Long horizon: free run for CLIMATE steps (about 50 Lyapunov times) and ask
# whether the model still lives on the right attractor. VPT asks whether it
# tracks one trajectory, which is a different and much shorter-lived question.
lo, hi = BOUNDS
CLIM = {}
for label, mdls in MODELS.items():
    with torch.no_grad():
        pred = mdls[0].Rollout(WarmC, CLIMATE).cpu().numpy()
    st = AttractorStats(FutCNp, pred)
    finite = np.isfinite(pred).all(axis=(1, 2))
    inrange = ((pred >= lo) & (pred <= hi)).all(axis=(1, 2))
    CLIM[label] = {k: float(v) for k, v in st.items()}
    CLIM[label]["divergence_50LT"] = float(1.0 - (finite & inrange).mean())
    CLIM[label]["nonfinite_frac"] = float(1.0 - finite.mean())

for label, pred in [("persistence", PersistenceRollout(WarmCNp, CLIMATE)),
                    ("climatology", MeanRollout(WarmCNp, CLIMATE, train.mean(0)))]:
    st = AttractorStats(FutCNp, np.asarray(pred))
    CLIM[label] = {k: float(v) for k, v in st.items()}
    CLIM[label]["divergence_50LT"] = 0.0
    CLIM[label]["nonfinite_frac"] = 0.0

OUT["climate_50LT"] = CLIM

# ---------------------------------------------------------------- lead-time profile
# Everything above is a summary. This is the profile: how the forecast decays
# with lead time, on four axes that fail in different ways.
LEADS_LT = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
LEAD_IDX = [min(int(round(t * SPL)), CLIMATE - 1) for t in LEADS_LT]
CLIM_MEAN = train.mean(0)
ACC_THRESH = 0.6


def Profile(pred, truth, persist):
    """MAE, anomaly correlation, skill against persistence, and variance ratio."""
    out = {"mae": [], "acc": [], "ss_persist": [], "var_ratio": []}
    for i in LEAD_IDX:
        p, t, q = pred[:, i, :], truth[:, i, :], persist[:, i, :]
        out["mae"].append(float(np.abs(p - t).mean()))
        ap, at = p - CLIM_MEAN, t - CLIM_MEAN
        den = np.sqrt((ap ** 2).sum() * (at ** 2).sum())
        out["acc"].append(float((ap * at).sum() / den) if den > 0 else 0.0)
        mse_m = float(((p - t) ** 2).mean())
        mse_p = float(((q - t) ** 2).mean())
        out["ss_persist"].append(float(1.0 - mse_m / mse_p) if mse_p > 0 else 0.0)
        out["var_ratio"].append(float(p.std() / t.std()) if t.std() > 0 else 0.0)
    return out


def AccHorizon(pred, truth, step=10):
    """First lead time where anomaly correlation drops below 0.6, in Lyapunov times."""
    for i in range(0, CLIMATE, step):
        ap, at = pred[:, i, :] - CLIM_MEAN, truth[:, i, :] - CLIM_MEAN
        den = np.sqrt((ap ** 2).sum() * (at ** 2).sum())
        acc = float((ap * at).sum() / den) if den > 0 else 0.0
        if acc < ACC_THRESH:
            return i / SPL
    return CLIMATE / SPL


PERSIST_C = np.asarray(PersistenceRollout(WarmCNp, CLIMATE))
PROF = {}
for label, mdls in MODELS.items():
    with torch.no_grad():
        pred = mdls[0].Rollout(WarmC, CLIMATE).cpu().numpy()
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    PROF[label] = Profile(pred, FutCNp, PERSIST_C)
    PROF[label]["acc_horizon_LT"] = AccHorizon(pred, FutCNp)

for label, pred in [("persistence", PERSIST_C),
                    ("climatology", np.asarray(MeanRollout(WarmCNp, CLIMATE, CLIM_MEAN)))]:
    PROF[label] = Profile(pred, FutCNp, PERSIST_C)
    PROF[label]["acc_horizon_LT"] = AccHorizon(pred, FutCNp)

OUT["profile"] = {"leads_LT": LEADS_LT, "lead_idx": LEAD_IDX,
                  "acc_threshold": ACC_THRESH, "by_model": PROF}

# boundedness of ours
with torch.no_grad():
    _, bs, cs = oursA[0].RolloutLatents(WarmC, CLIMATE)
csn, bsn = cs.cpu().numpy(), bs.cpu().numpy()
OUT["boundedness"] = {
    "C_outside_01": float(BoundViolation(csn)),
    "C_saturated": float(Saturation(csn)),
    "C_min": float(csn.min()), "C_max": float(csn.max()),
    "b_absmax": float(np.abs(bsn).max()),
    "steps": CLIMATE, "n_rollouts": int(WarmC.shape[0]),
}

# ---------------------------------------------------------------- capacity bound + passthrough
def passthrough(fn, eps=1e-3, reps=8, seed=0):
    """Fraction of a random input perturbation that survives to the output.

    Seeded, so the number is reproducible. Unseeded it drifts by about 0.001
    between runs, which is harmless for the argument but makes the paper's
    numbers impossible to check against a rerun.
    """
    g = torch.Generator().manual_seed(seed)
    rs = []
    with torch.no_grad():
        base = fn(Warm)
        for _ in range(reps):
            d = torch.randn(Warm.shape, generator=g) * eps
            rs.append((((fn(Warm + d) - base) ** 2).sum() / (d ** 2).sum()).item())
    return float(np.mean(rs))


# optimal linear rank-d reconstruction of the eval windows (PCA fit on train)
Xe = WarmNp.reshape(-1, N)
mu = train.mean(0)
_, _, Vt = np.linalg.svd(train - mu, full_matrices=False)
pca = {}
for d in range(1, N + 1):
    P = Vt[:d]
    Xr = (Xe - mu) @ P.T @ P + mu
    pca[d] = float(((Xr - Xe) ** 2).mean()) / FLOOR
OUT["pca_rank_over_floor"] = pca
OUT["capacity_bound_formula"] = {str(d): (N - d) / N for d in (3, 16)}

pt = {}
pt["Ours (three-phase)"] = {"d": K, "passthrough": passthrough(lambda t: oursA[0].Reconstruct(t))}
for phi, tag in [(0.0, "joint_phi0_seed0"), (0.5, "joint_phi0_5_seed0"), (1.0, "joint_phi1_seed0")]:
    j = load(lambda: JointAEGRU(N, 16, WJ), CK_M / f"{tag}.pt")
    pt[f"phi={phi:g}"] = {"d": 16,
                          "passthrough": passthrough((lambda jj: (lambda t: jj.Decode(jj.Encode(t))))(j))}
OUT["passthrough"] = pt

# ---------------------------------------------------------------- training schedule
OUT["schedule"] = {"epochs_per_phase": 60, "seeds": [0, 1, 2],
                   "threads": 8, "batch_p1": 4096, "batch_p2": 256, "batch_p3": 512,
                   "lr": 1e-3, "seq_stride": 4}
for ph in ("phase1", "phase2", "phase3A"):
    h = torch.load(CK_3 / f"{ph}_seed0.pt", weights_only=False, map_location=DEV)["hist"]
    OUT["schedule"][ph] = {"epochs_done": h["epochs_done"],
                           "total_time_s": h["total_time"],
                           "marks": sorted(h.get("marks", {}).keys())}

P = "C:/Users/ARPITG~1/AppData/Local/Temp/claude/c--Users-ArpitGoel-Documents-GitHub-Bounded-and-unbounded-latent-decoupling-for-forecast-and-reconstruction-quality/cdd20b34-1bb7-4cf8-a41f-3ee2fc81363c/scratchpad/paper_numbers.json"
io.open(P, "w", encoding="utf-8").write(json.dumps(OUT, indent=1))
print("wrote", P)
print("models evaluated:", len(RES))

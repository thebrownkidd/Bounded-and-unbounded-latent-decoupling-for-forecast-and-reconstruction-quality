"""
Multi-system experiment: bounded/unbounded latent decoupling on four new systems.

    Lorenz-96 N=8   Rössler   Kuramoto-Sivashinsky L=22   Thomas

For each system we train:
    Ours (three-phase)  x3 seeds
    Joint AE+GRU (φ sweep)  phi in [0.0, 0.1, 0.25, 0.5, 1.0]  x1 seed each

All models get matched parameters and 60 epochs per parameter group, mirroring
the Lorenz-63 protocol in Experimentation/ThreePhaseMapVariants.ipynb.

Usage
-----
    # all systems (from repo root, venv active):
    python Experimentation/run_multisystem.py

    # single system:
    python Experimentation/run_multisystem.py --system rossler

    # smoke test (5 epochs, 1 seed, 2 phi values):
    python Experimentation/run_multisystem.py --quick

Results are written to Paper/multisystem_numbers.json, one entry per system.
Trained model weights are cached under Checkpoints/multisystem/<system>/ so a
re-run picks up from where it left off.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from Src import DecoupledModel
from Comp import JointAEGRU
from SyntheticGenerators.MultiSystemLift import (
    SYSTEM_META, load_dataset, save_dataset
)
from Utils.Checkpoints import TrainOrLoad
from Utils.Metrics import (NRMSE, PerHorizonNRMSE, VPT, DivergenceRate, ToLyapunov)
from Utils.Rollout import (EvaluateModel, ForecastMetrics, MeanRollout,
                           EvalWindowsFromTrajectories, MakeSeriesWindows)
from Utils.Benchmark import (CountParams, CountInferenceParams, MatchWidth,
                             EpochTrain, PinThreads, SeedAll)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--system", nargs="+",
                    default=["lorenz96_n8", "rossler", "ks_l22", "thomas"])
parser.add_argument("--quick", action="store_true")
parser.add_argument("--ours-only", action="store_true",
                    help="Skip baseline training; load existing baseline "
                         "checkpoints for evaluation only.")
args = parser.parse_args()

QUICK   = args.quick
EPOCHS  = 5 if QUICK else 60
MARKS   = (1, 2, 5) if QUICK else (5, 10, 20, 40, 60)
SEEDS   = [0] if QUICK else [0, 1, 2]
PHIS    = [0.0, 0.5] if QUICK else [0.0, 0.1, 0.25, 0.5, 1.0]

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
PinThreads(8)

DATA_DIR  = ROOT / "Data"
CKPT_BASE = ROOT / "Checkpoints" / "multisystem"
OUT_PATH  = ROOT / "Paper" / (
    "multisystem_numbers_unbounded.json" if args.ours_only
    else "multisystem_numbers.json"
)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_BASE.mkdir(parents=True, exist_ok=True)

LR          = 1e-3
SEQ_STRIDE  = 4
W_CONSIST   = 1.0
MseT        = nn.MSELoss()

print(f"device={DEVICE}  epochs={EPOCHS}  seeds={SEEDS}  phis={PHIS}")


# ---------------------------------------------------------------------------
# Helpers shared across systems
# ---------------------------------------------------------------------------

def set_trainable(model, modules):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Three-phase training
# ---------------------------------------------------------------------------

def phase1_epoch(model, opt, X, batch):
    perm = torch.randperm(X.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), batch):
        x = X[perm[i:i + batch]]
        loss = MseT(model.D(model.R(model.E(x))), x)
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss.item(); nb += 1
    return {"recon": s / max(nb, 1)}


def phase2_epoch(model, opt, Hwarm, Cfut, k, unroll, batch, a, b_dim):
    perm = torch.randperm(Hwarm.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), batch):
        j = perm[i:i + batch]
        h, Ctgt = Hwarm[j], Cfut[j]
        Bn, T, n = h.shape
        bseq = model.F(h.reshape(Bn * T, n)).view(Bn, T, k)
        _, st = model.G(bseq[:, :-1], None)
        cur, out = bseq[:, -1], []
        for _ in range(unroll):
            cur, st = model.G.Step(cur, st)
            out.append(cur)
        Bt = torch.stack(out, 1)
        Chat = model.M(Bt.reshape(-1, k)).view(Bn, unroll, a, b_dim)
        loss = MseT(Chat, Ctgt)
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss.item(); nb += 1
    return {"c_fcst": s / max(nb, 1)}


def phase3_epoch(model, opt, bT, CT, Xseq, k, a, b_dim, batch):
    perm = torch.randperm(bT.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), batch):
        j = perm[i:i + batch]
        b, Ctgt, xseq = bT[j], CT[j], Xseq[j]
        Bn, T = b.shape[:2]
        Chat = model.M(b.reshape(Bn * T, k)).view(Bn, T, a, b_dim)
        N_obs = xseq.shape[-1]
        Xhat = model.D(Chat.reshape(Bn * T, a, b_dim)).view(Bn, T, N_obs)
        loss = MseT(Chat, Ctgt) + MseT(Xhat, xseq)
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss.item(); nb += 1
    return {"p3_loss": s / max(nb, 1)}


@torch.no_grad()
def val_scores(model, XvalWarm, XvalFut, unroll):
    model.eval()
    out = {
        "val_recon": MseT(model.Reconstruct(XvalWarm), XvalWarm).item(),
        "val_fcst":  MseT(model.Rollout(XvalWarm, unroll), XvalFut).item(),
    }
    model.train()
    return out


def train_three_phase(seed, cfg, XTrain, XvalWarm, XvalFut, TrainSeries,
                      ckpt_dir):
    """Train one seed of the three-phase decoupled model."""
    N, K, A, B_DIM = cfg["N"], cfg["K"], cfg["A"], cfg["B"]
    WARM, UNROLL = cfg["WARM"], cfg["UNROLL"]

    def build():
        SeedAll(seed)
        return DecoupledModel(N, K, A, B_DIM).to(DEVICE)

    def train(model):
        t0 = time.perf_counter()

        # ---- Phase 1: E, R, D ----------------------------------------
        params = set_trainable(model, [model.E, model.R, model.D])
        opt = torch.optim.Adam(params, lr=LR)
        h1 = EpochTrain(
            lambda: phase1_epoch(model, opt, XTrain, batch=4096),
            EvalFn=lambda: {"val_recon": MseT(
                model.D(model.R(model.E(XTrain[:256]))),
                XTrain[:256]).item()},
            epochs=EPOCHS, marks=MARKS, name=f"p1 seed{seed}", verbose=False)

        # ---- Precompute targets for phase 2 -------------------------
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            # h over warm window, C over forecast tail -- from training series
            XFullNp = MakeSeriesWindows(TrainSeries, WARM + UNROLL,
                                        stride=SEQ_STRIDE)
            XFull = torch.tensor(XFullNp, dtype=torch.float32, device=DEVICE)
            Bn, T, n = XFull.shape
            h_all = model.E(XFull.reshape(Bn * T, n))
            Hwarm = h_all.view(Bn, T, n)[:, :WARM].contiguous()
            Cfut = model.R(h_all).view(Bn, T, A, B_DIM)[:, WARM:].contiguous()

        # ---- Phase 2: F, G, M ----------------------------------------
        params = set_trainable(model, [model.F, model.G, model.M])
        opt = torch.optim.Adam(params, lr=LR)
        h2 = EpochTrain(
            lambda: phase2_epoch(model, opt, Hwarm, Cfut,
                                 K, UNROLL, 1024, A, B_DIM),
            EvalFn=lambda: val_scores(model, XvalWarm, XvalFut, UNROLL),
            epochs=EPOCHS, marks=MARKS, name=f"p2 seed{seed}", verbose=False)

        # ---- Precompute targets for phase 3 -------------------------
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            bT = model.F(model.E(XFull.reshape(Bn * T, n))).view(Bn, T, K)
            CT = model.R(model.E(XFull.reshape(Bn * T, n))).view(Bn, T, A, B_DIM)

        # ---- Phase 3: M, D -------------------------------------------
        params = set_trainable(model, [model.M, model.D])
        opt = torch.optim.Adam(params, lr=LR)
        h3 = EpochTrain(
            lambda: phase3_epoch(model, opt, bT, CT, XFull,
                                 K, A, B_DIM, 512),
            EvalFn=lambda: val_scores(model, XvalWarm, XvalFut, UNROLL),
            epochs=EPOCHS, marks=MARKS, name=f"p3 seed{seed}", verbose=False)

        # restore all gradients for eval
        for p in model.parameters():
            p.requires_grad_(True)

        return {"stageA": h1, "stageB": h2, "stageC": h3,
                "total_time": time.perf_counter() - t0}

    return TrainOrLoad(ckpt_dir / f"ours_unbounded_seed{seed}.pt", build, train, DEVICE)


# ---------------------------------------------------------------------------
# Joint AE+GRU baseline
# ---------------------------------------------------------------------------

def joint_epoch(model, opt, XFull, warm, batch):
    perm = torch.randperm(XFull.shape[0])
    s, nb = 0.0, 0
    for i in range(0, len(perm), batch):
        rec, fcst = model.Losses(XFull[perm[i:i + batch]], warm)
        loss = (1.0 - model._phi) * rec + model._phi * fcst
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss.item(); nb += 1
    return {"loss": s / max(nb, 1)}


def train_joint(phi, seed, cfg, XvalWarm, XvalFut, TrainSeries, ckpt_dir):
    N, WARM, UNROLL = cfg["N"], cfg["WARM"], cfg["UNROLL"]
    TARGET = cfg["TARGET"]

    width = MatchWidth(lambda w: JointAEGRU(N, latent=16, width=w), TARGET)

    def build():
        SeedAll(seed)
        m = JointAEGRU(N, latent=16, width=width).to(DEVICE)
        m._phi = phi
        return m

    def train(model):
        XFullNp = MakeSeriesWindows(TrainSeries, WARM + UNROLL, stride=SEQ_STRIDE)
        XFull = torch.tensor(XFullNp, dtype=torch.float32, device=DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        return EpochTrain(
            lambda: joint_epoch(model, opt, XFull, WARM, batch=256),
            EvalFn=lambda: val_scores(model, XvalWarm, XvalFut, UNROLL),
            epochs=EPOCHS, marks=MARKS,
            name=f"phi={phi:g} seed{seed}", verbose=False)

    tag = f"{phi:g}".replace(".", "_")
    return TrainOrLoad(ckpt_dir / f"joint_phi{tag}_seed{seed}.pt",
                       build, train, DEVICE)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def long_horizon_profile(model, WarmNp, FutNp, meta, scale, bounds):
    """MAE @ 1 LT, ACC horizon, variance ratio @ 50 LT."""
    WARM = WarmNp.shape[1]
    H = FutNp.shape[1]
    WarmT = torch.tensor(WarmNp, dtype=torch.float32, device=DEVICE)

    pred = model.Rollout(WarmT, H).cpu().numpy()

    slt = meta["steps_per_lt"]
    dt = meta["dt"]
    lam = meta["lam_max"]

    # MAE @ 1 LT
    h1 = min(slt, H)
    mae_1lt = float(np.abs(FutNp[:, :h1] - pred[:, :h1]).mean())

    # NRMSE per lead time
    curve = PerHorizonNRMSE(FutNp, pred, scale)

    # ACC horizon: first lead where ACC drops below 0.6
    clim = FutNp.mean(axis=(0, 2), keepdims=True)
    anom_t = FutNp - clim
    anom_p = pred - clim
    num = (anom_t * anom_p).mean(axis=(0, 2))
    den = np.sqrt((anom_t ** 2).mean(axis=(0, 2)) *
                  (anom_p ** 2).mean(axis=(0, 2)) + 1e-12)
    acc_curve = num / den
    below = np.nonzero(acc_curve < 0.6)[0]
    acc_hor_steps = int(below[0]) if len(below) else H
    acc_hor_lt = acc_hor_steps * dt * lam

    # Variance ratio at 50 LT (last 500 steps of CLIMATE)
    var_true = FutNp[:, -min(500, H):].std()
    var_pred = pred[:, -min(500, H):].std()
    var_ratio = float(var_pred / (var_true + 1e-12))

    return {
        "mae_1lt":      mae_1lt,
        "acc_horizon":  acc_hor_lt,
        "var_ratio_50lt": var_ratio,
        "curve":        curve.tolist(),
    }


def evaluate_all(models_dict, Warm, Fut, WarmLong, FutLong, meta, scale,
                 noise_floor, bounds):
    """Run EvaluateModel + long_horizon_profile on every model in models_dict.

    EvaluateModel's VPT defaults to Lorenz-63 lam_max, so we recompute it from
    the per-horizon NRMSE curve using the system's own lambda.
    """
    dt = meta["dt"]
    lam = meta["lam_max"]
    EvalKw = dict(dt=dt, threshold=0.4, scale=scale,
                  noise_floor=noise_floor, bounds=bounds)

    results = {}
    for label, model_list in tqdm(models_dict.items(), desc="evaluating"):
        runs = []
        for m in model_list:
            m.eval()
            r = EvaluateModel(m, Warm, Fut, **EvalKw)
            # Recompute VPT with the system's own Lyapunov exponent
            curve = np.asarray(r["curve"])
            over = np.nonzero(curve > 0.4)[0]
            steps = int(over[0]) if len(over) else len(curve)
            r["vpt"] = float(ToLyapunov(steps, dt, lam))
            r["vpt_censored"] = len(over) == 0
            lh = long_horizon_profile(m, WarmLong.cpu().numpy(),
                                       FutLong.cpu().numpy(), meta, scale, bounds)
            r.update(lh)
            runs.append(r)

        agg = {"model": label, "n_seeds": len(runs)}
        for k in runs[0]:
            if k == "curve":
                continue
            if isinstance(runs[0][k], (int, float, bool)):
                vals = [float(r[k]) for r in runs]
                agg[k] = float(np.mean(vals))
                agg[k + "_std"] = float(np.std(vals))
        results[label] = agg
    return results


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

all_results = {}

for SYSTEM in args.system:
    meta = SYSTEM_META[SYSTEM]
    print(f"\n{'='*60}")
    print(f"System: {meta['name']}")
    print(f"  n={meta['n_obs']}, k={meta['k']}, dt={meta['dt']}, "
          f"lam_max={meta['lam_max']}, steps/LT={meta['steps_per_lt']}")
    print(f"  horizon={meta['horizon']} ({meta['horizon']/meta['steps_per_lt']:.1f} LT), "
          f"climate={meta['climate']} ({meta['climate']/meta['steps_per_lt']:.1f} LT)")
    print(f"{'='*60}")

    ckpt_dir = CKPT_BASE / SYSTEM
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / f"{SYSTEM}.npz"

    # ---- Data ----------------------------------------------------------------
    if data_path.exists():
        print(f"Loading dataset from {data_path.name} ...")
        d = load_dataset(data_path)
    else:
        print(f"Generating dataset (this may take a few minutes for KS) ...")
        d = save_dataset(SYSTEM, path=str(data_path), verbose=True)

    TrainSeries = d["train"]   # (n_series, T_train, n_obs)
    ValSeries   = d["val"]
    HoldoutObs  = d["holdout_obs"]
    NOISE_FLOOR = d["noise_floor_mse"]

    N_OBS    = TrainSeries.shape[-1]
    N_SERIES = TrainSeries.shape[0]
    K        = meta["k"]
    A, B_DIM = 8, 8          # fixed carrier size
    WARM     = meta["warm"]
    UNROLL   = meta["unroll"]
    HORIZON  = meta["horizon"]
    CLIMATE  = meta["climate"]

    train_pool = TrainSeries.reshape(-1, N_OBS)
    SCALE  = float(HoldoutObs.std())
    BOUNDS = (float(train_pool.min()), float(train_pool.max()))

    print(f"  train {TrainSeries.shape}  holdout {HoldoutObs.shape}")
    print(f"  noise_floor_mse={NOISE_FLOOR:.6f}  scale={SCALE:.4f}")

    # Move to device
    XTrain  = torch.tensor(train_pool, dtype=torch.float32, device=DEVICE)

    val_pool = ValSeries.reshape(-1, N_OBS)
    XValFull_np = MakeSeriesWindows(ValSeries, WARM + UNROLL, stride=64,
                                    max_per_series=8)
    XValFull = torch.tensor(XValFull_np, dtype=torch.float32, device=DEVICE)
    XvalWarm, XvalFut = XValFull[:, :WARM], XValFull[:, WARM:]

    # Eval windows (short horizon)
    WarmNp, FutNp = EvalWindowsFromTrajectories(
        HoldoutObs, WARM, HORIZON, per_traj=2, seed=7)
    Warm = torch.tensor(WarmNp, dtype=torch.float32, device=DEVICE)
    Fut  = torch.tensor(FutNp,  dtype=torch.float32, device=DEVICE)

    # Long-horizon windows
    WarmLongNp, FutLongNp = EvalWindowsFromTrajectories(
        HoldoutObs, WARM, CLIMATE, per_traj=1, seed=11)
    WarmLong = torch.tensor(WarmLongNp, dtype=torch.float32, device=DEVICE)
    FutLong  = torch.tensor(FutLongNp,  dtype=torch.float32, device=DEVICE)

    # ---- Parameter budget ----------------------------------------------------
    SeedAll(0)
    ref = DecoupledModel(N_OBS, K, A, B_DIM)
    TARGET = CountParams(ref)

    # Baseline parameter budget: baselines were matched to the ORIGINAL A=B=8
    # model (not the kscaled one), so record that separately for loading.
    ref_orig = DecoupledModel(N_OBS, K, 8, 8)
    TARGET_ORIG = CountParams(ref_orig)

    cfg = dict(N=N_OBS, K=K, A=A, B=B_DIM, WARM=WARM, UNROLL=UNROLL,
               TARGET=TARGET, TARGET_ORIG=TARGET_ORIG)

    print(f"  TARGET params: {TARGET:,}  "
          f"(inference: {CountInferenceParams(ref):,})")

    # ---- Train Ours (three-phase) --------------------------------------------
    print(f"\n--- Ours (three-phase), {len(SEEDS)} seeds ---")
    ours_models = []
    for seed in SEEDS:
        print(f"  seed {seed} ...", end=" ", flush=True)
        m, h = train_three_phase(seed, cfg, XTrain, XvalWarm, XvalFut,
                                 TrainSeries, ckpt_dir)
        ours_models.append(m)
        if "total_time" in h:
            print(f"done ({h['total_time']:.0f}s)")
        else:
            print("loaded from checkpoint")

    # ---- Train Joint baselines (skipped when --ours-only) --------------------
    phi_models = {}
    if not args.ours_only:
        print(f"\n--- Joint AE+GRU phi sweep, 1 seed each ---")
        for phi in PHIS:
            print(f"  phi={phi:g} ...", end=" ", flush=True)
            m, _ = train_joint(phi, seed=0, cfg=cfg, XvalWarm=XvalWarm,
                               XvalFut=XvalFut, TrainSeries=TrainSeries,
                               ckpt_dir=ckpt_dir)
            phi_models[f"phi={phi:g}"] = [m]
            print("done/loaded")
    else:
        # Load existing baseline checkpoints for evaluation.
        # Baselines were width-matched to TARGET_ORIG (A=8,B=8), not the
        # kscaled model, so use the original target to reconstruct the right width.
        print(f"\n--- Loading existing Joint AE+GRU baselines (--ours-only) ---")
        N_baseline = cfg["N"]
        width = MatchWidth(lambda w: JointAEGRU(N_baseline, latent=16, width=w),
                           cfg["TARGET_ORIG"])
        for phi in PHIS:
            tag = f"{phi:g}".replace(".", "_")
            ckpt = ckpt_dir / f"joint_phi{tag}_seed0.pt"
            if ckpt.exists():
                SeedAll(0)
                m = JointAEGRU(N_baseline, latent=16, width=width).to(DEVICE)
                m._phi = phi
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["state_dict"])
                m.eval()
                phi_models[f"phi={phi:g}"] = [m]
                print(f"  phi={phi:g} ... loaded")
            else:
                print(f"  phi={phi:g} ... checkpoint not found, skipping")

    # ---- Evaluate ------------------------------------------------------------
    print(f"\n--- Evaluating ---")
    models_dict = {"Ours (three-phase)": ours_models}
    models_dict.update(phi_models)

    # Add reference baselines (no model, pure numpy)
    pers_pred = np.repeat(WarmNp[:, -1:, :], HORIZON, axis=1)
    mean_pred = np.broadcast_to(train_pool.mean(0), (WarmNp.shape[0], HORIZON, N_OBS)).copy()
    curve_pers = PerHorizonNRMSE(FutNp, pers_pred, SCALE)
    curve_clim = PerHorizonNRMSE(FutNp, mean_pred, SCALE)

    results = evaluate_all(models_dict, Warm, Fut, WarmLong, FutLong,
                           meta, SCALE, NOISE_FLOOR, BOUNDS)

    # Add trivial references
    def _vpt(curve):
        dt, lam = meta["dt"], meta["lam_max"]
        over = np.nonzero(np.asarray(curve) > 0.4)[0]
        steps = int(over[0]) if len(over) else len(curve)
        return float(ToLyapunov(steps, dt, lam))

    results["persistence"] = {
        "model": "persistence",
        "vpt": _vpt(curve_pers),
    }
    results["climatology"] = {
        "model": "climatology",
        "vpt": _vpt(curve_clim),
    }

    all_results[SYSTEM] = {
        "meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float))},
        "noise_floor_mse": NOISE_FLOOR,
        "target_params": TARGET,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "phis": PHIS,
        "results": {k: {ek: ev for ek, ev in v.items() if ek != "curve"}
                    for k, v in results.items()},
    }

    # Print summary table
    print(f"\n{'model':<30} {'VPT(LT)':>7} {'std':>5} {'recon*fl':>9} {'diverg':>7}")
    print("-" * 60)
    for label, r in results.items():
        vpt = r.get("vpt", float("nan"))
        vpt_std = r.get("vpt_std", 0.0)
        rxfl = r.get("recon_mse_over_floor", float("nan"))
        div  = r.get("divergence", float("nan"))
        print(f"  {label:<28} {vpt:6.3f} {vpt_std:5.3f} {rxfl:9.3f} {div:7.3f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults written to {OUT_PATH}")

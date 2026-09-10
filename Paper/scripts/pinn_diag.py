"""Is the PINN limited by its dynamics or by its state estimate?

The dynamics are exact (RK4 on the true Lorenz field, zero learned forecast
parameters). So if it forecasts badly, the only remaining suspect is the
initial condition it starts from, which comes from a learned encoder reading
noisy observations.
"""
import sys
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT))
import numpy as np, torch

from Src import DecoupledModel
from Comp import PhysicsLatentAE
from Comp.LorenzField import RK4Step, SIGMA, RHO, BETA
from Utils import LoadLorenzMultiSeries
from Utils.Rollout import EvalWindowsFromTrajectories
from Utils.Benchmark import CountParams, MatchWidth
from Utils.Metrics import StepsPerLyapunov

torch.set_num_threads(8)
M = LoadLorenzMultiSeries(ROOT / "Data" / "LorenzLiftMulti.npz")
Ho, HoS = M["holdout_obs"], M["holdout_states"]
WARM, HOR, DT = 64, 550, 0.01
SPL = StepsPerLyapunov(DT)

# identical seed and identical (N, T), so the start indices match exactly
Wo, Fo = EvalWindowsFromTrajectories(Ho, WARM, HOR, per_traj=2, seed=7)
Ws, Fs = EvalWindowsFromTrajectories(HoS, WARM, HOR, per_traj=2, seed=7)
print(f"aligned windows: obs {Wo.shape} states {Ws.shape}")

TARGET = CountParams(DecoupledModel(30, 3, 8, 8))
WP = MatchWidth(lambda w: PhysicsLatentAE(30, width=w), TARGET)
P = PhysicsLatentAE(30, width=WP, dt=DT, rho=28.0)
P.load_state_dict(torch.load(ROOT / "Checkpoints/multi_series/pinn_rho28_seed0.pt",
                             weights_only=False, map_location="cpu")["state_dict"])
P.eval()

with torch.no_grad():
    s_hat = P.ToPhysical(P.Encode(torch.tensor(Wo, dtype=torch.float32))[:, -1]).numpy()
s_true = Ws[:, -1]                      # true Lorenz state at the same instant

err = np.linalg.norm(s_hat - s_true, axis=1)
scale = HoS.reshape(-1, 3).std(0)
print(f"\nSTATE ESTIMATE at rollout start (encoder output vs truth)")
print(f"  mean |error|        {err.mean():.3f}")
print(f"  attractor scale     {np.linalg.norm(scale):.3f}")
print(f"  relative error      {err.mean()/np.linalg.norm(scale):.1%}")


def integrate(s0, steps):
    s = torch.tensor(s0, dtype=torch.float64)
    out = []
    for _ in range(steps):
        s = RK4Step(s, DT, SIGMA, RHO, BETA)
        out.append(s.clone())
    return torch.stack(out, 1).numpy()


roll_true = integrate(s_true, HOR)      # perfect dynamics, perfect initial state
roll_hat = integrate(s_hat, HOR)        # perfect dynamics, the PINN's initial state

d_true = np.linalg.norm(roll_true - Fs, axis=2)
d_hat = np.linalg.norm(roll_hat - Fs, axis=2)
thr = 0.4 * np.linalg.norm(scale)


def first_cross(d):
    v = []
    for row in d:
        idx = np.nonzero(row > thr)[0]
        v.append((idx[0] if len(idx) else len(row)) / SPL)
    return float(np.mean(v))


print(f"\nROLLOUT IN STATE SPACE, both using the exact RK4 integrator")
print(f"  from the TRUE state      VPT {first_cross(d_true):.3f} LT   "
      f"(ceiling for perfect dynamics)")
print(f"  from the PINN's estimate VPT {first_cross(d_hat):.3f} LT")
print(f"  PINN's reported VPT on observations: 0.405 LT")

print(f"\nERROR GROWTH from the PINN's initial state (Lyapunov times -> mean |error|)")
for lt in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0):
    i = min(int(lt * SPL), HOR - 1)
    print(f"  t = {lt:3.1f} LT   {d_hat[:, i].mean():8.3f}"
          f"   ({d_hat[:, i].mean()/np.linalg.norm(scale):6.1%} of attractor)")

"""Compute-budget instrumentation: parameter matching, wall-clock training,
FLOP accounting and rollout timing.

The comparison is run under two constraints at once -- equal trainable
parameters and equal wall-clock seconds -- because either alone is easy to
game. Matched parameters without matched time rewards whichever architecture
happens to be cheapest per parameter; matched time without matched parameters
rewards whichever is smallest.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import clear_output


# -- Parameters ---------------------------------------------------------------

def CountParams(model, trainable=True):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable)


def CountInferenceParams(model):
    """Parameters that actually run at test time.

    DecoupledModel carries LatentBounded, a training-only teacher worth ~26% of
    its Stage-A parameters and dead weight at inference. Counting it in a speed
    table would understate the model; counting it out of a *training* budget
    would overstate it. Hence two numbers.
    """
    if hasattr(model, "InferenceParams"):
        return sum(p.numel() for p in model.InferenceParams())
    return CountParams(model)


def MatchWidth(Build, target, lo=8, hi=1024):
    """Integer hidden width whose model comes closest to `target` params.

    Build(width) -> nn.Module. Parameter count is monotone in width, so a
    bisection is exact; the two neighbours are compared at the end because the
    count jumps in steps of a few hundred and the crossing point is rarely the
    nearest.
    """
    a, b = lo, hi
    while a < b:
        mid = (a + b) // 2
        if CountParams(Build(mid)) < target:
            a = mid + 1
        else:
            b = mid
    cands = [w for w in (a - 1, a) if w >= lo]
    return min(cands, key=lambda w: abs(CountParams(Build(w)) - target))


def CountFlops(model, seq_len=1):
    """Approximate multiply-accumulate FLOPs for one sample's forward pass.

    Linear and GRU only, which is everything in this comparison. Counts 2 FLOPs
    per MAC and ignores activations, so treat it as an architecture-cost
    ranking rather than a hardware prediction.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features
        elif isinstance(m, nn.GRU):
            for layer in range(m.num_layers):
                isz = m.input_size if layer == 0 else m.hidden_size
                total += seq_len * 6 * m.hidden_size * (isz + m.hidden_size)
    return total


# -- Wall-clock training ------------------------------------------------------

def TimedTrain(EpochFn, EvalFn=None, budget_s=180.0, max_epochs=100000,
               log_every=1, name="train", PlotFn=None, plot_every=10,
               verbose=True):
    """Run EpochFn until the wall-clock budget is spent.

    EpochFn(deadline) -> dict of training stats, and is expected to check the
    deadline inside its own batch loop and return early. Without that check the
    budget can only be enforced at epoch granularity, which quietly favours
    models with cheap epochs.

    Validation time is measured and refunded: the budget is a *training* compute
    budget, and a model should not be penalised for how often it is scored.
    'elapsed' is therefore training seconds only, and is the x-axis for every
    time-to-quality plot.

    The progress bar counts *seconds*, not epochs, because seconds are what is
    being spent. PlotFn(history) is called every plot_every epochs behind a
    clear_output, giving the live loss curve the autoencoder notebook uses.
    """
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    hist = {"epoch": [], "elapsed": []}
    epoch, eval_time = 0, 0.0

    Bar = tqdm(total=int(round(budget_s)), desc=name, unit="s", leave=True)

    while time.perf_counter() < deadline and epoch < max_epochs:
        epoch += 1
        stats = EpochFn(deadline)
        elapsed = time.perf_counter() - t0 - eval_time

        hist["epoch"].append(epoch)
        hist["elapsed"].append(elapsed)
        for k, v in (stats or {}).items():
            hist.setdefault(k, []).append(float(v))

        if EvalFn is not None:
            e0 = time.perf_counter()
            evals = EvalFn()
            spent = time.perf_counter() - e0
            eval_time += spent
            deadline += spent
            for k, v in evals.items():
                hist.setdefault(k, []).append(float(v))

        Bar.n = min(int(round(elapsed)), int(round(budget_s)))
        Bar.set_postfix({k: f"{hist[k][-1]:.2e}" for k in hist
                         if k not in ("epoch", "elapsed")})
        Bar.refresh()

        if PlotFn is not None and (epoch % plot_every == 0):
            clear_output(wait=True)
            PlotFn(hist)
            plt.show()

        if verbose and log_every and (epoch % log_every == 0):
            bits = " | ".join(f"{k} {hist[k][-1]:.3e}" for k in hist
                              if k not in ("epoch", "elapsed"))
            Bar.write(f"{name} epoch {epoch:04d} | {elapsed:6.1f}s | {bits}")

    Bar.n = Bar.total
    Bar.refresh()
    Bar.close()

    hist["total_time"] = time.perf_counter() - t0 - eval_time
    hist["eval_time"] = eval_time
    hist["epochs_done"] = epoch
    return hist


def TimeToTarget(hist, key, target):
    """Wall-clock seconds until `key` first reached `target`. None if never."""
    for t, v in zip(hist["elapsed"], hist.get(key, [])):
        if v <= target:
            return float(t)
    return None


# -- Inference timing ---------------------------------------------------------

@torch.no_grad()
def TimeRollout(model, Warm, horizon, repeats=3, warmup=1, **kw):
    """Median wall-clock of a free-running rollout, plus per-step cost."""
    model.eval()
    for _ in range(warmup):
        model.Rollout(Warm, min(horizon, 16), **kw)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.Rollout(Warm, horizon, **kw)
        times.append(time.perf_counter() - t0)

    total = float(np.median(times))
    B = Warm.shape[0]
    return {"rollout_s": total,
            "ms_per_step": 1000.0 * total / horizon,
            "ms_per_step_per_traj": 1000.0 * total / (horizon * B)}


def PinThreads(n=1):
    """Fix the thread count so wall-clock numbers mean something.

    Everything here runs on CPU, where torch will otherwise pick a thread count
    per process and make timings depend on what else the machine is doing.
    """
    torch.set_num_threads(n)
    return torch.get_num_threads()


def SeedAll(seed):
    """torch, numpy and random. The existing notebook seeds only torch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

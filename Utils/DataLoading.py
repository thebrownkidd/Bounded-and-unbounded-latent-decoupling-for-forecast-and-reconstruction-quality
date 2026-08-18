import numpy as np
import pandas as pd
import os

def LoadLorenzData(path):
    """Load the single-series Lorenz dataset built by SyntheticGenerators.LorenzLift.

    Returns train/val/test observations, the clean signal and true 3-d latent
    states for diagnostics (never train on those two), and the MEASURED
    reconstruction noise floor -- an empirical MSE, not the naive noise**2,
    which is only exact if the final standardisation left obs_sd at exactly 1.
    """
    data = np.load(path)
    train = data['train']
    val = data['val']
    test = data['test']
    clean = data['clean']
    states = data['states']
    noise_floor_mse = float(data['noise_floor_mse']) if 'noise_floor_mse' in data else 0.05 ** 2

    return train, val, test, clean, states, noise_floor_mse


def LoadLorenzMultiSeries(path):
    """Load the multi-series Lorenz dataset built by
    SyntheticGenerators.LorenzLift.save_multi_series_dataset.

    Returns a dict: train/val/test (n_series, T_i, n_obs) -- each training-pool
    series individually split 70/15/15, chronologically, exactly like the
    single-series pipeline; clean/states (n_series, T, ...) -- diagnostics only;
    holdout_obs/holdout_states (n_holdout, T_holdout, ...) -- trajectories never
    seen in any form during training, with no temporal split at all.
    """
    d = np.load(path)
    return {"train": d["train"], "val": d["val"], "test": d["test"],
           "clean": d["clean"], "states": d["states"],
           "holdout_obs": d["holdout_obs"], "holdout_states": d["holdout_states"],
           "obs_mu": d["obs_mu"], "obs_sd": d["obs_sd"],
           "noise_floor_mse": float(d["noise_floor_mse"])}

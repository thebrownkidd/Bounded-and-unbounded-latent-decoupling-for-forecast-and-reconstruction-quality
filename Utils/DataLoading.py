import numpy as np
import pandas as pd
import os

def LoadLorenzData(path):
    """Load Lorenz dataset from a CSV file.

    Returns dict with train/val/test observations, plus the clean signal
    and true 3-d latent states for diagnostics. Never train on those two.
    """
    data = np.load(path)
    train = data['train']
    val = data['val']
    test = data['test']
    clean = data['clean']
    states = data['states']


    return train, val, test, clean, states

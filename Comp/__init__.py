"""Comparison models. Nothing in here is ours -- these are the baselines the
bounded/unbounded architecture is measured against.
"""

from .JointAEGRU import JointAEGRU, Mlp
from .PhysicsLatentAE import PhysicsLatentAE
from .LorenzField import Field, RK4Step, SIGMA, RHO, BETA

__all__ = ["JointAEGRU", "Mlp", "PhysicsLatentAE",
           "Field", "RK4Step", "SIGMA", "RHO", "BETA"]

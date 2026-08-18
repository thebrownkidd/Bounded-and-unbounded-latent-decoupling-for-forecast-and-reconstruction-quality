"""The Lorenz-63 vector field in torch, for the physics-informed baseline.

Kept separate from SyntheticGenerators/LorenzLift.py on purpose: that file is
the data-generating truth and runs in numpy/scipy, this one is a differentiable
model component. They must agree on (sigma, rho, beta) or the PINN is being
handed the wrong physics.
"""

import torch

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def Field(s, sigma=SIGMA, rho=RHO, beta=BETA):
    """Lorenz-63 right-hand side. (..., 3) -> (..., 3)."""
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return torch.stack([sigma * (y - x),
                        x * (rho - z) - y,
                        x * y - beta * z], dim=-1)


def RK4Step(s, dt, sigma=SIGMA, rho=RHO, beta=BETA):
    """One classical RK4 step of the Lorenz field. (..., 3) -> (..., 3)."""
    k1 = Field(s, sigma, rho, beta)
    k2 = Field(s + 0.5 * dt * k1, sigma, rho, beta)
    k3 = Field(s + 0.5 * dt * k2, sigma, rho, beta)
    k4 = Field(s + dt * k3, sigma, rho, beta)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

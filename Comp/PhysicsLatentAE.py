import torch
import torch.nn as nn

from .JointAEGRU import Mlp
from .LorenzField import Field, RK4Step, SIGMA, RHO, BETA


class PhysicsLatentAE(nn.Module):
    """Physics-informed latent autoencoder -- the PINN comparison.

        x -> enc -> zhat in R^3 -> affine -> s (physical Lorenz coordinates)
        zhat -> dec -> xhat

        L = L_rec + lambda_phys * || ds/dt - Field(s) ||^2   (relative)

    The claim being tested is that physics decouples the tradeoff: if the
    governing equations supply the dynamics, the representation never has to
    encode them, so it is free to spend all its capacity on reconstruction.
    Accordingly the forecaster here has *no learned parameters at all* --
    rollout is RK4 integration of the known field, and only the encoder,
    decoder and coordinate gauge are trained.

    The gauge matters. SyntheticGenerators.lift z-scores the states before
    lifting, so the recoverable latent is standardised Lorenz, not physical
    Lorenz, and the field would be wrong by a per-channel scale if applied
    directly. mu/logsd carry that affine and should be initialised from the
    true state statistics via StatsInit.

    Two caveats to carry into any writeup. (1) This model is handed the exact
    true equations -- it is an oracle, and upper-bounds what physics-decoupling
    can buy; run it with a misspecified rho to see how much of the advantage
    survives. (2) states is diagnostics-only, so nothing here is supervised on
    it; only the affine *initialisation* uses its mean and std.
    """

    def __init__(self, n, width=245, dt=0.01, activation=nn.GELU,
                 sigma=SIGMA, rho=RHO, beta=BETA):
        super().__init__()
        self.n = n
        self.dt = dt
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.enc = Mlp([n, width, width, 3], activation)
        self.dec = Mlp([3, width, width, n], activation)
        self.mu = nn.Parameter(torch.zeros(3))
        self.logsd = nn.Parameter(torch.zeros(3))

    def StatsInit(self, states):
        """Seed the coordinate gauge from the true state statistics."""
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(states.mean(0), dtype=self.mu.dtype))
            self.logsd.copy_(torch.as_tensor(
                states.std(0), dtype=self.logsd.dtype).log())

    # -- Coordinate gauge ------------------------------------------------

    def ToPhysical(self, zhat):
        return zhat * self.logsd.exp() + self.mu

    def ToLatent(self, s):
        return (s - self.mu) / self.logsd.exp()

    def Encode(self, xseq):
        B, T, n = xseq.shape
        return self.enc(xseq.reshape(B * T, n)).view(B, T, 3)

    def Decode(self, zseq):
        B, T, d = zseq.shape
        return self.dec(zseq.reshape(B * T, d)).view(B, T, self.n)

    def InferenceParams(self):
        return self.parameters()

    # -- Training --------------------------------------------------------

    def Losses(self, triple):
        """triple (B, 3, n): three consecutive observations -> (rec, phys).

        ds/dt comes from a central difference over the outer two samples, so
        the residual is second-order accurate in dt without needing autograd
        through time. The residual is divided by the mean square of the field
        itself, making it a *relative* error: the raw Lorenz field is O(100)
        while reconstruction MSE is O(1e-3), and without this normalisation
        lambda_phys would have to absorb five orders of magnitude.
        """
        zhat = self.Encode(triple)
        rec = ((self.Decode(zhat) - triple) ** 2).mean()

        s = self.ToPhysical(zhat)
        dsdt = (s[:, 2] - s[:, 0]) / (2.0 * self.dt)
        target = Field(s[:, 1], self.sigma, self.rho, self.beta)
        phys = ((dsdt - target) ** 2).mean() / (target ** 2).mean().clamp_min(1e-12)
        return rec, phys

    # -- Shared evaluation interface -------------------------------------

    def Reconstruct(self, window):
        return self.Decode(self.Encode(window))

    def Rollout(self, window, horizon):
        """Free-running forecast by integrating the known field. No learned dynamics."""
        s = self.ToPhysical(self.Encode(window)[:, -1])
        preds = []
        for _ in range(horizon):
            s = RK4Step(s, self.dt, self.sigma, self.rho, self.beta)
            preds.append(self.ToLatent(s))
        return self.Decode(torch.stack(preds, dim=1))

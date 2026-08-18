import numpy as np
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

    The gauge matters, and getting it wrong is silent. `dec` reads `zhat`
    directly, never `s` -- so reconstruction supplies zero gradient pressure
    on zhat's scale. The only thing anchoring zhat to physical Lorenz
    coordinates is the physics residual, and because that residual is
    RELATIVE (normalised by the field's own magnitude, see Losses), gradient
    descent can satisfy it in a self-consistent but wrong coordinate system --
    zhat's natural scale drifts away from what mu/logsd were initialised for,
    `s = ToPhysical(zhat)` lands far outside the true attractor, and RK4
    integration of the (correct) field at a (wrong) state diverges immediately
    under chaos, even while reconstruction and the physics residual both keep
    looking healthy throughout training. This is diagnosed in the comparison
    notebook, Model C section.

    The fix: CalibrateGauge fits a general affine zhat <-> physical state by
    least squares against the true Lorenz state, on TRAINING data only, and
    is called every epoch (see the notebook's TrainPinn) so the physics
    residual is always computed in a coordinate system anchored to the real
    attractor rather than one gradient descent is free to drift away from.
    mu/logsd (still seeded by StatsInit) remain the fallback used only before
    the first calibration call.

    Two caveats to carry into any writeup. (1) This model is handed the exact
    true equations -- it is an oracle, and upper-bounds what physics-decoupling
    can buy; run it with a misspecified rho to see how much of the advantage
    survives. (2) states is diagnostics-only in the sense that no loss term
    ever backpropagates through it -- StatsInit and CalibrateGauge both use it,
    the first for a mean/std initialisation, the second for a closed-form
    least-squares fit recomputed from the CURRENT encoder each epoch; neither
    is a gradient step through the encoder's own parameters.
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
        # Least-squares gauge: [zhat, 1] @ GaugeFwd -> s, [s, 1] @ GaugeInv -> zhat.
        # Buffers (not Parameters), fit in closed form by CalibrateGauge, never by
        # gradient descent. All-zero means "not yet calibrated" -- ToPhysical/
        # ToLatent fall back to mu/logsd until the first CalibrateGauge call, and
        # this state (buffers, unlike a plain attribute) survives a checkpoint
        # round-trip, so a loaded model keeps using its calibrated gauge.
        self.register_buffer("_GaugeFwd", torch.zeros(4, 3))
        self.register_buffer("_GaugeInv", torch.zeros(4, 3))

    def StatsInit(self, states):
        """Seed the coordinate gauge from the true state statistics."""
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(states.mean(0), dtype=self.mu.dtype))
            self.logsd.copy_(torch.as_tensor(
                states.std(0), dtype=self.logsd.dtype).log())

    @torch.no_grad()
    def CalibrateGauge(self, Xtrain, StatesTrain):
        """Fit the least-squares gauge (zhat <-> physical state) on TRAINING
        data only, using the encoder's CURRENT weights. Closed-form, no
        gradient step -- cheap enough to call every epoch (one encoder forward
        pass plus two small `lstsq` solves), which is what keeps the physics
        residual anchored to the real attractor throughout training instead of
        only at initialisation.
        """
        Device = next(self.parameters()).device
        X = Xtrain if torch.is_tensor(Xtrain) else torch.as_tensor(Xtrain, dtype=torch.float32)
        Zhat = self.Encode(X.to(Device).unsqueeze(1)).squeeze(1).cpu().numpy()
        States = StatesTrain if isinstance(StatesTrain, np.ndarray) else np.asarray(StatesTrain)

        def _Fit(A, B):
            Design = np.concatenate([A, np.ones((len(A), 1))], axis=1)
            W, *_ = np.linalg.lstsq(Design, B, rcond=None)
            return W

        Fwd = _Fit(Zhat, States)
        Inv = _Fit(States, Zhat)
        self._GaugeFwd.copy_(torch.as_tensor(Fwd, dtype=self._GaugeFwd.dtype))
        self._GaugeInv.copy_(torch.as_tensor(Inv, dtype=self._GaugeInv.dtype))

    @property
    def _IsCalibrated(self):
        return bool(self._GaugeFwd.abs().sum() > 0)

    # -- Coordinate gauge ------------------------------------------------

    def ToPhysical(self, zhat):
        if self._IsCalibrated:
            Ones = torch.ones(*zhat.shape[:-1], 1, dtype=zhat.dtype, device=zhat.device)
            return torch.cat([zhat, Ones], dim=-1) @ self._GaugeFwd
        return zhat * self.logsd.exp() + self.mu

    def ToLatent(self, s):
        if self._IsCalibrated:
            Ones = torch.ones(*s.shape[:-1], 1, dtype=s.dtype, device=s.device)
            return torch.cat([s, Ones], dim=-1) @ self._GaugeInv
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

import torch
import torch.nn as nn

from .Encoder import Encoder
from .LatentRecon import LatentRecon
from .LatentMapping import LatentMapping
from .LatentBounded import LatentBounded
from .Decoder import Decoder
from .LatentForecast import LatentForecast


class DecoupledModel(nn.Module):
    """The bounded/unbounded chain plus its single forecaster.

    Reconstruction path (Stage A, unchanged from the autoencoder experiment):

        x -> E -> h -> f -> b (k, unbounded) -> m -> C (a x b, bounded) -> D -> x_hat

    Forecast path (Stage B). Only the unbounded latent is forecast; the bounded
    latent is recomputed from it by the *same* static m that reconstruction uses,
    so there is exactly one learned dynamics model in the chain:

        b_{t+1} = g(b_t)          unbounded latent carries the dynamics
        C_{t+1} = m(b_{t+1})      bounded latent is recomputed, not advanced
        x_{t+1} = D(C_{t+1})

    C stays in [0, 1] because m ends in a sigmoid, so boundedness survives an
    arbitrarily long rollout however far b drifts.

    r = LatentBounded is a training-only teacher for the consistency term and
    never runs at inference, so it is excluded from the inference parameter
    count.

    The submodules are exposed as attributes (E, F, M, R, D, G) so the notebook
    can freeze Stage A and optimise Stage B separately.
    """

    def __init__(self, n, k, a, b, rnn_hidden=64, rnn_layers=1):
        super().__init__()
        self.n, self.k, self.a, self.b = n, k, a, b
        self.E = Encoder(n)
        self.F = LatentRecon(n, k)
        self.M = LatentMapping(k, a, b)
        self.R = LatentBounded(n, a, b)
        self.D = Decoder(a, b, n)
        self.G = LatentForecast(k, hidden=rnn_hidden, layers=rnn_layers)

    # -- Stage A ---------------------------------------------------------

    def forward(self, x, with_teacher=True):
        """Autoencoder pass on (B, n). Returns (xhat, C, CTeach, b, h).

        `with_teacher=False` skips R, the training-only consistency teacher
        (never part of InferenceModules) -- used by Reconstruct, which is an
        inference-path call and has no business paying for it.
        """
        h = self.E(x)
        b = self.F(h)
        C = self.M(b)
        CTeach = self.R(h) if with_teacher else None
        return self.D(C), C, CTeach, b, h

    def StageAModules(self):
        return [self.E, self.F, self.M, self.R, self.D]

    def StageBModules(self):
        return [self.G]

    def InferenceModules(self):
        """Everything that runs at test time -- R is a teacher and is excluded."""
        return [self.E, self.F, self.M, self.D, self.G]

    def InferenceParams(self):
        for m in self.InferenceModules():
            yield from m.parameters()

    # -- Sequence helpers ------------------------------------------------

    def EncodeSeq(self, xseq, want_c=True):
        """(B, T, n) -> b (B, T, k) and C (B, T, a, b), or C=None if want_c=False.

        Rollout/RolloutLatents only need the warm-up's b; C over the warm-up
        window is never used there, so skipping it saves a M() call over the
        whole (B*T) window on every rollout.
        """
        B, T, n = xseq.shape
        h = self.E(xseq.reshape(B * T, n))
        b = self.F(h)
        C = self.M(b).view(B, T, self.a, self.b) if want_c else None
        return b.view(B, T, self.k), C

    # -- Shared evaluation interface -------------------------------------

    def Reconstruct(self, window):
        """(B, L, n) -> (B, L, n). The Stage-A inference path, per timestep."""
        B, L, n = window.shape
        xhat, _, _, _, _ = self.forward(window.reshape(B * L, n), with_teacher=False)
        return xhat.view(B, L, n)

    def Rollout(self, window, horizon):
        """(B, L, n) warm-up -> (B, horizon, n) free-running forecast.

        Only b is stepped. C is recomputed from it each step by the static m,
        so nothing but the GRU state carries across the rollout. M and D are
        applied once on the whole stacked b trajectory rather than once per
        step, which is a pure reassociation (same computation, same result).
        """
        b, _ = self.EncodeSeq(window, want_c=False)

        state = None
        if window.shape[1] > 1:
            _, state = self.G(b[:, :-1], None)

        bcur, bs = b[:, -1], []
        for _ in range(horizon):
            bcur, state = self.G.Step(bcur, state)
            bs.append(bcur)
        B = torch.stack(bs, dim=1)                      # (B, horizon, k)
        Bh, H, k = B.shape
        out = self.D(self.M(B.reshape(Bh * H, k))).view(Bh, H, self.n)
        return out

    def RolloutLatents(self, window, horizon):
        """As Rollout, but also returns the b and C trajectories.

        Needed for the boundedness audit (C must stay in [0, 1]) and for the
        affine coordinate map onto the true Lorenz state. Same batched-decode
        reassociation as Rollout.
        """
        b, _ = self.EncodeSeq(window, want_c=False)

        state = None
        if window.shape[1] > 1:
            _, state = self.G(b[:, :-1], None)

        bcur, bs = b[:, -1], []
        for _ in range(horizon):
            bcur, state = self.G.Step(bcur, state)
            bs.append(bcur)
        Bt = torch.stack(bs, dim=1)                      # (B, horizon, k)
        Bh, H, k = Bt.shape
        Cs = self.M(Bt.reshape(Bh * H, k)).view(Bh, H, self.a, self.b)
        Xs = self.D(Cs.reshape(Bh * H, self.a, self.b)).view(Bh, H, self.n)
        return Xs, Bt, Cs

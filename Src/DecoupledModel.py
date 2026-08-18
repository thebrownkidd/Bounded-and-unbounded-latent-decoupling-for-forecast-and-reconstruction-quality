import torch
import torch.nn as nn

from .Encoder import Encoder
from .LatentRecon import LatentRecon
from .LatentMapping import LatentMapping, LatentMappingDynamic
from .LatentBounded import LatentBounded
from .Decoder import Decoder
from .LatentForecast import LatentForecast


class DecoupledModel(nn.Module):
    """The bounded/unbounded chain plus its two forecasters.

    Reconstruction path (Stage A, unchanged from the autoencoder experiment):

        x -> E -> h -> f -> b (k, unbounded) -> m -> C (a x b, bounded) -> D -> x_hat

    Forecast path (Stage B), two models as specified:

        b_{t+1} = g(b_t)                 unbounded latent carries the dynamics
        C_{t+1} = m_dyn(b_{t+1}, C_t)    bounded latent is advanced, not recomputed
        x_{t+1} = D(C_{t+1})

    r = LatentBounded is a training-only teacher for the consistency term and
    never runs at inference, so it is excluded from the inference parameter
    count.

    The submodules are exposed as attributes (E, F, M, R, D, G, MD) so the
    notebook can freeze Stage A and optimise Stage B separately.
    """

    def __init__(self, n, k, a, b, rnn_hidden=64, rnn_layers=1,
                 dynamic_hidden=(64, 128), residual=True):
        super().__init__()
        self.n, self.k, self.a, self.b = n, k, a, b
        self.E = Encoder(n)
        self.F = LatentRecon(n, k)
        self.M = LatentMapping(k, a, b)
        self.R = LatentBounded(n, a, b)
        self.D = Decoder(a, b, n)
        self.G = LatentForecast(k, hidden=rnn_hidden, layers=rnn_layers)
        self.MD = LatentMappingDynamic(k, a, b, hidden=dynamic_hidden,
                                       residual=residual)

    # -- Stage A ---------------------------------------------------------

    def forward(self, x):
        """Autoencoder pass on (B, n). Returns (xhat, C, CTeach, b, h)."""
        h = self.E(x)
        b = self.F(h)
        C = self.M(b)
        return self.D(C), C, self.R(h), b, h

    def StageAModules(self):
        return [self.E, self.F, self.M, self.R, self.D]

    def StageBModules(self):
        return [self.G, self.MD]

    def InferenceModules(self):
        """Everything that runs at test time -- R is a teacher and is excluded."""
        return [self.E, self.F, self.M, self.D, self.G, self.MD]

    def InferenceParams(self):
        for m in self.InferenceModules():
            yield from m.parameters()

    # -- Sequence helpers ------------------------------------------------

    def EncodeSeq(self, xseq):
        """(B, T, n) -> b (B, T, k) and C (B, T, a, b)."""
        B, T, n = xseq.shape
        h = self.E(xseq.reshape(B * T, n))
        b = self.F(h)
        C = self.M(b)
        return b.view(B, T, self.k), C.view(B, T, self.a, self.b)

    # -- Shared evaluation interface -------------------------------------

    def Reconstruct(self, window):
        """(B, L, n) -> (B, L, n). The Stage-A inference path, per timestep."""
        B, L, n = window.shape
        xhat, _, _, _, _ = self.forward(window.reshape(B * L, n))
        return xhat.view(B, L, n)

    def Rollout(self, window, horizon, dynamic=True):
        """(B, L, n) warm-up -> (B, horizon, n) free-running forecast.

        dynamic=False swaps m_dyn for the static m, which is the ablation
        that asks whether carrying bounded-latent state across the rollout
        earns its parameters.
        """
        b, C = self.EncodeSeq(window)

        state = None
        if window.shape[1] > 1:
            _, state = self.G(b[:, :-1], None)

        bcur, cprev, out = b[:, -1], C[:, -1], []
        for _ in range(horizon):
            bnext, state = self.G.Step(bcur, state)
            cnext = self.MD(bnext, cprev) if dynamic else self.M(bnext)
            out.append(self.D(cnext))
            bcur, cprev = bnext, cnext
        return torch.stack(out, dim=1)

    def RolloutLatents(self, window, horizon, dynamic=True):
        """As Rollout, but also returns the b and C trajectories.

        Needed for the boundedness audit (C must stay in [0, 1]) and for the
        affine probe onto the true Lorenz state.
        """
        b, C = self.EncodeSeq(window)

        state = None
        if window.shape[1] > 1:
            _, state = self.G(b[:, :-1], None)

        bcur, cprev = b[:, -1], C[:, -1]
        xs, bs, cs = [], [], []
        for _ in range(horizon):
            bnext, state = self.G.Step(bcur, state)
            cnext = self.MD(bnext, cprev) if dynamic else self.M(bnext)
            xs.append(self.D(cnext))
            bs.append(bnext)
            cs.append(cnext)
            bcur, cprev = bnext, cnext
        return (torch.stack(xs, dim=1), torch.stack(bs, dim=1),
                torch.stack(cs, dim=1))

import torch
import torch.nn as nn

from .JointAEGRU import Mlp


class JointAEGRUSigmoid(nn.Module):
    """JointAEGRU with a sigmoid on the shared latent -- the ablation this
    project's headline claim needs: does merely BOUNDING the one latent that
    already serves both reconstruction and forecasting recover what Ours gets,
    or is separating it into two latents actually doing the work?

    Same topology and weighted loss as JointAEGRU:

        L = (1 - phi) * L_rec + phi * L_fcst

    except the latent is bounded, and kept bounded through an arbitrarily long
    rollout by construction rather than by penalty -- the same residual-in-
    logit-space trick Src/LatentMapping.py's LatentMappingDynamic uses:

        z = sigmoid(enc(x))                              encode into [0, 1]^latent
        z_{t+1} = sigmoid(logit(z_t) + delta(z_t, h_t))   GRU residual step

    The one thing this variant does NOT change is the coupling: z is still a
    single latent read by both the decoder and the forecaster, so if bounding
    alone were the source of Ours's stability, this model should show it too.
    """

    def __init__(self, n, latent=16, width=224, rnn_hidden=64, rnn_layers=1,
                activation=nn.GELU, eps=1e-6):
        super().__init__()
        self.n = n
        self.latent = latent
        self.eps = eps
        self.enc = nn.Sequential(Mlp([n, width, width, latent], activation), nn.Sigmoid())
        self.dec = Mlp([latent, width, width, n], activation)
        self.rnn = nn.GRU(latent, rnn_hidden, num_layers=rnn_layers, batch_first=True)
        self.head = nn.Linear(rnn_hidden, latent)
        # Zero-initialised, so the forecast step starts as the identity z_{t+1} = z_t
        # -- the right prior when consecutive latents are nearly identical, and
        # matches LatentMappingDynamic's own init.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def Encode(self, xseq):
        B, T, n = xseq.shape
        return self.enc(xseq.reshape(B * T, n)).view(B, T, self.latent)

    def Decode(self, zseq):
        B, T, d = zseq.shape
        return self.dec(zseq.reshape(B * T, d)).view(B, T, self.n)

    def StepLatent(self, z, state):
        out, state = self.rnn(z.unsqueeze(1), state)
        delta = self.head(out.squeeze(1))
        c = z.clamp(self.eps, 1.0 - self.eps)
        z_next = torch.sigmoid(torch.log(c) - torch.log1p(-c) + delta)
        return z_next, state

    def InferenceParams(self):
        return self.parameters()

    # -- Training --------------------------------------------------------

    def Losses(self, seq, warm):
        """seq (B, warm + unroll, n) -> (rec, fcst), both MSE in observation space.
        Identical protocol to JointAEGRU.Losses -- see there for the rationale.
        """
        z = self.Encode(seq)
        rec = ((self.Decode(z) - seq) ** 2).mean()

        state = None
        if warm > 1:
            _, state = self.rnn(z[:, :warm - 1], None)

        zc, preds = z[:, warm - 1], []
        for _ in range(seq.shape[1] - warm):
            zc, state = self.StepLatent(zc, state)
            preds.append(zc)
        pred = self.Decode(torch.stack(preds, dim=1))
        return rec, ((pred - seq[:, warm:]) ** 2).mean()

    # -- Shared evaluation interface -------------------------------------

    def Reconstruct(self, window):
        return self.Decode(self.Encode(window))

    def Rollout(self, window, horizon):
        z = self.Encode(window)
        state = None
        if window.shape[1] > 1:
            _, state = self.rnn(z[:, :-1], None)

        zc, preds = z[:, -1], []
        for _ in range(horizon):
            zc, state = self.StepLatent(zc, state)
            preds.append(zc)
        return self.Decode(torch.stack(preds, dim=1))

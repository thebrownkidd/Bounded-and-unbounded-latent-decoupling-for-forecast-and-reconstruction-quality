import torch
import torch.nn as nn


def Mlp(dims, activation=nn.GELU):
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class JointAEGRU(nn.Module):
    """Shared-latent autoencoder with a GRU forecast head, weighted by phi.

    The balancing approach in its plainest form, and the thing this project
    argues against: one latent serving both objectives, with a scalar that
    decides how much each matters.

        L = (1 - phi) * L_rec + phi * L_fcst

    This is the Coupled Attention Networks / MemAAE / CAAE family reduced to
    its load-bearing part. Sweeping phi from 0 to 1 traces the empirical
    reconstruction-forecast Pareto front, which is what turns every other
    model in the comparison from an isolated number into a point relative to
    a curve.

    The GRU head is residually parameterised, matching Src/LatentForecast, so
    the comparison does not turn on a parameterisation advantage.
    """

    def __init__(self, n, latent=16, width=224, rnn_hidden=64, rnn_layers=1,
                 activation=nn.GELU):
        super().__init__()
        self.n = n
        self.latent = latent
        self.enc = Mlp([n, width, width, latent], activation)
        self.dec = Mlp([latent, width, width, n], activation)
        self.rnn = nn.GRU(latent, rnn_hidden, num_layers=rnn_layers,
                          batch_first=True)
        self.head = nn.Linear(rnn_hidden, latent)

    def Encode(self, xseq):
        B, T, n = xseq.shape
        return self.enc(xseq.reshape(B * T, n)).view(B, T, self.latent)

    def Decode(self, zseq):
        B, T, d = zseq.shape
        return self.dec(zseq.reshape(B * T, d)).view(B, T, self.n)

    def StepLatent(self, z, state):
        out, state = self.rnn(z.unsqueeze(1), state)
        return z + self.head(out.squeeze(1)), state

    def InferenceParams(self):
        return self.parameters()

    # -- Training --------------------------------------------------------

    def Losses(self, seq, warm):
        """seq (B, warm + unroll, n) -> (rec, fcst), both MSE in observation space.

        Reconstruction is scored over the whole sequence; the forecast is a
        free-running unroll from the end of the warm-up, so the two losses
        compete for the same latent -- which is the entire point of the
        baseline.
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

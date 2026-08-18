import torch.nn as nn


class LatentForecast(nn.Module):
    """g : b_t in R^k  ->  b_{t+1} in R^k, the unbounded-latent forecaster.

    A GRU rather than a plain MLP because b_t is only approximately Markov:
    it is a learned 3-d summary, not the true Lorenz state, so the recurrent
    state carries whatever history the projection lost.

    The head predicts a residual. At dt = 0.01 the Lorenz state moves very
    little per step, so b_{t+1} ~ b_t; regressing the delta keeps the target
    away from a scale where the identity map looks like a good answer.
    """

    def __init__(self, k, hidden=64, layers=1):
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.layers = layers
        self.rnn = nn.GRU(k, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, k)

    def forward(self, bseq, state=None):
        """(B, T, k) -> one-step-ahead (B, T, k), plus the final RNN state."""
        out, state = self.rnn(bseq, state)
        return bseq + self.head(out), state

    def Step(self, bt, state=None):
        """(B, k) -> (B, k). One free-running step, carrying the RNN state."""
        pred, state = self.forward(bt.unsqueeze(1), state)
        return pred.squeeze(1), state

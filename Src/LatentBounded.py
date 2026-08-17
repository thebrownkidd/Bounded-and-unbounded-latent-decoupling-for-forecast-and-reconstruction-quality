import torch.nn as nn


class LatentBounded(nn.Module):
    """r : h_t in R^n  ->  C_t in [0, 1]^(a x b), the bounded latent."""

    def __init__(self, n, a, b, hidden=(128, 128), activation=nn.GELU):
        super().__init__()
        self.a = a
        self.b = b
        layers = []
        dims = [n] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], a * b))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h).view(-1, self.a, self.b)

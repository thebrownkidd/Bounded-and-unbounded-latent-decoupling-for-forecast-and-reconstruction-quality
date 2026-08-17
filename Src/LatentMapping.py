import torch.nn as nn


class LatentMapping(nn.Module):
    """f : h_t in R^n  ->  b_t in R^k   (k << n), the unbounded forecast latent."""

    def __init__(self, n, k, hidden=(128, 64), activation=nn.GELU):
        super().__init__()
        layers = []
        dims = [n] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], k))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)

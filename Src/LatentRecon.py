import torch.nn as nn


class LatentRecon(nn.Module):
    """f^{-1} : b_t in R^k  ->  h_t in R^n.

    Learned inverse of LatentMapping; train with a cycle loss
    ||f^{-1}(f(h)) - h|| to enforce invertibility of f.
    """

    def __init__(self, k, n, hidden=(64, 128), activation=nn.GELU):
        super().__init__()
        layers = []
        dims = [k] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], n))
        self.net = nn.Sequential(*layers)

    def forward(self, b):
        return self.net(b)

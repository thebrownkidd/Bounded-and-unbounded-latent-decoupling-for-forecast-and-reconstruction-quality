import torch.nn as nn


class Encoder(nn.Module):
    """E : x_t in R^n  ->  h_t in R^n"""

    def __init__(self, n, hidden=(128, 128), activation=nn.GELU):
        super().__init__()
        layers = []
        dims = [n] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], n))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

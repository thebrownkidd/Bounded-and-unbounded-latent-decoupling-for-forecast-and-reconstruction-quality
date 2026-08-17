import torch.nn as nn


class LatentMapping(nn.Module):
    """m : b_t in R^k  ->  C_t in [0, 1]^(a x b), the bounded latent.

    Replaces the composition r(f^-1(b_t)): the inference path is
    h_t -> b_t -> C_t, with no direct h_t -> C_t route.
    """

    def __init__(self, k, a, b, hidden=(64, 128), activation=nn.GELU):
        super().__init__()
        self.a = a
        self.b = b
        layers = []
        dims = [k] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], a * b))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, bt):
        return self.net(bt).view(-1, self.a, self.b)

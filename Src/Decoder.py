import torch.nn as nn


class Decoder(nn.Module):
    """D : C in [0, 1]^(a x b)  ->  y_hat in R^n (estimator of x_{t+1})."""

    def __init__(self, a, b, n, hidden=(128, 128), activation=nn.GELU):
        super().__init__()
        layers = [nn.Flatten()]
        dims = [a * b] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], n))
        self.net = nn.Sequential(*layers)

    def forward(self, c):
        return self.net(c)

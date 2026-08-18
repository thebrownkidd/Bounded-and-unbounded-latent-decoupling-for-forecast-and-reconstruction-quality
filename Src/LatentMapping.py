import torch
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


class LatentMappingDynamic(nn.Module):
    """m_dyn : (b_{t+1} in R^k, C_t in [0,1]^(a x b)) -> C_{t+1} in [0,1]^(a x b).

    The forecasting counterpart of LatentMapping. Where m recomputes the
    bounded latent from scratch every step, m_dyn *advances* it: C carries
    state across the rollout instead of being a pure function of b.

    With residual=True the update is applied in logit space,

        C_{t+1} = sigmoid(logit(C_t) + delta(b_{t+1}, C_t)),

    so C stays inside [0, 1] at every step of an arbitrarily long rollout by
    construction rather than by penalty. The final layer is zero-initialised,
    which makes the map start life as the identity C_{t+1} = C_t -- the right
    prior when consecutive bounded latents are nearly identical.
    """

    def __init__(self, k, a, b, hidden=(64, 128), activation=nn.GELU,
                 residual=True, eps=1e-6):
        super().__init__()
        self.a = a
        self.b = b
        self.residual = residual
        self.eps = eps

        layers = []
        dims = [k + a * b] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation())
        final = nn.Linear(dims[-1], a * b)
        layers.append(final)
        if not residual:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        if residual:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, bnext, cprev):
        """(B, k) and (B, a, b) -> (B, a, b)."""
        out = self.net(torch.cat([bnext, cprev.flatten(1)], dim=1))
        if self.residual:
            c = cprev.flatten(1).clamp(self.eps, 1.0 - self.eps)
            out = torch.sigmoid(torch.log(c) - torch.log1p(-c) + out)
        return out.view(-1, self.a, self.b)

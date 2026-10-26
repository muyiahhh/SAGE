"""Predictors used by ISG and PSA.

* ``build_predictor`` builds the ISG predictor that maps the SID view to the
  text view. The paper uses ``linear``; deeper variants are kept for ablations.
* ``PSAPredictor`` is the shared linear predictor applied to the decoding states
  during recommendation fine-tuning. It is initialized at the identity.
"""

import torch
import torch.nn as nn


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PredictorMLP(nn.Module):
    """N-layer MLP predictor: dim -> dim, hidden = dim * hidden_mult.

    ``num_layers`` counts Linear layers; num_layers=1 is a plain linear map.
    """

    def __init__(self, dim: int, num_layers: int = 1, hidden_mult: int = 4):
        super().__init__()
        if num_layers <= 1:
            self.net = nn.Linear(dim, dim)
        else:
            hidden = dim * hidden_mult
            layers = [nn.Linear(dim, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            layers += [nn.Linear(hidden, dim)]
            self.net = nn.Sequential(*layers)
        _init_linear(self)

    def forward(self, x):
        return self.net(x)


def build_predictor(predictor_type: str, dim: int, hidden_mult: int = 4) -> nn.Module:
    """ISG predictor factory: ``linear`` (paper), ``identity``, ``mlp2`` or ``mlp3``."""
    t = (predictor_type or "linear").lower()
    if t == "identity":
        return nn.Identity()
    if t == "linear":
        return PredictorMLP(dim, num_layers=1)
    if t == "mlp2":
        return PredictorMLP(dim, num_layers=2, hidden_mult=hidden_mult)
    if t == "mlp3":
        return PredictorMLP(dim, num_layers=3, hidden_mult=hidden_mult)
    raise ValueError(f"Unknown predictor_type: {predictor_type!r}")


class PSAPredictor(nn.Module):
    """Shared linear predictor g_psi of PSA, initialized near the identity.

    At initialization ``g(h) ~= h``, so both readouts start from the decoding
    states themselves.
    """

    def __init__(self, dim: int, init_noise: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))
            self.linear.weight.add_(torch.randn_like(self.linear.weight) * init_noise)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)

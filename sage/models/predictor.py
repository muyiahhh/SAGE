"""ISG predictor that maps a SID representation to the text-view space.

Selectable depth so we can ablate how strong the predictor should be
(LLM-JEPA finds predictor strength is task-dependent; a frozen-backbone ISG run
may not need a deep MLP). ``build_predictor`` is the single entry point.

Types:
  - "identity" : no predictor (align z_SID directly to z_text)
  - "linear"   : single Linear(d_proj, d_proj)
  - "mlp2"     : 2-layer MLP
  - "mlp3"     : 3-layer MLP (DEFAULT — current production predictor)
"""

import torch.nn as nn


def _init_linear(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PredictorMLP(nn.Module):
    """N-layer MLP predictor: d_proj -> d_proj, hidden = d_proj * hidden_mult.

    ``num_layers`` counts Linear layers; num_layers=1 is a plain linear map.
    The default (num_layers=3, hidden_mult=4) reproduces the original production
    predictor exactly.
    """

    def __init__(self, d_proj: int, num_layers: int = 3, hidden_mult: int = 4):
        super().__init__()
        if num_layers <= 1:
            self.net = nn.Linear(d_proj, d_proj)
        else:
            hidden = d_proj * hidden_mult
            layers = [nn.Linear(d_proj, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            layers += [nn.Linear(hidden, d_proj)]
            self.net = nn.Sequential(*layers)
        _init_linear(self)

    def forward(self, x):
        """x: (B, d_proj) -> (B, d_proj)"""
        return self.net(x)


def build_predictor(predictor_type: str, d_proj: int, hidden_mult: int = 4) -> nn.Module:
    """Factory for the JEPA predictor. See module docstring for the type list."""
    t = (predictor_type or "mlp3").lower()
    if t == "identity":
        return nn.Identity()
    if t == "linear":
        return PredictorMLP(d_proj, num_layers=1)
    if t == "mlp2":
        return PredictorMLP(d_proj, num_layers=2, hidden_mult=hidden_mult)
    if t == "mlp3":
        return PredictorMLP(d_proj, num_layers=3, hidden_mult=hidden_mult)
    raise ValueError(f"Unknown predictor_type: {predictor_type!r}")

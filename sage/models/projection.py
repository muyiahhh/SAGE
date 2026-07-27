"""Projection head shared by the online and target branches of ISG."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, d_proj: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_proj, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x):
        """x: (B, d_model) → (B, d_proj)"""
        return self.proj(x)

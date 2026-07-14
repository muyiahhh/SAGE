"""EMA (Exponential Moving Average) update for the JEPA target encoder."""

import torch
import torch.nn as nn
import copy


def build_ema_model(source_model: nn.Module) -> nn.Module:
    """Create an exact deep copy of source_model for use as the EMA target."""
    target = copy.deepcopy(source_model)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float = 0.999) -> None:
    """θ_target ← τ·θ_target + (1-τ)·θ_online.

    Only parameters are updated; buffers (e.g. running stats) are copied directly.
    """
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.data.copy_(b_o.data)

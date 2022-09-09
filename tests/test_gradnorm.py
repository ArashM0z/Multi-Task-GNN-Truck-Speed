"""GradNorm regulariser sanity tests."""
from __future__ import annotations

import torch
from torch import nn
from truck_speed_extensions.gradnorm import gradnorm_step


def test_gradnorm_returns_scalar() -> None:
    encoder = nn.Linear(8, 16)
    head_a = nn.Linear(16, 1)
    head_b = nn.Linear(16, 1)
    x = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 1)
    h = encoder(x)
    loss_a = ((head_a(h) - target) ** 2).mean()
    loss_b = ((head_b(h) - target) ** 2).mean()
    w_a = nn.Parameter(torch.tensor(1.0))
    w_b = nn.Parameter(torch.tensor(1.0))
    gn = gradnorm_step({"a": loss_a, "b": loss_b}, {"a": w_a, "b": w_b},
                       list(encoder.parameters()))
    assert gn.dim() == 0
    assert torch.isfinite(gn).all()

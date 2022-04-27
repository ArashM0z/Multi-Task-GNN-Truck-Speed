"""GradNorm cross-task gradient balancing (Chen et al. 2018).

Equalises per-task gradient norms during multi-task training so that no task
dominates the shared encoder. Plugged into the AST-MTL trainer to balance:
  - speed regression (primary)
  - flow regression (AST-MTL secondary)
  - weather classification (paper's new auxiliary)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def gradnorm_step(
    task_losses: dict[str, Tensor],
    task_weights: dict[str, Tensor],
    shared_params: list[Tensor],
    alpha: float = 1.5,
) -> Tensor:
    """Return the GradNorm regulariser loss.

    Args:
        task_losses: {task_name: scalar loss tensor}
        task_weights: {task_name: scalar learnable weight tensor}
        shared_params: parameters of the shared encoder
        alpha: restoring-force exponent (paper default 1.5)
    """
    g_norms: list[Tensor] = []
    initial: list[Tensor] = []
    for name, loss in task_losses.items():
        weighted = task_weights[name] * loss
        grads = torch.autograd.grad(weighted, shared_params, retain_graph=True, create_graph=True)
        g_norms.append(torch.cat([g.flatten() for g in grads]).norm())
        initial.append(loss.detach())

    g_norms_t = torch.stack(g_norms)
    initial_t = torch.stack(initial)
    avg_norm = g_norms_t.mean().detach()
    rel_inverse = initial_t / initial_t.mean()
    target = (avg_norm * rel_inverse ** alpha).detach()
    return (g_norms_t - target).abs().sum()

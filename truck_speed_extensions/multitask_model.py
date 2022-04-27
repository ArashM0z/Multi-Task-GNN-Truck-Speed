"""Multi-task wrapper combining the AST-MTL backbone with the weather head.

Imports the AST-MTL encoder from the upstream repo unchanged; adds the
auxiliary weather classification head, and exposes a single ``forward``
that returns all task outputs.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class MultiTaskTruckSpeed(nn.Module):
    """Shared encoder + (speed, flow, weather) heads.

    The shared encoder is whatever AST-MTL provides — we don't redefine it.
    This wrapper just bolts on the third (weather) head and the joint
    forward path used by the paper's training script.
    """

    def __init__(
        self,
        astmtl_encoder: nn.Module,
        speed_head: nn.Module,
        flow_head: nn.Module,
        weather_head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = astmtl_encoder
        self.speed_head = speed_head
        self.flow_head = flow_head
        self.weather_head = weather_head

    def forward(
        self, node_features: Tensor, edge_features: Tensor, adj: Tensor,
    ) -> Mapping[str, Tensor]:
        h = self.encoder(node_features, edge_features, adj)
        return {
            "speed": self.speed_head(h),
            "flow": self.flow_head(h),
            "weather": self.weather_head(h),
        }

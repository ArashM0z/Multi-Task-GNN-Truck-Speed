"""Auxiliary weather-classification head added in our SIGSPATIAL 2022 paper.

AST-MTL is built around two tasks (speed + flow). This module adds a third
auxiliary head that classifies the weather regime of each segment-hour.
Sharing the encoder across the three tasks transfers the snow/freezing-rain
patterns into the speed head — that is the main result of the paper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WeatherDisruptionHead(nn.Module):
    """Per-segment weather classification head.

    Classes (4): clear, rain, snow, freezing rain.
    Trained jointly with the speed head; cross-task gradient balancing via
    GradNorm (see ``gradnorm.py``).
    """

    def __init__(self, hidden_dim: int = 128, n_classes: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, segment_features: Tensor) -> Tensor:
        """(B, N, H) -> (B, N, n_classes)."""
        return self.head(segment_features)

"""Sanity tests for the auxiliary weather classification head."""
from __future__ import annotations

import torch
from truck_speed_extensions.weather_aux_head import WeatherDisruptionHead


def test_output_shape() -> None:
    head = WeatherDisruptionHead(hidden_dim=64, n_classes=4)
    x = torch.randn(2, 10, 64)
    y = head(x)
    assert y.shape == (2, 10, 4)


def test_logits_are_finite_and_differentiable() -> None:
    head = WeatherDisruptionHead(hidden_dim=32, n_classes=4)
    x = torch.randn(1, 5, 32, requires_grad=True)
    y = head(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(y).all()

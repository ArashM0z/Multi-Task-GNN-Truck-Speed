"""Paper training script: AST-MTL + weather head + GradNorm.

Wraps the upstream AST-MTL training loop with the paper-specific weather
auxiliary task. The actual encoder / data pipeline is whatever AST-MTL ships;
we only add the third head and the GradNorm regulariser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F

from truck_speed_extensions.gradnorm import gradnorm_step
from truck_speed_extensions.multitask_model import MultiTaskTruckSpeed
from truck_speed_extensions.weather_aux_head import WeatherDisruptionHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-speed", type=float, default=1.0)
    p.add_argument("--lambda-flow", type=float, default=0.5)
    p.add_argument("--lambda-weather", type=float, default=0.3)
    p.add_argument("--gradnorm-alpha", type=float, default=1.5)
    p.add_argument("--out", type=Path, default=Path("runs/truck_speed"))
    return p.parse_args()


def train_step(
    model: MultiTaskTruckSpeed,
    batch: dict[str, torch.Tensor],
    task_weights: dict[str, torch.Tensor],
    optimiser: optim.Optimizer,
    weight_optimiser: optim.Optimizer,
    gradnorm_alpha: float,
) -> dict[str, float]:
    outputs = model(batch["node_features"], batch["edge_features"], batch["adj"])
    losses = {
        "speed":   F.smooth_l1_loss(outputs["speed"],   batch["speed_target"]),
        "flow":    F.smooth_l1_loss(outputs["flow"],    batch["flow_target"]),
        "weather": F.cross_entropy(
            outputs["weather"].reshape(-1, outputs["weather"].size(-1)),
            batch["weather_target"].reshape(-1).long(),
        ),
    }
    total = sum(task_weights[name] * loss for name, loss in losses.items())
    optimiser.zero_grad()
    total.backward(retain_graph=True)

    weight_optimiser.zero_grad()
    gn = gradnorm_step(
        losses, task_weights, list(model.encoder.parameters()), alpha=gradnorm_alpha,
    )
    gn.backward()
    weight_optimiser.step()

    optimiser.step()
    return {f"loss_{k}": float(v.detach()) for k, v in losses.items()}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Placeholder encoder + heads; real instantiation pulls from the AST-MTL
    # package. This entrypoint just shows the training scaffolding.
    encoder = nn.Identity()
    speed_head = nn.Linear(128, 1)
    flow_head = nn.Linear(128, 1)
    weather_head = WeatherDisruptionHead(hidden_dim=128, n_classes=4)
    model = MultiTaskTruckSpeed(encoder, speed_head, flow_head, weather_head)

    task_weights = {
        "speed": nn.Parameter(torch.tensor(args.lambda_speed)),
        "flow": nn.Parameter(torch.tensor(args.lambda_flow)),
        "weather": nn.Parameter(torch.tensor(args.lambda_weather)),
    }
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    weight_optimiser = optim.Adam(list(task_weights.values()), lr=args.lr * 0.1)
    print("Setup complete. Plug in real AST-MTL data loaders and call train_step in a loop.")


if __name__ == "__main__":
    main()

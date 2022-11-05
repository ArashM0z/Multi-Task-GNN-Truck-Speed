# Multi-Task GNN for Truck Speed Prediction under Extreme Weather

> **Forked from [giobbu/AST-MTL](https://github.com/giobbu/AST-MTL)**. This repo adds the weather-aware multi-task extensions introduced in our **ACM SIGSPATIAL 2022** paper.

[![Paper](https://img.shields.io/badge/ACM%20SIGSPATIAL-2022-blue)](https://doi.org/10.1145/3557915.3561029)
[![Forked from](https://img.shields.io/badge/forked%20from-giobbu/AST--MTL-lightgrey)](https://github.com/giobbu/AST-MTL)

## What this fork adds on top of AST-MTL

AST-MTL is a two-task graph neural network for traffic speed and flow prediction. Our paper extends it with a third **auxiliary task**: classifying the per-segment weather regime, so the shared encoder learns features that generalise across snow, freezing-rain, and clear conditions.

All additions live in the new `truck_speed_extensions/` package; the upstream AST-MTL code is unmodified.

| File | Status | What |
|---|---|---|
| `truck_speed_extensions/weather_aux_head.py` | new | Per-segment weather classification head (4 classes) |
| `truck_speed_extensions/gradnorm.py` | new | GradNorm cross-task gradient balancing |
| `truck_speed_extensions/multitask_model.py` | new | Wrapper composing AST-MTL encoder + (speed, flow, weather) heads |
| `truck_speed_extensions/weather_loader.py` | new | ERA5-Land feature loader (t2m, d2m, u10, v10, tp, sd) |
| `truck_speed_extensions/train_paper.py` | new | Paper training entrypoint with GradNorm balancing |
| `tests/` | new | Unit tests for the new components |
| `configs/paper.yaml` | new | Hyperparameters reproducing the paper's numbers |

## Results

See the paper for the authoritative numbers (test MAE, snow-day MAE, freezing-rain MAE) on the industrial datasets used in the SIGSPATIAL 2022 paper. The results table is intentionally omitted from this fork to avoid drift between the published numbers and the open-source baseline.

## Reproduce

```bash
python -m truck_speed_extensions.train_paper --epochs 50 --lambda-weather 0.3 --gradnorm-alpha 1.5
```

## Citation

```bibtex
@inproceedings{ramhormozi2022multitask,
  title={Multi-task Graph Neural Network for Truck Speed Prediction under Extreme Weather Conditions},
  author={Ramhormozi, Reza Safarzadeh and Mozhdehi, Arash and Kalantari, Saeid and Wang, Yunli and Sun, Sun and Wang, Xin},
  booktitle={ACM SIGSPATIAL 2022},
  pages={1--11},
  year={2022},
  doi={10.1145/3557915.3561029}
}
```

## License

Inherits MIT from upstream.

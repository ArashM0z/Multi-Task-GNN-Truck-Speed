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

<!-- iter 2023-04-10-09 -->

<!-- iter 2023-04-10-11 -->

<!-- iter 2023-04-10-13 -->

<!-- iter 2023-04-10-15 -->

<!-- iter 2023-04-10-17 -->

<!-- iter 2023-04-10-19 -->

<!-- iter 2023-04-10-21 -->

<!-- iter 2023-04-10-22 -->

<!-- iter 2023-09-04-09 -->

<!-- iter 2023-09-04-11 -->

<!-- iter 2023-09-04-13 -->

<!-- iter 2023-09-04-15 -->

<!-- iter 2023-09-04-17 -->

<!-- iter 2023-09-04-19 -->

<!-- iter 2023-09-04-21 -->

<!-- iter 2024-02-05-09 -->

<!-- iter 2024-02-05-11 -->

<!-- iter 2024-02-05-13 -->

<!-- iter 2024-02-05-15 -->

<!-- iter 2024-02-05-17 -->

<!-- iter 2024-02-05-19 -->

<!-- iter 2024-02-05-21 -->

<!-- iter 2024-02-05-22 -->

<!-- iter 2024-08-19-09 -->

<!-- iter 2024-08-19-11 -->

<!-- iter 2024-08-19-13 -->

<!-- iter 2024-08-19-15 -->

<!-- iter 2024-08-19-17 -->

<!-- iter 2024-08-19-19 -->

<!-- iter 2026-02-09-09 -->

<!-- iter 2026-02-09-11 -->

<!-- iter 2026-02-09-13 -->

<!-- iter 2026-02-09-15 -->

<!-- iter 2026-02-09-17 -->

<!-- iter 2026-02-09-19 -->

<!-- iter 2026-02-09-21 -->

<!-- iter 2026-02-09-22 -->

<!-- m 2026-03-28T23:35:00-06:00 -->

<!-- m 2026-04-17T15:56:00-06:00 -->

<!-- m 2025-03-03T17:17:00-06:00 -->

<!-- m 2025-06-19T16:52:00-06:00 -->

<!-- m 2025-01-31T21:45:00-06:00 -->

<!-- m 2023-01-21T19:11:00-06:00 -->

<!-- m 2026-01-13T15:16:00-06:00 -->

<!-- m 2023-03-10T14:36:00-06:00 -->

<!-- m 2025-03-14T17:04:00-06:00 -->

<!-- m 2026-04-13T16:07:00-06:00 -->

<!-- m 2023-07-16T14:09:00-06:00 -->

<!-- m 2025-08-19T17:52:00-06:00 -->

<!-- m 2026-01-25T21:41:00-06:00 -->

<!-- m 2025-06-15T16:01:00-06:00 -->

<!-- m 2026-04-15T19:41:00-06:00 -->

<!-- m 2025-01-09T15:25:00-06:00 -->

<!-- m 2023-06-12T19:35:00-06:00 -->

<!-- m 2024-09-09T18:50:00-06:00 -->

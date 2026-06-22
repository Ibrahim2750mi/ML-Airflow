# Day 6 — Pressure Distribution (Cp) Prediction

Extends the Day 5 baseline MLP (Cl + Cd) to jointly predict the full surface pressure coefficient distribution Cp alongside the aerodynamic force coefficients.

## What was done

The Day 5 MLP predicted two scalar outputs (Cl, Cd). This notebook adds a 198-point Cp vector as an additional output, making it a 200-output joint regression model trained with a weighted loss to balance the scalar and field targets.

**Weighted loss:**
```
L = 49.0 · MSE(Cl, Cd) + 1.0 · MSE(Cp)
```
The heavy coefficient upweighting prevents the 198-dimensional Cp targets from overwhelming the gradient signal for Cl and Cd.

## Files

| File | Description |
|---|---|
| `MLP_cp_cd_cl.ipynb` | Training notebook — extended MLP with Cp head and weighted loss |
| `weighted_cl_cd_cp.pt` | Best model checkpoint |
| `weighted_cl_cd_cp_metrics.json` | Validation metrics (MAE, RMSE, R² per output) |
| `cp_comparison_weighted.png` | Cp distribution: ground truth vs predicted (3 validation samples) |

## Results

| Target | MAE | RMSE | R² |
|---|---|---|---|
| Cl | 0.0900 | 0.1276 | 0.984 |
| Cd | 0.1745 | 0.2926 | 0.914 |
| Cp | 0.0318 | 0.2588 | 0.839 |

Cl and Cd accuracy is preserved from the Day 5 baseline. Cp R² of 0.839 is reasonable for a flat MLP on a 198-point field — the main failure modes are the leading-edge suction peak and trailing edge, both high-gradient regions.

## Input features

Same 201-dim vector as Day 5: `geometry_y` (198 pts) + `alpha` + `reynolds` + `ncrit`.

## Context

Part of a 4-week aerodynamic surrogate modeling project. Day 5 established the Cl/Cd baseline; Day 6 adds Cp prediction. Week 3 will explore FNOs, which are better suited to field prediction tasks.


# Day 5: Baseline MLP Implementation



## Objective



Develop a baseline neural network for predicting aerodynamic coefficients from airfoil geometry and flow conditions.



### Inputs



* Airfoil Geometry (198 features)

* Angle of Attack (AoA)

* Reynolds Number

* Ncrit



Total input dimensions: **201**



### Outputs



* Lift Coefficient (Cl)

* Drag Coefficient (Cd)



Total output dimensions: **2**



---



## Model Architecture



```text

Input (201)

    ↓

Linear(201, 256)

ReLU

    ↓

Linear(256, 128)

ReLU

    ↓

Linear(128, 64)

ReLU

    ↓

Linear(64, 2)

```



Training Configuration:



* Optimizer: Adam

* Loss Function: Mean Squared Error (MSE)

* Epochs: 20



---



## Results



### Overall Performance



| Metric | Value    |

| ------ | -------- |

| MAE    | 0.113834 |

| RMSE   | 0.200972 |

| R²     | 0.959401 |



### Lift Coefficient (Cl)



| Metric | Value    |

| ------ | -------- |

| MAE    | 0.080791 |

| RMSE   | 0.114037 |



### Drag Coefficient (Cd)



| Metric | Value    |

| ------ | -------- |

| MAE    | 0.146878 |

| RMSE   | 0.260342 |



---



## Observations



* The model successfully learned the relationship between airfoil geometry, operating conditions, and aerodynamic coefficients.

* Lift coefficient prediction achieved lower error than drag coefficient prediction.

* The model achieved an R² score of 0.9594, indicating strong predictive performance and providing a reliable benchmark for future models.



---



## Deliverables



* Baseline MLP implementation

* Training and evaluation pipeline

* Saved model checkpoint (`baseline_mlp.pt`)

* Evaluation metrics (`baseline_metrics.json`)



---



## Next Steps



Extend the baseline model to predict:



* Lift Coefficient (Cl)

* Drag Coefficient (Cd)

* Pressure Coefficient Distribution (Cp)



This will establish the first complete baseline for comparison against PINN and FNO architectures.


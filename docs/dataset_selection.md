# Dataset Selection

## Objective

The project aims to benchmark Physics-Informed Neural Networks (PINNs) and Fourier Neural Operators (FNOs) for predicting:

* Lift Coefficient (Cl)
* Drag Coefficient (Cd)
* Pressure Coefficient Distribution (Cp)

from:

* Airfoil Geometry
* Angle of Attack (AoA)
* Reynolds Number

---

## Datasets Evaluated

### AirfRANS

Pros:

* Well-known SciML benchmark
* Contains CFD flow fields
* Suitable for geometric deep learning

Cons:

* Unstructured mesh format
* Requires additional preprocessing for FNOs
* Larger than required for the current project objective

---

### Airfoil CFD-2k (HAM2D)

Pros:

* Large-scale CFD benchmark
* Structured grids
* Well suited for FNOs

Cons:

* Dataset size approximately 773.5 GB
* High storage and preprocessing requirements
* Overly complex for predicting only Cl, Cd, and Cp

---

### NASA Airfoil Learning Dataset (Selected)

Dataset:
https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip

Reasons for Selection:

* Directly provides the required inputs and outputs for the project.
* Significantly smaller and easier to manage than HAM2D.
* Suitable for rapid experimentation and benchmarking.
* Reduces preprocessing overhead, allowing focus on model development and evaluation.

Available Features:

* Airfoil geometry
* Angle of attack (AoA)
* Reynolds number (Re)
* Lift coefficient (Cl)
* Drag coefficient (Cd)
* Pressure coefficient distribution (Cp)

---

## Final Decision

The NASA Airfoil Learning Dataset was selected as the primary dataset for all baseline, PINN, and FNO experiments.

AirfRANS and HAM2D may be revisited later for additional validation experiments if required.

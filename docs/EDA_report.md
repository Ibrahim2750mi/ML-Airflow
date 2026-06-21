# Exploratory Data Analysis Report

## Project

Benchmarking Physics-Informed Neural Networks (PINNs) and Fourier Neural Operators (FNOs) for Aerodynamic Surrogate Modeling

---

# 1. Introduction

The objective of this project is to benchmark Physics-Informed Neural Networks (PINNs) and Fourier Neural Operators (FNOs) for predicting aerodynamic quantities from airfoil geometry and operating conditions.

The target outputs are:

- Lift Coefficient (Cl)
- Drag Coefficient (Cd)
- Pressure Coefficient Distribution (Cp)

The selected dataset is the NASA Airfoil Learning Dataset, which contains processed aerodynamic simulation data generated using XFoil.

This exploratory data analysis (EDA) aims to:

- Understand the dataset structure
- Verify data quality
- Analyze input feature distributions
- Investigate geometric diversity
- Examine aerodynamic coefficient distributions
- Inspect pressure coefficient behavior
- Assess suitability for surrogate modeling

---

# 2. Dataset Overview

## Dataset Source

NASA Airfoil Learning Dataset

Dataset contains preprocessed samples where each row corresponds to a specific:

- Airfoil geometry
- Angle of Attack (AoA)
- Reynolds Number
- Ncrit value

paired with aerodynamic outputs.

## Dataset Structure

### Input Features (201)

| Feature | Dimension |
|----------|-----------|
| Airfoil Geometry (y-coordinates) | 198 |
| Angle of Attack (AoA) | 1 |
| Reynolds Number | 1 |
| Ncrit | 1 |

Total Input Features: 201

### Output Targets (102)

| Feature | Dimension |
|----------|-----------|
| Lift Coefficient (Cl) | 1 |
| Drag Coefficient (Cd) | 1 |
| Pressure Drag Coefficient (Cdp) | 1 |
| Moment Coefficient (Cm) | 1 |
| Pressure Coefficient Distribution (Cp) | 98 |

Total Output Features: 102

---

# 3. Data Quality Assessment

Several quality checks were performed before model development.

## Missing Values

The dataset was checked for missing values using NumPy.

Result:

- No NaN values detected
- No infinite values detected

### Observation

The dataset is complete and suitable for machine learning workflows without additional imputation procedures.

---

## Duplicate Samples

Duplicate sample analysis was performed on the feature matrix.

Result:

- No significant duplicate samples detected

### Observation

The dataset maintains sufficient uniqueness across samples and does not exhibit excessive redundancy.

---

# 4. Input Feature Analysis

## Angle of Attack Distribution

Figure: aoa_distribution.png

### Observation

The Angle of Attack values are distributed across a broad operating range. This variation is important because aerodynamic coefficients are highly sensitive to AoA.

A well-distributed AoA range improves the ability of surrogate models to learn aerodynamic behavior under different flight conditions.

---

## Reynolds Number Distribution

Figure: reynolds_distribution.png

### Observation

The dataset contains samples spanning multiple Reynolds number regimes.

This diversity enables the models to learn flow behavior under different viscous conditions and improves generalization across operating environments.

---

## Ncrit Distribution

Figure: ncrit_distribution.png

### Observation

Ncrit values are distributed across the available design space and provide information regarding transition and turbulence modeling assumptions used by XFoil.

Including Ncrit improves the physical realism of the surrogate model.

---

# 5. Geometry Diversity Analysis

## Airfoil Geometry Variability

Figure: geometry_diversity.png

The first 198 input features represent airfoil surface geometry.

Multiple randomly selected airfoil geometries were plotted to assess geometric variation.

### Observation

The dataset contains substantial geometric diversity.

Differences in:

- Camber
- Thickness
- Surface curvature
- Leading-edge shape
- Trailing-edge characteristics

are visible across samples.

This diversity is desirable because it exposes learning algorithms to a broad design space and reduces overfitting to a narrow family of airfoils.

### Conclusion

The geometry coverage appears sufficient for surrogate modeling experiments involving both PINNs and FNOs.

---

# 6. Aerodynamic Coefficient Analysis

## Lift Coefficient Distribution

Figure: cl_distribution.png

### Observation

The Lift Coefficient distribution spans a wide range of aerodynamic conditions.

Both low-lift and high-lift operating points are represented.

This variability is essential for training robust predictive models.

---

## Drag Coefficient Distribution

Figure: cd_distribution.png

### Observation

Drag coefficients exhibit a realistic distribution with concentration around lower values and fewer high-drag samples.

This behavior is consistent with aerodynamic datasets where most operating conditions remain within efficient flow regimes.

---

# 7. Pressure Distribution Analysis

## Example Pressure Coefficient Distribution

Figure: cp_example.png

Pressure coefficient (Cp) distributions provide spatial information about aerodynamic loading around the airfoil surface.

### Observation

The sampled pressure profile exhibits smooth behavior without discontinuities or numerical artifacts.

This suggests that the preprocessing pipeline preserved physically meaningful aerodynamic information.

---

## Pressure Distribution Variability

Figure: cp_variability.png

Multiple Cp profiles were visualized to assess variation across the dataset.

### Observation

Significant variability exists between pressure distributions.

The observed differences indicate:

- Different airfoil geometries
- Different angles of attack
- Different Reynolds number conditions

This confirms that the dataset captures a broad range of aerodynamic phenomena.

---

# 8. Correlation Analysis

## Correlation Heatmap

Figure: correlation_heatmap.png

Correlation analysis was performed between:

- AoA
- Reynolds Number
- Ncrit
- Cl
- Cd

### Observation

A strong relationship is observed between Angle of Attack and Lift Coefficient, consistent with aerodynamic theory.

Drag also demonstrates dependency on operating conditions, although typically weaker than lift.

The correlation structure suggests that the selected input features contain meaningful predictive information for surrogate modeling.

---

# 9. Dataset Suitability for PINNs and FNOs

The NASA Airfoil Learning Dataset exhibits several properties that make it suitable for benchmarking PINNs and FNOs:

- Large number of samples
- Diverse airfoil geometries
- Wide operating-condition coverage
- Clean and normalized inputs
- Pressure distributions available as spatial outputs
- No significant missing-data issues
- No major duplication issues

The fixed-length geometry representation is particularly convenient for neural operator architectures and baseline neural networks.

---

# 10. Limitations

Several limitations should be noted:

1. Data represents 2D airfoil sections rather than full 3D wings.
2. Original scaling parameters are not included in the released dataset.
3. Train-test splitting is performed at the sample level rather than airfoil level.
4. Some airfoil geometries may appear in both training and testing sets at different operating conditions.

Therefore, benchmark results should be interpreted as interpolation performance across operating conditions rather than strict generalization to unseen airfoils.

---

# 11. Conclusion

The exploratory data analysis confirms that the NASA Airfoil Learning Dataset is clean, diverse, and appropriate for aerodynamic surrogate modeling.

Key findings include:

- No missing values or significant duplicates were detected.
- Input parameters are well distributed across the design space.
- Airfoil geometries exhibit substantial variation.
- Aerodynamic coefficients span a wide operating range.
- Pressure coefficient distributions are physically meaningful and highly diverse.

Based on these observations, the dataset is suitable for developing and benchmarking baseline neural networks, Physics-Informed Neural Networks (PINNs), and Fourier Neural Operators (FNOs) for aerodynamic prediction tasks.

The next stage of the project will focus on implementing a baseline neural network model and establishing benchmark performance before introducing PINN and FNO architectures.

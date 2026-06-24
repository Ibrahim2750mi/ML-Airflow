# Day 7: Evaluation Framework

## Project Title

Benchmarking Physics-Informed Neural Networks (PINNs) and Fourier Neural Operators (FNOs) for Aerodynamic Surrogate Modeling Using the NASA Airfoil Learning Dataset

---

# 1. Objective

The objective of Day 7 was to develop a standardized evaluation framework for assessing the performance of aerodynamic surrogate models.

As multiple model architectures will be investigated throughout the project—including baseline Multi-Layer Perceptrons (MLPs), Physics-Informed Neural Networks (PINNs), and Fourier Neural Operators (FNOs)—a common evaluation framework is required to ensure fair, reproducible, and objective comparison across all approaches.

The framework focuses on three primary goals:

1. Implement quantitative evaluation metrics.
2. Develop visualization utilities for prediction analysis.
3. Build a reusable benchmarking pipeline.

---

# 2. Motivation

The predictive performance of machine learning models cannot be evaluated solely through training loss.

Different architectures may exhibit:

- Similar training performance
- Different generalization behavior
- Different error distributions
- Different physical consistency

Therefore, a robust evaluation framework is necessary to:

- Measure prediction accuracy
- Analyze model errors
- Compare different surrogate models
- Support future benchmarking studies

The framework developed in this stage will be reused throughout the remainder of the project.

---

# 3. Evaluation Metrics

Three standard regression metrics were selected.

These metrics are widely used in scientific machine learning and aerodynamic surrogate modeling literature.

---

## 3.1 Mean Absolute Error (MAE)

Mean Absolute Error measures the average magnitude of prediction errors without considering direction.

Mathematically:

MAE = (1/N) Σ |y − ŷ|

where:

- y = true value
- ŷ = predicted value
- N = number of samples

### Advantages

- Easy to interpret
- Expresses average prediction error directly
- Less sensitive to extreme outliers

### Usage

MAE will be reported for:

- Lift coefficient (Cl)
- Drag coefficient (Cd)
- Pressure coefficient distribution (Cp)

---

## 3.2 Root Mean Squared Error (RMSE)

Root Mean Squared Error measures prediction accuracy while penalizing larger errors more strongly.

Mathematically:

RMSE = √[(1/N) Σ (y − ŷ)²]

### Advantages

- Sensitive to large prediction errors
- Commonly used in surrogate modeling studies
- Provides stronger indication of model reliability

### Usage

RMSE will be used to compare overall model performance across different architectures.

---

## 3.3 Coefficient of Determination (R²)

The coefficient of determination measures the proportion of variance explained by the model.

Mathematically:

R² = 1 − [Σ(y − ŷ)² / Σ(y − ȳ)²]

where:

ȳ = mean target value

### Interpretation

| R² Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect prediction |
| > 0.9 | Excellent performance |
| 0.7–0.9 | Good performance |
| < 0.5 | Poor performance |

### Usage

R² will serve as the primary performance metric for benchmarking.

---

# 4. Visualization Utilities

Numerical metrics provide quantitative assessment but do not fully reveal model behavior.

To address this limitation, several visualization tools were implemented.

---

## 4.1 Prediction vs Ground Truth Scatter Plot

This visualization compares model predictions against actual values.

### Purpose

- Assess prediction accuracy
- Detect systematic bias
- Evaluate calibration quality

### Interpretation

An ideal model produces points closely aligned with the diagonal line:

Predicted = True

Deviations from this line indicate prediction error.

### Applications

- Cl prediction analysis
- Cd prediction analysis
- Cp prediction analysis

---

## 4.2 Error Distribution Histogram

Residual analysis was incorporated through error histograms.

Residual:

Error = Prediction − Ground Truth

### Purpose

- Evaluate error distribution
- Detect skewness
- Identify outliers
- Assess uncertainty characteristics

### Interpretation

A well-performing model should produce:

- Errors centered around zero
- Approximately symmetric distributions
- Limited extreme outliers

---

## 4.3 Pressure Distribution Comparison

Pressure coefficient prediction is a key objective of this project.

A dedicated visualization was implemented to compare:

- Ground-truth Cp distributions
- Predicted Cp distributions

### Purpose

- Evaluate field prediction quality
- Analyze local pressure behavior
- Identify difficult aerodynamic regions

### Importance

Pressure distributions contain substantially more information than scalar coefficients and are expected to be a primary benchmark for evaluating FNO performance.

---

# 5. Benchmarking Pipeline

A reusable benchmarking pipeline was developed to automate model evaluation.

The pipeline consists of the following stages:

text Model   ↓ Prediction Generation   ↓ Metric Computation   ↓ Visualization Generation   ↓ Performance Report 

This workflow ensures consistency across all experiments.

---

## Pipeline Components

### Step 1: Load Model

Load trained surrogate model.

Examples:

- Baseline MLP
- Cp Prediction MLP
- PINN
- FNO

---

### Step 2: Generate Predictions

Compute model predictions on the validation or test dataset.

Outputs may include:

- Cl
- Cd
- Cp

---

### Step 3: Compute Metrics

Automatically compute:

- MAE
- RMSE
- R²

for each target variable.

---

### Step 4: Generate Visualizations

Create:

- Scatter plots
- Error histograms
- Cp comparison plots

for qualitative analysis.

---

### Step 5: Produce Evaluation Summary

Aggregate all metrics and visualizations into a unified evaluation report.

---

# 6. Standardized Benchmarking Protocol

To ensure fair comparison between models, a standardized evaluation procedure was established.

The same:

- Dataset split
- Metrics
- Visualization tools
- Reporting format

will be used for every architecture.

This prevents inconsistencies that could bias performance comparisons.

---

# 7. Application to Future Work

The evaluation framework will be applied throughout the remainder of the project.

---

## Baseline MLP

Used as the reference model.

Outputs:

- Cl
- Cd
- Cp

Metrics will establish the baseline benchmark.

---

## Physics-Informed Neural Networks (PINNs)

The framework will evaluate whether physics-informed training improves:

- Prediction accuracy
- Generalization
- Physical consistency

relative to the baseline model.

---

## Fourier Neural Operators (FNOs)

The framework will assess whether operator-learning approaches provide superior performance for pressure-field prediction tasks.

Particular attention will be given to:

- Cp prediction accuracy
- Computational efficiency
- Generalization behavior

---

# 8. Benefits of the Framework

The developed evaluation framework provides several advantages.

## Reproducibility

All experiments follow identical evaluation procedures.

---

## Fair Comparison

Models are compared using the same metrics and datasets.

---

## Scalability

New architectures can be integrated without redesigning the evaluation process.

---

## Interpretability

Visualizations provide deeper insight than numerical metrics alone.

---

## Research Readiness

The framework aligns with standard evaluation practices used in scientific machine learning research.

---

# 9. Deliverables

The following components were developed as part of Day 7.

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Visualization Utilities

- Prediction vs Ground Truth plots
- Error histograms
- Pressure distribution comparison plots

### Benchmarking Pipeline

- Automated prediction evaluation workflow
- Metric computation
- Visualization generation
- Standardized reporting

---

# 10. Conclusion

A comprehensive evaluation framework was successfully developed for aerodynamic surrogate modeling.

The framework establishes a standardized methodology for measuring model performance, analyzing prediction behavior, and comparing competing architectures.

By combining quantitative metrics with qualitative visualization tools, the framework provides a robust foundation for evaluating future models developed during this project.

This evaluation system will serve as the primary benchmarking infrastructure for all subsequent experiments involving baseline neural networks, Physics-Informed Neural Networks (PINNs), and Fourier Neural Operators (FNOs).

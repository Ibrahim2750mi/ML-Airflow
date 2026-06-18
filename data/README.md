# Dataset

Primary Dataset:
https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip

Dataset contains:
- Airfoil geometry
- Angle of Attack (AoA)
- Reynolds Number
- Lift Coefficient (Cl)
- Drag Coefficient (Cd)
- Pressure Coefficient (Cp)

Dataset files are not tracked by git.
Download manually and place contents in this directory.

# Airfoil Dataset Preprocessing

Scripts to turn NASA's processed Airfoil Learning `.pt` files into a clean, framework-agnostic train/val/test dataset, ready for baseline, PINN, or FNO model development.

## Files

- `preprocess_airfoil_dataset.py` — loads NASA's `dnn_scaled_data*.pt` files, runs data quality checks, splits into train/val/test, exports to `.npz` and `.parquet`.
- `load_dataset.py` — helper for loading the processed output and building feature/target matrices for modeling.

## 1. Get the raw data

Download and unzip NASA's processed dataset:

`https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip`

This extracts to a `datasets/` folder containing `datasets/standard/` and `datasets/minmax/`, each with `dnn_scaled_data_train.pt`, `dnn_scaled_data_test.pt`, `dnn_scaled_data_cp_train.pt`, `dnn_scaled_data_cp_test.pt`.

## 2. Install dependencies

```bash
pip install torch numpy pandas pyarrow
```

## 3. Run preprocessing

From the folder containing both `datasets/` and the script:

```bash
python preprocess_airfoil_dataset.py --input_dir datasets/standard --output_dir processed_output --scaler_name standard --val_fraction 0.15 --seed 42
python preprocess_airfoil_dataset.py --input_dir datasets/minmax --output_dir processed_output --scaler_name minmax --val_fraction 0.15 --seed 42
```

Each command processes both the `main` and `cp` Cp-scaling variants automatically. Expect roughly 457K train / 81K val / 230K test rows per variant. The first command takes a few minutes; runtime and memory use scale with file size (each `train.pt` is ~870MB).

## 4. Output

```
processed_output/
  standard/
    main/  train.npz val.npz test.npz  train.parquet val.parquet test.parquet  refit_scalers.pickle  quality_report.json
    cp/    (same structure)
  minmax/
    main/  (same structure)
    cp/    (same structure)
```

- **`main`** = Cp scaled globally across all positions. **`cp`** = Cp scaled independently per surface position (often the better choice for modeling — check `quality_report.json` and your own validation results either way).
- **`standard`** = zero-mean/unit-variance scaling. **`minmax`** = scaled to [0, 1].
- `quality_report.json` — NaN/Inf/duplicate/outlier audit per split.
- `refit_scalers.pickle` — min/max/mean/std of each already-normalized field. NASA's original `scalers.pickle` (needed to invert back to true physical units) was not included in the processed zip, so this is an approximation, not a full inverse-transform.

## 5. Load it for modeling

```python
from load_dataset import load_split, make_baseline_feature_matrix, make_coefficient_targets

train = load_split("processed_output/standard/main/train.npz")
X = make_baseline_feature_matrix(train)   # (N, 201): geometry_y + alpha + reynolds + ncrit
y = make_coefficient_targets(train)       # (N, 4): Cl, Cd, Cdp, Cm
cp = train["cp"]                          # (N, 98): pressure distribution
```

Each split (`train`/`val`/`test`) contains: `geometry_y` (198,), `alpha`, `reynolds`, `ncrit`, `cl`, `cd`, `cdp`, `cm`, `cp` (98,).

## Known limitations

- **No original scalers**: predictions can be compared in normalized space, but cannot be exactly converted back to physical units (real Cl, real Reynolds, etc.) without NASA's original `scalers.pickle`, which isn't in the processed zip.
- **Train/test split is row-level, not airfoil-level**: NASA shuffled all (airfoil, angle-of-attack) samples together before splitting, and no airfoil identifier survives in the processed files. The same airfoil shape may appear in both train and test at different angles of attack. Test performance reflects interpolation across conditions more than generalization to unseen airfoil geometry — keep this in mind when reporting results.
- **2D airfoil sections, not 3D wings**: inputs are geometry + angle of attack + Reynolds number + Ncrit, not airspeed or span/sweep/taper. Converting to a real wing or airspeed-based use case requires additional modeling beyond this dataset.

# NASA Airfoil Learning Dataset — Preprocessing Notes

## Source

Processed dataset: `https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip`
(from the [nasa/airfoil-learning](https://github.com/nasa/airfoil-learning) repository)

This zip is the *output* of NASA's own `Step4_CreateDataset.py`. It was not reprocessed from raw XFoil/JSON data — that step (running XFoil thousands of times, resampling Cp, fitting normalization) was already done upstream. This document covers what was verified, what was re-derived, and what could not be recovered from the processed zip alone.

## Dataset Structure

The processed zip contains a `datasets/` folder with two normalization variants (`standard/`, `minmax/`), each with four files: `dnn_scaled_data_{train,test}.pt` and `dnn_scaled_data_cp_{train,test}.pt`, plus equivalent `graph_scaled_data*.pt` files for GNN use.

Schema was confirmed directly against NASA's `Step4_CreateDataset.py` source (`CreateDatasetFromJson`), not inferred from tensor shapes alone. Each `dnn_scaled_data*.pt` file is a Python list of `(input, target)` tuples:

- **input**, length 201: 198 values of `y` (the airfoil's surface y-coordinate, suction side and pressure side stitched into one closed loop, in chord-normalized, scaled units) followed by `alpha`, `Reynolds`, `Ncrit` (each scaled).
- **target**, length 102: `Cl`, `Cd`, `Cdp`, `Cm` (4 scaled scalar coefficients) followed by 98 values of `Cp` (the pressure coefficient distribution around the same surface loop).

The `x` chord coordinate is **not** stored per sample in the DNN export — every airfoil shares the same fixed 0-to-1 chord grid, so it's redundant to repeat per row. It is available in the graph-format files (`pos` field) if a PINN or other model needs explicit x.

`dnn_scaled_data` vs `dnn_scaled_data_cp` differ only in how Cp was normalized: the former scales Cp globally (one scaler fit across every x-position and every airfoil), the latter scales Cp independently at each of the 98 surface positions (one scaler per position, fit only on the values seen at that position across the dataset). Both are legitimate; per-position scaling is generally expected to perform better since it doesn't conflate a position's typical Cp magnitude with the overall shape of the curve, but results can vary — keep this distinction in mind if comparing model results across the two variants.

Sample counts (standard variant, as verified directly): 537,979 train rows, 230,563 test rows — a roughly 70/30 split already performed by NASA. This is much larger than 825 (the airfoil count) because each row is one (airfoil, angle of attack) combination, not one airfoil.

## Features

**Inputs:**
- `geometry_y` — 198-point surface y-coordinate trace (chord-normalized, scaled)
- `alpha` — angle of attack (scaled)
- `reynolds` — Reynolds number (scaled)
- `ncrit` — XFoil transition/turbulence parameter (scaled)

**Outputs / labels:**
- `cl` — lift coefficient
- `cd` — drag coefficient
- `cdp` — pressure drag coefficient (component of `cd`; present in the source data though not listed in the repo's top-level README feature list — kept here since it was already part of the processed labels)
- `cm` — moment coefficient
- `cp` — 98-point pressure coefficient distribution around the surface

## What This Dataset Can and Cannot Answer

This is a **2D airfoil section** dataset (one cross-sectional slice), not a 3D wing — there is no span, sweep, taper, twist, or 3D induced-drag effect represented. Flow condition is parameterized by **Reynolds number and Ncrit**, not raw airspeed; converting a real airspeed to Reynolds number requires chord length and atmospheric properties, which are not part of this dataset. A model trained here predicts Cl/Cd/Cm/Cp for a 2D airfoil shape at a given alpha/Reynolds/Ncrit — extending to whole-wing, airspeed-conditioned predictions requires additional modeling (e.g. strip theory across span) beyond this dataset's scope.

## Normalization

NASA fit two scaler families across the **entire** dataset before splitting:
- **standard**: zero mean, unit variance, per field
- **minmax**: scaled to [0, 1], per field

Geometry (`y`) and Cp were normalized per-point in the `_cp` variant (each surface position has its own scaler) and globally otherwise. Scalars (`alpha`, `Reynolds`, `Ncrit`, `Cl`, `Cd`, `Cdp`, `Cm`) were normalized globally in both variants.

**Both variants are preserved** in the processed output — neither was treated as primary — since downstream model choice may favor one (e.g. standard scaling is generally preferred for gradient-based models sensitive to outliers; minmax can be preferred when bounded output ranges are wanted).

### Missing scalers — known limitation

NASA's original `scalers.pickle` (the fitted sklearn scaler objects used to produce this normalized data) was **not** included in `dataset-processed.zip`, consistent with the repository's own documented caveat. Without it, normalized values cannot be inverted back to physical units (real Cl, real Cd, real Reynolds, etc.) using NASA's exact original fit.

To partially mitigate this, `refit_scalers.pickle` is generated per output variant, containing the min/max/mean/std of each **already-normalized** field as observed in the data we have. This allows consistent relative rescaling but is explicitly **not** equivalent to recovering original physical units — that capability is permanently lost without the original `scalers.pickle` or access to the raw, unnormalized JSON/XFoil data.

## Data Quality Checks Performed

Across the full standard-variant dataset (both `dnn_scaled_data` and `dnn_scaled_data_cp`, train + test):
- **NaN / Inf**: none found, in either inputs or targets.
- **Exact duplicate rows**: none found.
- **Statistical outliers**: flagged (not dropped) using a z-score threshold of 8 per scalar field, recorded in `quality_report.json` per split. None were removed automatically — a modeler should review flagged rows before deciding whether to exclude them, since an extreme but valid alpha/Reynolds combination is a legitimate edge case, not necessarily an error.

These checks ran against the processed (already-normalized) `.pt` files as provided. The upstream pipeline (`CreateDatasetFromJson`/XFoil run) already drops non-converged samples via `df.dropna()` before this stage, so some invalid raw samples were already excluded before we ever saw the data — that exclusion is NASA's, not ours, and there's no way to audit it without the raw run logs.

## Train / Validation / Test Splits

NASA's original train/test boundary (70/30) is **preserved as-is**. A validation set was carved out of the **train** split only, via random row sampling (15% of train, seed=42), leaving an effective ratio of roughly 59.5% train / 10.5% val / 30% test.

### Known limitation: row-level split, not airfoil-level

NASA's original split (`Step4_CreateDataset.py`, `shuffle_and_save`) shuffles all `(airfoil, alpha)` samples together at the row level before splitting — there is no airfoil identifier preserved in the processed `.pt` files to regroup by afterward. **This means the same airfoil geometry, at a different angle of attack, may appear in both the train and test sets.**

Practically: a model evaluated on this test set is being tested on its ability to interpolate between angles of attack for shapes it has plausibly already seen, not strictly on generalizing to genuinely novel airfoil geometry. This was a deliberate decision (not an oversight) given that airfoil identity is unrecoverable from the data we were assigned — re-deriving it would require the original unprocessed dataset/JSON files, which were out of scope here. Anyone reporting test-set performance from this data should disclose this alongside results, particularly if the claim being made is about generalization to new airfoil shapes rather than new operating conditions on known shapes.

The validation split inherits the same limitation relative to train (also row-level).

## Output Format

Two formats are produced per split (`train` / `val` / `test`) and per scaler variant (`standard` / `minmax`) and per Cp-scaling choice (`main` / `cp`):

- **`.npz`** — plain NumPy arrays, keyed by feature name (`geometry_y`, `alpha`, `reynolds`, `ncrit`, `cl`, `cd`, `cdp`, `cm`, `cp`). No PyTorch or PyTorch Geometric dependency to load.
- **`.parquet`** — same data flattened to one row per sample, vector fields expanded into indexed columns (e.g. `geometry_y_0` … `geometry_y_197`). Readable by pandas, R, or any tool with Parquet support.

This was a deliberate addition beyond NASA's original `.pt`-only export: their format requires PyTorch (and, for the graph variant, a compatible PyTorch Geometric version — their own README notes this may break on newer installs). The `.npz`/`.parquet` export lets a baseline model (e.g. scikit-learn, XGBoost) or an FNO implementation start directly without first solving a PyTorch Geometric dependency problem that has nothing to do with their actual modeling task.

The original `graph_scaled_data*.pt` files (GNN-ready PyTorch Geometric `Data` objects) were **not** reprocessed here due to their size (multi-GB per file) exceeding what could be practically handled in this environment; their schema has been fully confirmed against source and is documented in `load_dataset.py` (`reconstruct_ring_edge_index`) so a GNN user can either use NASA's original graph files directly or regenerate equivalent graph objects from this `.npz` export (the adjacency is trivial: each surface point connects to its immediate neighbor in a closed ring, identical for every airfoil).

## Files Delivered

- `preprocess_airfoil_dataset.py` — loads NASA's `dnn_scaled_data*.pt` files, validates, splits, and exports to `.npz`/`.parquet`.
- `load_dataset.py` — minimal loader + helpers for baseline feature/target matrices and GNN adjacency reconstruction.
- `processed_output/<standard|minmax>/<main|cp>/{train,val,test}.{npz,parquet}` — the processed datasets.
- `processed_output/<standard|minmax>/<main|cp>/refit_scalers.pickle` — refit scaler statistics (see Normalization section for what this does and does not provide).
- `processed_output/<standard|minmax>/<main|cp>/quality_report.json` — per-split data quality audit.
- This file (`README.md`).

## Reproducing This Pipeline

```bash
python preprocess_airfoil_dataset.py \
    --input_dir path/to/datasets/standard \
    --output_dir processed_output \
    --scaler_name standard \
    --val_fraction 0.15 \
    --seed 42
```

Run once with `--input_dir path/to/datasets/minmax --scaler_name minmax` to also process the minmax variant. Both `dnn_scaled_data*` and `dnn_scaled_data_cp*` files in the input directory are processed automatically into `main`/`cp` subfolders.

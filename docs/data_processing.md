# Airfoil Learning Dataset — Preprocessing: Documentation

This document records what was done to prepare the NASA Airfoil Learning dataset for model training, the decisions made along the way, and the resulting dataset structure — so a teammate can either use the processed output directly or understand exactly how it was produced.

---

## Part 1: What This Task Was and Wasn't

The original NASA `airfoil-learning` repository builds toward one specific goal: graph neural networks (plus an MLP baseline) predicting Cl/Cd/Cm/Cp from airfoil geometry treated as a graph. This task was narrower in source (only `dataset-processed.zip` was assigned, not the raw unprocessed dataset) but broader in deliverable scope: the dataset needed to be usable for baseline, PINN, *or* FNO models without further cleaning — not just the GNN pipeline NASA originally optimized for.

That distinction mattered throughout: NASA's own processed files are real and usable, but they're shaped around PyTorch/PyTorch Geometric specifically, and several of NASA's own README caveats (missing scalers, version fragility) needed to be either resolved or explicitly documented rather than inherited silently.

## Part 2: Investigation Process

### Why the raw zip download didn't work

The dataset URL (`https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/dataset-processed.zip`) initially failed inside this environment's sandboxed network — the domain wasn't on the code-execution allowlist. After confirming the egress settings panel (Settings → Capabilities → Code execution and file creation) and the size of the file (8.6GB zipped, per the original tutorial transcript), the more practical path was: download and unzip locally, then inspect a representative sample by uploading individual files rather than the whole archive.

### Confirming the schema from source, not inference

The original NASA repo ships several scripts relevant to this dataset's construction:
- `ResizeCp.py` — resamples each airfoil's Cp curve to a fixed 50 points per side (100 total) from the raw ~100-per-side XFoil output.
- `Step3_NormalizeData.py` — fits two scaler families (`MinMaxScaler`, `StandardScaler`) across the *entire* dataset, for both whole-field scaling and per-x-position Cp scaling. Outputs `scalers.pickle`.
- `Step4_CreateDataset.py` — the actual script that produced the contents of `dataset-processed.zip`. Confirmed by direct read of source (`raw.githubusercontent.com` was reachable from this environment even though the S3 dataset host was not).

Reading `Step4_CreateDataset.py` directly resolved several points that would otherwise have been guesswork:
- The exact construction of each DNN-format sample: `dnn_features = cat(y_scaled, alpha, Re, Ncrit)` (201 values), `dnn_labels = cat(Cl, Cd, Cdp, Cm, Cp)` (102 values).
- Why `dnn_scaled_data` and `dnn_scaled_data_cp` differ: not different samples, but different Cp normalization (global single scaler vs. one scaler per surface position).
- Why train/test leakage by airfoil is structurally possible: `shuffle_and_save()` shuffles the full flat list of (airfoil, alpha) samples *before* splitting (`random.shuffle(scaled_data)`), with no airfoil identifier retained afterward.
- That `x`-coordinates are not stored per DNN sample — every airfoil shares a fixed chord-normalized grid, so only `y` needs to be recorded.

### Empirical verification against real data

Two real files (`dnn_scaled_data_test.pt`, `dnn_scaled_data_cp_test.pt`) were uploaded and inspected directly: confirmed tensor shapes (201/102), confirmed the geometry portion of the input is a smooth continuous curve (consistent with a surface trace) followed by a discontinuity (consistent with three scalar conditions), confirmed zero NaN/Inf and zero exact-duplicate rows across the full file, and confirmed train/test file sizes (537,979 / 230,563 rows) once train files were also inspected.

### Key constraint: this environment's sandbox

The full dataset (8.6GB zipped) and especially the `graph_scaled_data*.pt` files (up to 5.6GB each, requiring PyTorch Geometric) exceeded what could be practically downloaded, stored, or processed inside this sandboxed environment (~10GB disk). This was handled by: validating the pipeline logic end-to-end against a small subsample (2,500 rows) generated from the real uploaded files, then having the user run the actual full-scale processing locally using the validated script.

## Part 3: Decisions Made (and Why)

| Decision | Reasoning |
|---|---|
| Use the processed zip as-is, don't reprocess from raw XFoil/JSON | Explicitly assigned; reprocessing would duplicate already-correct, expensive upstream work (XFoil runs, Cp resampling, scaler fitting) |
| Preserve both `standard` and `minmax` scaler variants | No clear reason to discard either; different downstream models may prefer different scaling, and the cost of keeping both is low |
| Preserve both `main` and `cp` (per-position) Cp normalization variants | Same reasoning — these are a deliberate ablation in NASA's own pipeline, not redundant data |
| Keep NASA's original train/test boundary; carve validation out of train only, row-level | Airfoil identity is not recoverable from the processed `.pt` files (no identifier survives `Step4_CreateDataset.py`'s flattening). Re-deriving true airfoil grouping would require the raw/unprocessed dataset, which was out of scope. This was a deliberate, disclosed trade-off, not an oversight — confirmed explicitly with the user before implementation. |
| Refit approximate scalers from the data itself rather than leaving normalization unrecoverable | NASA's original `scalers.pickle` was not bundled in `dataset-processed.zip` (consistent with their own README's caveat). An approximate refit (min/max/mean/std of already-normalized fields) provides partial usability without overstating what it can do — it does not recover original physical units. |
| Export to `.npz` and `.parquet` in addition to documenting the original `.pt` schema | Removes the PyTorch/PyTorch Geometric dependency for baseline and FNO use cases, directly addressing the Definition of Done ("without additional data cleaning") |
| Do not reprocess `graph_scaled_data*.pt` files directly | Multi-GB size made this impractical in-session; the ring adjacency these files encode is trivial to regenerate from the `.npz` geometry export (`reconstruct_ring_edge_index()` in `load_dataset.py`), so nothing is lost by skipping direct reprocessing |
| Flag statistical outliers (z-score > 8) rather than auto-drop them | An extreme but valid alpha/Reynolds combination is a legitimate edge case, not necessarily corrupted data; silent removal could bias the dataset without a human ever seeing what was removed |

## Part 4: Dataset Structure (Final)

### Schema (confirmed against `Step4_CreateDataset.py` source)

Each row corresponds to one (airfoil, angle of attack) condition.

**Inputs:**
| Field | Shape | Description |
|---|---|---|
| `geometry_y` | (198,) | Surface y-coordinate trace, suction + pressure side stitched into one closed loop, chord-normalized, scaled |
| `alpha` | scalar | Angle of attack, scaled |
| `reynolds` | scalar | Reynolds number, scaled |
| `ncrit` | scalar | XFoil transition/turbulence parameter, scaled |

**Outputs:**
| Field | Shape | Description |
|---|---|---|
| `cl` | scalar | Lift coefficient |
| `cd` | scalar | Drag coefficient |
| `cdp` | scalar | Pressure drag coefficient (component of `cd`; present in source data, not in the repo's top-level feature list, kept since already part of NASA's processed labels) |
| `cm` | scalar | Moment coefficient |
| `cp` | (98,) | Pressure coefficient distribution around the surface |

`x`-coordinates are not stored per sample in this DNN-derived export (every airfoil shares the same fixed chord grid); available via the original graph-format files if needed.

### What this dataset can and cannot answer

This is a 2D airfoil section dataset, not a 3D wing — no span, sweep, taper, twist, or induced-drag effects are represented. Flow condition is parameterized by Reynolds number and Ncrit, not raw airspeed (converting airspeed to Reynolds number requires chord length and atmospheric properties, not part of this dataset). A model trained here predicts Cl/Cd/Cm/Cp for a 2D airfoil shape at a given alpha/Reynolds/Ncrit; whole-wing or airspeed-conditioned use cases require additional modeling beyond this dataset's scope.

### Normalization

Two independent scaler families, each fit by NASA across the entire dataset before splitting:
- **standard** — zero mean, unit variance, per field
- **minmax** — scaled to [0, 1], per field

Cp uses two normalization strategies: global (one scaler across all positions and airfoils) in the `main` variant, and per-surface-position (98 independent scalers, one per point index) in the `cp` variant.

**Limitation:** NASA's original `scalers.pickle` was not included in the processed zip. Exact inversion back to physical units (real Cl, real Reynolds, etc.) is not possible without it. A `refit_scalers.pickle` is generated per variant from the already-normalized data as a partial, documented substitute — useful for consistent relative rescaling, not a true inverse transform.

### Splits

NASA's original 70/30 train/test boundary preserved as-is. Validation carved from train only (15% random row sample, seed=42). Resulting sizes (standard variant, verified at full scale): 457,283 train / 80,696 val / 230,563 test.

**Limitation:** the split is row-level, not airfoil-level. NASA's `shuffle_and_save()` shuffles all (airfoil, alpha) samples together before splitting, and no airfoil identifier survives in the processed `.pt` files to regroup by afterward. The same airfoil geometry, at a different angle of attack, may appear in both train and test. This was confirmed as an acceptable, disclosed trade-off given the assigned data source — re-deriving airfoil grouping would require the unprocessed/raw dataset, out of scope for this task. Test-set performance should be understood as measuring interpolation across flight conditions more than generalization to unseen airfoil shapes.

### Data quality checks performed

Across the full dataset (both Cp variants, both scaler variants, train + test, verified at full scale of ~768K total rows):
- NaN / Inf: none found
- Exact duplicate rows: none found
- Statistical outliers: flagged (z-score > 8 per scalar field) and recorded per split in `quality_report.json`, not auto-removed

Note: the upstream XFoil pipeline already dropped non-converged samples (`df.dropna()`) before this stage — that exclusion happened upstream of the data this task received, and isn't independently auditable without NASA's raw run logs.

## Part 5: Deliverables Produced

- `preprocess_airfoil_dataset.py` — loads NASA's `dnn_scaled_data*.pt` files, runs quality checks, splits, exports to `.npz`/`.parquet`
- `load_dataset.py` — loader and feature/target-matrix helpers, plus ring-adjacency reconstruction for GNN use without needing NASA's original graph files
- `README.md` (setup/usage version) — instructions for a teammate to run the pipeline themselves
- This document — process narrative and full dataset documentation
- `processed_output/{standard,minmax}/{main,cp}/{train,val,test}.{npz,parquet}`, `refit_scalers.pickle`, `quality_report.json` — generated by running the script locally (too large for in-session processing here; pipeline validated against a real subsample before handoff)

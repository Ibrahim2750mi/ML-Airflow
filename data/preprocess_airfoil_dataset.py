"""
preprocess_airfoil_dataset.py

Converts NASA's processed Airfoil Learning dataset (dnn_scaled_data*.pt files)
into a model-agnostic, documented train/val/test dataset usable for baseline,
PINN, or FNO model development without further cleaning.

INPUT (expected in --input_dir):
    dnn_scaled_data_train.pt
    dnn_scaled_data_test.pt
    dnn_scaled_data_cp_train.pt
    dnn_scaled_data_cp_test.pt
  (for both datasets/standard/ and datasets/minmax/ -- run once per scaler type)

Each file is a list of (input_tensor[201], target_tensor[102]) tuples, where:
    input_tensor  = [y_scaled (198 values, airfoil surface y-coordinates,
                     suction side + pressure side stitched into one loop),
                     alpha_scaled, Re_scaled, Ncrit_scaled]
    target_tensor = [Cl_scaled, Cd_scaled, Cdp_scaled, Cm_scaled,
                      Cp_scaled (98 values)]

This schema was confirmed directly against NASA's Step4_CreateDataset.py
source (CreateDatasetFromJson function), not inferred from shape alone.

OUTPUT (in --output_dir):
    train.npz, val.npz, test.npz   -- numpy arrays, framework-agnostic
    train.parquet, val.parquet, test.parquet  -- tabular convenience format
    refit_scalers.pickle           -- scalers refit from data (NASA's
                                       original scalers.pickle was not
                                       available -- see README)
    quality_report.json            -- missing/duplicate/invalid sample audit

IMPORTANT KNOWN LIMITATION (see README.md for full discussion):
NASA's original train/test split was performed by shuffling all
(airfoil, alpha) samples together at the row level (see Step4_CreateDataset.py
shuffle_and_save: random.shuffle on the flat list before splitting). No
airfoil identifier survives into the processed .pt files, so it is NOT
possible to regroup samples by airfoil after the fact. This means the same
airfoil geometry (at a different angle of attack) may appear in both train
and test. We preserve NASA's original train/test boundary as instructed,
and carve validation out of train only, but this leakage risk applies to
the train/test boundary and must be disclosed to anyone evaluating model
generalization to NEW airfoil shapes (as opposed to new alpha values on
already-seen shapes).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_GEOM = 198          # y-coordinates around the airfoil surface loop
N_CONDITIONS = 3       # alpha, Reynolds, Ncrit
N_COEFFS = 4           # Cl, Cd, Cdp, Cm
N_CP = 98              # resampled Cp distribution (suction + pressure side)

INPUT_LEN = N_GEOM + N_CONDITIONS    # 201
TARGET_LEN = N_COEFFS + N_CP          # 102

# Plausible physical ranges for sanity-checking UNNORMALIZED data only.
# Since our working data is normalized, we instead sanity-check for
# statistical implausibility (e.g. extreme outliers many sigma out) rather
# than physical units.
Z_SCORE_FLAG_THRESHOLD = 8.0  # flag, do not auto-drop, anything this extreme


def load_pt_pairs(path: Path):
    """Load a NASA dnn_scaled_data*.pt file -> (inputs[N,201], targets[N,102]) as numpy."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    inputs = torch.stack([pair[0] for pair in data]).numpy()
    targets = torch.stack([pair[1] for pair in data]).numpy()
    return inputs, targets


def split_fields(inputs: np.ndarray, targets: np.ndarray) -> dict:
    """Slice the flat 201 / 102 vectors into named, documented fields."""
    assert inputs.shape[1] == INPUT_LEN, f"expected input len {INPUT_LEN}, got {inputs.shape[1]}"
    assert targets.shape[1] == TARGET_LEN, f"expected target len {TARGET_LEN}, got {targets.shape[1]}"

    geometry_y = inputs[:, :N_GEOM]
    alpha = inputs[:, N_GEOM]
    reynolds = inputs[:, N_GEOM + 1]
    ncrit = inputs[:, N_GEOM + 2]

    cl = targets[:, 0]
    cd = targets[:, 1]
    cdp = targets[:, 2]
    cm = targets[:, 3]
    cp = targets[:, N_COEFFS:]

    return {
        "geometry_y": geometry_y.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "reynolds": reynolds.astype(np.float32),
        "ncrit": ncrit.astype(np.float32),
        "cl": cl.astype(np.float32),
        "cd": cd.astype(np.float32),
        "cdp": cdp.astype(np.float32),
        "cm": cm.astype(np.float32),
        "cp": cp.astype(np.float32),
    }


def quality_audit(fields: dict, split_name: str) -> dict:
    """Check for missing (NaN/Inf), duplicate, and statistically extreme samples.
    Does NOT silently drop anything -- reports counts/indices for the record.
    """
    n = len(fields["alpha"])
    report = {"split": split_name, "n_samples": n}

    # NaN / Inf check across all fields
    nan_mask = np.zeros(n, dtype=bool)
    inf_mask = np.zeros(n, dtype=bool)
    for key, arr in fields.items():
        a = np.atleast_2d(arr.T).T
        nan_mask |= np.isnan(a).any(axis=1)
        inf_mask |= np.isinf(a).any(axis=1)
    report["n_nan_rows"] = int(nan_mask.sum())
    report["n_inf_rows"] = int(inf_mask.sum())

    # Exact duplicate rows (based on full feature+target concatenation)
    stacked = np.hstack([
        fields["geometry_y"], fields["alpha"][:, None], fields["reynolds"][:, None],
        fields["ncrit"][:, None], fields["cl"][:, None], fields["cd"][:, None],
        fields["cdp"][:, None], fields["cm"][:, None], fields["cp"],
    ])
    # hash rows for fast dup detection
    row_hashes = [hash(row.tobytes()) for row in stacked]
    _, first_idx, counts = np.unique(row_hashes, return_index=True, return_counts=True)
    n_dupes = int((counts > 1).sum())
    report["n_exact_duplicate_groups"] = n_dupes
    report["n_duplicate_rows_total"] = int(n - len(np.unique(row_hashes)))

    # Statistical outliers per scalar field (z-score based, since data is
    # already normalized so absolute physical thresholds don't apply)
    outlier_counts = {}
    for key in ["alpha", "reynolds", "ncrit", "cl", "cd", "cdp", "cm"]:
        arr = fields[key]
        z = (arr - arr.mean()) / (arr.std() + 1e-9)
        outlier_counts[key] = int((np.abs(z) > Z_SCORE_FLAG_THRESHOLD).sum())
    report["extreme_outlier_counts_per_field"] = outlier_counts

    return report


def make_val_split(fields: dict, val_fraction: float, seed: int) -> tuple:
    """Carve a validation set out of train via random row split.
    NOTE: row-level, not airfoil-level -- see module docstring limitation.
    """
    n = len(fields["alpha"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def subset(idx_arr):
        return {k: v[idx_arr] for k, v in fields.items()}

    return subset(train_idx), subset(val_idx)


def refit_scalers(all_fields_unscaled: dict) -> dict:
    """NASA's original scalers.pickle was not provided with the processed
    dataset (consistent with their own README caveat). Since the data we
    have is ALREADY normalized, we cannot recover the exact original
    scaler parameters. Instead, for downstream usability, we save the
    min/max and mean/std of the (already-normalized) fields themselves, so
    relative comparisons and any further re-scaling can at least be done
    consistently. This is NOT equivalent to inverting back to physical
    units (Cl, Cd in real aerodynamic coefficients, Reynolds in real flow
    units) -- that capability is lost without NASA's original
    scalers.pickle. This is documented as an explicit limitation.
    """
    scalers = {}
    for key, arr in all_fields_unscaled.items():
        flat = arr.reshape(-1)
        scalers[key] = {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
        }
    return scalers


def save_split_npz(fields: dict, path: Path):
    np.savez_compressed(path, **fields)


def save_split_parquet(fields: dict, path: Path):
    """Flatten vector fields (geometry_y, cp) into indexed columns for a
    single flat table; this is the most universally-readable format
    (pandas, R, Excel, anything) at the cost of wide column count.
    """
    df_dict = {}
    for key, arr in fields.items():
        if arr.ndim == 1:
            df_dict[key] = arr
        else:
            for i in range(arr.shape[1]):
                df_dict[f"{key}_{i}"] = arr[:, i]
    df = pd.DataFrame(df_dict)
    df.to_parquet(path, index=False)


def process_one_scaler_variant(input_dir: Path, output_dir: Path, scaler_name: str,
                                 cp_variant: str, val_fraction: float, seed: int):
    """
    scaler_name: 'standard' or 'minmax'
    cp_variant: 'main' (global Cp scaler) or 'cp' (per-position Cp scaler)
    """
    suffix = "_cp" if cp_variant == "cp" else ""
    train_path = input_dir / f"dnn_scaled_data{suffix}_train.pt"
    test_path = input_dir / f"dnn_scaled_data{suffix}_test.pt"

    if not train_path.exists() or not test_path.exists():
        print(f"  [skip] missing files for scaler={scaler_name} variant={cp_variant}: "
              f"{train_path.name} / {test_path.name}")
        return None

    print(f"  Loading {train_path.name} ...")
    train_inputs, train_targets = load_pt_pairs(train_path)
    print(f"  Loading {test_path.name} ...")
    test_inputs, test_targets = load_pt_pairs(test_path)

    train_fields_full = split_fields(train_inputs, train_targets)
    test_fields = split_fields(test_inputs, test_targets)

    train_fields, val_fields = make_val_split(train_fields_full, val_fraction, seed)

    reports = [
        quality_audit(train_fields, "train"),
        quality_audit(val_fields, "val"),
        quality_audit(test_fields, "test"),
    ]

    variant_dir = output_dir / scaler_name / cp_variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    for name, fields in [("train", train_fields), ("val", val_fields), ("test", test_fields)]:
        save_split_npz(fields, variant_dir / f"{name}.npz")
        save_split_parquet(fields, variant_dir / f"{name}.parquet")

    scalers = refit_scalers({
        "geometry_y": np.concatenate([train_fields["geometry_y"], val_fields["geometry_y"], test_fields["geometry_y"]]),
        "alpha": np.concatenate([train_fields["alpha"], val_fields["alpha"], test_fields["alpha"]]),
        "reynolds": np.concatenate([train_fields["reynolds"], val_fields["reynolds"], test_fields["reynolds"]]),
        "ncrit": np.concatenate([train_fields["ncrit"], val_fields["ncrit"], test_fields["ncrit"]]),
        "cl": np.concatenate([train_fields["cl"], val_fields["cl"], test_fields["cl"]]),
        "cd": np.concatenate([train_fields["cd"], val_fields["cd"], test_fields["cd"]]),
        "cdp": np.concatenate([train_fields["cdp"], val_fields["cdp"], test_fields["cdp"]]),
        "cm": np.concatenate([train_fields["cm"], val_fields["cm"], test_fields["cm"]]),
        "cp": np.concatenate([train_fields["cp"], val_fields["cp"], test_fields["cp"]]),
    })
    with open(variant_dir / "refit_scalers.pickle", "wb") as f:
        pickle.dump(scalers, f)

    with open(variant_dir / "quality_report.json", "w") as f:
        json.dump(reports, f, indent=2)

    print(f"  -> wrote {variant_dir} "
          f"(train={len(train_fields['alpha'])}, val={len(val_fields['alpha'])}, "
          f"test={len(test_fields['alpha'])})")

    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing dnn_scaled_data*.pt files "
                              "(e.g. path to datasets/standard/ or datasets/minmax/)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scaler_name", type=str, default="standard",
                         choices=["standard", "minmax"],
                         help="Label for which NASA scaler variant this input_dir holds "
                              "(for naming output subfolders only -- does not affect processing)")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                         help="Fraction of TRAIN rows to carve out as validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing scaler variant: {args.scaler_name}")
    for cp_variant in ["main", "cp"]:
        print(f" cp_variant={cp_variant}")
        process_one_scaler_variant(input_dir, output_dir, args.scaler_name,
                                     cp_variant, args.val_fraction, args.seed)

    print("Done.")


if __name__ == "__main__":
    main()

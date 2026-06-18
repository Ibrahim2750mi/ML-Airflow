"""
load_dataset.py

Minimal, framework-agnostic loader for the processed Airfoil Learning dataset.
No PyTorch or PyTorch Geometric required -- works with plain numpy/pandas.

Usage:
    from load_dataset import load_split
    train = load_split("processed_output/standard/main/train.npz")
    train["geometry_y"]   # (N, 198) float32
    train["alpha"]        # (N,) float32
    train["reynolds"]     # (N,) float32
    train["ncrit"]        # (N,) float32
    train["cl"], train["cd"], train["cdp"], train["cm"]   # (N,) float32 each
    train["cp"]           # (N, 98) float32

For a baseline model: concatenate geometry_y + alpha + reynolds + ncrit as
your feature vector, predict cl/cd/cm (and optionally cp) as targets.

For an FNO: geometry_y and cp are already fixed-length, fixed-order vectors
(198 and 98 points respectively, consistently ordered around the airfoil
surface) -- treat them as 1D grids directly.

For a PINN: pos (x,y) is NOT included in this DNN-derived export, only y
(x is implicitly a fixed 0..1 chord-normalized grid shared by all airfoils
in the original dataset -- see README "Known Limitations" for why x isn't
stored per-sample, and recover it from the original JSON/graph files if
your PINN's physics loss needs explicit x).

For a GNN: use the original graph_scaled_data*.pt files directly (requires
torch + torch_geometric) -- this export does not duplicate the graph
edge_index/edge_attr structure, since that is redundant with geometry_y
(every airfoil uses the same ring adjacency) and is cheap to regenerate.
See README for the one-line adjacency reconstruction.
"""

import numpy as np


def load_split(npz_path: str) -> dict:
    """Load one split (train/val/test) as a dict of numpy arrays."""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def make_baseline_feature_matrix(split: dict) -> np.ndarray:
    """Flatten a split into a single (N, 201) feature matrix:
    [geometry_y (198), alpha, reynolds, ncrit] -- ready for sklearn/XGBoost/MLP.
    """
    return np.hstack([
        split["geometry_y"],
        split["alpha"][:, None],
        split["reynolds"][:, None],
        split["ncrit"][:, None],
    ])


def make_coefficient_targets(split: dict) -> np.ndarray:
    """Returns (N, 4) array of [Cl, Cd, Cdp, Cm]."""
    return np.stack([split["cl"], split["cd"], split["cdp"], split["cm"]], axis=1)


def reconstruct_ring_edge_index(n_points: int = 198) -> np.ndarray:
    """Recreate the airfoil surface adjacency used by the GNN variant of this
    dataset: every point connects to its immediate neighbor around the closed
    surface loop. Matches create_edge_adjacency() in NASA's libs/utils.py.
    Returns shape (2, n_points) edge_index, PyG convention.
    """
    src = np.arange(n_points)
    dst = (np.arange(n_points) + 1) % n_points
    edge_index = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),  # undirected: both directions
    ])
    return edge_index


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "processed_output/standard/main/train.npz"
    split = load_split(path)
    print(f"Loaded {path}")
    for k, v in split.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    X = make_baseline_feature_matrix(split)
    y = make_coefficient_targets(split)
    print(f"Baseline feature matrix: {X.shape}")
    print(f"Coefficient targets: {y.shape}")

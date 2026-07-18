"""
train_fno.py

FNO1d training script for aerodynamic surrogate modeling.
Predicts Cl, Cd, and Cp (98 points) from airfoil geometry + flow conditions.

Called from notebooks/FNO.ipynb via:
    %run ../train_fno.py --data_dir processed_output/standard/main \
                         --output_dir artifacts \
                         --epochs 20 \
                         --modes 16 \
                         --width 64

Architecture: FNO1d
    - Input:  (batch, 201, 1) — 198 geometry y-coords + alpha + Re + Ncrit,
              each broadcast as a channel value at every position. The 201
              points serve as the spatial dimension of the 1D operator.
    - Output: (batch, 100) — Cl, Cd, Cp×98, same layout as MLP/PINN.

Why FNO1d on this data:
    The airfoil geometry is a 1D signal (198 surface y-coords) and the
    target Cp distribution is also a 1D signal (98 chord-wise values).
    FNO1d applies spectral convolutions over the spatial (position) axis,
    learning the mapping between these two 1D signals conditioned on the
    scalar flow parameters (alpha, Re, Ncrit). This is exactly the problem
    the original Li et al. 2021 FNO1d was designed for (1D function-to-
    function mapping), so we use that variant and nothing fancier.

Data layout (inherited from preprocess_airfoil_dataset.py):
    X : (N, 201) — [geometry_y(198), alpha, reynolds, ncrit]
    y : (N, 100) — [Cl, Cd, Cp×98]

Loss: same weighted MSE as Day 6 (W_COEFF=49 for Cl/Cd, W_CP=1 for Cp).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# FNO1d implementation
# Reference: Li et al. 2021 "Fourier Neural Operator for Parametric PDEs"
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer.

    Takes input of shape (batch, width, n_points), applies:
        1. FFT along the spatial dimension
        2. Truncate to the lowest `modes` frequency components
        3. Apply a learnable complex-valued weight matrix R to those modes
        4. Inverse FFT back to spatial domain

    The output has the same shape as the input.

    Args:
        in_channels  (int): number of input channels (= FNO width)
        out_channels (int): number of output channels (= FNO width)
        modes        (int): number of Fourier modes to keep (K in the paper)
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        # Complex weight tensor: (in_channels, out_channels, modes)
        # Initialised with small random values scaled by 1/sqrt(in*out)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        x : (batch, in_channels, n_points)
        returns : (batch, out_channels, n_points)
        """
        batch, _, n = x.shape

        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)           # (batch, in_ch, n//2+1)

        # Truncate to lowest `modes` frequency components and apply weights
        out_ft = torch.zeros(
            batch, self.out_channels, n // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        # einsum: bix,iox->box  (batch, in, modes) x (in, out, modes) -> (batch, out, modes)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :self.modes],
            self.weights
        )

        # Inverse FFT back to spatial domain
        return torch.fft.irfft(out_ft, n=n, dim=-1)  # (batch, out_ch, n_points)


class FNOBlock1d(nn.Module):
    """
    Single FNO layer = SpectralConv1d + pointwise bypass W + activation.

    output = activation( SpectralConv(x) + W(x) )

    The bypass W is a 1x1 convolution (pointwise linear transform in the
    spatial dimension, applied independently at each position). This lets
    the layer represent features that aren't captured by the low-frequency
    spectral path alone.
    """
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass   = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.bypass(x))


class FNO1d(nn.Module):
    """
    Full 1D Fourier Neural Operator.

    Architecture:
        1. Lift input channels to `width` via a linear layer (P)
        2. Apply `n_layers` FNO blocks (spectral conv + bypass)
        3. Project from `width` to `out_dim` via two linear layers (Q)

    Input/output convention for this project:
        x_in  : (batch, n_points, in_channels)  — spatial last for embedding
        output : (batch, out_dim)                — global prediction

    Args:
        in_channels (int): number of input channels per position
                           = 1 (geometry_y value) + 3 (alpha, Re, Ncrit) = 4
                           (scalars broadcast to all 201 positions)
        out_dim     (int): total output size = 100 (Cl + Cd + Cp×98)
        n_points    (int): spatial dimension = 201
        modes       (int): number of Fourier modes to retain (K)
        width       (int): channel width throughout the FNO layers
        n_layers    (int): number of FNO blocks
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_dim:     int = 100,
        n_points:    int = 201,
        modes:       int = 16,
        width:       int = 64,
        n_layers:    int = 4,
    ):
        super().__init__()
        self.n_points = n_points
        self.width    = width

        # P: lift from in_channels → width
        self.lift = nn.Linear(in_channels, width)

        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock1d(width, modes) for _ in range(n_layers)
        ])

        # Q: project from (n_points * width) → out_dim
        self.proj1 = nn.Linear(n_points * width, 256)
        self.proj2 = nn.Linear(256, out_dim)

    def forward(self, x):
        """
        x : (batch, n_points, in_channels)
        returns : (batch, out_dim)
        """
        # Lift: (batch, n_points, in_ch) → (batch, n_points, width)
        x = self.lift(x)

        # Permute for Conv1d: (batch, width, n_points)
        x = x.permute(0, 2, 1)

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Flatten spatial + channel dims: (batch, n_points * width)
        x = x.reshape(x.shape[0], -1)

        # Project to output
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# Use the canonical loader from data/load_dataset.py rather than
# reimplementing it here. load_split() returns the same dict of numpy arrays
# that all other notebooks in this repo use:
#   split["geometry_y"]  (N, 198)
#   split["alpha"]       (N,)
#   split["reynolds"]    (N,)
#   split["ncrit"]       (N,)
#   split["cl/cd/cdp/cm"] (N,)
#   split["cp"]          (N, 98)
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data"))
from load_dataset import load_split  # noqa: E402


def make_X_fno(split, n_points=201):
    """
    Build the FNO input tensor: (N, n_points, in_channels=4).

    Channel layout at every position:
        ch 0: geometry_y value at that position (only first 198 positions
              have a meaningful geometry value; positions 198-200 hold
              the scalar conditions broadcast uniformly)
        ch 1: alpha  (broadcast to all n_points positions)
        ch 2: reynolds (broadcast)
        ch 3: ncrit   (broadcast)

    This gives the FNO spatial context (geometry shape) alongside global
    conditioning at every node — the standard approach for scalar-conditioned
    FNOs (see Li et al. 2021 Appendix A).
    """
    N = split["alpha"].shape[0]

    # Geometry channel: pad to n_points if geometry is shorter
    geom = split["geometry_y"]                           # (N, 198)
    pad  = np.zeros((N, n_points - geom.shape[1]), dtype=np.float32)
    geom_padded = np.concatenate([geom, pad], axis=1)   # (N, 201)

    alpha   = np.broadcast_to(split["alpha"][:, None],   (N, n_points))
    reynolds = np.broadcast_to(split["reynolds"][:, None], (N, n_points))
    ncrit   = np.broadcast_to(split["ncrit"][:, None],   (N, n_points))

    X = np.stack([geom_padded, alpha, reynolds, ncrit], axis=-1)  # (N, 201, 4)
    return X.astype(np.float32)


def make_y(split):
    """(N, 100) — [Cl, Cd, Cp×98]. Identical layout to MLP/PINN."""
    return np.concatenate([
        split["cl"].reshape(-1, 1),
        split["cd"].reshape(-1, 1),
        split["cp"]
    ], axis=1).astype(np.float32)


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2,
    )


# ---------------------------------------------------------------------------
# Loss — same weighted MSE as Day 6
# ---------------------------------------------------------------------------

W_COEFF = 49.0
W_CP    = 1.0

def build_weights(device):
    w = torch.tensor([W_COEFF, W_COEFF] + [W_CP] * 98, dtype=torch.float32)
    return (w / w.sum()).to(device)


def weighted_mse(pred, target, w_norm):
    return ((pred - target) ** 2 * w_norm).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, w_norm, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += weighted_mse(model(x), y, w_norm).item() * x.shape[0]
            n     += x.shape[0]
    return total / n


def compute_metrics(model, X, y, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()

    pred_cl = pred[:, 0];  true_cl = y[:, 0]
    pred_cd = pred[:, 1];  true_cd = y[:, 1]
    pred_cp = pred[:, 2:]; true_cp = y[:, 2:]

    def m(t, p):
        return {
            "mae":  float(mean_absolute_error(t, p)),
            "rmse": float(np.sqrt(mean_squared_error(t, p))),
            "r2":   float(r2_score(t, p)),
        }

    return {
        "cl": m(true_cl, pred_cl),
        "cd": m(true_cd, pred_cd),
        "cp": m(true_cp.reshape(-1), pred_cp.reshape(-1)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_split = load_split(os.path.join(args.data_dir, "train.npz"))
    val_split   = load_split(os.path.join(args.data_dir, "val.npz"))

    X_train = make_X_fno(train_split)
    X_val   = make_X_fno(val_split)
    y_train = make_y(train_split)
    y_val   = make_y(val_split)

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FNO1d(
        in_channels=4,
        out_dim=100,
        n_points=201,
        modes=args.modes,
        width=args.width,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO1d | modes={args.modes} width={args.width} "
          f"n_layers={args.n_layers} | params={n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )
    w_norm = build_weights(device)

    # ── Training ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history       = []
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "fno_v1.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = weighted_mse(model(x), y, w_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        train_loss = total_loss / n_batches

        val_loss = evaluate(model, val_loader, w_norm, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ saved best checkpoint (val={val_loss:.6f})")

    # ── Final metrics on val set ───────────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    metrics = compute_metrics(model, X_val, y_val, device)

    print(f"\n── FNO1d Val Metrics (best checkpoint, epoch {ckpt['epoch']}) ──")
    print(f"  Cl | MAE={metrics['cl']['mae']:.4f}  "
          f"RMSE={metrics['cl']['rmse']:.4f}  R²={metrics['cl']['r2']:.4f}  "
          f"(MLP baseline: MAE=0.0900 R²=0.984)")
    print(f"  Cd | MAE={metrics['cd']['mae']:.4f}  "
          f"RMSE={metrics['cd']['rmse']:.4f}  R²={metrics['cd']['r2']:.4f}  "
          f"(MLP baseline: MAE=0.1745 R²=0.914)")
    print(f"  Cp | MAE={metrics['cp']['mae']:.4f}  "
          f"RMSE={metrics['cp']['rmse']:.4f}  R²={metrics['cp']['r2']:.4f}  "
          f"(MLP baseline: MAE=0.0318 R²=0.839)")

    output = {
        "model": "FNO1d",
        "hyperparams": vars(args),
        "n_params": n_params,
        "best_epoch": ckpt["epoch"],
        "best_val_weighted_mse": best_val_loss,
        "val_metrics": metrics,
        "history": history,
        "mlp_baseline": {
            "cl": {"mae": 0.0900, "rmse": 0.1276, "r2": 0.984},
            "cd": {"mae": 0.1745, "rmse": 0.2926, "r2": 0.914},
            "cp": {"mae": 0.0318, "rmse": 0.2588, "r2": 0.839},
        },
    }

    metrics_path = os.path.join(args.output_dir, "fno_v1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {ckpt_path}")
    print(f"Saved: {metrics_path}")

    return model, output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train FNO1d on NASA airfoil dataset")
    p.add_argument("--data_dir",   default="processed_output/standard/main")
    p.add_argument("--output_dir", default="artifacts")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--modes",      type=int,   default=16,
                   help="Number of Fourier modes to retain (K)")
    p.add_argument("--width",      type=int,   default=64,
                   help="Channel width throughout FNO layers")
    p.add_argument("--n_layers",   type=int,   default=4,
                   help="Number of FNO blocks")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=1024)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(args)

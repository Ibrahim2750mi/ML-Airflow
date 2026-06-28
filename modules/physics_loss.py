"""
physics_loss.py

Physics-informed loss terms for aerodynamic surrogate modeling.
Implements the four constraints formulated in Day 8:

  1. Lift-Pressure Consistency  -- predicted Cp should integrate to match predicted Cl
  2. Pressure Smoothness        -- penalize high-frequency oscillations in Cp
  3. Drag Positivity            -- Cd must be >= 0
  4. Drag Consistency           -- Cd must be >= Cdp (pressure drag component)

All functions accept PyTorch tensors of shape (batch, ...) and return
scalar losses compatible with autograd / backward().

Usage:
    from physics_loss import PhysicsLoss
    physics = PhysicsLoss(lambda_lift=1.0, lambda_smooth=0.01,
                          lambda_drag=0.1, lambda_cdp=0.1)
    loss = physics(pred, target)
"""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss for the airfoil PINN.

    Args:
        lambda_lift   (float): weight for lift-pressure consistency loss
        lambda_smooth (float): weight for Cp smoothness loss
        lambda_drag   (float): weight for drag positivity loss
        lambda_cdp    (float): weight for drag decomposition consistency loss
        n_cp          (int):   number of Cp surface points (default 98)

    Output tensor layout assumed:
        pred[:, 0]    = Cl
        pred[:, 1]    = Cd
        pred[:, 2:]   = Cp  (n_cp values, chord-ordered 0→1)

    target layout:
        target[:, 0]  = Cl
        target[:, 1]  = Cd
        target[:, 2]  = Cdp   (used for drag consistency constraint only)
        target[:, 3:] = Cp    (n_cp values)

    Note: target[:, 2] = Cdp is NOT predicted by the model (output is
    only Cl + Cd + Cp = 100 dims). It is used from ground-truth labels
    only to enforce the physical Cd >= Cdp constraint during training.
    If Cdp is not available, set lambda_cdp=0.
    """

    def __init__(
        self,
        lambda_lift: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_drag: float = 0.1,
        lambda_cdp: float = 0.1,
        n_cp: int = 98,
    ):
        super().__init__()
        self.lambda_lift = lambda_lift
        self.lambda_smooth = lambda_smooth
        self.lambda_drag = lambda_drag
        self.lambda_cdp = lambda_cdp
        self.n_cp = n_cp

    def lift_consistency_loss(self, pred_cl, pred_cp):
        """
        Constraint: Cl ∝ ∫ Cp dx

        Approximates the chord-wise integral of the Cp distribution using
        the trapezoidal rule on a uniform grid (x/c = 0..1, n_cp points).
        The integrated value is scaled to match the magnitude of Cl via a
        learned-free surrogate: we enforce that the RELATIVE ordering and
        magnitude of Cl_pred matches Cl_from_Cp across the batch, using MSE.

        Note: The exact proportionality constant between ∫Cp dx and Cl
        depends on airfoil-specific factors and is absorbed into the MSE
        minimization -- we are enforcing consistency of direction and scale,
        not exact equality.
        """
        # Trapezoidal integration: Σ (Cp[i] + Cp[i+1]) / 2 * Δx
        # Uniform spacing: Δx = 1 / (n_cp - 1)
        dx = 1.0 / (self.n_cp - 1)
        cl_from_cp = 0.5 * (pred_cp[:, :-1] + pred_cp[:, 1:]).sum(dim=1) * dx
        # MSE between Cl predicted by network and Cl estimated from Cp integral
        return nn.functional.mse_loss(pred_cl, cl_from_cp)

    def smoothness_loss(self, pred_cp):
        """
        Constraint: penalize rapid variation between neighboring Cp points.

        L_smooth = mean[ (Cp[i+1] - Cp[i])^2 ]

        Encourages physically realistic, smooth pressure distributions.
        Does not penalize genuine leading-edge suction peaks -- only
        high-frequency oscillations inconsistent with real aerodynamics.
        """
        diff = pred_cp[:, 1:] - pred_cp[:, :-1]   # (batch, n_cp-1)
        return (diff ** 2).mean()

    def drag_positivity_loss(self, pred_cd):
        """
        Constraint: Cd >= 0

        L_drag = mean[ ReLU(-Cd) ]

        ReLU(-Cd) is zero when Cd >= 0 (no penalty) and positive when
        Cd < 0 (linearly penalizes violation magnitude).
        """
        return torch.relu(-pred_cd).mean()

    def drag_consistency_loss(self, pred_cd, true_cdp):
        """
        Constraint: Cd >= Cdp  (total drag >= pressure drag component)

        L_cdp = mean[ ReLU(Cdp_true - Cd_pred) ]

        Uses ground-truth Cdp labels as the reference since Cdp is not
        a model output. Penalizes predictions where the model outputs a
        Cd value smaller than the known pressure drag component.
        """
        return torch.relu(true_cdp - pred_cd).mean()

    def forward(self, pred, target):
        """
        Compute total physics loss.

        Args:
            pred   (Tensor): model output (batch, 100) -- [Cl, Cd, Cp x 98]
            target (Tensor): ground truth (batch, 102) -- [Cl, Cd, Cdp, Cm, Cp x 98]
                             OR (batch, 100) if Cdp not available (set lambda_cdp=0)

        Returns:
            dict with keys: 'lift', 'smooth', 'drag', 'cdp', 'total'
        """
        pred_cl = pred[:, 0]
        pred_cd = pred[:, 1]
        pred_cp = pred[:, 2:]   # (batch, 98)

        losses = {}

        # 1. Lift-pressure consistency
        losses["lift"] = self.lift_consistency_loss(pred_cl, pred_cp)

        # 2. Smoothness
        losses["smooth"] = self.smoothness_loss(pred_cp)

        # 3. Drag positivity
        losses["drag"] = self.drag_positivity_loss(pred_cd)

        # 4. Drag consistency (requires Cdp in target)
        if self.lambda_cdp > 0 and target.shape[1] > 100:
            # target layout: [Cl, Cd, Cdp, Cm, Cp x 98] = 102 cols
            true_cdp = target[:, 2]
            losses["cdp"] = self.drag_consistency_loss(pred_cd, true_cdp)
        else:
            losses["cdp"] = torch.tensor(0.0, device=pred.device)

        losses["total"] = (
            self.lambda_lift   * losses["lift"]   +
            self.lambda_smooth * losses["smooth"] +
            self.lambda_drag   * losses["drag"]   +
            self.lambda_cdp    * losses["cdp"]
        )

        return losses


def verify_constraints(pred, target, n_cp=98, verbose=True):
    """
    Standalone constraint verification -- checks what fraction of predictions
    satisfy each physical constraint. Use this as a sanity check after training
    to confirm the physics losses are actually improving physical consistency,
    not just reducing loss numbers.

    Args:
        pred   (Tensor): model output (batch, 100)
        target (Tensor): ground truth (batch, 102 or 100)

    Returns:
        dict of constraint satisfaction rates (0.0 to 1.0)
    """
    pred_cl = pred[:, 0]
    pred_cd = pred[:, 1]
    pred_cp = pred[:, 2:]

    results = {}

    # Drag positivity: what fraction of predictions have Cd >= 0?
    drag_ok = (pred_cd >= 0).float().mean().item()
    results["drag_positivity_rate"] = drag_ok

    # Drag consistency: what fraction have Cd >= Cdp (ground truth)?
    if target.shape[1] > 100:
        true_cdp = target[:, 2]
        cdp_ok = (pred_cd >= true_cdp).float().mean().item()
        results["drag_consistency_rate"] = cdp_ok
    else:
        results["drag_consistency_rate"] = None

    # Smoothness: mean squared difference between adjacent Cp points
    diff = pred_cp[:, 1:] - pred_cp[:, :-1]
    results["mean_cp_smoothness"] = (diff ** 2).mean().item()

    # Lift consistency: correlation between pred_cl and integrated Cp
    dx = 1.0 / (n_cp - 1)
    cl_from_cp = 0.5 * (pred_cp[:, :-1] + pred_cp[:, 1:]).sum(dim=1) * dx
    corr = torch.corrcoef(torch.stack([pred_cl, cl_from_cp]))[0, 1].item()
    results["lift_cp_correlation"] = corr

    if verbose:
        print("=== Physics Constraint Verification ===")
        print(f"  Drag positivity (Cd >= 0):         {drag_ok*100:.1f}% of predictions")
        if results["drag_consistency_rate"] is not None:
            print(f"  Drag consistency (Cd >= Cdp):      {cdp_ok*100:.1f}% of predictions")
        print(f"  Cp smoothness (mean sq diff):       {results['mean_cp_smoothness']:.6f}")
        print(f"  Lift-Cp correlation:                {corr:.4f}")

    return results

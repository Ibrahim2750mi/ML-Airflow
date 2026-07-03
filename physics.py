import torch


def gradients(y, x):
    """
    Compute dy/dx using autograd.
    """
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,
    )[0]


def second_gradient(y, x, dim):
    """
    Compute second derivative:
    d²y / dx_dim²
    """
    g = gradients(y, x)
    return gradients(g[:, dim:dim+1], x)[:, dim:dim+1]


def navier_stokes_residual(
    model,
    inputs,
    nu=1e-5,
):
    """
    Compute Navier-Stokes residuals.

    Inputs
    ------
    inputs : (N,7)

    Columns
    -------
    0 x
    1 y
    2 inlet_u
    3 inlet_v
    4 sdf
    5 normal_x
    6 normal_y

    Returns
    -------
    continuity
    momentum_x
    momentum_y
    u
    v
    p
    """

    inputs = inputs.clone().detach().requires_grad_(True)

    u, v, p = model(inputs)

    # ==========================
    # First derivatives
    # ==========================

    du = gradients(u, inputs)
    dv = gradients(v, inputs)
    dp = gradients(p, inputs)

    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]

    dv_dx = dv[:, 0:1]
    dv_dy = dv[:, 1:2]

    dp_dx = dp[:, 0:1]
    dp_dy = dp[:, 1:2]

    # ==========================
    # Second derivatives
    # ==========================

    d2u_dx2 = second_gradient(u, inputs, 0)
    d2u_dy2 = second_gradient(u, inputs, 1)

    d2v_dx2 = second_gradient(v, inputs, 0)
    d2v_dy2 = second_gradient(v, inputs, 1)

    # ==========================
    # Continuity
    # ==========================

    continuity = du_dx + dv_dy

    # ==========================
    # x momentum
    # ==========================

    momentum_x = (
        u * du_dx
        + v * du_dy
        + dp_dx
        - nu * (d2u_dx2 + d2u_dy2)
    )

    # ==========================
    # y momentum
    # ==========================

    momentum_y = (
        u * dv_dx
        + v * dv_dy
        + dp_dy
        - nu * (d2v_dx2 + d2v_dy2)
    )

    return (
        continuity,
        momentum_x,
        momentum_y,
        u,
        v,
        p,
    )


def physics_loss(
    continuity,
    momentum_x,
    momentum_y,
):
    """
    PINN physics loss.
    """

    loss_c = torch.mean(continuity ** 2)

    loss_mx = torch.mean(momentum_x ** 2)

    loss_my = torch.mean(momentum_y ** 2)

    return loss_c + loss_mx + loss_my
# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------


from   typing import Tuple  # Dict, Sequence  #, List, Optional

import torch

import numpy  as np


# import utils





# Pinball (quantile) loss
# ----------------------------------------------------------------------

def quantile_with_crossing_torch(
    y_pred:         torch.Tensor,     # (B, H, Q)
    y_true:         torch.Tensor,     # (B, H) or (B, H, 1)
    quantiles:      Tuple[float, ...],
    lambda_cross:   float,
    lambda_coverage:float,
    smoothing     : float
) -> torch.Tensor:
    """
    Joint quantile loss with crossing penalty.
    """

    loss = 0.

    for i, tau in enumerate(quantiles):
        diff = y_true - y_pred[..., i]
        pin = torch.maximum(tau * diff, -(1 - tau) * diff)
        loss += pin.mean(dim=0).mean()   # mean per horizon
        # loss += torch.mean(torch.maximum(tau * diff, -(1-tau) * diff))

        # Coverage penalty
        if lambda_coverage > 0.:
            # scale per horizon using target variability
            scale_h    = torch.std(y_true, dim=0, unbiased=False)  # (H,)
            tau_smooth = smoothing * scale_h.clamp_min(1e-3)       # (H,)
            # tau_smooth = smoothing * torch.std(y_true)

            z = (y_pred[..., i] - y_true) / tau_smooth   # broadcast over B
            z = torch.clamp(z, -20., 20.)  # preventing overflow
            soft_ind   = torch.sigmoid(z)         # (B, H)
            coverage_h = soft_ind.mean(dim=0)     # (H,)
            # coverage = torch.sigmoid(z).mean()

            err  = coverage_h - tau
            w    = torch.where(err > 0,  tau,  1 - tau)
            alpha = 1. / (tau * (1-tau))   # emphasizes tails
            loss+= lambda_coverage * alpha * (w * err**2).mean()
            # loss += lambda_coverage * alpha * (coverage - tau) ** 2

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = torch.relu(y_pred[..., :-1] - y_pred[..., 1:])
        loss   += lambda_cross * penalty.sum(dim=-1).mean()


    return loss


def quantile_with_crossing_numpy(
        y_pred        : np.ndarray,     # (B, H, Q)
        y_true        : np.ndarray,     # (B, H)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        smoothing     : float
    ) -> float:

    loss = 0.

    def sigmoid(x: float) -> float:
        return 1. / (1. + np.exp(-x))

    # print(f"[quantile_loss_with_crossing_numpy] y_pred.shape = {y_pred.shape}")
    # print(f"[quantile_loss_with_crossing_numpy] y_true.shape = {y_true.shape}")

    for i, tau in enumerate(quantiles):
        diff = y_true - y_pred[..., i]
        # print(f"[quantile_loss_with_crossing_numpy] {tau} diff.shape = {diff.shape}"

        pin = np.maximum(tau * diff, -(1 - tau) * diff)
        loss += pin.mean(axis=0).mean()   # mean per horizon
        # loss += np.mean(np.maximum(tau * diff, -(1-tau) * diff))

        # Coverage penalty
        if lambda_coverage > 0.:
            # scale per horizon using target variability
            scale_h    = np.std(y_true, axis=0, correction=0)     # (H,)
            tau_smooth = smoothing * np.clip(scale_h, 1e-3, None)  # (H,)
            # tau_smooth = smoothing * np.std(y_true)

            z = (y_pred[..., i] - y_true) / tau_smooth
            z = np.clip(z, -20., 20.)  # preventing overflow
            soft_ind   = sigmoid(z)
            coverage_h = soft_ind.mean(axis=0)   # (H,)
            # coverage = sigmoid(z).mean()

            err  = coverage_h - tau
            w    = np.where(err > 0,  tau,  1 - tau)
            alpha = 1. / (tau * (1-tau))   # emphasizes tails
            loss+= lambda_coverage * alpha * (w * err**2).mean()
            # loss += lambda_coverage * alpha * (coverage - tau) ** 2

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = np.maximum(0., y_pred[..., :-1] - y_pred[..., 1:])
        loss   += lambda_cross * np.mean(np.sum(penalty, axis=-1))

    return float(loss)


# losses with derivatives
# ----------------------------------------------------------------------

# /!\ These two MUST remain equivalent.
#    When making modifications, we modify both in parallel.

def derivative_torch(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
    """
    First-order finite-difference derivative loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Shape (B, H, Q) or (B, H)
    y_true : torch.Tensor
        Shape (B, H)

    Returns
    -------
    torch.Tensor
        Scalar loss
    """

    # No horizon → no derivative loss
    if y_pred.dim() < 2:
        return y_pred.new_zeros(())

    # Ensure (B, H, Q)
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(-1)

    # Now we REQUIRE a horizon
    if y_pred.shape[1] < 2:
        return y_pred.new_zeros(())

    assert y_pred.shape[:2] == y_true.shape, (y_pred.shape, y_true.shape)

    # Temporal finite differences (within each sample)
    dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]   # (B, H-1, Q)
    dy_true = y_true[:, 1:]    - y_true[:, :-1]      # (B, H-1)

    # Broadcast true derivatives over quantiles
    dy_true = dy_true.unsqueeze(-1)                  # (B, H-1, 1)

    return torch.mean((dy_pred - dy_true) ** 2)  # MSE on derivative mismatch


def derivative_numpy(
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
    """
    NumPy version of first-order finite-difference derivative loss.
    Returns 0. if no temporal dimension is present.
    Must match derivative_loss_torch exactly.

    Parameters
    ----------
    y_pred : np.ndarray
         Shape (B, H, Q), (B, H), or (B,)
    y_true : np.ndarray
         Shape (B, H) or (B,)

    Returns
    -------
    float
         Scalar loss
    """

    # No horizon → no derivative loss
    if y_pred.ndim < 2 or y_true.ndim < 2:
        return 0.

    # Ensure (B, H, Q)
    if y_pred.ndim == 2:
        y_pred = y_pred[..., np.newaxis]   # (B, H, 1)

    # Horizon must exist
    if y_pred.shape[1] < 2:
        return 0.

    assert y_pred.shape[:2] == y_true.shape, (
        y_pred.shape, y_true.shape
    )

    # Temporal finite differences
    dy_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]   # (B, H-1, Q)
    dy_true = y_true[:, 1:]    - y_true[:, :-1]      # (B, H-1)

    dy_true = dy_true[..., np.newaxis]               # (B, H-1, 1)

    return float(np.mean((dy_pred - dy_true) ** 2))


# wrappers (add together all components to the loss)
# ----------------------------------------------------------------------

def quantile_torch(
        y_pred        : torch.Tensor,   # (B, H, Q)
        y_true        : torch.Tensor,   # (B, H) or (B, H, 1)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
        lambda_median : float,
        smoothing_cross:float
    ) -> torch.Tensor:
    """
    Torch loss wrapper for quantile forecasts.
    """
    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    # Base quantile + crossing loss
    loss = quantile_with_crossing_torch(
        y_pred       = y_pred,
        y_true       = y_true,
        quantiles    = quantiles,
        lambda_cross = lambda_cross,
        lambda_coverage=lambda_coverage,
        smoothing    = smoothing_cross
    )

    # Optional derivative loss (per quantile)
    if lambda_deriv > 0.:
        # _y_true = y_true.squeeze(-1) if y_true.ndim == 3 else y_true
        loss += lambda_deriv * derivative_torch(y_pred, y_true)

    q50_pred = y_pred[..., len(quantiles)//2]
    # print(f"shapes: {y_pred.shape} -> {q50_pred.shape}, {y_true.shape}")
    loss += lambda_median * ((q50_pred - y_true).mean(dim=0)**2).mean()

    return loss


def quantile_numpy(
        y_pred        : np.ndarray,     # (B, H, Q)
        y_true        : np.ndarray,     # (B, H)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
        lambda_median : float,
        smoothing_cross:float
    ) -> float:
    """
    NumPy loss wrapper for quantile forecasts.
    MUST match torch version exactly.
    """

    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    loss = quantile_with_crossing_numpy(
        y_pred       = y_pred,
        y_true       = y_true,
        quantiles    = quantiles,
        lambda_cross = lambda_cross,
        lambda_coverage=lambda_coverage,
        smoothing    = smoothing_cross
    )

    if lambda_deriv > 0.:
        loss += lambda_deriv * derivative_numpy(y_pred, y_true)

    q50_pred = y_pred[..., len(quantiles)//2]
    loss += lambda_median * ((q50_pred - y_true).mean(axis=0)**2).mean()

    return float(loss)



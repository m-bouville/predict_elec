# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------


from   typing import Dict, Tuple  # , List Sequence  #, Optional

import torch

import numpy  as np
# import pandas as pd


# import utils





# Pinball (quantile) loss
# ----------------------------------------------------------------------

def penalty_nation_cold_torch(
        saturation_cold_degC: float,
        threshold_cold_degC : float,
        Tavg_current        : torch.Tensor,  # /!\ assumes a single one (correct for a single day)
    ) -> torch.Tensor:  # returns shape (B,)

    # linear ramp
    penalty = (Tavg_current - threshold_cold_degC) / \
              (saturation_cold_degC - threshold_cold_degC)

    # clip to [0, 1]
    penalty = torch.clamp(penalty, 0., 1.)

    return penalty


def quantile_with_crossing_torch(
    y_nation_pred   :  torch.Tensor,     # (B, V, Q)
    y_nation_true   :  torch.Tensor,     # (B, V) or (B, V, 1)

    # constants
    quantiles       :  Tuple[float, ...],
    lambda_cross    :  float,
    lambda_coverage : float,
    smoothing       : float,
        # temperature-dependence (pinball loss, coverage penalty)
    saturation_cold_degC:float,
    threshold_cold_degC: float,
    lambda_cold     : float,
    Tavg_current    : torch.Tensor,  # (B, V, 1)
            # /!\ assumes a single one (correct for a single day)

) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
    """
    Joint quantile loss with crossing penalty.
    """

    B, V, Q = y_nation_pred.shape
    device  = y_nation_pred.device

    loss_pinball_h  = torch.zeros(V, device=device)
    loss_coverage_h = torch.zeros(V, device=device)
    loss_crossing_h = torch.zeros(V, device=device)

    _penalty_nation_cold = lambda_cold * penalty_nation_cold_torch(
            saturation_cold_degC, threshold_cold_degC, Tavg_current.to(device)).mean()

    for i, tau in enumerate(quantiles):
        diff = y_nation_true - y_nation_pred[..., i]
        pin = torch.maximum(tau * diff, -(1 - tau) * diff)
        loss_pinball_h += ((1 + _penalty_nation_cold) * pin).mean(dim=0)

        # Coverage penalty
        if lambda_coverage > 0.:
            # scale per horizon using target variability
            scale_h    = torch.std(y_nation_true, dim=0, unbiased=False)  # (V,)
            tau_smooth = smoothing * scale_h.clamp_min(1e-3)       # (V,)

            z = -diff / tau_smooth   # broadcast over B
            z = torch.clamp(z, -20., 20.)  # preventing overflow
            soft_ind   = torch.sigmoid(z)         # (B, V)
            coverage_h = ((1 + _penalty_nation_cold) * soft_ind).mean(dim=0)     # (V,)

            err  = coverage_h - tau
            w    = torch.where(err > 0,  tau,  1 - tau)
            alpha = 1. / (tau * (1-tau))   # emphasizes tails

            loss_coverage_h += lambda_coverage * alpha * w * err.abs()

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = torch.relu(y_nation_pred[..., :-1] - y_nation_pred[..., 1:])
        loss_crossing_h += lambda_cross * penalty.sum(dim=-1).mean(dim=0)

    loss_h = loss_pinball_h + loss_coverage_h + loss_crossing_h

    return loss_h, {'pinball':  loss_pinball_h,
                    'coverage': loss_coverage_h, 'crossing': loss_crossing_h}


def penalty_nation_cold_numpy(
        saturation_cold_degC: float,
        threshold_cold_degC : float,
        Tavg_current        : np.ndarray,  # /!\ assumes a single one (correct for a single day)
    ) -> np.ndarray:  # returns shape (B,)

    # linear ramp
    penalty = (Tavg_current - threshold_cold_degC) / \
              (saturation_cold_degC - threshold_cold_degC)

    # clip to [0, 1]
    penalty = np.clip(penalty, 0., 1.)

    return penalty



def quantile_with_crossing_numpy(
        y_nation_pred   : np.ndarray,     # (B, V, Q)
        y_nation_true   : np.ndarray,     # (B, V)

        # constants
        quantiles       : Tuple[float, ...],
        lambda_cross    : float,
        lambda_coverage : float,
        smoothing       : float,
            # temperature-dependence (pinball loss, coverage penalty)
        saturation_cold_degC:float,
        threshold_cold_degC: float,
        lambda_cold     : float,
        Tavg_current    : np.ndarray,  # /!\ assumes a single one (correct for a single day)

    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

    B, V, Q = y_nation_pred.shape

    loss_pinball_h  = np.zeros(V)
    loss_coverage_h = np.zeros(V)
    loss_crossing_h = np.zeros(V)

    _penalty_nation_cold = lambda_cold * penalty_nation_cold_numpy(
                saturation_cold_degC, threshold_cold_degC, Tavg_current).mean()

    def sigmoid(x: float) -> float:
        return 1. / (1. + np.exp(-x))

    # print(f"[quantile_loss_with_crossing_numpy] y_nation_pred.shape = {y_nation_pred.shape}")
    # print(f"[quantile_loss_with_crossing_numpy] y_nation_true.shape = {y_nation_true.shape}")

    for i, tau in enumerate(quantiles):
        diff = y_nation_true - y_nation_pred[..., i]      # (B, V)
        # print(f"[quantile_loss_with_crossing_numpy] {tau} diff.shape = {diff.shape}"

        pin = np.maximum(tau * diff, -(1 - tau) * diff)
        loss_pinball_h += ((1 + _penalty_nation_cold) * pin).mean(axis=0)   # (V,)

        # Coverage penalty
        if lambda_coverage > 0.:
            # scale per horizon using target variability
            scale_h    = np.std(y_nation_true, axis=0, correction=0)      # (V,)
            tau_smooth = smoothing * np.clip(scale_h, 1e-3, None)  # (V,)

            z = -diff / tau_smooth                # (B, V)
            z = np.clip(z, -20., 20.)  # preventing overflow
            soft_ind   = sigmoid(z)              # (B, V)
            coverage_h = ((1 + _penalty_nation_cold) * soft_ind).mean(axis=0)   # (V,)

            err  = coverage_h - tau              # (V,)
            w    = np.where(err > 0,  tau,  1 - tau)
            alpha = 1. / (tau * (1-tau))   # emphasizes tails

            loss_coverage_h += lambda_coverage * alpha * w * np.abs(err)     # (V,)

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = np.maximum(0., y_nation_pred[..., :-1] - y_nation_pred[..., 1:])  # (B, V, Q-1)
        loss_crossing_h += lambda_cross * np.sum(penalty, axis=-1).mean(axis=0) # (V,)

    loss_h = loss_pinball_h + loss_coverage_h + loss_crossing_h
    # print(pd.DataFrame({'quantile_with_crossing': loss_h, 'pinball': loss_pinball_h,
    #        'coverage': loss_coverage_h, 'crossing': loss_crossing_h}).round(2))


    return loss_h, {'pinball':  loss_pinball_h,
                    'coverage': loss_coverage_h, 'crossing': loss_crossing_h}




# losses with derivatives
# ----------------------------------------------------------------------

# /!\ These two MUST remain equivalent.
#    When making modifications, we modify both in parallel.

def derivative_torch(
        y_nation_pred: torch.Tensor,
        y_nation_true: torch.Tensor,
    ) -> torch.Tensor:
    """
    First-order finite-difference derivative loss.

    Parameters
    ----------
    y_nation_pred : torch.Tensor
        Shape (B, V, Q) or (B, V)
    y_nation_true : torch.Tensor
        Shape (B, V)

    Returns
    -------
    torch.Tensor
        Shape (V)
    """

    # Ensure (B, V, Q)
    if y_nation_pred.dim() == 2:
        y_nation_pred = y_nation_pred.unsqueeze(-1)    # (B, V, 1)

    B, V, Q = y_nation_pred.shape
    device  = y_nation_pred.device

    # No horizon → no derivative loss
    if V < 2:
        return torch.zeros(V, device=device)

    assert y_nation_true.shape == (B, V), (y_nation_true.shape, B, V)


    # Temporal finite differences (within each sample)
    dy_nation_pred = y_nation_pred[:, 1:, :] - y_nation_pred[:, :-1, :]   # (B, V-1, Q)
    dy_nation_true = y_nation_true[:, 1:]    - y_nation_true[:, :-1]      # (B, V-1)

    # Broadcast true derivatives over quantiles
    dy_nation_true = dy_nation_true.unsqueeze(-1)                  # (B, V-1, 1)

    # print(f"(dy_nation_pred - dy_nation_true) ** 2: {((dy_nation_pred - dy_nation_true) ** 2).shape}")

    # average over quantiles
    deriv_err = ((dy_nation_pred - dy_nation_true) ** 2).mean(dim=-1)    # (B, V-1)

    # average over batch
    deriv_h = deriv_err.mean(dim=0)                  # (V-1,)

    # Map to horizons: prepend zero for h=0
    loss_h     = torch.zeros(V, device=device)
    loss_h[1:] = deriv_h

    return loss_h


def derivative_numpy(
        y_nation_pred: np.ndarray,
        y_nation_true: np.ndarray,
    ) -> np.ndarray:
    """
    NumPy version of first-order finite-difference derivative loss.
    Returns 0. if no temporal dimension is present.
    Must match derivative_loss_torch exactly.

    Parameters
    ----------
    y_nation_pred : np.ndarray
         Shape (B, V, Q), (B, V), or (B,)
    y_nation_true : np.ndarray
         Shape (B, V) or (B,)

    Returns
    -------
    np.ndarray
         Shape (V)
    """

    # Ensure (B, V, Q)
    if y_nation_pred.ndim == 2:
        y_nation_pred = y_nation_pred[..., np.newaxis]   # (B, V, 1)

    B, V, Q = y_nation_pred.shape

    # No horizon → no derivative loss
    if V < 2:
        return np.zeros(V)

    assert y_nation_true.shape == (B, V), (y_nation_true.shape, B, V)

    # Temporal finite differences
    dy_nation_pred = y_nation_pred[:, 1:, :] - y_nation_pred[:, :-1, :]   # (B, V-1, Q)
    dy_nation_true = y_nation_true[:, 1:]    - y_nation_true[:, :-1]      # (B, V-1)

    dy_nation_true = dy_nation_true[..., np.newaxis]               # (B, V-1, 1)

    # average over quantiles
    deriv_err = ((dy_nation_pred - dy_nation_true) ** 2).mean(axis=-1) # (B, V-1)

    # average over batch
    deriv_h = deriv_err.mean(axis=0)                  # (V-1,)

    # Map to horizons: prepend zero for h=0
    loss_h     = np.zeros(V)
    loss_h[1:] = deriv_h

    return loss_h


# wrappers (add together all components to the loss)
# ----------------------------------------------------------------------

def quantile_torch(
        y_nation_pred   : torch.Tensor,   # (B, V, Q)
        y_nation_true   : torch.Tensor,   # (B, V) or (B, V, 1)

        # constants
        quantiles       : Tuple[float, ...],
        lambda_cross    : float,
        lambda_coverage : float,
        lambda_deriv    : float,
        lambda_median   : float,
        smoothing_cross : float,

        # temperature-dependence (pinball loss, coverage penalty)
        saturation_cold_degC:float,
        threshold_cold_degC: float,
        lambda_cold     : float,
        Tavg_current    : torch.Tensor,   # (B, V, 1)
            # /!\ assumes a single one (correct for a single day)
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
    """
    Torch loss wrapper for quantile forecasts.
    """

    # print("[quantile_torch] Tavg_current", Tavg_current.shape)
    if y_nation_true.ndim == 3:
        y_nation_true = y_nation_true.squeeze(-1)

    # Base quantile + crossing loss
    loss_quantile_with_crossing_h, dict_loss_quantile_with_crossing_h = \
        quantile_with_crossing_torch(
            y_nation_pred    = y_nation_pred,
            y_nation_true    = y_nation_true,
            quantiles        = quantiles,
            lambda_cross     = lambda_cross,
            lambda_coverage  = lambda_coverage,
            smoothing        = smoothing_cross,
            saturation_cold_degC=saturation_cold_degC,
            threshold_cold_degC=threshold_cold_degC,
            lambda_cold      = lambda_cold,
            Tavg_current     = Tavg_current
        )

    # Optional derivative loss (per quantile)
    if lambda_deriv > 0.:
        loss_deriv_h = lambda_deriv * derivative_torch(y_nation_pred, y_nation_true)
    else:
        loss_deriv_h = torch.zeros_like(loss_quantile_with_crossing_h)

    if lambda_median > 0.:
        q50_pred = y_nation_pred[..., len(quantiles)//2]
        loss_median_h = lambda_median * (q50_pred - y_nation_true).mean(dim=0).abs()
    else:
        loss_median_h = torch.zeros_like(loss_quantile_with_crossing_h)

    loss_h = loss_quantile_with_crossing_h + loss_deriv_h + loss_median_h
    # print(f"loss_h (torch): {loss_h.shape}: {loss_h}")

    return loss_h, dict({#'quantile_with_crossing': loss_quantile_with_crossing_h,
                         'derivative': loss_deriv_h, 'median': loss_median_h},
                        **dict_loss_quantile_with_crossing_h)


def quantile_numpy(
        y_nation_pred   : np.ndarray,     # (B, V, Q)
        y_nation_true   : np.ndarray,     # (B, V)

        # constants
        quantiles       : Tuple[float, ...],
        lambda_cross    : float,
        lambda_coverage : float,
        lambda_deriv    : float,
        lambda_median   : float,
        smoothing_cross : float,

        # temperature-dependence (pinball loss, coverage penalty)
        saturation_cold_degC:float,
        threshold_cold_degC: float,
        lambda_cold     : float,
        Tavg_current    : np.ndarray,     # (B, V)
                # /!\ assumes a single one (correct for a single day)
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    NumPy loss wrapper for quantile forecasts.
    MUST match torch version exactly.
    """

    # print("[quantile_numpy] Tavg_current", Tavg_current.shape)
    if y_nation_true.ndim == 3:
        y_nation_true = y_nation_true.squeeze(-1)

    loss_quantile_with_crossing_h, dict_loss_quantile_with_crossing_h = \
        quantile_with_crossing_numpy(
            y_nation_pred           = y_nation_pred,
            y_nation_true           = y_nation_true,
            quantiles        = quantiles,
            lambda_cross     = lambda_cross,
            lambda_coverage  = lambda_coverage,
            smoothing        = smoothing_cross,
            saturation_cold_degC=saturation_cold_degC,
            threshold_cold_degC=threshold_cold_degC,
            lambda_cold      = lambda_cold,
            Tavg_current     = Tavg_current
        )

    if lambda_deriv > 0.:
        loss_deriv_h = lambda_deriv * derivative_numpy(y_nation_pred, y_nation_true)
    else:
        loss_deriv_h = np.zeros_like(loss_quantile_with_crossing_h)

    if lambda_median > 0.:
        q50_pred = y_nation_pred[..., len(quantiles)//2]
        loss_median_h = lambda_median * np.abs((q50_pred - y_nation_true).mean(axis=0))
    else:
        loss_median_h = np.zeros_like(loss_quantile_with_crossing_h)

    loss_h = loss_quantile_with_crossing_h + loss_deriv_h + loss_median_h
    # print(f"loss_h (numpy): {loss_h.shape}: {loss_h}")

    return loss_h, dict({#'quantile_with_crossing': loss_quantile_with_crossing_h,
                         'derivative': loss_deriv_h, 'median': loss_median_h},
                        **dict_loss_quantile_with_crossing_h)



# losses for région consumptions
# ----------------------------------------------------------------------

# NB: this is where we set to which region each number in Y_regions_XX corresponds

def regions_torch(
        Y_regions_pred    : torch.Tensor,     # (B, V, R)
        Y_regions_true    : torch.Tensor,     # (B, V, R)
        lambda_regions    : float,   # comparing pred and true region by region
        lambda_regions_sum: float    # comparing pred and true nationally
    ) -> torch.Tensor:
    """
    Torch loss for régional consumption forecasts.
    y_nation_pred, y_nation_true: (B, V, R)
    MUST match numpy version exactly.
    """
    # print(f"shapes Y_regions_pred {Y_regions_pred.shape}, "
    #       f"Y_regions_true {Y_regions_true.shape}")

    if lambda_regions <= 0.:
        return torch.zeros(Y_regions_pred.shape[1]).to(Y_regions_true.device)

    # MAE region by region
    abs_err= (Y_regions_pred - Y_regions_true).abs().mean(dim=0)   # (V, R)
    out    = torch.sum(abs_err, dim=1)                             # (V)

    # national total
    err    = (Y_regions_pred - Y_regions_true)      .mean(dim=0)   # (V, R)
    out   += torch.sum(err,      dim=1).abs() * lambda_regions_sum # (V)

    return lambda_regions * out


def regions_numpy(
        Y_regions_pred    : np.ndarray,     # (B, V, R)
        Y_regions_true    : np.ndarray,     # (B, V, R)
        lambda_regions    : float,   # comparing pred and true region by region
        lambda_regions_sum: float    # comparing pred and true nationally
    ) -> np.ndarray:
    """
    NumPy loss for régional consumption forecasts.
    y_nation_pred, y_nation_true: (B, V, R)
    MUST match torch version exactly.
    """
    # print(f"shapes Y_regions_pred {Y_regions_pred.shape}, "
    #       f"Y_regions_true {Y_regions_true.shape}")

    if lambda_regions <= 0.:
        return np.zeros(Y_regions_pred.shape[1])

    # MAE region by region
    abs_err= np.abs(Y_regions_pred - Y_regions_true).mean(axis=0)   # (V, R)
    out    =        np.sum(abs_err, axis=1)                         # (V)

    # national total
    err    =       (Y_regions_pred - Y_regions_true).mean(axis=0)   # (V, R)
    out   += np.abs(np.sum(err,     axis=1)) * lambda_regions_sum   # (V)

    return lambda_regions * out

###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# losses.py
# Loss functions for the NNTQ model
#
###############################################################################


from   typing import Dict, Tuple, Optional  # , List Sequence

import torch

# import pandas as pd


# import utils





# Pinball (quantile) loss
# ----------------------------------------------------------------------

def penalty_nation_cold_torch(
        saturation_cold_degC: float,
        threshold_cold_degC : float,
        Tavg_current        : torch.Tensor,  # (B, V): temperature per sample/horizon
    ) -> torch.Tensor:  # returns the same shape as Tavg_current

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
    Tavg_current    : torch.Tensor,  # (B, V): per-sample temperatures

) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
    """
    Joint quantile loss with crossing penalty.
    """

    B, V, Q = y_nation_pred.shape
    device  = y_nation_pred.device

    loss_pinball_h  = torch.zeros(V, device=device)
    loss_coverage_h = torch.zeros(V, device=device)
    loss_crossing_h = torch.zeros(V, device=device)

    # one weight per SAMPLE (= per forecast day), from that day's mean temperature
    # /!\ was `.mean()` over the whole (shuffled) batch: a near-constant scalar,
    #     identical for cold and mild days, which neutralized the cold weighting
    _penalty_per_day = lambda_cold * penalty_nation_cold_torch(
            saturation_cold_degC, threshold_cold_degC, Tavg_current.to(device))
    if _penalty_per_day.dim() == 1:                              # (B,)
        _penalty_per_day = _penalty_per_day.unsqueeze(-1)        # (B, 1)
    else:                                                        # (B, V)
        _penalty_per_day = _penalty_per_day.mean(dim=-1, keepdim=True)  # (B, 1)

    w_cold = 1. + _penalty_per_day                               # (B, 1)

    if lambda_coverage > 0.:
        # smoothing scale per horizon, from the target variability
        #   (/!\ was recomputed for every quantile)
        scale_h    = torch.std(y_nation_true, dim=0, unbiased=False)  # (V,)
        tau_smooth = smoothing * scale_h.clamp_min(1e-3)              # (V,)

    for i, tau in enumerate(quantiles):
        diff = y_nation_true - y_nation_pred[..., i]
        pin = torch.maximum(tau * diff, -(1 - tau) * diff)
        loss_pinball_h += (w_cold * pin).mean(dim=0)   # cold days weigh more

        # Coverage penalty
        if lambda_coverage > 0.:
            z = -diff / tau_smooth   # broadcast over B
            z = torch.clamp(z, -20., 20.)  # preventing overflow
            soft_ind   = torch.sigmoid(z)         # (B, V)
            coverage_h = (w_cold * soft_ind).mean(dim=0) / w_cold.mean(dim=0) # (V,)
                # weighted (normalized) coverage: cold days count more, but a
                # perfectly calibrated forecast still yields coverage == tau
                # (the previous, unnormalized product biased coverage upward)

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





# losses with derivatives
# ----------------------------------------------------------------------

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

    # No horizon => no derivative loss
    if V < 2:
        return torch.zeros(V, device=device)

    assert y_nation_true.shape == (B, V), (y_nation_true.shape, B, V)


    # Temporal finite differences (within each sample)
    dy_nation_pred = y_nation_pred[:, 1:, :] - y_nation_pred[:, :-1, :] # (B, V-1, Q)
    dy_nation_true = y_nation_true[:, 1:]    - y_nation_true[:, :-1]    # (B, V-1)

    # Broadcast true derivatives over quantiles
    dy_nation_true = dy_nation_true.unsqueeze(-1)                       # (B, V-1, 1)

    # print(f"(dy_nation_pred - dy_nation_true) ** 2: "
    #       f"{((dy_nation_pred - dy_nation_true) ** 2).shape}")

    # average over quantiles
    deriv_err = ((dy_nation_pred - dy_nation_true) ** 2).mean(dim=-1)   # (B, V-1)

    # average over batch
    deriv_h = deriv_err.mean(dim=0)                  # (V-1,)

    # Map to horizons: prepend zero for h=0
    loss_h     = torch.zeros(V, device=device)
    loss_h[1:] = deriv_h

    return loss_h


# wrappers (add together all components to the loss)
# ----------------------------------------------------------------------

def quantile_torch(
        y_nation_pred     : torch.Tensor,   # (B, V, Q)
        y_nation_true     : torch.Tensor,   # (B, V) or (B, V, 1)

        # constants
        quantiles         : Tuple[float, ...],
        lambda_cross      : float,
        lambda_coverage   : float,
        lambda_deriv      : float,
        lambda_median     : float,
        smoothing_cross   : float,

            # temperature-dependence (pinball loss, coverage penalty)
        saturation_cold_degC:float,
        threshold_cold_degC:float,
        lambda_cold       : float,
        Tavg_current      : torch.Tensor,   # (B, V): per-sample temperatures
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

            # constants
            quantiles        = quantiles,
            lambda_cross     = lambda_cross,
            lambda_coverage  = lambda_coverage,
            smoothing        = smoothing_cross,
                # temperature-dependence (pinball loss, coverage penalty)
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


# losses for région consumptions
# ----------------------------------------------------------------------

# NB: this is where we set to which region each number in Y_regions_XX corresponds

def regions_torch(
        Y_regions_pred    : torch.Tensor,     # (B, V, R)
        Y_regions_true    : torch.Tensor,     # (B, V, R)
        lambda_regions    : float,   # comparing pred and true region by region
        lambda_regions_sum: float,   # comparing pred and true nationally
        regions_to_nation : Optional[torch.Tensor] = None   # (R,)
    ) -> torch.Tensor:
    """
    Torch loss for régional consumption forecasts.
    y_nation_pred, y_nation_true: (B, V, R), each region in its own scaled
    units. `regions_to_nation` (region std / national std) converts them to
    national scaled units, so that errors are in GW (up to the national std):
    the sum over the regions is then the national error.
    (/!\ without it, as before: errors in each region's own std units, whose
     sum is not the national error)
    """
    # print(f"shapes Y_regions_pred {Y_regions_pred.shape}, "
    #       f"Y_regions_true {Y_regions_true.shape}")

    if lambda_regions <= 0.:
        return torch.zeros(Y_regions_pred.shape[1]).to(Y_regions_true.device)

    diff = Y_regions_pred - Y_regions_true                           # (B, V, R)
    if regions_to_nation is not None:
        diff = diff * regions_to_nation.to(diff)                     # national units

    # MAE region by region
    abs_err= diff.abs().mean(dim=0)                                  # (V, R)
    out    = torch.sum(abs_err, dim=1)                               # (V)

    # national total
    err    = diff.mean(dim=0)                                        # (V, R)
    out   += torch.sum(err,      dim=1).abs() * lambda_regions_sum   # (V)

    return lambda_regions * out


from   typing import Tuple, Dict  #, Sequence, List, Optional
# from   collections import defaultdict

import torch
import torch.nn as nn
from   torch.utils.data         import Dataset, DataLoader
# from   torch.optim.lr_scheduler import ReduceLROnPlateau

from   sklearn.preprocessing   import StandardScaler

import numpy  as np
import pandas as pd


import  day_ahead, utils



# ----------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------

# Pinball (quantile) loss
# ----------------------------------------------------------------------

def quantile_loss_with_crossing_torch(
    y_pred:         torch.Tensor,     # (B, H, 3)
    y_true:         torch.Tensor,     # (B, H) or (B, H, 1)
    quantiles:      Tuple[float, ...],
    lambda_cross:   float,
    lambda_coverage:float
) -> torch.Tensor:
    """
    Joint quantile loss with crossing penalty.
    """

    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    loss = 0.

    for i, tau in enumerate(quantiles):
        diff = y_true - y_pred[..., i]
        loss += torch.mean(torch.maximum(tau * diff, -(1-tau) * diff))

        # Coverage penalty
        if lambda_coverage > 0.:
            coverage = (y_true <= y_pred[..., i]).float().mean()
            alpha = 1. / (tau * (1-tau))   # emphasizes tails
            loss += lambda_coverage * alpha * (coverage - tau) ** 2

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = torch.relu(y_pred[..., :-1] - y_pred[..., 1:])
        loss   += lambda_cross * penalty.sum(dim=-1).mean()


    return loss


def quantile_loss_with_crossing_numpy(
        y_pred        : np.ndarray,     # (B, H, Q)
        y_true        : np.ndarray,     # (B, H)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float
    ) -> float:

    loss = 0.

    for i, tau in enumerate(quantiles):
        diff = y_true - y_pred[..., i]
        loss += np.mean(np.maximum(tau * diff, -(1-tau) * diff))

        # Coverage penalty
        if lambda_coverage > 0.:
            coverage = (y_true <= y_pred[..., i]).mean()
            alpha = 1. / (tau * (1-tau))   # emphasizes tails
            loss += lambda_coverage * alpha * (coverage - tau) ** 2

    # Crossing penalty
    if lambda_cross > 0.:
        penalty = np.maximum(0., y_pred[..., :-1] - y_pred[..., 1:])
        loss   += lambda_cross * np.mean(np.sum(penalty, axis=-1))

    return float(loss)


# losses with derivatives
# ----------------------------------------------------------------------

# /!\ These two MUST remain equivalent.
#    When making modifications, we modify both in parallel.

def derivative_loss_torch(
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


def derivative_loss_numpy(
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

def loss_wrapper_quantile_torch(
        y_pred        : torch.Tensor,   # (B, H, Q)
        y_true        : torch.Tensor,   # (B, H) or (B, H, 1)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
    ) -> torch.Tensor:
    """
    Torch loss wrapper for quantile forecasts.
    """

    # Base quantile + crossing loss
    loss = quantile_loss_with_crossing_torch(
        y_pred       = y_pred,
        y_true       = y_true,
        quantiles    = quantiles,
        lambda_cross = lambda_cross,
        lambda_coverage=lambda_coverage
    )

    # Optional derivative loss (per quantile)
    if lambda_deriv > 0.:
        _y_true = y_true.squeeze(-1) if y_true.ndim == 3 else y_true
        loss += lambda_deriv * derivative_loss_torch(y_pred, _y_true)

    return loss


def loss_wrapper_quantile_numpy(
        y_pred        : np.ndarray,     # (B, H, Q)
        y_true        : np.ndarray,     # (B, H)
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
    ) -> float:
    """
    NumPy loss wrapper for quantile forecasts.
    MUST match torch version exactly.
    """

    loss = quantile_loss_with_crossing_numpy(
        y_pred       = y_pred,
        y_true       = y_true,
        quantiles    = quantiles,
        lambda_cross = lambda_cross,
        lambda_coverage=lambda_coverage
    )

    if lambda_deriv > 0.:
        loss += lambda_deriv * derivative_loss_numpy(y_pred, y_true)

    return float(loss)



# Metamodel: losses (predictions are in utils.py)
# ----------------------------------------------------------------------

def compute_meta_loss(
        pred_scaled   : torch.Tensor,   # (B, H)
        x_scaled      : torch.Tensor,   # (B, L, F)
        y_scaled      : torch.Tensor,   # (B, H, 1)
        baseline_idx  : Dict[str, int],
        weights_meta  : Dict[str, float],
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float
    ) -> torch.Tensor:
    """
    Returns:
        meta_loss_scaled : torch scalar tensor
    """

    B, _, _ = x_scaled.shape

    pred_meta_scaled = utils.compute_meta_prediction_torch(
        pred_scaled, x_scaled, baseline_idx, weights_meta, len(quantiles)//2)

    # Match target shape
    y_scaled_1 = y_scaled[:, 0, 0]  # .reshape(B)

    return loss_wrapper_quantile_torch(pred_meta_scaled, y_scaled_1,
                    quantiles, lambda_cross, lambda_coverage, lambda_deriv)





# ============================================================
# 3. DATASET CLASS (multivariate input → multistep target + future features)
# ============================================================

# class MultiVarDataset(Dataset):
#     def __init__(self, data, split:str, input_len:int, pred_len:int, target_index=0):
#         """
#         data: numpy array of shape (T, F)
#             column 0 must be the target (log-real_TR)
#         input_len: number of past steps (L)
#         pred_len: number of future steps (H)
#         target_index: which column to forecast (0 by default)
#         """
#         self.data        = data.astype(np.float32)
#         self.split       = split
#         self.input_len   = input_len
#         self.pred_len    = pred_len
#         self.target_index= target_index

#     def __len__(self):
#         # +1 to match original sliding window count
#         return len(self.data) - self.input_len - self.pred_len + 1

#     def __getitem__(self, idx):
#         # Input window (all features)
#         x = self.data[idx : idx + self.input_len]              # shape (L, F)

#         # Multi-step target (only the 1 column)
#         y = self.data[
#             idx + self.input_len : idx + self.input_len + self.pred_len,
#             self.target_index
#         ]                                                      # shape (H,)

#         # # ground-truth future non-target features (for auxiliary loss)
#         # y_features = self.data[
#         #     idx + self.input_len : idx + self.input_len + self.pred_len,
#         #     1:
#         # ]                                                      # shape (H, F-1)

#         if self.split == "test" or self.split == "valid":
#             return (
#                 torch.tensor(x),                # (L, F)
#                 torch.tensor(y).unsqueeze(-1),  # (H, 1)
#                 # torch.tensor(y_features),       # (H, F-1)
#                 idx                             # the true global window index
#             )
#         else:
#             return (
#                 torch.tensor(x),                # (L, F)
#                 torch.tensor(y).unsqueeze(-1),  # (H, 1)
#                 # torch.tensor(y_features),       # (H, F-1)
#             )




# def make_X_and_y(series, dates,
#                  train_split, n_valid,
#                  feature_cols, target_col,
#                  input_length:int, pred_length:int, batch_size:int,
#                  verbose: int = 0):


#     # TODO: DayAheadDataset needs INPUT_LENGTH history before the first validation noon.
#     #    val_start = TRAIN_SPLIT - INPUT_LENGTH - PRED_LENGTH


#     # Map column names -> column indices in train_data
#     all_cols   = [target_col] + feature_cols
#     col_to_idx = {col: i for i, col in enumerate(all_cols)}

#     feature_idx= [col_to_idx[c] for c in feature_cols]
#     target_idx =  col_to_idx[target_col]


#     # 1. Extract X and y using names
#     X_GW = series[:, feature_idx];  y_GW = series[:, target_idx]

#     # if verbose >= 2:
#     #     print(f"y_train: mean{y_train_GW.mean():6.2f} GW, std{y_train_GW.std():6.2f} GW")
#     #     print(f"y_valid: mean{y_valid_GW.mean():6.2f} GW, std{y_valid_GW.std():6.2f} GW")


#     # 2. Fit two different scalers (on training set)
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()

#     X_train_GW = X_GW[:train_split][:-n_valid]
#     y_train_GW = y_GW[:train_split][:-n_valid]
#     scaler_x.fit(X_train_GW)
#     scaler_y.fit(y_train_GW.reshape(-1, 1))


#     # 3. Transform X and y separately
#     X_scaled = scaler_x.transform(X_GW)
#     y_scaled = scaler_y.transform(y_GW.reshape(-1, 1)).ravel()

#     df_scaled = np.column_stack([y_scaled, X_scaled])


#     # 5. SAFETY CHECKS
#     print("scaler_y.mean_.shape:", scaler_y.mean_.shape)
#     print("scaler_y.scale_.shape:", scaler_y.scale_.shape)

#     assert scaler_y.mean_.shape[0] == 1, "scaler_y must be fitted on ONE target only"
#     assert df_scaled     .shape[1] == 1 + len(feature_cols), "scaled feature count mismatch"


#     # 4. Rebuild scaled arrays for your pipeline
#     #    (target in column 0, features after)
#     train_scaled = df_scaled[:train_split]
#     test_scaled  = df_scaled[train_split:]

#      # validation
#     valid_scaled = train_scaled[-n_valid:]
#     train_scaled = train_scaled[:-n_valid]


#     # 5. DATASET FOR TRANSFORMER
#     train_dataset_scaled = MultiVarDataset(
#         train_scaled, "train", input_length, pred_length, target_index=0)
#     valid_dataset_scaled = MultiVarDataset(
#         valid_scaled, "valid", input_length, pred_length, target_index=0)
#     test_dataset_scaled  = MultiVarDataset(
#         test_scaled,  "test",  input_length, pred_length, target_index=0)

#     # Note: DataLoader now yields tuples (x, y, y_features)
#     train_loader = DataLoader(train_dataset_scaled, batch_size=batch_size,
#                               shuffle=True, drop_last=True)
#     valid_loader = DataLoader(valid_dataset_scaled, batch_size=batch_size*4,
#                               shuffle=False, drop_last=False) # was drop_last=True
#     test_loader  = DataLoader(test_dataset_scaled , batch_size=64,
#                               shuffle=False, drop_last=False) # was w/o drop_last


#     train_dates = dates[:train_split]
#     test_dates  = dates[train_split:]
#     valid_dates = train_dates[-n_valid:]
#     train_dates = train_dates[:-n_valid]

#     test_data  = series[train_split:]
#     X_test_GW  = test_data[:, feature_idx];  y_test_GW  = test_data [:, target_idx]


#     return [train_loader,valid_loader,test_loader], \
#            [train_dates, valid_dates, test_dates ],\
#             scaler_y, X_test_GW, y_test_GW, test_dataset_scaled, test_scaled



def make_X_and_y(series, dates,
                 train_split, n_valid,
                 feature_cols, target_col,
                 input_length:int, pred_length:int, batch_size:int,
                 do_day_ahead:bool=False, forecast_hour:int=12,
                 verbose: int = 0):

    assert series.shape[0] == len(dates), \
        f"series.shape ({series.shape}) != len(dates) ({len(dates)})"

    # print(f"Forecast hour: {forecast_hour:2n}:00")
    # print(f"input_length:{input_length:4n} [half-hours]")
    # print(f"pred_length: {pred_length :4n} [half-hours]")
    # print(f"batch_size:  {batch_size  :4n}")
    # print()

    train_dates = dates[:train_split]
    test_dates  = dates[train_split:]
    valid_dates = train_dates[-n_valid:]
    train_dates = train_dates[:-n_valid]


    # TODO: DayAheadDataset needs INPUT_LENGTH history before the first validation noon.
    #    val_start = TRAIN_SPLIT - INPUT_LENGTH - PRED_LENGTH


    # Map column names -> column indices in train_data
    all_cols   = [target_col] + feature_cols
    col_to_idx = {col: i for i, col in enumerate(all_cols)}

    feature_idx= [col_to_idx[c] for c in feature_cols]
    target_idx =  col_to_idx[target_col]

    test_data  = series[train_split:]
    X_test_GW  = test_data[:, feature_idx];  y_test_GW  = test_data [:, target_idx]


    # 1. Extract X and y using names
    X_GW = series[:, feature_idx];  y_GW = series[:, target_idx]

    # if verbose >= 2:
    #     print(f"y_train: mean{y_train_GW.mean():6.2f} GW, std{y_train_GW.std():6.2f} GW")
    #     print(f"y_valid: mean{y_valid_GW.mean():6.2f} GW, std{y_valid_GW.std():6.2f} GW")


    # 2. Fit two different scalers (on training set)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_GW = X_GW[:train_split][:-n_valid]
    y_train_GW = y_GW[:train_split][:-n_valid]
    scaler_x.fit(X_train_GW)
    scaler_y.fit(y_train_GW.reshape(-1, 1))


    # 3. Transform X and y separately
    X_scaled = scaler_x.transform(X_GW)
    y_scaled = scaler_y.transform(y_GW.reshape(-1, 1)).ravel()

    df_scaled = np.column_stack([y_scaled, X_scaled])


    # 5. SAFETY CHECKS
    # print("scaler_y.mean_.shape:", scaler_y.mean_.shape)
    # print("scaler_y.scale_.shape:", scaler_y.scale_.shape)

    assert scaler_y.mean_.shape[0] == 1, "scaler_y must be fitted on ONE target only"
    assert df_scaled     .shape[1] == 1 + len(feature_cols), \
        "scaled feature count mismatch"


    # 4. Rebuild scaled arrays for your pipeline
    #    (target in column 0, features after)
    train_scaled = df_scaled[:train_split]
    test_scaled  = df_scaled[train_split:]

     # validation
    valid_scaled = train_scaled[-n_valid:]
    train_scaled = train_scaled[:-n_valid]

    # print("len(X_scaled) [half-hours]", len(train_scaled),len(valid_scaled),len(test_scaled))
    # print("len(X_dates)  [half-hours]", len(train_dates), len(valid_dates), len(test_dates))



    # DAY-AHEAD PATH (SCALED, DELEGATED)

    if do_day_ahead:

        assert pred_length == 48,\
            f"Day-ahead forecasting requires pred_length == 48, not {pred_length}"

        def build_day_ahead(data, date_slice):
            return day_ahead.DayAheadDataset(
                data        = data,
                dates       = date_slice,
                input_length= input_length,
                pred_length = pred_length,
                forecast_hour=forecast_hour,
                target_index= target_idx
            ) # X_list, y_list, origin_list, target_dates_list

        train_dataset_scaled = build_day_ahead(train_scaled, train_dates)
        valid_dataset_scaled = build_day_ahead(valid_scaled, valid_dates)
        test_dataset_scaled  = build_day_ahead(test_scaled,  test_dates )

        print("len(X_dataset_scaled[0]) [days]", len(train_dataset_scaled[0]),
              len(valid_dataset_scaled[0]), len(test_dataset_scaled[0]))

        # print("DATASET TYPE:", type   (train_dataset_scaled))
        # print("DATASET LEN :", len    (train_dataset_scaled))
        # print("HAS __len__ :",   hasattr(train_dataset_scaled, "__len__"))
        # print("HAS __getitem__:",hasattr(train_dataset_scaled, "__getitem__"))


        # if verbose >= 1:
        #     print("\n[make_X_and_y] Day-ahead mode (scaled)")
            # print(f"  Train samples: {len(train_dataset_scaled)}")
            # print(f"  Valid samples: {len(valid_dataset_scaled)}")
            # print(f"  Test samples:  {len(test_dataset_scaled )}")

        # print(train_dataset_scaled[0][0][:5])

        def build_loader(dataset, shuffle, drop_last):
            return DataLoader(
                dataset,
                batch_size = batch_size if shuffle else batch_size * 2,
                shuffle    = shuffle,
                drop_last  = drop_last,
                # num_workers= num_workers,
                # pin_memory = pin_memory,
            )

        train_loader = build_loader(train_dataset_scaled, shuffle=True, drop_last=True )
        valid_loader = build_loader(valid_dataset_scaled, shuffle=False,drop_last=False)
        test_loader  = build_loader(test_dataset_scaled,  shuffle=False,drop_last=False)

        print("len(X_loader):", len(train_loader), len(valid_loader), len(test_loader))

        # print(); print("valid_loader:")
        # for batch_idx, (x_scaled, y_scaled, origin_unix) in enumerate(valid_loader):
            # # Convert back to timestamps if needed
            # origins = [pd.Timestamp(t, unit='s') for t in origin_unix.tolist()]
            # print(batch_idx, x_scaled, y_scaled, origins[0], "to", origins[-1])

        return (
            [train_loader,valid_loader,test_loader], \
            [train_dates, valid_dates, test_dates ],\
             scaler_y, X_test_GW, y_test_GW,
             test_dataset_scaled, test_scaled[:, feature_idx]
        )


    # SLIDING-WINDOW PATH (UNCHANGED, uses *_scaled)
    raise NotImplementedError(
        "Original sliding-window implementation is obsolete."
    )

    # # 5. DATASET FOR TRANSFORMER
    # train_dataset_scaled = MultiVarDataset(
    #     train_scaled, "train", input_length, pred_length, target_index=0)
    # valid_dataset_scaled = MultiVarDataset(
    #     valid_scaled, "valid", input_length, pred_length, target_index=0)
    # test_dataset_scaled  = MultiVarDataset(
    #     test_scaled,  "test",  input_length, pred_length, target_index=0)

    # # Note: DataLoader now yields tuples (x, y, y_features)
    # train_loader = DataLoader(train_dataset_scaled, batch_size=batch_size,
    #                           shuffle=True, drop_last=True)
    # valid_loader = DataLoader(valid_dataset_scaled, batch_size=batch_size*4,
    #                           shuffle=False, drop_last=False) # was drop_last=True
    # test_loader  = DataLoader(test_dataset_scaled , batch_size=64,
    #                           shuffle=False, drop_last=False) # was w/o drop_last



    # return [train_loader,valid_loader,test_loader], \
    #        [train_dates, valid_dates, test_dates ],\
    #         scaler_y, X_test_GW, y_test_GW, test_dataset_scaled, test_scaled




# def make_X_and_y(train_data, valid_data, test_data,
#                  feature_cols, target_col,
#                  input_length:int, pred_length:int, batch_size:int,
#                  day_ahead:bool=False, forecast_hour:int=12, dates=None,
#                  verbose: int = 0):

#     # Helper: dataset + loader
#     def build_loader(dataset, shuffle, drop_last):
#         return DataLoader(
#             dataset,
#             batch_size=batch_size if shuffle else batch_size * 2,
#             shuffle    = shuffle,
#             drop_last  = drop_last,
#             # num_workers= num_workers,
#             # pin_memory = pin_memory,
#         )

#     if day_ahead:
#         assert dates is not None
#         assert pred_length == 48,\
#             f"Day-ahead forecasting requires pred_length == 48, not {pred_length}"

#     # Map column names -> column indices in train_data
#     all_cols   = [target_col] + feature_cols
#     col_to_idx = {col: i for i, col in enumerate(all_cols)}


#     feature_idx= [col_to_idx[c] for c in feature_cols]
#     target_idx =  col_to_idx[target_col]

#     # TODO: 1. run scaler_x and scaler_y, transform, 2. split into train, valide, test
#     #    not in the opposite order.

#     # 1. Extract X and y using names
#     X_train_GW = train_data[:, feature_idx];  y_train_GW = train_data[:, target_idx]
#     X_valid_GW = valid_data[:, feature_idx];  y_valid_GW = valid_data[:, target_idx]
#     X_test_GW  = test_data [:, feature_idx];  y_test_GW  = test_data [:, target_idx]

#     if verbose >= 2:
#         print(f"y_train: mean{y_train_GW.mean():6.2f} GW, std{y_train_GW.std():6.2f} GW")
#         print(f"y_valid: mean{y_valid_GW.mean():6.2f} GW, std{y_valid_GW.std():6.2f} GW")


#     # 2. Fit TWO different scalers
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()

#     scaler_x.fit(X_train_GW)
#     scaler_y.fit(y_train_GW.reshape(-1, 1))


#     # print("scaler_y.mean_.shape:", scaler_y.mean_.shape)
#     # print("scaler_y.scale_.shape:", scaler_y.scale_.shape)


#     # 3. Transform X and y separately
#     X_train_scaled = scaler_x.transform(X_train_GW)
#     X_valid_scaled = scaler_x.transform(X_valid_GW)
#     X_test_scaled  = scaler_x.transform(X_test_GW)

#     y_train_scaled = scaler_y.transform(y_train_GW.reshape(-1, 1)).ravel()
#     y_valid_scaled = scaler_y.transform(y_valid_GW.reshape(-1, 1)).ravel()
#     y_test_scaled  = scaler_y.transform(y_test_GW .reshape(-1, 1)).ravel()


#     # 4. Rebuild scaled arrays for your pipeline
#     #    (target in column 0, features after)
#     train_scaled = np.column_stack([y_train_scaled, X_train_scaled])
#     valid_scaled = np.column_stack([y_valid_scaled, X_valid_scaled])
#     test_scaled  = np.column_stack([y_test_scaled , X_test_scaled ])


#     # 5. SAFETY CHECKS (important)
#     assert scaler_y.mean_.shape[0] == 1, "scaler_y must be fitted on ONE target only"
#     assert train_scaled  .shape[1] == 1 + len(feature_cols),\
#         "scaled feature count mismatch"


#     # DAY-AHEAD PATH (SCALED, DELEGATED)

#     if day_ahead:
#         from day_ahead_forecast import create_day_ahead_dataset

#         assert pred_length == 48, "Day-ahead forecasting requires pred_length=48"

#         def build_day_ahead(data, date_slice):
#             return create_day_ahead_dataset(
#                 data=data,
#                 dates=date_slice,
#                 input_length=input_length,
#                 pred_length=pred_length,
#                 forecast_hour=forecast_hour,
#             )

#         train_dataset_scaled = build_day_ahead(train_scaled, train_dates)
#         valid_dataset_scaled = build_day_ahead(valid_scaled, valid_dates)
#         test_dataset_scaled  = build_day_ahead(test_scaled,  test_dates )

#         train_loader = build_loader(train_dataset_scaled, True,  True)
#         valid_loader = build_loader(valid_dataset_scaled, False, False)
#         test_loader  = build_loader(test_dataset_scaled,  False, False)

#         print("\n[make_X_and_y] Day-ahead mode (scaled)")
#         print(f"  Forecast hour: {forecast_hour}:00")
#         print(f"  Train samples: {len(train_dataset_scaled)}")
#         print(f"  Valid samples: {len(valid_dataset_scaled)}")
#         print(f"  Test samples:  {len(test_dataset_scaled )}")

#         return (
#             train_loader,
#             valid_loader,
#             test_loader,
#             scaler_y,       # <-- IMPORTANT: preserved
#             None,
#             None,
#             test_dataset_scaled,
#             test_scaled,
#         )


#     # SLIDING-WINDOW PATH (UNCHANGED, uses *_scaled)
#     raise NotImplementedError(
#         "Original sliding-window implementation is obsolete."
#     )

#     # train_dataset_scaled = MultiVarDataset(
#     #     train_scaled, "train", input_length, pred_length, target_index=0)
#     # valid_dataset_scaled = MultiVarDataset(
#     #     valid_scaled, "valid", input_length, pred_length, target_index=0)
#     # test_dataset_scaled  = MultiVarDataset(
#     #     test_scaled,  "test",  input_length, pred_length, target_index=0)


#     # # Note: DataLoader now yields tuples (x, y, y_features)
#     # train_loader = DataLoader(train_dataset_scaled, batch_size=batch_size,
#     #                           shuffle=True, drop_last=True)
#     # valid_loader = DataLoader(valid_dataset_scaled, batch_size=batch_size*4,
#     #                           shuffle=False, drop_last=False) # was drop_last=True
#     # test_loader  = DataLoader(test_dataset_scaled , batch_size=64,
#     #                           shuffle=False, drop_last=False) # was w/o drop_last

#     return train_loader, valid_loader, test_loader, scaler_y, \
#         X_test_GW, y_test_GW, test_dataset_scaled, test_scaled





# ============================================================
# 4. Collects Attention
# ============================================================

class TransformerEncoderLayerWithAttn(nn.Module):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-8):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            norm = x.norm(2, dim=-1, keepdim=True)
            return x * self.weight / (norm / (x.shape[-1]**0.5 + self.eps))


    def __init__(self, d_model, nhead, dropout, ffn_mult):
        super().__init__()

        # MultiheadAttention with batch_first keeps shapes (B, L, D)
        self.self_attn = nn.MultiheadAttention(
            embed_dim  = d_model,
            num_heads  = nhead,
            dropout    = dropout,
            batch_first= True  # ok in recent PyTorch
        )

        self.linear1 = nn.Linear(d_model, ffn_mult*d_model)
        self.linear2 = nn.Linear(ffn_mult*d_model, d_model)
        self.norm1   = self.RMSNorm(d_model)
        self.norm2   = self.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attn
        )

        # attn_weights shape depends on PyTorch version:
        # - recent PyTorch (when average_attn_weights arg available in call)
        #   => attn_weights shape: (batch, num_heads, L, L)
        # - older PyTorch => attn_weights shape: (batch * num_heads, L, L)
        if return_attn and attn_weights is not None:
            if attn_weights.dim() == 4:
                attn = attn_weights  # already (B, H, L, L)
            elif attn_weights.dim() == 3:
                # reshape (B*H, L, L) -> (B, H, L, L)
                B = x.size(0)
                L = x.size(1)
                H = attn_weights.size(0) // B
                attn = attn_weights.view(B, H, L, L)
            else:
                # unexpected shape; pass through
                attn = attn_weights
        else:
            attn = None

        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff = self.linear2(torch.relu(self.linear1(x)))
        x = x + self.dropout(ff)
        x = self.norm2(x)

        return x, attn


# ============================================================
# 5. TRANSFORMER MODEL (ENCODER → REPEAT LAST STATE)
# - Modified minimally: add a small head to predict future non-target features
# ============================================================


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, in_channels, d_model):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(1, 2)        # (B, C, L)
        x = self.proj(x)             # (B, D, T)
        return x.transpose(1, 2)     # (B, T, D)



class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features:int, dim_model:int, nhead:int, num_layers:int,
                 input_len:int, patch_len:int, stride:int, pred_len:int,
                 dropout:float, ffn_mult:int, num_quantiles:int,
                 num_geo_blocks, geo_block_ratio):
        super().__init__()

        self.num_features   = num_features
        self.dim_model      = dim_model
        self.pred_len       = pred_len
        self.num_quantiles  = num_quantiles

        self.input_len      = input_len
        self.patch_len      = patch_len
        self.stride         = stride
        self.num_patches    = (input_len - patch_len) // stride + 1

        total_covered = ((input_len - patch_len) // stride) * stride + patch_len
        self.pad_len  = input_len - total_covered

        self.patch_embed = PatchEmbedding(patch_len, stride, num_features, dim_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(dim_model, nhead, dropout, ffn_mult)
            for _ in range(num_layers)
        ])

        # blocks
        self.num_geo_blocks = num_geo_blocks
        self.geo_block_ratio= geo_block_ratio

        self.block_sizes    = geometric_block_sizes(
            self.num_patches,
            self.num_geo_blocks,
            self.geo_block_ratio
        )
        assert sum(self.block_sizes) == self.num_patches

        self.block_ranges = []
        idx = 0
        for size in self.block_sizes:
            self.block_ranges.append((idx, idx + size))
            idx += size

        self.block_weighting = BlockWeighting(
            num_blocks = num_geo_blocks,
            model_dim  = dim_model
        )


        # fc_out
        self.fc_out = nn.Sequential(
            nn.Linear(2 * dim_model, dim_model),
            nn.GELU(),
            nn.Linear(dim_model, pred_len * num_quantiles)
        )
        # self.fc_out = nn.Linear(dim_model, pred_len)



    def forward(self, x, *args, **kwargs):
        B, L, F = x.shape    # x: (B, input_len, features)
        assert L == self.input_len,    (L, self.input_len)
        assert F == self.num_features, (F, self.num_features)

        # guaranteeing: last patch ends exactly at t = L
        if self.pad_len > 0:
            x = torch.nn.functional.pad(
                x,
                pad  = (0, 0, 0, self.pad_len),  # pad time dimension on the right
                mode = "constant",
                value= 0.
        )

        # 1. Patch embedding
        h = self.patch_embed(x)                     # (B, num_patches, model_dim)

        # 2. Transformer encoder
        for layer in self.layers:
            h, _ = layer(h, return_attn=False)      # (B, num_patches, model_dim)

        # Geometric block pooling
        B, T, D = h.shape   # (batch_size = 256, num_tokens ≈ 160, model_dim = 256)
        assert T == self.num_patches, (T, self.num_patches)
        assert D == self.dim_model,   (D, self.dim_model)

        # block_sizes = geometric_block_sizes(
        #     T, self.num_geo_blocks, self.geo_block_ratio)

        assert sum(self.block_sizes) == T, \
            "Geometric block sizes must sum to num_tokens ({T}), not {block_sizes}"
        # block_sizes = compute_geometric_block_sizes(
        #     num_tokens=T,
        #     num_blocks=NUM_GEO_BLOCKS,
        #     ratio=GEO_BLOCK_RATIO
        # )

        # hybrid representation
        h_last = h[:, -1, :]          # (B, D)


        h_blocks = torch.stack([
                h[:, start:end, :].mean(dim=1)
                for start, end in self.block_ranges
            ], dim=1)  # (B, num_blocks, D)

        # blocks = []
        # idx = 0

        # for size in self.block_sizes:
        #     block = h[:, idx:idx + size, :].mean(dim=1)   # (B, D)
        #     blocks.append(block)
        #     idx += size

        # # h_geo = torch.cat(blocks, dim=-1)
        # # h_geo shape = (B=256, K*D = 4*192 = 768)

        # # blocks is a Python list of K tensors of shape (B, D)
        # h_blocks = torch.stack(blocks, dim=1)     # (B, K, D)
        # # (B, K=4, D=192)

        # # h_geo = torch.cat(blocks, dim=-1)
        # # (B, 768)


        h_weighted = self.block_weighting(h_blocks)    # (B, D)

        h_final = torch.cat([h_last, h_weighted], dim=-1)   # TODO remove h_geo
        # (B, 960)

        expected_in = self.fc_out[0].in_features   # first Linear in your Sequential
        assert h_final.shape[1] == expected_in, (
            f"fc_out expects {expected_in} features but got {h_final.shape[1]}"
        )

        z = self.fc_out(h_final)                 # (B, H*Q  )
        z = z.view(z.shape[0], self.pred_len, self.num_quantiles) # (B, H, Q)
        return z   # .unsqueeze(-1)



def lr_warmup_cosine(step, warmup_steps, epochs, num_steps):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, (epochs * num_steps - warmup_steps))
    return 0.5 * (1 + np.cos(np.pi * progress))



# Define early stopping class
class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Signal to stop training
        return False




def geometric_block_sizes(num_tokens, num_blocks, ratio):
    """
    Returns a list of block sizes that:
    - follow a geometric progression
    - sum exactly to num_tokens
    - give highest resolution to the most recent block
    """
    weights = [ratio ** i for i in reversed(range(num_blocks))]
    total = sum(weights)

    sizes = [int(round(num_tokens * w / total)) for w in weights]

    # Fix rounding drift so sum == num_tokens
    drift = num_tokens - sum(sizes)
    sizes[-1] += drift   # push correction into most recent block

    return sizes


class BlockWeighting(nn.Module):
    """
    Lightweight learned weighting across K geometric blocks.

    Input:
        h_blocks: (B, K, D)

    Output (by default):
        proj(weighted): (B, D)   # a D-dim vector computed as weighted sum across blocks
    Optional (if return_concat=True):
        (weighted_proj, concat) where concat is the flattened (B, K*D) tensor.
    """
    def __init__(self, num_blocks, model_dim, use_proj=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.model_dim = model_dim
        self.use_proj = use_proj

        # one scalar logit per block -> softmaxed to get weights
        self.logit = nn.Parameter(torch.zeros(num_blocks))

        # Project the weighted (D) vector back to D if requested.
        # Note: this MUST be (model_dim -> model_dim) because forward() returns
        # a weighted SUM across blocks (shape = (B, D)).
        if self.use_proj:
            self.proj = nn.Linear(model_dim, model_dim)
        else:
            self.proj = None

    def forward(self, h_blocks):  #, return_concat=False):
        """
        h_blocks: (B, K, D)
        # return_concat: if True, returns (proj(weighted), concat_flat)
        #                where concat_flat = h_blocks.view(B, K*D)
        """
        B, K, D = h_blocks.shape
        assert K == self.num_blocks, f"Expected {self.num_blocks} blocks, got {K}"

        # softmax weights across blocks -> shape (K,)
        w = torch.softmax(self.logit, dim=0)   # (K,)

        # weighted sum: (B, K, D) * (1, K, 1) -> (B, K, D) -> sum -> (B, D)
        weighted = (h_blocks * w.view(1, K, 1)).sum(dim=1)  # (B, D)

        if self.use_proj:
            out = self.proj(weighted)   # (B, D)
        else:
            out = weighted

        # if return_concat:
        #     concat = h_blocks.reshape(B, K * D)   # (B, K*D)
        #     return out, concat

        return out


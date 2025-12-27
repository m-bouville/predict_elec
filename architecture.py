
from   typing import Tuple, List, Sequence  #, Dict, Optional
# from   collections import defaultdict

import torch
import torch.nn as nn
from   torch.utils.data         import DataLoader  # Dataset
# from   torch.optim.lr_scheduler import ReduceLROnPlateau

from   sklearn.preprocessing   import StandardScaler

import numpy  as np
import pandas as pd


import losses  # utils



# ==============================================================================
# 2. DATASET CLASS FOR DAY-AHEAD FORECASTING
# ==============================================================================

class DayAheadDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for day-ahead market forecasting.

    Each sample represents one forecast made at noon, predicting the next
    48 half-hourly values.
    """

    def __init__(
        self,
        data_subset  :  np.ndarray,
        dates_subset : pd.DatetimeIndex,
        input_length : int,
        pred_length  : int,
        features_in_future:int,
        forecast_hour: int,
        target_index : int
    ):
        """
        Parameters
        ----------
        data_subset : np.ndarray
            Shape (T, F+1): includes target in column 0
        dates_subset : pd.DatetimeIndex
            Timestamps for each row
        input_length : int
            Number of historical half-hours
        pred_length : int
            Number of future half-hours to predict
        forecast_hour : int
            Hour when forecast is made
        target_index : int
            Column index of target variable
        """
        self.data_subset  = data_subset.astype(np.float32)
        self.dates_subset = dates_subset
        self.input_length = input_length
        self.pred_length  = pred_length
        self.features_in_future=int(features_in_future)
        self.forecast_hour= forecast_hour
        self.target_index = target_index

        # Pre-compute valid forecast indices
        self.start_indices_subset= []
        self.forecast_origins    = []

        for idx_steps_subset, date in enumerate(dates_subset):
            if (date.hour   == forecast_hour and
                date.minute == 0 and
                idx_steps_subset >= input_length and
                idx_steps_subset + pred_length < len(data_subset)):

                self.start_indices_subset.append(idx_steps_subset)
                self.forecast_origins    .append(date)

        # print("forecast_origins:", type(self.forecast_origins[0]))

    def __len__(self):
        return len(self.start_indices_subset)

    def __getitem__(self, idx_days_subset):
        """
        Returns
        -------
        X : torch.Tensor
            Input features, shape (input_length, F)
        y : torch.Tensor
            Target values,  shape (pred_length,  1)
        idx_steps : List[int]
            Acceptable indices in self.start_indices_subset
        forecast_origin : List[int]
            When this forecast was made (converted to int)
        """
        idx_subset = self.start_indices_subset[idx_days_subset]

        # Input: all features (=> excluding consumption) from past
        X = self.data_subset[idx_subset - self.input_length :
                             idx_subset + self.features_in_future*self.pred_length]  # (L+H, F)
        X = np.delete(X, self.target_index, axis=1)

        # Target: only consumption, future values (excluding present datetime)
        y = self.data_subset[idx_subset+1 : idx_subset+1 + self.pred_length,
                             self.target_index]

        # origin = self.forecast_origins[idx_days]
        # print(f"Type: {type(origin)}, Value: {origin}")
        # origin_int = int(origin.timestamp())
        # print(f"After conversion: {type(origin_int)}, Value: {origin_int}")

        return (
            torch.tensor(X),
            torch.tensor(y).unsqueeze(-1),  # (pred_length, 1)
            idx_subset,
            int(self.forecast_origins[idx_days_subset].timestamp())
                # pd.Timestamp != batchable
        )


# ============================================================
# 3. make_X_and_y
# ============================================================

def make_X_and_y(series, dates,
                 train_split, n_valid,
                 feature_cols, target_col,
                 input_length:int, pred_length:int, features_in_future:bool,
                 batch_size:int, forecast_hour:int=12,
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
    # print("scaler_y.mean_.shape: ", scaler_y.mean_.shape)
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
    assert pred_length == 48,\
        f"Day-ahead forecasting requires pred_length == 48, not {pred_length}"

    def build_day_ahead(data_subset, date_slice):
        return DayAheadDataset(
            data_subset  = data_subset,
            dates_subset = date_slice,
            input_length = input_length,
            pred_length  = pred_length,
            features_in_future=features_in_future,
            forecast_hour= forecast_hour,
            target_index = target_idx
        ) # X_list, y_list, origin_list, target_dates_list

    train_dataset_scaled = build_day_ahead(train_scaled, train_dates)
    valid_dataset_scaled = build_day_ahead(valid_scaled, valid_dates)
    test_dataset_scaled  = build_day_ahead(test_scaled,  test_dates )

    # print("len(X_dataset_scaled[0]) [days]", len(train_dataset_scaled[0]),
    #       len(valid_dataset_scaled[0]), len(test_dataset_scaled[0]))

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

    # print("len(X_loader):", len(train_loader), len(valid_loader), len(test_loader))

    # print(); print("valid_loader:")
    # for batch_idx, (x_scaled, y_scaled, origin_unix) in enumerate(valid_loader):
        # # Convert back to timestamps if needed
        # origins = [pd.Timestamp(t, unit='s') for t in origin_unix.tolist()]
        # print(batch_idx, x_scaled, y_scaled, origins[0], "to", origins[-1])

    return (
        [train_loader,valid_loader,test_loader], \
        [train_dates, valid_dates, test_dates ],\
         scaler_y, [X_GW, y_GW],
         [X_train_GW, y_train_GW, train_dataset_scaled],
         [X_GW[:train_split][-n_valid:], y_GW[:train_split][-n_valid:], valid_dataset_scaled],
         [X_test_GW, y_test_GW, test_dataset_scaled],
         test_scaled[:, feature_idx]
    )






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
                 features_in_future: bool,
                 dropout:float, ffn_mult:int, num_quantiles:int,
                 num_geo_blocks, geo_block_ratio):
        super().__init__()

        self.num_features   = num_features
        self.dim_model      = dim_model
        self.pred_len       = pred_len
        self.num_quantiles  = num_quantiles

        self.input_len      = input_len
        self.features_in_future=int(features_in_future)
        self.patch_len      = patch_len
        self.stride         = stride
        self.num_patches    = ((input_len+self.features_in_future*pred_len) \
                               - patch_len) // stride + 1

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

        self.block_sizes    = block_sizes(
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



    def forward(self, X, *args, **kwargs):
        B, L_plus_H, F = X.shape      # x: (B, L+H, F)
        L = L_plus_H - self.features_in_future*self.pred_len  # (L+H) - H
        assert L == self.input_len,    (L, self.input_len)
        assert F == self.num_features, (F, self.num_features)

        # guaranteeing: last patch ends exactly at t = L
        if self.pad_len > 0:
            X = torch.nn.functional.pad(
                X,
                pad  = (0, 0, 0, self.pad_len),  # pad time dimension on the right
                mode = "constant",
                value= 0.
        )

        # 1. Patch embedding
        h = self.patch_embed(X)                     # (B, num_patches, model_dim)

        # 2. Transformer encoder
        for layer in self.layers:
            h, _ = layer(h, return_attn=False)      # (B, num_patches, model_dim)

        # Geometric block pooling
        B, T, D = h.shape                   # (batch_size, num_tokens, model_dim)
        assert T == self.num_patches, (T, self.num_patches)
        assert D == self.dim_model,   (D, self.dim_model)

        assert sum(self.block_sizes) == T, \
            "Geometric block sizes must sum to num_tokens ({T}), not {block_sizes}"

        # hybrid representation
        h_last = h[:, -1, :]          # (B, D)

        h_blocks = torch.stack([
                h[:, start:end, :].mean(dim=1)
                for start, end in self.block_ranges
            ], dim=1)  # (B, num_blocks, D)

        # h_final = h_blocks.reshape(B, len(self.block_sizes) * D)   # (B, K·D)

        h_weighted = self.block_weighting(h_blocks)    # (B, D)

        # h_final = h_weighted                              # (B, 2D)
        h_final = torch.cat([h_last, h_weighted], dim=-1)   # (B,  D)

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


def constant_block_sizes(num_tokens: int, num_blocks: int) -> List[float]:
    # print("constant", num_tokens, num_blocks)
    sizes = [int(round(num_tokens / num_blocks))] * num_blocks

    # Fix rounding drift so sum == num_tokens
    drift: int = num_tokens - sum(sizes)
    for i in range(abs(drift)):  # push correction of +/- 1 into enough blocks
        sizes[-(i+1)] += int(np.sign(drift))
    # print(sizes, sum(sizes), num_tokens)

    return sizes


def geometric_block_sizes(num_tokens: int, num_blocks: int, ratio: float) -> List[float]:
    """
    Returns a list of block sizes that:
    - follow a geometric progression
    - sum exactly to num_tokens
    - give highest resolution to the most recent block
    """
    # print("geometric", num_tokens, num_blocks, ratio)
    weights = [max(ratio ** i, 1./num_tokens) for i in reversed(range(num_blocks))]
    # print([round(w, 3) for w in weights])
    total = sum(weights)
    # print([round(num_tokens * w / total, 3) for w in weights])
    sizes = [max(int(round(num_tokens * w / total)), 1) for w in weights]

    # Fix rounding drift so sum == num_tokens
    drift = num_tokens - sum(sizes)
    sizes[-1] += drift   # push correction into largest block

    return sizes


def block_sizes(num_tokens: int, num_blocks: int, ratio: float) -> List[float]:
    if np.isclose(ratio, 1):
        return constant_block_sizes(num_tokens, num_blocks)
    return geometric_block_sizes(num_tokens, num_blocks, ratio)


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
        self.model_dim  = model_dim
        self.use_proj   = use_proj

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





# ----------------------------------------------------------------------
# calculate losses for:
#    - training (torch)
#    - validation and testing (numpy)
# ----------------------------------------------------------------------


def subset_evolution_torch(
        model         : nn.Module,
        amp_scaler,
        optimizer,
        scheduler,
        subset_loader : DataLoader,
        subset_dates  : Sequence,
        # scaler_y      : StandardScaler,
        # constants
        device        : torch.device,
        # input_length  : int,
        # pred_length   : int,
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
        lambda_median : float,
        smoothing_cross:float
    ) -> Tuple[float, float]:
    """
    Returns:
        nn_loss_scaled   : float
        meta_loss_scaled : float
    """

    model.train()

    loss_quantile_scaled     = 0.

    # for batch_idx, (x_scaled, y_scaled, origins) in enumerate(train_loader):
    for (X_scaled, y_scaled, _, _) in subset_loader:
        X_scaled_dev = X_scaled.to(device)   # (B, L, F)
        y_scaled_dev = y_scaled.to(device)   # (B, H, 1)

        # origins = [pd.Timestamp(t, unit='s') for t in origin_unix.tolist()]
        # print(batch_idx, x_scaled, y_scaled, origins[0], "to", origins[-1])

        # optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type): # mixed precision
            pred_scaled_dev = model(X_scaled_dev)
            loss_quantile_scaled_dev = losses.quantile_torch(
                pred_scaled_dev, y_scaled_dev, quantiles,
                lambda_cross, lambda_coverage, lambda_deriv,
                lambda_median, smoothing_cross)

        amp_scaler.scale(loss_quantile_scaled_dev).backward()       # full precision
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        loss_quantile_scaled += loss_quantile_scaled_dev.item()

    scheduler.step()
    loss_quantile_scaled     /= len(subset_loader)

    return loss_quantile_scaled


@torch.no_grad()
def subset_evolution_numpy(
        model         : nn.Module,
        subset_loader : DataLoader,
        subset_dates  : Sequence,
        # scaler_y      : StandardScaler,
        # constants
        device        : torch.device,
        # input_length  : int,
        # pred_length   : int,
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float,
        lambda_median : float,
        smoothing_cross:float
    ) -> Tuple[float, float]:
    """
    Returns:
        nn_loss_scaled   : float
        meta_loss_scaled : float
    """
    model.eval()

    # Q = len(quantiles)
    # T = len(dates)

    loss_quantile_scaled     = 0.

    # main loop
    for (X_scaled, y_scaled, _, _) in subset_loader:
        X_scaled_dev = X_scaled.to(device)
        y_scaled_cpu = y_scaled[:, :, 0].cpu().numpy()  # (B, H)

        # NN forward
        pred_scaled_cpu = model(X_scaled_dev).cpu().numpy() # (B, H, Q)

        # loss
        loss_quantile_scaled_cpu = losses.quantile_numpy(
            pred_scaled_cpu, y_scaled_cpu, quantiles,
            lambda_cross, lambda_coverage, lambda_deriv,
            lambda_median, smoothing_cross)
        loss_quantile_scaled += loss_quantile_scaled_cpu

        # print(f"X_scaled.shape:        {X_scaled.shape} -- theory: (B, L, F)")
        # print(f"y_scaled.shape:        {y_scaled.shape}   -- theory: (B, H, 1)")
        # print(f"pred_scaled_dev.shape: {pred_scaled_dev.shape}-- theory: (B, H, Q)")

    loss_quantile_scaled     /= len(subset_loader)

    return loss_quantile_scaled


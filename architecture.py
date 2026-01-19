
from   typing import Tuple, List, Dict  # Sequence  #, Optional
# from   collections import defaultdict

import torch
import torch.nn as nn
from   torch.utils.data         import DataLoader  # Dataset
# from   torch.optim.lr_scheduler import ReduceLROnPlateau

from   sklearn.preprocessing   import StandardScaler

import numpy  as np
import pandas as pd


import losses, containers  # utils
from   constants   import Split



# ==============================================================================
# 2. DATASET CLASS FOR DAY-AHEAD FORECASTING
# ==============================================================================

class DayAheadDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for day-ahead market forecasting.

    Each sample represents one forecast made at noon, predicting the next-day
    48 half-hourly values.
    """

    def __init__(
        self,
        data_subset  : np.ndarray,
        dates_subset : pd.DatetimeIndex,
        temperatures_subset: np.ndarray,
        input_length : int,
        pred_length  : int,
        features_in_future:int,
        forecast_hour: int,

        index_y_nation : int,
        indices_Y_regions: List[int]
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
        index_y_nation : int
            Column index of target variable
        """
        self.data_subset    = data_subset.astype(np.float32)
        self.dates_subset   = dates_subset
        self.temperatures_subset=temperatures_subset
        self.input_length   = input_length
        self.pred_length    = pred_length
        self.features_in_future=int(features_in_future)  # 0 or 1
        self.future_length  = self.features_in_future * self.pred_length
        self.forecast_hour  = forecast_hour

        self.index_y_nation = index_y_nation
        self.indices_Y_regions= indices_Y_regions

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
        subset_loader

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

        _range_future = range(idx_subset, idx_subset + self.pred_length)

        # Input: all features (=> excluding consumption) from past
        X = self.data_subset[idx_subset - self.input_length :
                             idx_subset + self.future_length]  # (L+H, F)
        X = np.delete(X, self.index_y_nation + self.indices_Y_regions, axis=1)
            # now a true X (w/o y)

        # Target: only consumption, future values (excluding present datetime)
        # print(self.data_subset.shape)
        # print("indices:", self.index_y_nation, self.indices_Y_regions)
        y_nation = self.data_subset[_range_future, self.index_y_nation]
        Y_regions= self.data_subset[_range_future][:, self.indices_Y_regions] #(L+H, R)

        # Temperatures
        T = self.temperatures_subset[_range_future]

        # origin = self.forecast_origins[idx_days]
        # print(f"Type: {type(origin)}, Value: {origin}")
        # origin_int = int(origin.timestamp())
        # print(f"After conversion: {type(origin_int)}, Value: {origin_int}")

        return (
            torch.tensor(X),
            torch.tensor(Y_regions),
            torch.tensor(y_nation).unsqueeze(-1),  # (pred_length, 1)
            torch.tensor(T)       .unsqueeze(-1),  # (pred_length, 1),
            idx_subset,
            int(self.forecast_origins[idx_days_subset].timestamp())
                # pd.Timestamp != batchable
        )


# ============================================================
# 3. make_X_and_y
# ============================================================

def make_X_and_y(array           : np.ndarray,
                 dates           : pd.DatetimeIndex,
                 temperatures    : np.ndarray,
                 train_split     : int,
                 n_valid         : int,
                 names_cols      : Dict[str, List[str]],
                 use_ML_features : bool,

                 weights_regions : Dict[str, float],
                 minutes_per_step: int,
                 input_length    : int,
                 pred_length     : int,
                 features_in_future:bool,
                 batch_size      : int,
                 forecast_hour   : int = 12,
                 verbose         : int =  0):

    assert array.shape[0] == len(dates), \
        f"array.shape ({array.shape}) != len(dates) ({len(dates)})"

    # print(f"Forecast hour: {forecast_hour:2n}:00")
    # print(f"input_length:{input_length:4n} [half-hours]")
    # print(f"pred_length: {pred_length :4n} [half-hours]")
    # print(f"batch_size:  {batch_size  :4n}")
    # print()

    #indices
    idx_all   = pd.RangeIndex(len(dates))
    idx_train = idx_all[:train_split]
    idx_test  = idx_all[train_split:]

    idx_valid = idx_train[-n_valid:]
    idx_train = idx_train[:-n_valid]


    # dates
    train_dates = dates[idx_train]
    valid_dates = dates[idx_valid]
    test_dates  = dates[idx_test ]

    # temperatures
    train_Tavg_degC = temperatures[idx_train]
    valid_Tavg_degC = temperatures[idx_valid]
    test_Tavg_degC  = temperatures[idx_test ]


    # Map column names -> column indices in train_data
    # print(names_cols)
    # print({k: len(w) for (k, w) in names_cols.items()})
    all_cols  = names_cols['y_nation'] + names_cols['Y_regions'] + \
                names_cols['features'] + names_cols['ML_preds' ]
    col_to_idx= {col: i for i, col in enumerate(all_cols)}

    indices = dict()
    for (_name, _list) in names_cols.items():
        indices[_name] = [col_to_idx[_col] for _col in _list]
        # print(_name, indices[_name])
    if use_ML_features:
        names_cols['features'].extend(names_cols['ML_preds'])
        indices   ['features'].extend(indices   ['ML_preds'])
        # print('features', indices['features'])

    test_data       = array[idx_test]
    X_test_GW       = test_data[:, indices['features' ]]
    y_nation_test_GW= test_data[:, indices['y_nation' ]]
    Y_regions_test_GW=test_data[:, indices['Y_regions']]
    preds_ML_test   = test_data[:, indices['ML_preds' ]]


    # 1. Extract X and y using names
    X_GW         = array[:, indices['features' ]]
    y_nation_GW  = array[:, indices['y_nation' ]].squeeze()
    Y_regions_GW = array[:, indices['Y_regions']]
    preds_ML     = array[:, indices['ML_preds' ]]
    # print("y_nation_GW.shape:", y_nation_GW.shape)

    # 2. Fit two different scalers (on training set)
    scaler_X        = StandardScaler()
    scaler_y_nation = StandardScaler()
    scaler_Y_regions= StandardScaler()

    X_train_GW        = X_GW        [idx_train]
    y_nation_train_GW = y_nation_GW [idx_train]
    Y_regions_train_GW= Y_regions_GW[idx_train]
    preds_ML_train    = preds_ML    [idx_train]

    scaler_X        .fit(X_train_GW)
    scaler_y_nation .fit(y_nation_train_GW.reshape(-1, 1))
    scaler_Y_regions.fit(Y_regions_train_GW)

    X_valid_GW        = X_GW        [idx_valid]  # for return only
    y_nation_valid_GW = y_nation_GW [idx_valid]  # for return only
    Y_regions_valid_GW= Y_regions_GW[idx_valid]  # for return only
    preds_ML_valid    = preds_ML    [idx_valid]

    # 3. Transform X and y separately
    X_scaled        = scaler_X        .transform(X_GW)
    y_nation_scaled = scaler_y_nation .transform(y_nation_GW.reshape(-1, 1)).ravel()
    Y_regions_scaled= scaler_Y_regions.transform(Y_regions_GW)

    df_scaled = np.column_stack([y_nation_scaled, Y_regions_scaled, X_scaled])


    # 5. SAFETY CHECKS
    # print("scaler_y.mean_.shape: ", scaler_y.mean_.shape)
    # print("scaler_y.scale_.shape:", scaler_y.scale_.shape)

    # assert scaler_y.mean_.shape[0]= 1,"scaler_y must be fitted on ONE target only"
    assert df_scaled.shape[1] == 1 + len(indices['Y_regions']) \
                    + len(indices['features']), "scaled feature count mismatch"


    # 4. Rebuild scaled arrays for your pipeline
    #    (target in column 0, features after)
    train_scaled = df_scaled[:train_split]
    test_scaled  = df_scaled[idx_test]

     # validation
    valid_scaled = train_scaled[-n_valid:]
    train_scaled = train_scaled[:-n_valid]

    # print("len(X_scaled)",len(train_scaled),len(valid_scaled),len(test_scaled))
    # print("len(X_dates) ",len(train_dates), len(valid_dates), len(test_dates))



    def build_day_ahead(data_subset, date_slice, temperatures_subset):
        return DayAheadDataset(
            data_subset      = data_subset,
            dates_subset     = date_slice,
            temperatures_subset=temperatures_subset,
            input_length     = input_length,
            pred_length      = pred_length,
            features_in_future=features_in_future,
            forecast_hour    = forecast_hour,
            index_y_nation   = indices['y_nation' ],
            indices_Y_regions= indices['Y_regions']
        ) # X_list, y_list, origin_list, target_dates_list

    train_dataset_scaled= build_day_ahead(train_scaled,train_dates,train_Tavg_degC)
    valid_dataset_scaled= build_day_ahead(valid_scaled,valid_dates,valid_Tavg_degC)
    test_dataset_scaled = build_day_ahead(test_scaled, test_dates, test_Tavg_degC)
    # complete_dataset_scaled=build_day_ahead(df_scaled, dates, temperatures)

    # print("len(X_dataset_scaled[0]) [days]", len(train_dataset_scaled[0]),
    #       len(valid_dataset_scaled[0]), len(test_dataset_scaled[0]))


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

    names_models = [e.split('_')[1] for e in names_cols['ML_preds']]
    train = containers.DataSplit(Split.train, "training",
            idx_train, X_train_GW, y_nation_train_GW, Y_regions_train_GW,
            train_dates, train_Tavg_degC, names_cols['features'],
            dict_preds_ML=pd.DataFrame(preds_ML_train, index=train_dates,
                                       columns=names_models).to_dict(),
            loader=train_loader, dataset_scaled=train_dataset_scaled)
    valid = containers.DataSplit(Split.valid, "validation",
            idx_valid, X_valid_GW, y_nation_valid_GW, Y_regions_valid_GW,
            valid_dates, valid_Tavg_degC, names_cols['features'],
            dict_preds_ML=pd.DataFrame(preds_ML_valid, index=valid_dates,
                                       columns=names_models).to_dict(),
            loader=valid_loader, dataset_scaled=valid_dataset_scaled)
    test  = containers.DataSplit(Split.test,  "testing",
            idx_test,  X_test_GW,  y_nation_test_GW,  Y_regions_test_GW,
            test_dates,  test_Tavg_degC, names_cols['features'],
            dict_preds_ML=pd.DataFrame(preds_ML_test, index=test_dates,
                                       columns=names_models).to_dict(),
            loader=test_loader,  dataset_scaled=test_dataset_scaled)
    complete=containers.DataSplit(Split.complete,  "all data",
            idx_all,  X_GW,   y_nation_GW,  Y_regions_GW,
            dates,  temperatures, names_cols['features'],
            dict_preds_ML=pd.DataFrame(preds_ML, index=dates,
                                       columns=names_models).to_dict(),
            loader=None,  dataset_scaled=None)

    data = containers.DatasetBundle(
            train, valid, test, complete=complete,
            scaler_y_nation=scaler_y_nation,
            X=X_GW, y_nation=y_nation_GW, Y_regions=Y_regions_GW,
            minutes_per_step=minutes_per_step,
            num_steps_per_day = int(round(24*60/minutes_per_step)),
            num_features = len(names_cols['features']), weights_regions=weights_regions,
            num_time_steps=array.shape[0]
        )

    return data, test_scaled[:, indices['features']]





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
# 5. TRANSFORMER MODEL
# ============================================================


class PatchEmbedding(nn.Module):
    def __init__(self, patch_length, stride, in_channels, dim_model):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels = in_channels,
            out_channels= dim_model,
            kernel_size = patch_length,
            stride      = stride
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.transpose(1, 2)        # (B, C, L)
        x = self.proj(x)             # (B, D, T)
        return x.transpose(1, 2)     # (B, T, D)



class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features:int, dim_model:int, num_heads:int, num_layers:int,
                 input_length:int, patch_length:int, stride:int, pred_length:int,
                 features_in_future: bool,
                 dropout:float, ffn_mult:int, num_quantiles:int, num_regions:int,
                 num_geo_blocks, geo_block_ratio):
        super().__init__()

        self.num_features   = num_features
        self.dim_model      = dim_model
        self.pred_length    = pred_length
        self.num_quantiles  = num_quantiles
        self.num_regions    = num_regions

        self.input_length   = input_length
        self.patch_length   = patch_length
        self.stride         = stride

        if input_length is not None:
            self.features_in_future=int(features_in_future)
            self.num_patches  = ((input_length+self.features_in_future*pred_length) \
                                 - patch_length) // stride + 1

            remainder       = (input_length - patch_length) % stride
            self.pad_length = (stride - remainder) % stride
            # total_covered= ((input_length-patch_length)//stride) * stride + patch_length
            # self.pad_length  = input_length - total_covered

            self.patch_embed= PatchEmbedding(patch_length,stride,num_features,dim_model)
            self.layers = nn.ModuleList([
                TransformerEncoderLayerWithAttn(dim_model, num_heads, dropout, ffn_mult)
                for _ in range(num_layers)
            ])

        # blocks
        self.num_geo_blocks = num_geo_blocks
        self.geo_block_ratio= geo_block_ratio

        if num_geo_blocks is not None:
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
                nn.Linear(dim_model, pred_length * (num_quantiles + num_regions))
            )
            # self.fc_out = nn.Linear(dim_model, pred_len)



    def forward(self, X, *args, **kwargs):
        B, L_plus_H, F = X.shape      # x: (B, L+H, F)
        L = L_plus_H - self.features_in_future*self.pred_length  # (L+H) - H
        assert L == self.input_length, (L, self.input_length)
        assert F == self.num_features, (F, self.num_features)

        # guaranteeing: last patch ends exactly at t = L
        if self.pad_length > 0:
            # _old_shape = X.shape
            X = torch.nn.functional.pad(
                X,
                pad  = (0, 0, 0, self.pad_length),  # pad time dimension on the right
                mode = "constant",
                value= 0.
                )
            # print(f"pad_length = {self.pad_length}: {_old_shape} -> {X.shape}")

        # 1. Patch embedding
        h = self.patch_embed(X)                     # (B, num_patches, model_dim)
        B, T, D = h.shape
        assert T == self.num_patches, (T, self.num_patches)
        assert D == self.dim_model,   (D, self.dim_model)

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

        h_final = torch.cat([h_last, h_weighted], dim=-1)   # (B, 2D)

        expected_in = self.fc_out[0].in_features   # first Linear in your Sequential
        assert h_final.shape[1] == expected_in, (
            f"fc_out expects {expected_in} features but got {h_final.shape[1]}"
        )

        z = self.fc_out(h_final)                 # (B, H*Q  )
        z = z.view(z.shape[0], self.pred_length,
                   self.num_quantiles + self.num_regions) # (B, H, Q+R)
        return (z[:, :, :self.num_quantiles], z[:, :, self.num_quantiles:])



def lr_warmup_cosine(step, warmup_steps, epochs, num_steps) -> float:
    # No warmup: pure cosine decay
    if warmup_steps is None or warmup_steps == 0:
        progress = step / max(1, epochs * num_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    # Warmup phase
    if step < warmup_steps:
        return step / warmup_steps

    # Cosine decay after warmup
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
        model_NN, #: containers.NeuralNet
        subset_loader: DataLoader
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
    """
    Returns:
        loss_quantile_scaled_h: torch.tensor
            shape (V, ), its average is used for gradients
        dict_losses_h: Dict[str, torch.tensor]
            components of the loss, each of shape (V, ): used for diagnostics
    """

    model       = model_NN.model
    amp_scaler  = model_NN.amp_scaler
    device      = model_NN.device
    valid_length= model_NN.valid_length


    model.train()

    loss_quantile_scaled_h = torch.zeros(valid_length, device=device)
    dict_losses_h = {#'quantile_with_crossing':torch.zeros(valid_length,device=device),
                     'pinball':   torch.zeros(valid_length, device=device),
                     'coverage':  torch.zeros(valid_length, device=device),
                     'crossing':  torch.zeros(valid_length, device=device),
                     'derivative':torch.zeros(valid_length, device=device),
                     'median':    torch.zeros(valid_length, device=device)}

    # for batch_idx, (x_scaled, y_scaled, origins) in enumerate(train_loader):
    for (X_scaled, Y_regions_scaled, y_nation_scaled,
         T_degC, _, origin_unix) in subset_loader:
        X_scaled_dev        = X_scaled        .to(device)   # (B, L, F)
        Y_regions_scaled_dev= Y_regions_scaled.to(device)   # (B, H, R)
        y_nation_scaled_dev = y_nation_scaled .to(device)   # (B, H, 1)
        T_degC_dev          = T_degC          .to(device)   # (B, H, 1)

        # print("X_scaled_dev", X_scaled_dev.shape)

        # origins_dev = [pd.Timestamp(t, unit='s')
        #                for t in origin_unix.tolist()].to(device)
        # print(batch_idx, x_scaled, y_scaled, origins[0], "to", origins[-1])

        # model_NN.optimizer).zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type): # mixed precision
            (pred_nation_scaled, pred_regions_scaled) = model(X_scaled_dev)
            pred_nation_scaled_dev = pred_nation_scaled .to(device) # (B, H, Q)
            pred_regions_scaled_dev= pred_regions_scaled.to(device) # (B, H, R)

            # assert y_scaled_dev.shape[1] == pred_scaled_dev.shape[1] == pred_length

            # validation and plotting will be over VALID_LENGTH, not PRED_LENGTH
            loss_quantile_scaled_h_batch, dict_losses_h_batch = losses.quantile_torch(
                pred_nation_scaled_dev[:, -valid_length:, :],
                y_nation_scaled_dev   [:, -valid_length:, 0], model_NN.quantiles,
                **{_name: getattr(model_NN, _name) for _name in ['lambda_cross', \
                     'lambda_coverage','lambda_deriv','lambda_median','smoothing_cross',
                     'saturation_cold_degC', 'threshold_cold_degC', 'lambda_cold']},
                Tavg_current=T_degC_dev[:, -valid_length:, 0])

            if model_NN.lambda_regions > 0:
                loss_region_scaled_h_batch = losses.regions_torch(
                        pred_regions_scaled_dev[:, -valid_length:, :],     # (B, V, R)
                        Y_regions_scaled_dev   [:, -valid_length:, :],     # (B, V, R)
                        model_NN.lambda_regions, model_NN.lambda_regions_sum
                    )
                loss_scaled_h_batch= loss_quantile_scaled_h_batch+loss_region_scaled_h_batch
            else:
                loss_scaled_h_batch= loss_quantile_scaled_h_batch

        amp_scaler.scale(loss_scaled_h_batch.mean()).backward()# full precision
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        amp_scaler.step(model_NN.optimizer)
        amp_scaler.update()

        loss_quantile_scaled_h += loss_scaled_h_batch
        dict_losses_h = {key: dict_losses_h[key] + dict_losses_h_batch[key]
                         for key in dict_losses_h}

        # loss_quantile_scaled += loss_quantile_scaled_dev.item()

    model_NN.scheduler.step()

    loss_quantile_scaled_h      /= len(subset_loader)
    dict_losses_h = {key: value /  len(subset_loader)
                     for (key, value) in dict_losses_h.items()}

    return loss_quantile_scaled_h, dict_losses_h


@torch.no_grad()
def subset_evolution_numpy(
        model_NN, #: containers.NeuralNet
        subset_loader: DataLoader
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
        loss_quantile_scaled_h: np.ndarray
            shape (V, ), its average is used for gradients
        dict_losses_h: Dict[str, np.ndarray]
            components of the loss, each of shape (V, ): used for diagnostics
    """

    model       = model_NN.model
    device      = model_NN.device
    valid_length= model_NN.valid_length


    model.eval()

    # Q = len(quantiles)
    # T = len(dates)

    loss_quantile_scaled_h = np.zeros(valid_length)
    dict_losses_h = {#'quantile_with_crossing': np.zeros(valid_length),
            'pinball':   np.zeros(valid_length),
            'coverage':  np.zeros(valid_length), 'crossing':np.zeros(valid_length),
            'derivative':np.zeros(valid_length), 'median':  np.zeros(valid_length)}
    # print("initial:\n", loss_quantile_scaled_h.round(2))
    # print("initial:\n", pd.DataFrame(dict_losses_h).round(2).head())

    # main loop
    for (X_scaled, Y_regions_scaled, y_nation_scaled,
         T_degC, _, origin_unix) in subset_loader:
        X_scaled_dev             = X_scaled.to(device)
        Y_regions_scaled_cpu     = Y_regions_scaled[:, :, :].cpu().numpy() # (B, H, R)
        y_median_nation_scaled_cpu=y_nation_scaled [:, :, 0].cpu().numpy() # (B, H)
        T_degC_cpu               = T_degC          [:, :, 0].cpu().numpy() # (B, H)

        # origins_cpu = [pd.Timestamp(t, unit='s') for t in origin_unix.tolist()].cpu()

        # NN forward
        (pred_nation_scaled, pred_regions_scaled) = model(X_scaled_dev)
        pred_nation_scaled_cpu = pred_nation_scaled .cpu().numpy() # (B, H, Q)
        pred_regions_scaled_cpu= pred_regions_scaled.cpu().numpy() # (B, H, R)

        # assert y_scaled_cpu.shape[1] == pred_scaled_cpu.shape[1] == pred_length

        # validation and plotting will be over VALID_LENGTH, not PRED_LENGTH
        # loss
        loss_quantile_scaled_h_batch, dict_losses_h_batch = losses.quantile_numpy(
            pred_nation_scaled_cpu    [:, -valid_length:],
            y_median_nation_scaled_cpu[:, -valid_length:],
            **{_name: getattr(model_NN, _name) for _name in ['quantiles', \
                 'lambda_cross', 'lambda_coverage', 'lambda_deriv', \
                 'lambda_median', 'smoothing_cross',
                 'saturation_cold_degC', 'threshold_cold_degC', 'lambda_cold']},
                Tavg_current=T_degC_cpu[:, -valid_length:])

        if model_NN.lambda_regions > 0:
            loss_region_scaled_h_batch = losses.regions_numpy(
                    pred_regions_scaled_cpu[:, -valid_length:, :],     # (B, V, R)
                    Y_regions_scaled_cpu   [:, -valid_length:, :],     # (B, V, R)
                    model_NN.lambda_regions, model_NN.lambda_regions_sum
                )
            loss_scaled_h_batch= loss_quantile_scaled_h_batch+loss_region_scaled_h_batch
        else:
            loss_scaled_h_batch= loss_quantile_scaled_h_batch


        # print("batch:\n", loss_scaled_h_batch.round(2))

        # print("previous total:\n", pd.DataFrame(dict_losses_h).round(2).head())
        # print("batch:\n", pd.DataFrame(dict_losses_h_batch).round(2).head())

        loss_quantile_scaled_h += loss_scaled_h_batch
        dict_losses_h = {key: dict_losses_h[key] + dict_losses_h_batch[key]
                         for key in dict_losses_h}

        # print("running total:\n", loss_quantile_scaled_h.round(2))
        # print("running total:\n", pd.DataFrame(dict_losses_h).round(2).head())

    loss_quantile_scaled_h /= len(subset_loader)
    dict_losses_h = {key: value / len(subset_loader)
                     for (key, value) in dict_losses_h.items()}

    # print("after norm:\n", pd.DataFrame(dict_losses_h).round(2).head())

    return loss_quantile_scaled_h, dict_losses_h


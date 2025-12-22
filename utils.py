import os

import json
import hashlib
import pickle

from   typing import Sequence, List, Tuple, Dict  #, Optional
# from   collections import defaultdict

import torch
import torch.nn as nn
from   torch.utils.data         import DataLoader  # Dataset

import numpy  as np
import pandas as pd

from   sklearn.linear_model    import Ridge
from   sklearn.ensemble        import RandomForestRegressor
from   sklearn.preprocessing   import StandardScaler
# from   sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt


import losses, IO  # architecture



# ----------------------------------------------------------------------
# Create dataframes and vectors
# ----------------------------------------------------------------------

def df_features_calendar(dates: pd.DatetimeIndex,
                verbose: int = 0) -> pd.DataFrame:

    df = pd.DataFrame(index=dates)
    assert isinstance(df.index, pd.DatetimeIndex)

    # periods
    df['hour_norm'] = (df.index.hour + df.index.minute/60) / 24
    df['dow_norm']  = (df.index.dayofweek + df['hour_norm']) / 7
    df['doy_norm']  =  df.index.dayofyear / 365

    # sine waves
    for _hours in [12, 24]:  # several periods per day
        df['sin_'+str(_hours)+'h'] = np.sin(24/_hours * 2*np.pi * df['hour_norm'])

    df['sin_1wk']  = np.sin(2*np.pi*df['dow_norm'])
    # df['cos_1wk']  = np.cos(2*np.pi*df['dow_norm'])
    # df['sin_12mo']  = np.sin(2*np.pi*df['doy_norm'])
    df['cos_12mo'] = np.cos(2*np.pi*df['doy_norm'])
    # df['cos_6mo']  = np.cos(4*np.pi*df['doy_norm'])  # 2 periods per year

    # remove temporary variables
    df.drop(columns=['hour_norm', 'dow_norm', 'doy_norm'], inplace=True)

    df['is_Friday'  ] = (df.index.dayofweek == 4).astype(np.int16)   # (0: Monday)
    df['is_Saturday'] = (df.index.dayofweek == 5).astype(np.int16)
    df['is_Sunday'  ] = (df.index.dayofweek == 6).astype(np.int16)

    if verbose >= 2:
        print(df.head().to_string())

    return df


def df_features_consumption(consumption: pd.Series,
                verbose: int = 0) -> pd.DataFrame:

    df = consumption.to_frame('consumption_GW')

    LAG = 1   # Base lag (30 min), lest we implicitly leak future information
    _shifted_conso = df['consumption_GW'].shift(LAG)

    # diff
    # df["consumption_diff_30min_GW"]=_shifted_conso.diff( 1)
    for _hours in [1, 3, 6, 12, 24]:  # n hours (2n half-hours)
        df["consumption_diff_"+str(_hours)+"h_GW"] = \
            _shifted_conso - _shifted_conso.shift(freq=f"{_hours}h")

    # df["Tavg_diff_24h_degC"] = df['Tavg_degC'] - df['Tavg_degC'].shift(freq="1D")

    # moving average
    df["consumption_SMA_6h_GW" ] = _shifted_conso.rolling(
         "6h", min_periods= 5*2).mean()
    df["consumption_SMA_24h_GW"] = _shifted_conso.rolling(
        "24h", min_periods=22*2).mean()

    if verbose >= 2:
        print(df.tail().to_string())  # head() would be made of NaNs

    return df.drop(columns=['consumption_GW'])


def df_features(dict_fnames: Dict[str, str], output_fname: str,
                verbose: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, dates_df = IO.load_data(dict_fnames, output_fname)
    assert isinstance(df.index, pd.DatetimeIndex)

    # Fourier-like sine waves, weekends
    df_calendar    = df_features_calendar(df.index, verbose)

    # # moving averages and offsets based on consumption [/!\ risk of leak]
    # df_consumption = df_features_consumption(df['consumption_GW'], verbose)
    df_consumption = pd.DataFrame()


    # school holidays (one column per holiday type)
    df_holiday, dates_df.loc["school_holidays"] = \
        IO.make_school_holidays_indicator(df.index, verbose)
    first_available = pd.Timestamp("2014-09-15", tz="UTC")
        # upper bound for return to class


    # remove columns that are not features
    df.drop(columns=['year', 'month', 'timeofday', 'dateofyear'], inplace=True)

    # merge types of features
    df = pd.concat([df.loc[df.index >= first_available], df_holiday,
                    df_calendar, df_consumption], axis=1)


    dates_df.loc["df"]= [df.index.min(), df.index.max()]
    # start date: next full day (eg 2011-12-31 23:00 -> 2012-01-01)
    dates_df["start"] = (dates_df["start"] + pd.Timedelta(hours=2)).dt.floor("D").dt.date
    dates_df["end"]   = dates_df["end"  ].dt.date

    # df['date'] = df.index.year - 2000 + df.index.dayofyear / 365  # for long-term drift


    if verbose >= 1:
        print(dates_df)

    if verbose >= 2:
        # print(df.head().to_string())

        plt.figure(figsize=(10,6))
        df.drop(columns=['consumption_GW']).iloc[-(8*24*2):].plot()
        plt.show()

        # no time modification
        plt.figure(figsize=(10,6))
        (df[['Tavg_degC', 'solar_kW_per_m2', 'wind_m_per_s']]).plot()
        plt.show()

    return df, dates_df


# ----------------------------------------------------------------------
# Metamodel: prediction (losses are in architecture.py)
# ----------------------------------------------------------------------

# /!\ These two MUST remain equivalent.
#    When making modifications, we modify both in parallel.

def compute_meta_prediction_torch(
        pred_scaled   : torch.Tensor,   # (B, H, Q)
        x_scaled      : torch.Tensor,   # (B, L, F)
        baseline_idx  : Dict[str, int],
        weights_meta  : Dict[str, float],
        idx_median    : int
    ) -> torch.Tensor:                  # (B, Q)
    """
    Median-only metamodel prediction.

    Returns:
        y_meta_scaled : torch.Tensor, shape (B, Q)
    """

    pred_scaled1 = dict()
    # Baseline predictions from input features
    # First horizon only (t+1)
    for _name in ['lr', 'rf']:
        if _name in baseline_idx:
            pred_scaled1[_name]= x_scaled[:, -1, baseline_idx[_name]]
        elif weights_meta[_name] == 0:
            pred_scaled1[_name]= 0
        else:
            raise ValueError(f"{_name} not in baseline_idx, "
                             f"yet weight == {weights_meta[_name]} != 0")

    # B, _, _ = x_scaled.shape

    pred_scaled1['nn'] = pred_scaled[:, 0, idx_median]   # (B,)

    # Weighted meta prediction
    pred_meta_scaled = (
        weights_meta['nn'] * pred_scaled1['nn'] +
        weights_meta['lr'] * pred_scaled1.get('lr', 0) +
        weights_meta['rf'] * pred_scaled1.get('rf', 0)
    )

    return pred_meta_scaled


def compute_meta_prediction_numpy(
        pred_scaled   : np.ndarray,   # (B, Q) or (B, H, Q)
        x_scaled      : np.ndarray,   # (B, L, F)
        baseline_idx  : Dict[str, int],
        weights_meta  : Dict[str, float],
        idx_median    : int
    ) -> np.ndarray:                  # (B,)
    """
    NumPy-only meta prediction for validation.
    """

    # B, Q = pred_scaled.shape

    # NN: take median only
    if pred_scaled.ndim == 3:
        # (B, H, Q) → horizon 0, quantile index 1
        nn_pred = pred_scaled[:, 0, idx_median]   # (B,)
    elif pred_scaled.ndim == 2:
        # (B, Q)
        nn_pred = pred_scaled[:, idx_median]      # (B,)
    else:
        raise ValueError(f"Unexpected pred_scaled shape {pred_scaled.shape}")


    y_meta = weights_meta.get('nn', 0.) * nn_pred  # (B,)

    for _name in ['lr', 'rf']:
        w = weights_meta.get(_name, 0.)
        if w == 0.:
            continue
        if _name not in baseline_idx:
            raise ValueError(f"{_name} not in baseline_idx, yet weight={w} != 0")

        baseline = x_scaled[:, -1, baseline_idx[_name]]  # (B,)
        y_meta  += w * baseline                          # (B,)

    assert y_meta.ndim == 1, y_meta.shape

    return y_meta

# ----------------------------------------------------------------------
# validate_day_ahead
# ----------------------------------------------------------------------


@torch.no_grad()
def validate_day_ahead(
        model         : nn.Module,
        valid_loader  : DataLoader,
        valid_dates   : Sequence,
        scaler_y      : StandardScaler,
        baseline_idx  : Dict[str, int],
        device        : torch.device,
        input_length  : int,
        pred_length   : int,
        # incr_steps    : int,
        weights_meta  : Dict [str, float],
        quantiles     : Tuple[float, ...],
        lambda_cross  : float,
        lambda_coverage:float,
        lambda_deriv  : float
    ) -> Tuple[float, float]:
    """
    Returns:
        nn_loss_scaled   : float
        meta_loss_scaled : float
    """

    model.eval()

    Q = len(quantiles)
    # T = len(dates)


    # main loop
    for (x_scaled, y_scaled, _, _) in valid_loader:

        x_scaled_dev = x_scaled.to(device)
        x_scaled_cpu = x_scaled.cpu().numpy()
        # idx_np       = idx.cpu().numpy()              # shape (B,)

        # NN forward
        pred_scaled_dev = model(x_scaled_dev)
        pred_scaled_cpu = pred_scaled_dev[:, 0, :].detach().cpu().numpy()

        # inverse scale NN
        pred_nn = pred_scaled_cpu                # (B, Q), scaled

        # meta prediction
        pred_meta_scaled = compute_meta_prediction_numpy(
            pred_scaled_cpu,
            x_scaled_cpu,
            baseline_idx,
            weights_meta,
            Q // 2,
        )

        # true values
        y_true  = y_scaled[:, 0, 0].cpu().numpy()  # (B,), scaled


    nn_loss_scaled   = losses.loss_wrapper_quantile_numpy(
        pred_nn,   y_true, quantiles, lambda_cross,
        lambda_coverage,lambda_deriv)
    meta_loss_scaled = losses.loss_wrapper_quantile_numpy(
        pred_meta_scaled, y_true, quantiles, lambda_cross,
        lambda_coverage, lambda_deriv)
    # nn_loss_scaled   = np.mean((y_nn_scaled   - y_true_scaled) ** 2)
    # meta_loss_scaled = np.mean((y_meta_scaled - y_true_scaled) ** 2) # BUG: by hand

    return nn_loss_scaled, meta_loss_scaled



# ----------------------------------------------------------------------
# linear regression & random forest
# ----------------------------------------------------------------------

def temperature_correlation_matrix(df, verbose: int = 1) -> None:
    temp_cols = ["Tmin_degC", "Tmax_degC", "Tavg_degC", "Tavg_sat15_degC"]

    # Ensure they exist in the DataFrame
    temp_cols = [c for c in temp_cols if c in df.columns]

    print("Temperature Features Correlation Matrix (%):")
    corr = df[temp_cols].corr() * 100
    print(corr.round(1))

    if verbose > 1:
        # Also show pairs sorted by absolute correlation (most correlated first)
        print("Highly Correlated Temperature Feature Pairs (%):")
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
        )
        print(pairs.round(1))

def _build_model_from_cfg(cfg: Dict[str, dict]):
    """Factory that builds a model strictly from cfg."""
    cfg = cfg.copy()
    model_type = cfg.pop("type")

    if model_type == "ridge":
        return Ridge(**cfg)

    elif model_type == "rf":
        return RandomForestRegressor(**cfg)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

baseline_cfg = [
    {  # 'DEBUG'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":     50,     # was 300 -> fewer trees
        "max_depth":         6,     # shallower trees
        "min_samples_leaf": 10,     # more regularization
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'SMALL'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    250,
        "max_depth":        10,
        "min_samples_leaf": 15,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'LARGE'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    400,
        "max_depth":        15,
        "min_samples_leaf": 20,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'HUGE'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    500,
        "max_depth":        20,
        "min_samples_leaf": 20,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
}
]



def load_or_compute_rf_predictions(
    compute_kwargs,
    cache_dir,
    cache_id_dict,
    verbose: int = 0,
):
    """
    Generic cache wrapper for RandomForest predictions.

    Parameters
    ----------
    compute_kwargs : dict
        Keyword arguments passed to compute_fn
    cache_dir : str
        Directory to store cached predictions
    cache_id_dict : dict
        Dict describing data + RF config (used to build cache key)
    verbose : int
        Print cache hit/miss messages
    """
    os.makedirs(cache_dir, exist_ok=True)

    key_str    = json.dumps(cache_id_dict, sort_keys=True)
    cache_key  = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"rf_preds_{cache_key}.pkl")

    # either load...
    if os.path.exists(cache_path):
        if verbose > 0:
            print("Loaded RandomForest predictions from cache")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ... or compute
    if verbose > 0:
        print("Training RandomForest (no cache found)")

    out = regression_and_forest(**compute_kwargs)

    # Save
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)

    if verbose > 0:
        print("Saved RandomForest predictions to cache")

    return out



## Version 3 -- simplifying to be sure I understand it
##      older versions in: archives/utils-old-LR_RF-test_predictions.py
def regression_and_forest(
    df:          pd.DataFrame,
    target_col:  str,
    feature_cols:List[str],
    train_end:   int,   # end of training set (exclusive)
    val_end:     int,     # end of validation set (exclusive)
    quantiles:   Tuple[float, ...],
    lambda_cross:float,
    lambda_coverage:float,
    lambda_deriv:float,
    models_cfg:  Dict[str, dict] = baseline_cfg,
    verbose:     int = 0
) -> Tuple[Dict[str, pd.Series], Ridge, Dict[str, Dict[str, float]]]:
    """
    Leakage-safe contemporaneous tabular baselines:
        y_t ~ features_t   (NO LAG)

    Parameters
    ----------
    models_cfg : dict
        Example:
        {
            "lr": {"type": "ridge", "alpha": 1.0},
            "rf": {"type": "rf", "n_estimators": 300, "max_depth": 12}
        }

    Returns
    -------
    features : Dict[str, pd.Series]
        {name -> OOF feature aligned with df.index}
    final_models : Dict[str, fitted model]
    losses : Dict[str, Dict[str, float]]
        {name -> {"train": mse, "valid": mse}}
    """


    # -------------------------
    # 1. Extract matrices
    # -------------------------
    X_GW: np.ndarray = df[feature_cols].values.astype(np.float32)
    y_GW: np.ndarray = df[target_col  ].values.astype(np.float32)

    N: int = len(df)
    # print(f"indices: 0 < {train_end} < {val_end} < {N}")
    assert 0 < train_end < val_end <= N, "Invalid split indices"

    # -------------------------
    # 2. Define ranges
    # -------------------------
    train_idx: np.ndarray = np.arange(0,        train_end)
    valid_idx: np.ndarray = np.arange(train_end,val_end)
    test_idx : np.ndarray = np.arange(val_end,  N)

    X_train_GW: np.ndarray = X_GW[train_idx];  y_train_GW: np.ndarray = y_GW[train_idx]
    X_valid_GW: np.ndarray = X_GW[valid_idx];  y_valid_GW: np.ndarray = y_GW[valid_idx]
    X_test_GW : np.ndarray = X_GW[ test_idx];  y_test_GW : np.ndarray = y_GW[ test_idx]

    # -------------------------
    # 3. predictions on TRAIN only
    # -------------------------

    # def mse(a,b): return float(np.mean((a-b)**2))

    models    = dict()
    preds_GW  = dict()
    losses_GW = dict()
    series_pred_GW= pd.Series()

    for name, cfg in models_cfg.items():  # name = e.g. 'lr', 'rf'
        preds_GW [name] = pd.Series()
        losses_GW[name] = dict()

        models[name] = _build_model_from_cfg(cfg)
        models[name].fit(X_train_GW, y_train_GW)

        pred_train_GW = models[name].predict(X_train_GW)
        pred_valid_GW = models[name].predict(X_valid_GW)
        pred_test_GW  = models[name].predict(X_test_GW )

        # Calculate losses
        losses_GW[name]['train'] = losses.loss_wrapper_quantile_numpy(
            pred_train_GW, y_train_GW, quantiles, lambda_cross,
            lambda_coverage, lambda_deriv)
        losses_GW[name]['valid'] = losses.loss_wrapper_quantile_numpy(
            pred_valid_GW, y_valid_GW, quantiles, lambda_cross,
            lambda_coverage, lambda_deriv)
        losses_GW[name]['test' ] = losses.loss_wrapper_quantile_numpy(
            pred_test_GW,  y_test_GW,  quantiles, lambda_cross,
            lambda_coverage, lambda_deriv)

        series_pred_GW[name] = pd.Series(
            np.concatenate([pred_train_GW, pred_valid_GW, pred_test_GW]),
                            index = df.index)


    if verbose >= 2 and {"lr", "rf"} <= models.keys():

        ridge = pd.Series(
            models["lr"].coef_ * 100.,
            index=feature_cols,
            name="ridge_coef_pc"
        ).round(2)

        rf = pd.Series(
            models["rf"].feature_importances_ * 100.,
            index=feature_cols,
            name="rf_importance_pc"
        )

        df_imp = pd.concat([ridge, rf], axis=1)

        # Normalize for comparability
        _source = df_imp["ridge_coef_pc"].abs()
        df_imp["ridge_norm"] = _source / _source.max()

        _source = df_imp["rf_importance_pc"]
        df_imp["rf_norm"]    = _source / _source.max()

        # Overall relevance score
        df_imp["score_pc"] = 100. * df_imp[["ridge_norm", "rf_norm"]].mean(axis=1)

        # Final ordering
        df_imp = (
            df_imp
            .sort_values("score_pc", ascending=False)
            .drop(columns=["ridge_norm", "rf_norm"])
        )

        print("\n[Model diagnostics] Top features (Ridge + RF):")
        print(df_imp.round(2))  # .head(20)


    print("MSE [GW²]:\n", pd.DataFrame(losses_GW).round(2).T)

    return series_pred_GW, models, losses_GW



# -------------------------------------------------------
# Testing
# -------------------------------------------------------


@torch.no_grad()
def test_predictions_day_ahead(
    X_test_GW, y_test_GW, test_loader, model, scaler_y,
    num_test_windows, feature_cols, test_dates, device,
    input_length: int, pred_length: int,
    quantiles: Tuple[float, ...], is_validation: bool = False
) -> Tuple[pd.Series, Dict[str, pd.Series], Dict[str, pd.Series]]:

    model.eval()

    # -------------------------------
    # Baselines available
    # -------------------------------
    baseline_names = [
        nm for nm in ("lr", "rf")
        if f"consumption_{nm}" in feature_cols
    ]

    records = []  # one record per (origin, horizon)

    # Iterate once: no aggregation
    for batch_idx, batch_scaled in enumerate(test_loader):
        x_batch_scaled, y_batch_scaled, idx_batch, datetime_int = batch_scaled
        idx_batch   = idx_batch   .cpu().numpy()   # origin indices
        datetime_int= datetime_int.cpu().numpy()

        # NN prediction
        x_scaled_dev = x_batch_scaled.to(device)
        pred_scaled = model(x_scaled_dev).cpu().numpy()  # (B, H, Q)

        B, H, Q = pred_scaled.shape
        # print(B, H, Q)
        assert H == pred_length
        assert Q == len(quantiles)

        # Inverse-scale ground truth
        y_true_GW = scaler_y.inverse_transform(
            y_batch_scaled[:, :, 0].cpu().numpy().reshape(-1, 1)
        ).reshape(B, H)

        # Inverse-scale predictions
        y_pred_GW = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, Q)
        ).reshape(B, H, Q)

        # One prediction per target timestamp
        for b in range(B):
            # origin_time = pd.Timestamp(datetime_int[b], unit='s')

            for h in range(H):
                target_idx = idx_batch[b] + h
                if target_idx >= len(test_dates):
                    continue

                target_time = test_dates[target_idx]

                row = {
                    "target_time": target_time,
                    "y_true": y_true_GW[b, h],
                }

                for qi, tau in enumerate(quantiles):
                    row[f"q{int(100*tau)}"] = y_pred_GW[b, h, qi]

                # Baselines at the same target time
                for nm in baseline_names:
                    col = feature_cols.index(f"consumption_{nm}")
                    row[nm] = X_test_GW[target_idx, col]

                records.append(row)

    # Build DataFrame
    df = pd.DataFrame.from_records(records).set_index("target_time").sort_index()

    # Output format
    true_series_GW = df["y_true"]

    dict_pred_series_GW = {
        f"q{int(100*tau)}": df[f"q{int(100*tau)}"]
        for tau in quantiles
    }

    dict_baseline_series_GW = {
        nm: df[nm]
        for nm in baseline_names
    }

    return true_series_GW, dict_pred_series_GW, dict_baseline_series_GW



def quantile_coverage(y_true: pd.Series,
                      y_pred: pd.Series) -> float:
    """
    Fraction of points where y_true <= y_pred
    """
    idx = y_true.index.intersection(y_pred.index)
    return float(np.mean(y_true.loc[idx] <= y_pred.loc[idx]))




def compare_models(true_series, dict_pred_series, dict_baseline_series,
                   weights_meta : Dict[str, float], unit:str="",
                   verbose: int = 0) -> None:
    if verbose < 1: return  # this function does nothing if it cannot display

    def rmse(a,b): return round(np.sqrt(np.mean((a-b)**2)),2)
    def mae (a,b): return round(np.mean(np.abs  (a-b)),    2)

    # print("weights_meta:", weights_meta)

    list_baselines = []
    for _name in ['lr', 'rf']:
        if _name in dict_baseline_series:  # keep only those we trained
            list_baselines.append(_name)
    # print("list_baselines:", list_baselines)

    df_eval = pd.DataFrame({
        "true":true_series,
        "nn":  dict_pred_series.get("q50")
    })
    for _name in list_baselines:
        df_eval[_name] = dict_baseline_series.get(_name)
    df_eval.dropna(inplace=True)

    _meta = df_eval['nn'] * weights_meta['nn']
    for _name in list_baselines:
        _meta += df_eval[_name] * weights_meta[_name]
    df_eval['meta'] = _meta

    names_models = ['nn'] + list_baselines + ['meta']
    y_true = df_eval["true"]

    rows = []
    for name in names_models:
        y_pred = df_eval[name]
        res    = y_pred - y_true

        rows.append({
            "model": name,
            "RMSE":  rmse(y_pred, y_true),
            "MAE":   mae (y_pred, y_true),
            "bias":  res.mean(),
            "std":   res.std(),
        })

    df_metrics = (
        pd.DataFrame(rows)
          .set_index("model")
          .sort_values("RMSE")
    )

    print("\nTesting metrics:")
    print(df_metrics.round(3))



    if verbose >= 2:
        print("\n[Diagnostics] RMSE by hour of day")

        df_eval['hour'] = df_eval.index.hour

        df_rmse_hour = pd.DataFrame({
            name: (
                df_eval
                .groupby("hour")
                .apply(lambda d: rmse(d[name], d["true"]), include_groups=False)
            )
            for name in ["nn"] + list_baselines + ["meta"]
        })
        print(df_rmse_hour.round(2))

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for col in df_rmse_hour.columns:
            plt.plot(df_rmse_hour.index, df_rmse_hour[col], label=col)
        plt.xlabel("Hour of day")
        plt.ylabel("RMSE [GW]")
        plt.title("RMSE by hour of day")
        plt.xticks(range(0, 24))
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


        print("\n[Diagnostics] RMSE by month of year")

        df_eval["month"] = df_eval.index.month

        df_rmse_month = pd.DataFrame({
            name: (
                df_eval
                .groupby("month")
                .apply(lambda d: rmse(d[name], d["true"]), include_groups=False)
            )
            for name in ["nn"] + list_baselines + ["meta"]
        })
        print(df_rmse_month.round(2))

        plt.figure(figsize=(10, 6))
        for col in df_rmse_month.columns:
            plt.plot(df_rmse_month.index, df_rmse_month[col], marker="o", label=col)
        plt.xlabel("Month")
        plt.ylabel("RMSE [GW]")
        plt.title("RMSE by month")
        plt.xticks(range(1, 13))
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()



# -------------------------------------------------------
# Diagnosing outliers
# -------------------------------------------------------

def per_timestamp_loss_after_aggregation(
    y_true,           # shape (T,)
    y_pred,           # shape (T, Q) or (T,) if median only
    quantiles,
):
    """
    Returns per-timestamp pinball loss (T,)
    """
    if y_pred.ndim == 1:  # median only
        q50_idx = quantiles.index(0.5)
        y_pred = y_pred[:, None]
        quantiles = [0.5]

    losses = []
    for q, tau in enumerate(quantiles):
        e = y_true - y_pred[:, q]
        losses.append(np.maximum(tau * e, (tau - 1) * e))

    return np.mean(np.stack(losses, axis=1), axis=1)  # (T,)


def worst_days_by_loss(
    dates,
    y_true,
    y_pred,
    quantiles,
    temperature,
    top_n=10
):
    """
    Returns DataFrame of worst days by mean loss
    """
    ts_loss = per_timestamp_loss_after_aggregation(
        y_true, y_pred, quantiles
    )

    df = pd.DataFrame({
        'date':      dates.normalize(),  # midnight per day
        'loss_pc':   (ts_loss * 100).round(2),
        'Tavg_degC': temperature,
    })

    daily = (
        df.groupby('date', as_index=False)
          .agg(
              mean_loss_pc= ('loss_pc',  "mean"),
              max_loss_pc = ('loss_pc',  "max"),
              ramp_pc     = ("loss_pc", lambda x: np.max(np.abs(np.diff(x)))),
              n_points    = ('loss_pc',  "size"),
              Tavg_degC   = ('Tavg_degC',"mean"),
          )
          .sort_values('mean_loss_pc', ascending=False)
    )

    daily = daily[daily['n_points'] == 48]  # incommensurable

    daily['day_name'] = daily['date'].dt.day_name()
    daily['day'     ] = daily['date'].dt.day
    daily['month'   ] = daily['date'].dt.month
    daily['year'    ] = daily['date'].dt.year

    daily['mean_loss_pc'] = daily['mean_loss_pc'].round(1)
    daily['Tavg_degC'   ] = daily['Tavg_degC'   ].round(1)


    return daily[['day_name', 'day', 'month', 'year',
                  'mean_loss_pc', 'max_loss_pc', 'ramp_pc', 'Tavg_degC']].head(top_n)


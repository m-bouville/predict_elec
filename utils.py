# ----------------------------------------------------------------------
#
# utils.py
#
# ----------------------------------------------------------------------


# import os
import time

from   typing import List, Tuple, Dict, Optional  #, Sequence

import torch

import numpy  as np
import pandas as pd

# from   sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt


import IO  # losses, architecture,  plots



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
    for _hours in [6, 8, 12, 24]:  # several periods per day
        df['sin_'+str(_hours)+'h'] = np.sin(24/_hours * 2*np.pi * df['hour_norm'])
    df['cos_'+str(_hours)+'h'] = np.cos(24/_hours * 2*np.pi * df['hour_norm'])

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
    df['is_Monday'  ] = (df.index.dayofweek == 0).astype(np.int16) # for morning

    df['is_August'  ] = ((df.index.month     == 8) \
                         & (df.index.day >= 5) & (df.index.day >= 25)).astype(np.int16)
    df['is_Christmas']= (((df.index.month == 12) & (df.index.day >= 24)) | \
                         ((df.index.month ==  1) & (df.index.day <=  2))).astype(np.int16)

    if verbose >= 3:
        print(df.head().to_string())

    return df


def df_features_past_consumption(consumption: pd.Series,
                lag: int, num_steps_per_day: int,
                verbose: int = 0) -> pd.DataFrame:

    df = consumption.to_frame('consumption_GW')

    # lag avoids implicitly leaking future information
    _shifted_conso = df['consumption_GW'].shift(lag)

    # # diff
    # for _weeks in [1, 2]:
    #     _hours = int(round(num_steps_per_day * 7 * _weeks))
    #     df[f"consumption_diff_{_weeks}wk_GW"] = \
    #         _shifted_conso - _shifted_conso.shift(freq=f"{_hours}h")

    # moving averages
    for _weeks in [1, 2, 4, 52]:
        _hours = int(round(num_steps_per_day * 7 * _weeks))
        df[f"consumption_SMA_{_weeks}wk_GW"] = _shifted_conso.rolling(
            f"{_hours}h", min_periods=int(round(_hours*.8))).mean()

    if verbose >= 3:
        print(df.tail().to_string())  # head() would be made of NaNs

    return df.drop(columns=['consumption_GW'])


def df_features(dict_fnames: Dict[str, str], cache_fname: str,
        lag: int, num_steps_per_day: int, minutes_per_step: int, verbose: int = 0) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, dates_df = IO.load_data(dict_fnames, cache_fname,
            num_steps_per_day=num_steps_per_day, minutes_per_step=minutes_per_step)
    assert isinstance(df.index, pd.DatetimeIndex)

    # Fourier-like sine waves, weekends
    df_calendar    = df_features_calendar(df.index, verbose)

    # /!\ moving averages and offsets based on consumption may leak
    df_consumption = df_features_past_consumption(
            df['consumption_GW'], lag, num_steps_per_day, verbose)
    # df_consumption = pd.DataFrame()


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


    # start date: next full day (eg 2011-12-31 23:00 -> 2012-01-01)
    dates_df["start"] = (dates_df["start"] + pd.Timedelta(hours=2)).dt.floor("D").dt.date
    dates_df["end"]   =  dates_df["end"  ].dt.date

    # df['date'] = df.index.year - 2000 + df.index.dayofyear / 365  # for long-term drift

    if verbose >= 3:
        # print(df.head().to_string())

        plt.figure(figsize=(10,6))
        df.drop(columns=['consumption_GW']).iloc[-(8*24*2):].plot()
        plt.show()

        # no time modification
        plt.figure(figsize=(10,6))
        (df[['Tmax_degC', 'Tavg_degC']]).plot()
        # (df[['Tavg_degC', 'solar_kW_per_m2', 'wind_m_per_s']]).plot()
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



# -------------------------------------------------------
# Testing
# -------------------------------------------------------


@torch.no_grad()
def subset_predictions_day_ahead(
    X_subset_GW, subset_loader, model, scaler_y,
    feature_cols, device,
    input_length: int, pred_length: int, valid_length: int, minutes_per_step: int,
    quantiles: Tuple[float, ...]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Tuple]:
    # subset: train, valid ot test

    # print(f"X_subset_GW.shape = {X_subset_GW.shape}")

    model.eval()

    # Baselines available
    baseline_names = [
        nm for nm in ('lr', 'rf', 'oracle')
        if f"consumption_{nm}" in feature_cols
    ]
    offset_steps = pred_length - valid_length + 1
    print(f"{pred_length} - {valid_length} + 1 = {offset_steps}")

    records = []  # one record per (origin, horizon)
    min_origin_time =  np.inf
    max_origin_time = -np.inf

    # Iterate once: no aggregation
    for (X_scaled, y_scaled, idx_subset, forecast_origin_int) in subset_loader:
        idx_subset         = idx_subset         .cpu().numpy()   # origin indices
        forecast_origin_int= forecast_origin_int.cpu().numpy()

        # NN prediction
        X_scaled_dev = X_scaled.to(device)


        # pred_scaled = model(X_scaled_dev).cpu().numpy()  # (B, H, Q)
        # assert pred_scaled.shape[1] == valid_length
        # pred_scaled = pred_scaled[:, -valid_length:]     # (B, V, Q)
        pred_scaled = model(X_scaled_dev)[:, -valid_length:].cpu().numpy() #(B, V, Q)

        B, V, Q = pred_scaled.shape
        # print(B, V, Q)
        assert V == valid_length
        assert Q == len(quantiles)
        # assert B <= batch_size

        # Inverse-scale ground truth                 # (B, [1..H])
        y_true_GW = scaler_y.inverse_transform(
            y_scaled[:, -valid_length:, 0].cpu().numpy().reshape(-1, 1)
        ).reshape(B, V)

        # Inverse-scale predictions
        y_pred_GW = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, Q)
        ).reshape(B, V, Q)

        # One prediction per target timestamp
        for sample in range(B):
            forecast_origin_time = pd.Timestamp(
                forecast_origin_int[sample], unit='s', tz='UTC')
            # print (f"sample:{sample:3n}, {forecast_origin_time}")
            min_origin_time = min(min_origin_time, forecast_origin_int[sample])
            max_origin_time = max(max_origin_time, forecast_origin_int[sample])

            for h in range(V):
                    # /!\ after reference: h == 0 <=> offset_steps after noon
                idx_subset_current = idx_subset[sample] + h + offset_steps
                if idx_subset_current >= X_subset_GW.shape[0]:
                    continue     # otherwise, would be after end of dataset

                # target_time = test_dates[target_idx]

                time_current = forecast_origin_time + \
                    pd.Timedelta(minutes = minutes_per_step * (h + offset_steps))
                # print (f"sample{sample:4n}, h ={h:3n}: "
                #        f"idx_subset_current = {idx_subset_current}, "
                #        f"time_current = {time_current}")

                row = {
                    "time_current": time_current,
                    "y_true"      : y_true_GW[sample, h],  # starts at 12:30
                }

                for qi, tau in enumerate(quantiles):
                    row[f"q{int(100*tau)}"] = y_pred_GW[sample, h, qi] # starts at 12:30

                # Baselines at the same target time
                for nm in baseline_names:
                    col    = feature_cols.index(f"consumption_{nm}")
                    row[nm]= X_subset_GW[idx_subset_current, col]

                row['h'] = h
                row['idx_subset_current'] = idx_subset_current

                records.append(row)

    # Build DataFrame
    df = pd.DataFrame.from_records(records).set_index("time_current").sort_index()

    # print(df.head(10).astype(np.float32).round(2))

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

    return true_series_GW, dict_pred_series_GW, dict_baseline_series_GW, \
        (pd.Timestamp(min_origin_time, unit='s', tz='UTC'), \
         pd.Timestamp(max_origin_time, unit='s', tz='UTC'))


def quantile_coverage(y_true: pd.Series,
                      y_pred: pd.Series) -> float:
    """
    Fraction of points where y_true <= y_pred
    """
    idx = y_true.index.intersection(y_pred.index)
    return float(np.mean(y_true.loc[idx] <= y_pred.loc[idx]))


def index_summary(name   : str,
                  idx    : pd.DatetimeIndex,
                  ref_idx: Optional[pd.DatetimeIndex] = None):
    return {
        "series":    name,
        "start":     idx.min().date(),
        "end":       idx.max().date(),
        "n":         len(idx),
        "n_common":  len(idx.intersection(ref_idx))if ref_idx is not None else None,
        "start_diff":idx.min() != ref_idx.min()    if ref_idx is not None else None,
        "end_diff":  idx.max() != ref_idx.max()    if ref_idx is not None else None,
    }


def compare_models(true_series, dict_pred_series,
                   dict_baseline_series: Dict[str, List[float]] or None,
                   list_pred_meta: List[List[float]], subset: str="", unit:str="",
                   max_RMSE: float = 4,
                   verbose: int = 0) -> pd.DataFrame:
    # if verbose < 1: return  # this function does nothing if it cannot display

    def rmse(a,b): return round(np.sqrt(np.mean((a-b)**2)),2)
    def mae (a,b): return round(np.mean(np.abs  (a-b)),    2)

    # print("weights_meta:", weights_meta)

    if verbose >= 3:
        print("shapes:")
        print(f"true: {true_series.shape} ({true_series.index.min()} -> {true_series.index.max()})")
        print(f"nn:   {dict_pred_series.get('q50').shape} "
              f"({dict_pred_series.get('q50').index.min()} -> "
              f"{dict_pred_series.get('q50').index.max()})")
        if list_pred_meta is not None:
            for (i, _pred_meta) in enumerate(list_pred_meta):
                print(f"meta {i+1}: {_pred_meta.shape}")
                #" ({pred_meta.index.min()} -> {pred_meta.index.max()})")

    names_models = []

    df_eval = pd.DataFrame({"true": true_series})

    if dict_pred_series is not None:
        df_eval['nn'] = dict_pred_series.get("q50")
        names_models += ['nn']

    # baselines (LR, RF)
    list_baselines = []
    if dict_baseline_series is not None:
        for _name in ['lr', 'rf', 'oracle']:
            if _name in dict_baseline_series:  # keep only those we trained
                if verbose >= 3:
                    print(f"{_name}:   {dict_baseline_series.get(_name).shape} "
                          f"({dict_baseline_series.get(_name).index.min()} -> "
                          f"{dict_baseline_series.get(_name).index.max()})")
                list_baselines.append(_name)
                df_eval[_name] = dict_baseline_series.get(_name)
        # print("list_baselines:", list_baselines)

    df_eval.dropna(inplace=True)

    names_models += list_baselines

    # print("pred_meta:", pred_meta)
    if list_pred_meta is not None:
        for (i, _pred_meta) in enumerate(list_pred_meta):
            _name = "meta" if len(list_pred_meta) == 1 else f"meta{i+1}"
            df_eval[_name] = _pred_meta.reindex(df_eval.index)
            names_models += [_name]

    y_true = df_eval["true"]

    rows = []
    for name in names_models:
        y_pred = df_eval[name]
        res    = y_pred - y_true

        rows.append({
            "model": name,
            "bias":  res.mean(),
            # "std":   res.std(),
            "RMSE":  rmse(y_pred, y_true),
            "MAE":   mae (y_pred, y_true),
        })

    df_metrics = (
        pd.DataFrame(rows)
          .set_index("model")
          # .sort_values("RMSE")
    )

    if verbose >= 1:
        print(df_metrics.round(3))

        # plotting RMSE as a function of bias for the different models
        plt.figure(figsize=(10,6))
        for model in df_metrics.index:
            plt.scatter(df_metrics.loc[model, 'bias'],
                        df_metrics.loc[model, 'RMSE'], label=model)
        plt.xlabel(subset + ' bias [GW]'); plt.xlim(-0.5, 1.5)
        plt.ylabel(subset + ' RMSE [GW]'); plt.ylim( 0., max_RMSE) # plt.ylim(bottom=0.)
        plt.legend()
        plt.show()




    if verbose >= 3:
        subset_str = "" if unit is None else f" {subset}"
        unit_str   = "" if unit is None else f" [{unit}]"
        print(f"\n[Diagnostics]{subset_str} RMSE by hour of day{unit_str}")

        df_eval['hour'] = df_eval.index.hour

        df_rmse_hour = pd.DataFrame({
            name: (
                df_eval
                .groupby("hour")
                .apply(lambda d: rmse(d[name], d["true"]), include_groups=False)
            )
            for name in names_models
        })
        print(df_rmse_hour.round(2))


        plt.figure(figsize=(10, 6))
        for col in df_rmse_hour.columns:
            plt.plot(df_rmse_hour.index, df_rmse_hour[col], label=col)
        plt.xlabel( "hour of day")
        plt.ylabel(f"RMSE{unit_str}")
        plt.title (f"{subset_str} RMSE by hour of day")
        plt.xticks(range(0, 24))
        plt.ylim(bottom=0, top=max_RMSE)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


        print(f"\n[Diagnostics]{subset_str} RMSE by month of year{unit_str}")

        df_eval["month"] = df_eval.index.month

        df_rmse_month = pd.DataFrame({
            name: (
                df_eval
                .groupby("month")
                .apply(lambda d: rmse(d[name], d["true"]), include_groups=False)
            )
            for name in names_models
        })
        print(df_rmse_month.round(2))

        plt.figure(figsize=(10, 6))
        for col in df_rmse_month.columns:
            plt.plot(df_rmse_month.index, df_rmse_month[col], marker="o", label=col)
        plt.xlabel("Month")
        plt.ylabel(f"RMSE{unit_str}")
        plt.title (f"{subset_str} RMSE by month")
        plt.xticks(range(1, 13))
        plt.ylim(bottom=0, top=max_RMSE)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_metrics



# -------------------------------------------------------
# Display evolution
# -------------------------------------------------------

def display_evolution(
        epoch: int, t_epoch_start,
        train_loss_scaled: float, valid_loss_scaled: float,
        list_of_min_losses: List[float], list_of_lists: List[List[float]],
        # constants
        num_epochs: int, display_every: int, plot_conv_every: int,
        min_delta: float, verbose:int=0)\
            -> [[float, float, float],
                [List[float], List[float], List[float], List[float]]]:

    (min_train_loss_scaled,  min_valid_loss_scaled, min_loss_display_scaled) = \
        list_of_min_losses

    (list_train_loss_scaled, list_min_train_loss_scaled,
    list_valid_loss_scaled, list_min_valid_loss_scaled) = list_of_lists

    if ((epoch+1) % display_every == 0) | (epoch == 0):
        # comparing latest loss to lowest so far
        if valid_loss_scaled <= min_loss_display_scaled - min_delta:
            is_better = '**'
        elif valid_loss_scaled <= min_loss_display_scaled:
            is_better = '*'
        else:
            is_better = ''

        min_loss_display_scaled = min_valid_loss_scaled

        t_epoch = time.perf_counter() - t_epoch_start

        if verbose >= 1:
            print(f"{epoch+1:3n} /{num_epochs:3n} ={(epoch+1)/num_epochs*100:3.0f}%,"
                  f"{t_epoch/60*(num_epochs/(epoch+1)-1)+.5:3.0f} min left, "
                  f"loss (1e-3): "
                  f"train{train_loss_scaled*1000:5.0f} (best{min_train_loss_scaled*1000:5.0f}), "
                  f"valid{valid_loss_scaled*1000:5.0f} ({    min_valid_loss_scaled*1000:5.0f})"
                  f" {is_better}")

    min_train_loss_scaled = min(min_train_loss_scaled, train_loss_scaled)
    list_train_loss_scaled    .append(train_loss_scaled)
    list_min_train_loss_scaled.append(min_train_loss_scaled)

    min_valid_loss_scaled = min(min_valid_loss_scaled, valid_loss_scaled)
    list_valid_loss_scaled    .append(valid_loss_scaled)
    list_min_valid_loss_scaled.append(min_valid_loss_scaled)


    # if ((epoch+1 == plot_conv_every) | ((epoch+1) % plot_conv_every == 0))\
    #         & (epoch < num_epochs-2):
    #     plots.convergence(list_train_loss_scaled, list_min_train_loss_scaled,
    #                       list_valid_loss_scaled, list_min_valid_loss_scaled,
    #                       baseline_losses_scaled,
    #                       None, None, None, None,
    #                       # list_meta_train_loss_scaled, list_meta_min_train_loss_scaled,
    #                       # list_meta_valid_loss_scaled, list_meta_min_valid_loss_scaled,
    #                       partial=True, verbose=verbose)

    return ((min_train_loss_scaled, min_valid_loss_scaled, min_loss_display_scaled),
            # list of lists
            (list_train_loss_scaled, list_min_train_loss_scaled,
            list_valid_loss_scaled, list_min_valid_loss_scaled)
            )



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
    num_steps_per_day,
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

    daily = daily[daily['n_points'] == num_steps_per_day]  # incommensurable

    daily['day_name'] = daily['date'].dt.day_name()
    daily['day'     ] = daily['date'].dt.day
    daily['month'   ] = daily['date'].dt.month
    daily['year'    ] = daily['date'].dt.year

    daily['mean_loss_pc'] = daily['mean_loss_pc'].round(1)
    daily['Tavg_degC'   ] = daily['Tavg_degC'   ].round(1)


    return daily[['day_name', 'day', 'month', 'year',
                  'mean_loss_pc', 'max_loss_pc', 'ramp_pc', 'Tavg_degC']].head(top_n)


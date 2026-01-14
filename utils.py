# ----------------------------------------------------------------------
#
# utils.py
#
# ----------------------------------------------------------------------


# import os
import time

from   typing import List, Tuple, Dict, Sequence, Optional, Any

import torch

import numpy  as np
import pandas as pd

# from   sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt
import seaborn as sns

import holidays


import IO,  plots  # losses, architecture



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

    for _days, _name in zip([1.75, 3.5, 7], ["1_75", "3_5", "7"]): # several per week
        df['sin_'+_name+'day'] = np.sin(7/_days * 2*np.pi * df['dow_norm'])
    # df['sin_1wk']  = np.sin(2*np.pi*df['dow_norm'])

    for _months in [3, 4, 6, 12]:  # several periods per year
        df['cos_'+str(_months)+'mo'] = np.cos(12/_months * 2*np.pi * df['doy_norm'])


    # remove temporary variables
    df.drop(columns=['hour_norm', 'dow_norm', 'doy_norm'], inplace=True)


    # day of week: 7 days => 6 degrees of freedom (convention: 0 = Monday)
    df['is_Monday'  ] = (df.index.dayofweek == 0).astype(np.int16) # esp. morning
    df['is_Tuesday' ] = (df.index.dayofweek == 1).astype(np.int16)
    df['is_Wednesday']= (df.index.dayofweek == 2).astype(np.int16)
    df['is_Friday'  ] = (df.index.dayofweek == 4).astype(np.int16) # esp. evening
    df['is_Saturday'] = (df.index.dayofweek == 5).astype(np.int16)
    df['is_Sunday'  ] = (df.index.dayofweek == 6).astype(np.int16)
    df['is_weekend' ] = (((df.index.dayofweek == 4) & (df.index.hour >= 16)) |
                          (df.index.dayofweek.isin([5, 6]))).astype(np.int16)

    # peak hours. /!\ UTC: [6, 8) means [7, 9) local in winter
    df['is_morning_peak']=((df.index.hour >= 6) & (df.index.hour < 8)).astype(np.int16)
    df['is_evening_peak']=((df.index.hour >=17) & (df.index.hour <19)).astype(np.int16)

    # there is a peak of coverage loss between about 9pm and midninght, UTC
    df['is_evening'    ] = (df.index.hour >= 21).astype(np.int16)

    # people go on holiday
    df['is_August'  ] = ((df.index.month    == 8) \
                    & (df.index.day >= 5) & (df.index.day <= 25)).astype(np.int16)
    # df['is_Christmas']=(((df.index.month==12) & (df.index.day>=23)) | \
    #                  ((df.index.month== 1) & (df.index.day<= 4))).astype(np.int16)
            # redundent with school holiday

    # covid lockdown periods
    _tz = 'Europe/Paris'
    lockdown_periods = [
        (pd.Timestamp('2020-03-17', tz=_tz), pd.Timestamp('2020-05-11', tz=_tz)),
        (pd.Timestamp('2020-10-30', tz=_tz), pd.Timestamp('2020-12-15', tz=_tz)),
        (pd.Timestamp('2021-04-03', tz=_tz), pd.Timestamp('2021-05-03', tz=_tz))
    ]
    df['lockdown'] = 0
    for start, end in lockdown_periods:
        df.loc[start:end, 'lockdown'] = 1


    # public holidays for France
    fr_holidays = holidays.France(years=range(2012, 2027))
    dates_holidays = set(fr_holidays.keys())
    df['is_holiday'] = np.isin(df.index.date, list(dates_holidays)).astype(np.int16)


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


def df_features(dict_input_csv_fnames: Dict[str, str], cache_fname: str,
        lag: int, num_steps_per_day: int, minutes_per_step: int, verbose: int = 0) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
    df, dates_df, weights_regions = IO.load_data(dict_input_csv_fnames, cache_fname,
            num_steps_per_day=num_steps_per_day, minutes_per_step=minutes_per_step)
    assert isinstance(df.index, pd.DatetimeIndex)

    # Fourier-like sine waves, weekends
    df_calendar    = df_features_calendar(df.index, verbose)

    # `lag` ensures moving averages and offsets based on consumption do not leak
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
    dates_df["start"]= (dates_df["start"] + pd.Timedelta(hours=2)) \
        .dt.floor("D").dt.date
    dates_df["end"]  =  dates_df["end"  ].dt.date

    df['date'] = df.index.year - 2000 + df.index.dayofyear/365  # for long-term drift

    if verbose >= 3:
        # print(df.head().to_string())

        plt.figure(figsize=(10,6))
        df.drop(columns=['consumption_GW']).iloc[-(8*24*2):].plot()
        plt.show()

        # # no time modification
        # plt.figure(figsize=(10,6))
        # (df[['Tmax_degC', 'Tavg_degC']]).plot()
        # # (df[['Tavg_degC', 'solar_kW_per_m2', 'wind_m_per_s']]).plot()
        # plt.show()

    return df, dates_df, weights_regions




# -------------------------------------------------------
# Testing
# -------------------------------------------------------


@torch.no_grad()
def subset_predictions_day_ahead(
    X_subset_GW, subset_loader, model, scaler_y_nation,
    cols_features: List[str], device,
    input_length: int, pred_length: int, valid_length: int, minutes_per_step: int,
    quantiles: Tuple[float, ...]
) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series],
           Tuple[pd.Timestamp, pd.Timestamp]]:
    # subset: train, valid ot test

    # print(f"X_subset_GW.shape = {X_subset_GW.shape}")

    model.eval()

    # Baselines available
    baseline_names = [
        nm for nm in ('LR', 'RF', 'LGBM', 'oracle')
        if f"consumption_{nm}" in cols_features
    ]
    offset_steps = pred_length - valid_length + 1
    # print(f"{pred_length} - {valid_length} + 1 = {offset_steps}")

    records = []  # one record per (origin, horizon)
    min_origin_time =  np.inf
    max_origin_time = -np.inf

    # Iterate once: no aggregation
    for (X_scaled, regions_scaled, y_scaled, T_degC, idx_subset, forecast_origin_int) in subset_loader:
        idx_subset         = idx_subset         .cpu().numpy()   # origin indices
        forecast_origin_int= forecast_origin_int.cpu().numpy()

        # NN prediction
        X_scaled_dev = X_scaled.to(device)


        # pred_scaled = model(X_scaled_dev).cpu().numpy()  # (B, H, Q)
        # assert pred_scaled.shape[1] == valid_length
        # pred_scaled = pred_scaled[:, -valid_length:]     # (B, V, Q)
        (pred_nation_scaled_dev, _) = model(X_scaled_dev)
        pred_nation_scaled_cpu = pred_nation_scaled_dev[:, -valid_length:] .cpu().numpy() # (B, V, Q)
        # pred_regions_scaled_cpu= pred_regions_scaled_dev[:, -valid_length:].cpu().numpy() # (B, V, R)

        B, V, Q = pred_nation_scaled_cpu.shape
        # print(B, V, Q)
        assert V == valid_length,   f"{V} != {valid_length}"
        assert Q == len(quantiles), f"{Q} != {len(quantiles)}"
        # assert B <= batch_size

        # Inverse-scale ground truth                 # (B, [1..H])
        true_nation_GW = scaler_y_nation.inverse_transform(
            y_scaled[:, -valid_length:, 0].cpu().numpy().reshape(-1, 1)
        ).reshape(B, V)

        # Inverse-scale predictions
        pred_nation_GW = scaler_y_nation.inverse_transform(
            pred_nation_scaled_cpu.reshape(-1, Q)
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
                    "y_true"      : true_nation_GW[sample, h],  # starts at 12:30
                }

                for qi, tau in enumerate(quantiles):
                    row[f"q{int(100*tau)}"] = pred_nation_GW[sample, h, qi] # starts at 12:30

                # Baselines at the same target time
                for nm in baseline_names:
                    col    = cols_features.index(f"consumption_{nm}")
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


def compare_models(true_series:     pd.Series,
                   dict_pred_series:        Dict[str, pd.Series],
                   dict_preds_ML:  Optional[Dict[str, pd.Series]] = None,
                   dict_preds_meta:Optional[Dict[str, pd.Series]] = None,
                   subset:   str   = "", unit: str = "",
                   max_RMSE: float = 4,
                   verbose:  int   = 0) -> pd.DataFrame:
    # if verbose < 1: return  # this function does nothing if it cannot display

    def rmse(a,b): return round(np.sqrt(np.mean((a-b)**2)),4)
    def mae (a,b): return round(np.mean(np.abs  (a-b)),    4)

    # print("weights_meta:", weights_meta)

    if verbose >= 3:
        print("shapes:")
        print(f"true: {true_series.shape} ({true_series.index.min()}"
              f" -> {true_series.index.max()})")
        print(f"nn:   {dict_pred_series.get('q50').shape} "
              f"({dict_pred_series.get('q50').index.min()} -> "
              f"{dict_pred_series.get('q50').index.max()})")
        if dict_preds_meta is not None:
            for (_name, _pred_meta) in dict_preds_meta.items():
                print(f"{_name}: {_pred_meta.shape}")
                #" ({pred_meta.index.min()} -> {pred_meta.index.max()})")


    df_eval = pd.DataFrame({"true": true_series})

    if dict_pred_series is not None:
        df_eval['NNTQ'] = dict_pred_series.get("q50")

    # baselines (LR, RF)
    if dict_preds_ML is not None:
        for (_name, _pred_ML) in dict_preds_ML.items():
            if verbose >= 3:
                print(f"{_name}:   {dict_preds_ML.get(_name).shape} "
                      f"({dict_preds_ML.get(_name).index.min()} -> "
                      f"{dict_preds_ML.get(_name).index.max()})")
            df_eval[_name] = dict_preds_ML.get(_name)
        # print("list_baselines:", list_baselines)

    df_eval.dropna(inplace=True)

    # print("pred_meta:", pred_meta)
    if dict_preds_meta is not None:
        for (_name, _pred_meta) in dict_preds_meta.items():
            df_eval[f"meta {_name}"] = _pred_meta.reindex(df_eval.index)

    y_true = df_eval['true']


    rows = []
    for name in df_eval.columns:
        if name != 'true':
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
        print(df_metrics.round(2))
        plots.metrics(df_metrics, subset, max_RMSE)


    if verbose >= 3:
        subset_str = "" if unit is None else f" {subset}"
        unit_str   = "" if unit is None else f" [{unit}]"
        print(f"\n[Diagnostics]{subset_str} RMSE by hour of day{unit_str}")

        df_eval['hour'] = df_eval.index.hour

        df_rmse_hour = pd.DataFrame({
            name: (
                df_eval
                .groupby('hour')
                .apply(lambda d: rmse(d[name], d["true"]), include_groups=False)
            )
            for name in df_eval.columns if name not in ['true', 'hour']
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
            for name in df_eval.columns if name not in ['true', 'month']
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
        train_loss: float, valid_loss: float,
        list_of_min_losses: List[float], list_of_lists: List[List[float]],
        # constants
        num_epochs: int, display_every: int, plot_conv_every: int,
        min_delta: float, verbose:int=0)\
            -> [[float, float, float],
                [List[float], List[float], List[float], List[float]]]:

    (min_train_loss,  min_valid_loss, min_loss_display) = \
        list_of_min_losses

    (list_train_loss, list_min_train_loss,
    list_valid_loss, list_min_valid_loss) = list_of_lists

    if ((epoch+1) % display_every == 0) | (epoch == 0):
        # comparing latest loss to lowest so far
        if valid_loss <= min_loss_display - min_delta:
            is_better = '**'
        elif valid_loss <= min_loss_display:
            is_better = '*'
        else:
            is_better = ''

        min_loss_display = min_valid_loss

        t_epoch = time.perf_counter() - t_epoch_start

        if verbose >= 1:
            print(f"{epoch+1:3n} /{num_epochs:3n} ={(epoch+1)/num_epochs*100:3.0f}%,"
                  f"{t_epoch/60*(num_epochs/(epoch+1)-1)+.5:3.0f} min left, "
                  f"loss (1e-3): "
                  f"train{train_loss*1000:5.0f} (best{min_train_loss*1000:5.0f}), "
                  f"valid{valid_loss*1000:5.0f} ({    min_valid_loss*1000:5.0f})"
                  f" {is_better}")

    min_train_loss = min(min_train_loss, train_loss)
    list_train_loss    .append(train_loss)
    list_min_train_loss.append(min_train_loss)

    min_valid_loss = min(min_valid_loss, valid_loss)
    list_valid_loss    .append(valid_loss)
    list_min_valid_loss.append(min_valid_loss)


    # if ((epoch+1 == plot_conv_every) | ((epoch+1) % plot_conv_every == 0))\
    #         & (epoch < num_epochs-2):
    #     plots.convergence(list_train_loss, list_min_train_loss,
    #                       list_valid_loss, list_min_valid_loss,
    #                       baseline_losses,
    #                       None, None, None, None,
    #                       # list_meta_train_loss, list_meta_min_train_loss,
    #                       # list_meta_valid_loss, list_meta_min_valid_loss,
    #                       partial=True, verbose=verbose)

    return ((min_train_loss, min_valid_loss, min_loss_display),
            # list of lists
            (list_train_loss, list_min_train_loss,
            list_valid_loss, list_min_valid_loss)
            )



# -------------------------------------------------------
# Diagnosing outliers
# -------------------------------------------------------

def worst_days_by_loss(
    split      : str,
    y_true     : np.ndarray,   # shape (T,)
    y_pred     : np.ndarray,   # shape (T,)
    temperature: np.ndarray,
    holidays   : np.ndarray,
    num_steps_per_day: int,
    top_n      : int,
    verbose    : int = 0,
) -> (pd.DataFrame, float):
    """
    Returns DataFrame of worst days by mean loss
    """

    if verbose > 0:
        print(f"\nWorst days ({split})")

    diff   = y_pred - y_true
    diff_pc= diff / y_pred * 100
    # ts_loss = np.maximum(0.5 * diff, -0.5 * diff)  # 0.5: median

    df_aligned =pd.concat([diff,  diff_pc,  temperature,  holidays],
                          axis=1, join='inner').astype(np.float32)
    df_aligned.columns = ['diff','diff_pc','temperature','holidays']

    df = pd.DataFrame({
        'date':     df_aligned.index.normalize(),  # midnight per day
        'diff':     df_aligned['diff']       .astype(np.float32).round(2),
        'abs_diff': df_aligned['diff'].abs() .astype(np.float32).round(2),
        'diff_pc':  df_aligned['diff_pc']    .astype(np.float32).round(2),
        'Tavg_degC':df_aligned['temperature'].astype(np.float32),
        'holiday':  df_aligned['holidays']   .astype(np.int16),
    })

    daily = (
        df.groupby('date', as_index=False)
          .agg(
              diff     = ('diff',      "mean"),
              abs_diff = ('abs_diff',  "mean"),  # for sorting only
              diff_pc  = ('diff_pc',   "mean"),
              max_diff = ('abs_diff',  "max" ),
              ramp     = ('diff', lambda x: np.max(np.abs(np.diff(x)))
                             if len(x) > 1 else np.nan),
              n_points = ('diff',      "size"),
              Tavg_degC= ('Tavg_degC', "mean"),
              holiday  = ('holiday',   "mean"),
          )
          .sort_values('abs_diff', ascending=False)
    )

    daily = daily[daily['n_points'] == num_steps_per_day]  # incommensurable

    avg_abs_diff = float(daily['abs_diff'].mean())

    daily['day_name'] = daily['date'].dt.day_name()
    daily['day'     ] = daily['date'].dt.day
    daily['month'   ] = daily['date'].dt.month
    daily['year'    ] = daily['date'].dt.year

    daily['diff'    ] = daily['diff'     ].astype(np.float32).round(1)
    daily['diff_pc' ] = daily['diff_pc'  ].astype(np.float32).round(1)
    daily['max_diff'] = daily['max_diff' ].astype(np.float32).round(1)
    daily['ramp'    ] = daily['ramp'     ].astype(np.float32).round(2)

    daily['Tavg_degC']= daily['Tavg_degC'].astype(np.float32).round(1)
    daily['holiday' ] = daily['holiday'  ].astype(np.int16)


    daily = daily[['day_name', 'day', 'month', # 'year', 'holiday',
                   'Tavg_degC', 'diff', 'diff_pc', # 'max_diff',
                   'ramp']].head(top_n)


    # plots
    if verbose > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=daily, x='month', bins=12, discrete=True)
        plt.title('Histogram of Months for Bad Days')
        plt.xlabel('Month')
        plt.ylabel('Frequency')
        plt.xticks(range(1, 13))
        plt.show()

        # plt.figure(figsize=(10, 6))
        # sns.histplot(data=daily, x='year',bins=len(daily['year'].unique()),discrete=True)
        # plt.title('Histogram of Years for Bad Days')
        # plt.xlabel('Year')
        # plt.ylabel('Frequency')
        # plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=daily, x='Tavg_degC', bins=20, kde=True)
        plt.title('Histogram of Average Temperature for Bad Days')
        plt.xlabel('Average Temperature (°C)')
        plt.ylabel('Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=daily, x='day_name', shrink=0.8)
        plt.title('Histogram of Days of the Week for Bad Days')
        plt.xlabel('Day of the Week')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.show()


    return daily, avg_abs_diff


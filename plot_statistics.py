# import sys

from   typing   import Dict, Optional, Tuple, Sequence, List  # , Any,

import numpy  as np
import pandas as pd

# import matplotlib.pyplot as plt


import plots  # containers




def apply_threshold(df            : pd.DataFrame,
                    threshold_degC: float,
                    direction     : str) -> pd.DataFrame:
    if threshold_degC is not None:
        if direction == '<=':
            df = df[df['T_degC'] <= threshold_degC]
        elif direction == '>=':
            df = df[df['T_degC'] >= threshold_degC]
        elif direction in ['==', '=']:
            df = df[(df['T_degC'] - threshold_degC).abs() < 2.]
        else:
            raise ValueError(f"`{direction}` not a valid direction")

    return df


def create_df(
    true_series         : pd.Series,
    dict_pred_series    : Dict[str, pd.Series],
    dict_baseline_series: Dict[str, pd.Series],
    dict_meta_series    : Dict[str, pd.Series],
    T_degC              : pd.Series,
    dates               : pd.DatetimeIndex,
    threshold_degC      : Optional[Tuple[str, float]]) -> pd.DataFrame:

    df = pd.concat([true_series] + list(dict_pred_series.values()) + \
                   list(dict_baseline_series.values()) + \
                   list(dict_meta_series.values()) + \
                   [pd.Series(T_degC,index=dates, name='T_degC')],
                   axis=1, join='inner').dropna()
    columns_preds = ['true'] + ["NNTQ " + e for e in dict_pred_series.keys()] + \
                   list(dict_baseline_series.keys()) + \
                   ["meta " + e for e in dict_meta_series.keys()]
    df.columns = columns_preds + ['T_degC']

    if threshold_degC is not None:
        df = apply_threshold(df, threshold_degC[1], threshold_degC[0])

    return (df, columns_preds)



def time_of_day_temp_sensitivity(
    true_series         : pd.Series,
    dict_pred_series    : Dict[str, pd.Series],
    dict_baseline_series: Dict[str, pd.Series],
    dict_meta_series    : Dict[str, pd.Series],
    T_degC              : pd.Series,
    dates               : pd.DatetimeIndex,
    threshold_degC      : Optional[Tuple[str, float]],
    num_steps_per_day   : int
) -> [pd.Series, float, float]:
    """
    Time-of-day–dependent temperature sensitivity:
    d E[y_pred | T, hod] / dT

    Parameters
    ----------
    y     : pd.Series
        consumption
    T_avg : pd.Series
        Average temperature
    num_steps_per_day : int
        Number of time bins per day (48 = half-hourly)

    Returns
    -------
    pd.Series
        Temperature sensitivity per time-of-day bin (GW / K)
    """
    (df, columns_preds) = create_df(
        true_series, dict_pred_series, dict_baseline_series,
        dict_meta_series, T_degC, dates, threshold_degC)

    # Map timestamps to time-of-day bins
    hod = (df.index.hour  * num_steps_per_day //   24 +
           df.index.minute* num_steps_per_day // 1440)
    df["hod"] = hod.astype(int)

    rows = dict()
    sensitivity_GW_per_K = dict()

    for _col in columns_preds:
        cov = np.cov(df["T_degC"], df[_col], bias=True)[0, 1]
        var = df["T_degC"].var()
        sensitivity_GW_per_K[_col] = round(float(cov / var), 3)

        slopes = []
        for h in range(num_steps_per_day):
            sub = df[df["hod"] == h]

            # Not enough variation -> undefined slope
            if len(sub) < 10 or sub["T_degC"].var() == 0:
                slopes.append(np.nan)
                continue

            cov = np.cov(sub["T_degC"], sub[_col], bias=True)[0, 1]
            var = sub["T_degC"].var()
            slopes.append(round(float(cov / var), 3))

        rows[_col] = slopes

    return pd.DataFrame(rows, index=np.arange(24, step=24/num_steps_per_day)), \
        sensitivity_GW_per_K, len(df) / num_steps_per_day


def thermosensitivity_per_time_of_day(
     data_split,  # : containers.DataSplit,
     thresholds_degC  : List[Tuple[str, float]],  # eg [('<=', 10), ('<=', 4), ('>=', 23)]
     num_steps_per_day: int,
     ylim             : [float, float] = [0, 3]
)   -> None:

    for _threshold_degC in thresholds_degC:
        sensitivity_df, sensitivity_GW_per_K, num_days = \
            time_of_day_temp_sensitivity(
                data_split.true_nation_GW, data_split.dict_preds_NNTQ,
                data_split.dict_preds_ML, data_split.dict_preds_meta,
                data_split.Tavg_degC.round(1), data_split.dates,
                threshold_degC=_threshold_degC,
                num_steps_per_day=num_steps_per_day)
        threshold_str = f"T_avg {_threshold_degC[0]}{_threshold_degC[1]:3n} °C"
        # print(f"{num_days:5.1f} test days with {threshold_str}")
              # f"{sensitivity_GW_per_K} GW/K")

        _sign = (-1) ** (_threshold_degC[0] == '<=')
        plots.curves(
             sensitivity_df['true'] * _sign,
            {_col: sensitivity_df['NNTQ '+_col]*_sign for _col in ['q10', 'q50', 'q90']} \
                if data_split.name != 'complete' else None,
            None,  # {_col: sensitivity_df[        _col]*_sign for _col in ['LR', 'RF', 'LGBM']},
            {_col: sensitivity_df['meta '+_col]*_sign for _col in ['LR', 'NN']} \
                if data_split.name == 'test' else None,
             xlabel="time of day [UTC]", ylabel="thermosensitivity [GW/K]",
             title=f"{data_split.name_display}, {round(num_days):n} days with {threshold_str}",
             ylim=ylim, date_range=None, moving_average=None, groupby=None)



def threshold_temp_sensitivity(
    true_series         : pd.Series,
    dict_pred_series    : Dict[str, pd.Series],
    dict_baseline_series: Dict[str, pd.Series],
    dict_meta_series    : Dict[str, pd.Series],
    T_degC              : pd.Series,
    dates               : pd.DatetimeIndex,
    thresholds_degC     : Sequence[float],
    direction           : str,  # '<=', '>=' or '=='
    num_steps_per_day   : int
) -> pd.DataFrame:
    """
    Time-of-day–dependent temperature sensitivity:
    d E[y_pred | T, hod] / dT

    Parameters
    ----------
    y     : pd.Series
        consumption
    T_avg : pd.Series
        Average temperature
    num_steps_per_day : int
        Number of time bins per day (48 = half-hourly)

    Returns
    -------
    pd.Series
        Temperature sensitivity per time-of-day bin (GW / K)
    """
    (df, columns_preds) = create_df(
        true_series, dict_pred_series, dict_baseline_series,
        dict_meta_series, T_degC, dates, threshold_degC=None)

    # print(columns_preds)
    # print(df.shape)

    rows = dict()
    for _T in thresholds_degC:
        _df = apply_threshold(df, _T, direction)

        # Not enough variation -> undefined slope
        if len(_df) < 10*num_steps_per_day or _df["T_degC"].var() == 0:
            rows[_T] = [np.nan] * len(columns_preds)
            continue

        slopes = []
        for _col in columns_preds:
            cov = np.cov(_df["T_degC"], _df[_col], bias=True)[0, 1]
            var = _df["T_degC"].var()
            slopes.append(round(float(cov / var), 3))
        rows[_T] = slopes

    _df = pd.DataFrame(rows).T
    _df.columns = columns_preds

    return _df

def thermosensitivity_per_temperature(
     data_split,  # : containers.DataSplit,
     thresholds_degC  : Sequence[float],
     num_steps_per_day: int,
     ylim             : [float, float] = [-3, 3]
     )   -> None:
    sensitivity_df = threshold_temp_sensitivity(
            data_split.true_nation_GW, data_split.dict_preds_NNTQ,
            data_split.dict_preds_ML, data_split.dict_preds_meta,
            data_split.Tavg_degC.round(1), data_split.dates,
            thresholds_degC=thresholds_degC, direction='==',   # _direction
            num_steps_per_day=num_steps_per_day)

    plots.curves(
         sensitivity_df['true'],
        {_col: sensitivity_df['NNTQ '+_col] for _col in ['q10', 'q50', 'q90']} \
            if data_split.name != 'complete' else None,
        None,  #{_col: sensitivity_df[_col] for _col in ['LR', 'RF', 'LGBM']},
        {_col: sensitivity_df['meta '+_col] for _col in ['LR', 'NN']} \
            if data_split.name == 'test' else None,
         xlabel="threshold T_avg [°C]",
         ylabel="thermosensitivity [GW/K]",
         title=f"{data_split.name_display}",
             ylim=ylim, date_range=None, moving_average=7, groupby=None)




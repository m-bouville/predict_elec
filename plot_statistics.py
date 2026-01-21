import sys

from   typing   import Dict, Optional, Tuple, Sequence, List  # , Any,

import numpy  as np
import pandas as pd

from   sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


import plots  # containers
from   constants   import Split




# -------------------------------------------------------
# thermosensitivity by time of day or by T°
# -------------------------------------------------------

def drift_with_time(
     consumption      : pd.Series,
     temperature      : pd.Series,
     num_steps_per_day: int,
     )   -> None:

    # print("consumption:", consumption)
    # print("temperature:", temperature)

    (start_date, end_date) = [consumption.index[ 0] + pd.DateOffset(months=12), \
                              consumption.index[-1] - pd.DateOffset(months= 7)]
    moving_average=365
    print("date_range:", [start_date, end_date])
    print("moving_average:", moving_average, "days")


    _consumption  = consumption.resample('D').mean()
    _temperature_sat = (temperature.clip(upper=15) - 15)
    # _temperature  = (temperature.rolling(moving_average, center=True) \
    #                  .mean()).loc[start_date:end_date]

    # thermosensitive model
    common_indices = _consumption.dropna().index.intersection(
        _temperature_sat.dropna().index)
    _temperature_common = _temperature_sat.reindex(common_indices)
    _consumption_common = _consumption    .reindex(common_indices)
    _year = pd.Series(common_indices.year - 2020 + common_indices.dayofyear/365,
                      index = common_indices, name="year")  # for long-term drift


    # linear regression to quantify thermosensitivity in winter
    model_LR = LinearRegression()
    model_LR.fit(pd.concat([_temperature_common, _year], axis=1),
                 _consumption_common)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model = f"{model_LR.intercept_:.1f} GW  " \
                     f"{model_LR.coef_[0]:.2f} * min(T, 15 °C)  " \
                     f"{model_LR.coef_[1]:.2f} * (year - 2020)"
    print(_formula_model)

    conso_model =  _temperature_common * model_LR.coef_[0] + \
                    _year * model_LR.coef_[1] + model_LR.intercept_

    _consumption  = (consumption.resample('D').mean()
                     .rolling(moving_average, center=True) \
                     .mean()).loc[start_date:end_date]

    _conso_model  = (conso_model.rolling(moving_average, center=True) \
                     .mean()).loc[start_date:end_date]
    # _temperature_sat = ((temperature.clip(upper=15) - 15). \
    #                     rolling(moving_average, center=True) \
    #                  .mean()).loc[start_date:end_date]
    # _temperature  = (temperature.rolling(moving_average, center=True) \
    #                  .mean()).loc[start_date:end_date]

    plt.figure(figsize=(10,6))
    # print("_consumption:", _consumption)
    plt.plot(_consumption.index, _consumption.values,
             label="real", color="red")
    plt.plot(_conso_model.index, _conso_model.values,
             label=f"model: {_formula_model}", color="blue")
    plt.ylabel("consumption [GW], annual moving average")

    # ax2 = plt.twinx()
    # ax2.plot(_temperature.index, _temperature.values,
    #          label="temperature", color="grey")
    # ax2.set_ylabel("temperature [°C], annual moving average")


    # if ylim   is not None:  plt.ylim  (ylim)
    plt.xlabel("date")
    plt.legend()
    # if title  is not None:  plt.title (title)

    # lines1, labels1 = plt.gca().get_legend_handles_labels()
    # lines2, labels2 = ax2      .get_legend_handles_labels()
    # plt.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.show()


    # print(_consumption.index.has_duplicates)
    # print(_temperature.index.has_duplicates)



    # plt.figure(figsize=(10,6))
    # plt.scatter(_temperature_common, _consumption_common, s=300, alpha=0.3)
    # plt.scatter(_temperature_common, conso_model, s=300, alpha=0.3)
    # plt.xlabel("temperature [°C], annual moving average")
    # plt.ylabel("consumption [GW], annual moving average")
    # plt.show()





    plt.figure(figsize=(10,6))
    # print("_consumption:", _consumption)
    plt.plot(_consumption.index, (_consumption - _conso_model).values, color="blue")
    plt.hlines(0, common_indices[0], common_indices[-1], color="black")
    plt.xlabel("date")
    plt.ylabel("consumption residual [GW], annual moving average")
    plt.show()



    sys.exit()




# -------------------------------------------------------
# thermosensitivity by région
# -------------------------------------------------------

def thermosensitivity_regions(df_consumption       : pd.DataFrame,
                              df_temperature       : pd.DataFrame,
                              threshold_winter_degC: float = 15.,
                              threshold_summer_degC: float = 20.) -> None:
    df_consumption.drop(columns=['year', 'month', 'dateofyear', 'timeofday'],
                        inplace=True)
    df_consumption.columns = [_col.split("_")[1]
                              for _col in list(df_consumption.columns)]
    df_consumption = df_consumption.resample('D').mean().dropna()  # half-hour -> day

    df_temperature = df_temperature.dropna()

    assert set(df_consumption.columns) == set(df_temperature.columns), \
        f"{df_consumption.columns} != {df_temperature.columns}"

    # print("df_consumption:", df_consumption.shape, list(df_consumption.columns))
    # print("df_temperature:", df_temperature.shape, list(df_temperature.columns))

    # Reindex both DataFrames to keep only common indices
    common_indices = df_consumption.index.intersection(df_temperature.index)
    df_consumption = df_consumption.reindex(common_indices)
    df_temperature = df_temperature.reindex(common_indices)

    # print("common shape:", df_consumption.shape)
    # print("df_consumption:", df_consumption.shape, list(df_consumption.columns))
    # print("df_temperature:", df_temperature.shape, list(df_temperature.columns))


    # national
    _avg_conso = df_consumption.mean().mean()
    plt.scatter(df_temperature.mean(axis=1),
                df_consumption.mean(axis=1) / _avg_conso,
                s=10, alpha=0.15)
    plt.xlim(-2, 27)
    plt.ylim(0.6, 1.7)
    plt.xlabel("average daily temperature [°C]")
    plt.ylabel("national consumption / its average")
    plt.show()


    # per région

    for _col in list(df_consumption.columns):
        _avg_conso = df_consumption[_col].mean()
        _consumption_norm = df_consumption[_col]/_avg_conso
        plt.scatter(df_temperature[_col].rolling(7, min_periods=6).mean(),
                    _consumption_norm   .rolling(7, min_periods=6).mean(),
                    label=_col, s=10, alpha=0.05)

    plt.legend(loc='upper right')
    plt.xlim(-5, 30)
    plt.ylim(0.6, 1.8)
    plt.xlabel("average daily temperature [°C]")
    plt.ylabel("consumption / its average")
    plt.show()

    list_regions_plots = ['Occi.', 'HdF']


    # per région, winter
    _slopes_winter_pc = pd.Series()

    for _col in list(df_consumption.columns):
        _avg_conso = df_consumption[_col].mean()
        _consumption_norm = df_consumption[_col]/_avg_conso

        # saturating T°
        _temp_sat_winter = df_temperature[_col].clip(upper=threshold_winter_degC)
        _conso_winter   = _consumption_norm[_temp_sat_winter < threshold_winter_degC]
        _temp_sat_winter= _temp_sat_winter [_temp_sat_winter < threshold_winter_degC]

        # linear regression to quantify thermosensitivity in winter
        model_winter = LinearRegression()
        model_winter.fit(_temp_sat_winter.to_frame(), _conso_winter)
        _slopes_winter_pc[_col] = -round(float(model_winter.coef_[0]) * 100, 2)

        if _col in list_regions_plots:
            plt.scatter(_temp_sat_winter, _conso_winter, label=_col, s=30, alpha=0.3)

    plt.legend(loc='lower left')
    plt.xlim(-6, threshold_winter_degC)
    plt.ylim(0.6, 1.9)
    plt.xlabel(f"average daily temperature < {threshold_winter_degC:n} °C")
    plt.ylabel("consumption / its average")
    plt.show()


    # per région, summer
    _slopes_summer_pc = pd.Series()

    for _col in list(df_consumption.columns):
        _avg_conso = df_consumption[_col].mean()
        _consumption_norm = df_consumption[_col]/_avg_conso

        # saturating T°
        _temp_sat_summer= df_temperature[_col].clip(lower=threshold_summer_degC)
        _conso_summer   = _consumption_norm[_temp_sat_summer > threshold_summer_degC]
        _temp_sat_summer= _temp_sat_summer [_temp_sat_summer > threshold_summer_degC]

        # linear regression to quantify thermosensitivity in summer
        model_summer = LinearRegression()
        model_summer.fit(_temp_sat_summer.to_frame(), _conso_summer)
        _slopes_summer_pc[_col]  = round(float(model_summer.coef_[0]) * 100, 2)

        if _col in list_regions_plots:
            plt.scatter(_temp_sat_summer, _conso_summer, label=_col, s=30, alpha=0.5)

    plt.legend(loc='lower right')
    plt.xlim(threshold_summer_degC, 31)
    plt.ylim(0.6, 1.1)
    plt.xlabel(f"average daily temperature > {threshold_summer_degC:n} °C")
    plt.ylabel("consumption / its average")
    plt.show()


    _df_slopes_pc = pd.concat([_slopes_winter_pc.T, _slopes_summer_pc.T], axis=1)
    _df_slopes_pc.columns = ["winter", "summer"]
    print(_df_slopes_pc.T.to_string())


    # plot summer f° winter
    plt.scatter(_df_slopes_pc['winter'], _df_slopes_pc['summer'], s=100)

    # Annotate each point with its key
    for _col in _df_slopes_pc.index:
        plt.annotate(_col, (_df_slopes_pc['winter'].loc[_col] + 0.05,
                            _df_slopes_pc['summer'].loc[_col]))

    plt.xlabel(f"winter (T_avg < {threshold_winter_degC:n} °C)")
    plt.ylabel(f"summer (T_avg > {threshold_summer_degC:n} °C)")
    if plt.ylim()[0] > 0:
        plt.ylim(bottom=0)
    plt.ylim(top=plt.ylim()[1]+0.05)
    # plt.xlim(top=plt.xlim()[1]+0.05)
    plt.title("Seasonal thermosensitivities [% avg demand per K]")




# -------------------------------------------------------
# thermosensitivity by time of day or by T°
# -------------------------------------------------------

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

    # print("true_series:", true_series.shape)
    # print(pd.DataFrame(dict_pred_series    ).shape)
    # print(pd.DataFrame(dict_baseline_series).shape)
    # print(pd.DataFrame(dict_meta_series    ).shape)
    # print(T_degC.shape)
    # print(dates.shape)

    df = pd.concat([true_series,
                   pd.DataFrame(dict_pred_series),
                   pd.DataFrame(dict_baseline_series),
                   pd.DataFrame(dict_meta_series),
                   pd.Series(T_degC, index=dates, name='T_degC')],
                   axis=1).dropna()

    df.index = pd.to_datetime(df.index)
    columns_preds = ['true'] + ["NNTQ " + e for e in dict_pred_series.keys()] + \
                   list(dict_baseline_series.keys()) + \
                   ["meta " + e for e in dict_meta_series.keys()]
    df.columns = columns_preds + ['T_degC']

    if threshold_degC is not None:
        df = apply_threshold(df, threshold_degC[1], threshold_degC[0])
    # print(df)

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
                data_split.dict_preds_ML,  data_split.dict_preds_meta,
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
                if data_split.name != Split.complete else None,
            None,  # {_col: sensitivity_df[        _col]*_sign for _col in ['LR', 'RF', 'LGBM']},
            {_col: sensitivity_df['meta '+_col]*_sign for _col in ['LR', 'NN']} \
                if data_split.name == Split.test else None,
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
            if data_split.name != Split.complete else None,
        None,  #{_col: sensitivity_df[_col] for _col in ['LR', 'RF', 'LGBM']},
        {_col: sensitivity_df['meta '+_col] for _col in ['LR', 'NN']} \
            if data_split.name == Split.test else None,
         xlabel="threshold T_avg [°C]",
         ylabel="thermosensitivity [GW/K]",
         title=f"{data_split.name_display}",
         ylim=ylim, date_range=None, moving_average=7, groupby=None)




import sys

from   typing   import Dict, Optional, Tuple, Sequence, List  # , Any,

import numpy  as np
import pandas as pd

from   sklearn.linear_model import LinearRegression, HuberRegressor

import matplotlib.pyplot as plt


import plots  # containers
from   constants   import Split



year_ref = 2021




# -------------------------------------------------------
# drift over time
# -------------------------------------------------------

def drift_with_time(
     consumption      : pd.Series,
     temperature      : pd.Series,
     num_steps_per_day: int,
     )   -> None:

    # print("consumption:", consumption)
    # print("temperature:", temperature)

    moving_average=365
    print("moving_average:", moving_average, "days")


    _consumption = consumption.resample('D').mean().dropna()
    _temperature = temperature.resample('D').mean().dropna()


    # temperature
    # ------------------


    _temperature_annual = temperature.rolling(365, min_periods=365, center=True) \
                     .mean().dropna()
    _year_annual = pd.Series(_temperature_annual.index.year - year_ref + _temperature_annual.index.dayofyear/365,
                      index = _temperature_annual.index, name="year")  # for long-term drift

    # linear regression to quantify thermosensitivity in winter
    model_T = HuberRegressor(
                epsilon=2.,  # outlier threshold, in the range [1, inf), def 1.35
                alpha  =0.   # no regularization unless needed
              )
    model_T.fit(_year_annual.to_frame(), _temperature_annual)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model_T = f"{model_T.intercept_:.2f} °C + " \
                       f"{model_T.coef_[0]:.2f} * (year - {year_ref})"
    print(f"T [R² ={model_T.score(_year_annual.to_frame(), _temperature_annual)*100:3.0f}%] = "
          f"{_formula_model_T}")


    T_model = _year_annual * model_T.coef_[0] + model_T.intercept_
    T_model            = (T_model            .rolling(30, center=True).mean())
    _temperature_annual= (_temperature_annual.rolling(30, center=True).mean())

    plt.figure(figsize=(10,6))
    plt.plot(_temperature_annual.index, _temperature_annual.values,
             label="real", color="black")
    plt.plot(T_model.index, T_model.values,
             label=f"model: {_formula_model_T}", color="red")
    plt.ylabel("temperature [°C], annual moving average")
    plt.xlabel("year")
    plt.legend()
    plt.show()



    # consumption
    # ------------------

    # _temperature_sat = (temperature.clip(upper=15) - 15)
    # _temperature  = (temperature.rolling(moving_average, center=True) \
    #                  .mean()).loc[start_date:end_date]

    _temperature_sat_annual = ((temperature.clip(upper=15) - 15).rolling(365, min_periods=365, center=True) \
                     .mean()).dropna()
    _consumption_annual = (_consumption.rolling(365, min_periods=365, center=True) \
                     .mean()).dropna()
    _year_annual = pd.Series(_consumption_annual.index.year - year_ref + _consumption_annual.index.dayofyear/365,
                      index = _consumption_annual.index, name="year")  # for long-term drift



    # model w/o T°
    model_no_T = LinearRegression()
    model_no_T.fit(_year_annual.to_frame(), _consumption_annual)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model_no_T = f"{model_no_T.intercept_:.1f} GW  " \
                     f"{model_no_T.coef_[0]:.2f} * (year - {year_ref})"
    print(f"consumption [R² ={model_no_T.score(_year_annual.to_frame(), _consumption_annual)*100:3.0f}%] = "
          f"{_formula_model_no_T}")

    conso_model_no_T = _year_annual * model_no_T.coef_[0] + model_no_T.intercept_

    # _consumption  = (consumption.resample('D').mean()
    #                  .rolling(moving_average, center=True) \
    #                  .mean()).loc[start_date:end_date]




    # thermosensitive model
    common_indices = _consumption_annual.dropna().index.intersection(
        _temperature_sat_annual.dropna().index)
    _temperature_common = _temperature_sat_annual.reindex(common_indices)
    _consumption_common = _consumption_annual    .reindex(common_indices)
    _year_common = pd.Series(common_indices.year - year_ref + common_indices.dayofyear/365,
                      index = common_indices, name="year")  # for long-term drift


    # linear regression to quantify thermosensitivity in winter
    model_with_T = LinearRegression()
    X = pd.concat([_temperature_common, _year_common], axis=1)
    model_with_T.fit(X, _consumption_common)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model_with_T = f"{model_with_T.intercept_:.1f} GW  " \
                     f"{model_with_T.coef_[1]:.2f} * (year - {year_ref}) " \
                     f"{model_with_T.coef_[0]:.2f} * min(T-15 °C, 0)"
    print(f"consumption [R² ={model_with_T.score(X, _consumption_common)*100:3.0f}%] = "
          f"{_formula_model_with_T}")

    conso_model_with_T =  _temperature_common * model_with_T.coef_[0] + \
                    _year_common * model_with_T.coef_[1] + model_with_T.intercept_


    # plotting
    _consumption_annual= _consumption_annual.rolling(30, center=True).mean()
    _conso_model_no_T  = conso_model_no_T   .rolling(30, center=True).mean()
    _conso_model_with_T= conso_model_with_T .rolling(30, center=True).mean()

    plt.figure(figsize=(10,6))
    plt.plot(_consumption_annual.index, _consumption_annual.values,
             label="real", color="black")
    plt.plot(_conso_model_no_T.index, _conso_model_no_T.values,
             label=f"model: {_formula_model_no_T}", color="green")
    plt.plot(_conso_model_with_T.index, _conso_model_with_T.values,
             label=f"model: {_formula_model_with_T}", color="red")
    plt.ylabel("consumption [GW], annual moving average")
    plt.xlabel("year")
    plt.legend()
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
    plt.plot(_consumption_annual.index, (_conso_model_no_T   - _consumption_annual).values,
             label=f"model: {_formula_model_no_T}",  color="green")
    plt.plot(_consumption_annual.index, (_conso_model_with_T - _consumption_annual).values,
             label=f"model: {_formula_model_with_T}",color="red")
    plt.hlines(0, _consumption_annual.index[0], _consumption_annual.index[-1], color="black")
    plt.xlabel("year")
    plt.ylabel("consumption residual [GW], annual moving average")
    plt.legend()
    plt.show()





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
    _slopes_winter_pc = pd.Series(dtype=float)

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
    _slopes_summer_pc = pd.Series(dtype=float)

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
                    name_col      : str,
                    threshold     : float,
                    direction     : str,
                    width         : Optional[float]) -> pd.DataFrame:
    if name_col not in ['index', 'date']:
        _col = df[name_col]
    else:
        _col = df.index

    if threshold is not None:
        # print("threshold:", threshold)
        if direction == '<=':
            df = df[_col <= threshold]
        elif direction == '>=':
            df = df[_col >= threshold]
        elif direction in ['==', '=']:
            df = df[np.abs(_col - threshold) < width]

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


    list_df_to_concat = [true_series]

    # add dicts if not empty
    if dict_pred_series:
        list_df_to_concat.append(pd.DataFrame(dict_pred_series))
    if dict_baseline_series:
        list_df_to_concat.append(pd.DataFrame(dict_baseline_series))
    if dict_meta_series:
        list_df_to_concat.append(pd.DataFrame(dict_meta_series))

    # add T_degC
    list_df_to_concat.append(pd.Series(T_degC, index=dates, name='T_degC'))

    # Concaténation
    if len(list_df_to_concat) > 0:
        df = pd.concat(list_df_to_concat, axis=1).dropna()
    else:
        df = pd.DataFrame()

    # df = pd.concat([true_series,
    #                pd.DataFrame(dict_pred_series),
    #                pd.DataFrame(dict_baseline_series),
    #                pd.DataFrame(dict_meta_series),
    #                pd.Series(T_degC, index=dates, name='T_degC')],
    #                axis=1).dropna()

    df.index = pd.to_datetime(df.index)
    columns_preds = ['true'] + ["NNTQ " + e for e in dict_pred_series.keys()] + \
                   list(dict_baseline_series.keys()) + \
                   ["meta " + e for e in dict_meta_series.keys()]
    df.columns = columns_preds + ['T_degC']

    if threshold_degC is not None:
        df = apply_threshold(df, 'T_degC', threshold_degC[1], threshold_degC[0], 2)
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
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
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
        sensitivity_GW_per_K[_col] = \
            round(LinearRegression().fit(df[["T_degC"]], df[_col]).coef_[0], 3)

        slopes = []
        for h in range(num_steps_per_day):
            sub = df[df["hod"] == h]

            # Not enough variation -> undefined slope
            if len(sub) < 10 or sub["T_degC"].var() == 0:
                slopes.append(np.nan)
                continue

            slopes.append(round(LinearRegression().fit(
                sub[["T_degC"]], sub[_col]).coef_[0], 3))

        rows[_col] = slopes

    return pd.DataFrame(rows, index=np.arange(24, step=24/num_steps_per_day)), \
        sensitivity_GW_per_K, len(df) / num_steps_per_day


def thermosensitivity_per_time_of_day(
     data_split,  # : containers.DataSplit,
     thresholds_degC  : List[Tuple[str, float]],  # eg [('<=', 10), ('>=', 23)]
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


        # /!\ quantiles are meaningless
        #   they are slopes of quantiles, not quantiles of slopes

        print(list(sensitivity_df.columns))
        _sign = (-1) ** (_threshold_degC[0] == '<=')
        plots.curves(
             sensitivity_df['true'] * _sign,
            {_col: sensitivity_df['NNTQ '+_col]*_sign for _col in ['q50']},
                # if data_split.name != Split.complete else None,
            {_col: sensitivity_df[        _col]*_sign for _col in ['RF', 'LGBM']} \
                if data_split.name in [Split.train, Split.complete] else None,
            {_col: sensitivity_df['meta '+_col]*_sign for _col in ['LR', 'NN']} \
                if data_split.name in [Split.valid, Split.test] else None,
             xlabel="time of day [UTC]", ylabel="thermosensitivity [GW/K]",
             title=f"{data_split.name_display}, {round(num_days):n} days "
                   f"with {threshold_str}",
             ylim=ylim, date_range=None, moving_average=None, groupby=None)



def threshold_temp_sensitivity(
    true_series         : pd.Series,
    dict_pred_series    : Dict[str, pd.Series],
    dict_baseline_series: Dict[str, pd.Series],
    dict_meta_series    : Dict[str, pd.Series],
    T_degC              : pd.Series,
    dates               : pd.DatetimeIndex,
    name_col            : str,
    thresholds          : Sequence[float],
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
    # print(thresholds)
    # print(df.head())

    width = {'T_degC': 2, 'date': pd.Timedelta(days=6*30)}

    rows = dict()
    for _target in thresholds:
        _df = apply_threshold(df, name_col, _target, direction, width[name_col])

        # # Not enough variation -> undefined slope
        # if len(_df) < 10*num_steps_per_day or _df['T_degC'].var() == 0:
        #     rows[_target] = [np.nan] * len(columns_preds)
        #     continue

        slopes = []
        for _col in columns_preds:
            slopes.append(round(LinearRegression().fit(
                _df[['T_degC']], _df[_col]).coef_[0], 3))
            # print(_target, _col, slopes)

        rows[_target] = slopes

    _df = pd.DataFrame(rows).T
    _df.columns = columns_preds

    return _df


def thermosensitivity_per_temperature(
     consumption      : pd.Series,
     temperature      : pd.Series,
     thresholds_degC  : Sequence[float],
     num_steps_per_day: int,
     ylim             : [float, float] = [-3., 0.5]
     )   -> None:
    sensitivity_df = threshold_temp_sensitivity(
            consumption, {}, {}, {},
            temperature.round(1), consumption.index, name_col='T_degC',
            thresholds=thresholds_degC, direction='==',
            num_steps_per_day=num_steps_per_day)

    plots.curves(
         sensitivity_df['true'], None, None, None,
         xlabel="threshold T_avg [°C]",
         ylabel="thermosensitivity [GW/K]",
         title=None,
         ylim=ylim, date_range=None, moving_average=7, groupby=None)


def thermosensitivity_per_date(
     consumption      : pd.Series,
     temperature      : pd.Series,
     list_dates       : Sequence[pd.DatetimeIndex],
     num_steps_per_day: int,
     ylim             : [float, float] = [-1.65, -1.25],
     moving_average   = None
     )   -> None:

    year_ref = 2021

    _list_coef     = []
    _list_intercept= []

    for threshold_degC in np.arange(13, 16, 0.5):
        # _temperature = temperature.resample('D').mean().dropna()
        _temperature_sat_fit = (temperature.clip(upper=threshold_degC, )- threshold_degC).\
            resample('D').mean().dropna()
        _temperature_sat_fit = _temperature_sat_fit[_temperature_sat_fit<0]
        # _temperature_sat_plot= (temperature.clip(upper=13.5)- 13.5).resample('D').mean().dropna()
        _consumption = consumption.resample('D').mean().dropna(). \
                                                reindex(_temperature_sat_fit.index)

        common_indices = _consumption.dropna().index.intersection(
            _temperature_sat_fit.dropna().index)
        _temperature_common = _temperature_sat_fit.reindex(common_indices)
        _consumption_common = _consumption        .reindex(common_indices)
        _year_common = pd.Series(common_indices.year - year_ref + common_indices.dayofyear/365,
                          index = common_indices, name="year")  # for long-term drift



        # _year= pd.Series(_temperature_sat_fit.index.year - year_ref + \
        #                  _temperature_sat_fit.index.dayofyear/365,
        #                  index= _temperature_sat_fit.index, name="year")  # for long-term drift

        sensitivity_df = threshold_temp_sensitivity(
                _consumption, {}, {}, {},
                _temperature_common.round(1), _consumption_common.index, name_col='date',
                thresholds=list_dates, direction='==',
                num_steps_per_day=num_steps_per_day)

        _sensitivity = sensitivity_df['true'] #.resample('D').mean().dropna()


        # linear regression to quantify thermosensitivity in winter
        model = LinearRegression()
        X_year = pd.DataFrame(_sensitivity.index.year - year_ref + \
                              _sensitivity.index.dayofyear/365,
                                   index = _sensitivity.index
                              )
        model.fit(X_year, _sensitivity)
        # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
        # _formula_model = f"{model.intercept_:.2f} GW/K + " \
        #                  f"{model.coef_[0]:.3f} * (year - {year_ref})"
        # print(f"thermosensitivity "
        #       f"({sum(_temperature_sat_fit<0):4n} with T < {threshold_degC:4.1f} °C, "
        #       f"R² ={model.score(X_year, _sensitivity)*100:3.0f}%) = "
        #       f"{_formula_model}")

        _list_coef     = model.coef_[0]
        _list_intercept= model.intercept_


    # sensitivity_model = _year_common * np.mean(_list_coef) + np.mean(_list_intercept)

    _formula_model = f"{np.mean(_list_intercept):.2f} GW/K + " \
                     f"{np.mean(_list_coef):.3f} * (year - {year_ref})"
    print(f"thermosensitivity = {_formula_model}")

    # plt.figure(figsize=(10,6))
    # # print("_consumption:", _consumption)
    # plt.plot(_sensitivity.index, _sensitivity.rolling(
    #                         6, min_periods=5, center=True).mean().values,
    #          label="actual",  color="black")
    # plt.plot(sensitivity_model.index, sensitivity_model.values,
    #          label=f"model: {_formula_model}",color="red")
    # # plt.hlines(0, _consumption.index[0], _consumption.index[-1], color="black")
    # plt.xlabel("year")
    # plt.ylabel(f"thermosensitivity [GW/K], T < {threshold_degC:4.1f} °C, annual moving average")
    # plt.legend()
    # plt.show()


    # I calculate thermsensitivity for T < 12 °C to avoid the bend around 15 °C
    #   but for the net consumption, I avoid excluding days with (a little) heating

    _temperature_sat = ((temperature.clip(upper=15.75) - 15.75).rolling(30, min_periods=30, center=True) \
                     .mean()).dropna()
    _consumption = consumption.resample('D').mean().dropna()
    _consumption = (_consumption.rolling(30, min_periods=30, center=True) \
                     .mean()).dropna()
    _year= pd.Series(_temperature_sat.index.year - year_ref + \
                     _temperature_sat.index.dayofyear/365,
                     index= _temperature_sat.index, name="year")  # for long-term drift
    _sensitivity = np.mean(_list_intercept) + _year * np.mean(_list_coef)


    _consumption_net = (_consumption - (_sensitivity * _temperature_sat)).dropna()
    # print("shapes:", _consumption_net.shape, _year.shape,
    #       pd.DataFrame(_year.reindex(_consumption_net.index)).shape)

    # linear regression to quantify long-term drift
    model = LinearRegression()
    X_year = pd.DataFrame(_year.reindex(_consumption_net.index))
    model.fit(X_year, _consumption_net)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model = f"{model.intercept_:.1f} GW " \
                     f"{model.coef_[0]:.2f} * (year - {year_ref})"
    print(f"non-thermosensitive consumption "
          f"[R² ={model.score(X_year, _consumption_net)*100:3.0f}%] = {_formula_model}")

    consumption_net_model = _year * model.coef_[0] + model.intercept_



    plt.figure(figsize=(10,6))
    sma_days = 3*30
    # plt.plot(_consumption.index, plots._apply_moving_average(
    #     _consumption, sma_days).values, label="complete",color="grey")
    plt.plot(_consumption_net.index, plots._apply_moving_average(
        _consumption_net, sma_days).values, label="actual",color="black")
    plt.plot(consumption_net_model.index, consumption_net_model.values,
             label=f"trend: {_formula_model}",color="red")
    plt.ylim(38, 46)
    plt.xlabel("year")
    plt.ylabel("non-thermosensitive consumption [GW], trimester moving average")
    plt.legend(loc='lower left')
    plt.show()


    # residual
    plt.figure(figsize=(10,6))
    sma_days = 3*30
    plt.plot(_consumption_net.index, plots._apply_moving_average(
        _consumption_net - consumption_net_model.reindex(_consumption_net.index),
        sma_days).values,
             label="actual - trend",color="black")
    plt.hlines(0, _year.index[0], _year.index[-1], color="black")
    plt.ylim(-5., 3.)
    plt.xlabel("year")
    plt.ylabel("residual non-thermosensitive consumption [GW], trimester avg")
    plt.show()




def thermosensitivity_per_temperature_model(
     data_split,  # : containers.DataSplit,
     thresholds_degC  : Sequence[float],
     num_steps_per_day: int,
     ylim             : [float, float] = [-3.5, 1.5]
     )   -> None:
    sensitivity_df = threshold_temp_sensitivity(
            data_split.true_nation_GW, data_split.dict_preds_NNTQ,
            data_split.dict_preds_ML, data_split.dict_preds_meta,
            data_split.Tavg_degC.round(1), data_split.dates,
            thresholds=thresholds_degC, direction='==',
            num_steps_per_day=num_steps_per_day)

    # /!\ quantiles are meaningless
    #   they are slopes of quantiles, not quantiles of slopes

    plots.curves(
         sensitivity_df['true'],
        {_col: sensitivity_df['NNTQ '+_col] for _col in ['q50']},
            # if data_split.name != Split.complete else None,
        {_col: sensitivity_df[_col] for _col in ['RF', 'LGBM']} \
            if data_split.name in [Split.train, Split.complete] else None,
        {_col: sensitivity_df['meta '+_col] for _col in ['LR', 'NN']} \
            if data_split.name in [Split.valid, Split.test] else None,
         xlabel="threshold T_avg [°C]",
         ylabel="thermosensitivity [GW/K]",
         title=f"{data_split.name_display}",
         ylim=ylim, date_range=None, moving_average=7, groupby=None)





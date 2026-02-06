###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# plot_statistics.py
# plotting statistics of electricity data (with or without model)
#
###############################################################################

import sys

from   typing   import Dict, Optional, Tuple, Sequence, List  # , Any

import numpy  as np
import pandas as pd

from   sklearn.linear_model import LinearRegression, HuberRegressor

import matplotlib.pyplot as plt


import plots  # containers
from   constants   import Split



year_ref = 2021

months_seasons = {'spring': [3, 4, 5], 'summer': [ 6, 7, 8],
                  'autumn': [9,10,11], 'winter': [12, 1, 2]}
colors_seasons = {'spring': 'green', 'summer': 'orange',
                  'autumn': 'brown', 'winter': 'skyblue',
                  'all': 'black',    'mid'   : 'grey'
                  }




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
    _year_annual = pd.Series(_temperature_annual.index.year - year_ref + \
                             _temperature_annual.index.dayofyear/365,
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

    _temperature_sat_annual = ((temperature.clip(upper=15) - 15).\
                rolling(365, min_periods=365, center=True).mean()).dropna()
    _consumption_annual = (_consumption.rolling(365, min_periods=365, center=True) \
                     .mean()).dropna()
    _year_annual = pd.Series(_consumption_annual.index.year - year_ref + \
                             _consumption_annual.index.dayofyear/365,
                      index = _consumption_annual.index, name="year")  # for long-term drift



    # model w/o T°
    model_no_T = LinearRegression()
    model_no_T.fit(_year_annual.to_frame(), _consumption_annual)
    # _slope = -round(float(model_LR.coef_[0]) * 100, 2)
    _formula_model_no_T = f"{model_no_T.intercept_:.1f} GW  " \
                     f"{model_no_T.coef_[0]:.2f} * (year - {year_ref})"
    _R2 = model_no_T.score(_year_annual.to_frame(), _consumption_annual)
    print(f"consumption [R² ={_R2*100:3.0f}%] = {_formula_model_no_T}")

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
# prices
# -------------------------------------------------------

def prices_per_season(
             price: pd.Series,
        ) -> None:

    print("from", price.index.date.min(), "to", price.index.date.max())

    winter = price[price.index.month.isin([12, 1, 2])]
    winter['year_as_January'] = (winter.index + pd.DateOffset(months=2)).year.astype(int)
    # winter['year_pair']= winter.apply(lambda row:f"{row['year_as_January']-1}-"
    #                                              f"{row['year_as_January']-2000}",axis=1)
    # print(winter)

    summer = price[price.index.month.isin([ 6, 7, 8])]


    ranges = {'winter': range(2016, 2027), 'summer': range(2015, 2026)}
    colors = colors_seasons | \
             {2020: 'blue', 2021: 'cornflowerblue',2022: 'red',   2023: 'hotpink',
              2024: 'green',2025: 'chartreuse',    2026: 'yellowgreen'}
    styles = {'avg': 'solid',   'std': 'dotted',  'range': 'dashed'}
    names  = {'avg': "average", 'std': "std dev", 'range': "amplitude"}
    styles.pop('std');  names.pop('std')

    _ylim = [0, 150]
    _dict_stats = {'avg_winter': [], 'std_winter': [], 'range_winter': [],
                   'avg_summer': [], 'std_summer': [], 'range_summer': []}


    # winter
    plt.figure(figsize=(10,6))
    for _year in ranges['winter']:
        _winter = winter[winter['year_as_January'] == _year].\
            drop(columns=['year_as_January'])
        _winter = _winter.groupby(_winter.index.hour).mean()

        _dict_stats['avg_winter'  ].append(float(_winter.mean().iloc[0]))
        # _dict_stats['std_winter'  ].append(float(_winter.std ().iloc[0]))
        _dict_stats['range_winter'].append(float((_winter.max()-_winter.min()).iloc[0]))

        _winter = _winter.rename(lambda x: x + 1).rename_axis('hour_local')
        _winter.loc[0] = _winter.loc[24]
        _winter.sort_index(inplace=True)
        # print(_year, _winter)

        if _year >= 2020:
            plt.plot(_winter.index, _winter.values,
                     color=colors[_year], label=f"{_year-1}-{_year-2000}")

    plt.ylabel("winter spot price [€/MWh]")
    plt.xlabel("local time of day")
    plt.ylim(_ylim)
    plt.legend()
    plt.show()


    # summer
    plt.figure(figsize=(10,6))
    for _year in ranges['summer']:
        _summer = summer[summer.index.year == _year]
        _summer = _summer.groupby(_summer.index.hour).mean()

        _dict_stats['avg_summer'  ].append(float(_summer.mean().iloc[0]))
        # _dict_stats['std_summer'  ].append(float(_summer.std ().iloc[0]))
        _dict_stats['range_summer'].append(float((_summer.max()-_summer.min()).iloc[0]))

        _summer = _summer.rename(lambda x: x + 2).rename_axis('hour_local')
        _summer.loc[0] = _summer.loc[24]
        _summer.loc[1] = _summer.loc[25]
        _summer.drop(index=[25], inplace=True)
        _summer.sort_index(inplace=True)
        # print(_year, _summer)

        if _year >= 2020:
            plt.plot(_summer.index, _summer['price_euro_per_MWh'].values,
                     color=colors[_year], label=_year)

    plt.ylabel("summer spot price [€/MWh]")
    plt.xlabel("local time of day")
    plt.ylim(_ylim)
    plt.legend()
    plt.show()


    # avg and std dev as functions of year
    plt.figure(figsize=(10, 6))
    for _season in list(ranges.keys()):
        for _stat in list(names.keys()):
            plt.plot(ranges[_season], _dict_stats[_stat+'_'+_season],
                     label = names[_stat] + ", " + _season,
                     color=colors[_season], linestyle=styles[_stat])
    plt.xlabel('year (winter: year of January)')
    plt.ylabel('spot price [€/MWh]')
    plt.xlim(2014, 2027)
    plt.ylim(_ylim)
    plt.legend(ncols=2)
    plt.show()

    _dict_ref = {}
    for key in list(_dict_stats.keys()):
        _dict_ref[key] = np.mean(_dict_stats[key][:6])

    plt.figure(figsize=(10, 6))

    for _season in list(ranges.keys()):
        for _stat in list(names.keys()):
            _complete = _stat+'_'+_season
            plt.plot(ranges[_season], _dict_stats[_complete]/_dict_ref[_complete],
                     label = names[_stat] + ", " + _season,
                     color=colors[_season], linestyle=styles[_stat])
    plt.xlabel('year (winter: year of January)')
    plt.ylabel('spot price, normalized')
    plt.xlim(2014, 2027)
    plt.ylim(0, 6)
    plt.legend(ncols=2)
    plt.show()


    _dict_ref = {}
    for key in list(_dict_stats.keys()):
        _dict_ref[key] = np.mean(_dict_stats[key][:6])

    plt.figure(figsize=(10, 6))

    normalized_amplitude = {_season:
    [round(rg/avg, 3) for (rg, avg) in
     zip(_dict_stats['range_'+ _season], _dict_stats['avg_'  + _season])]
                        for _season in list(ranges.keys())}
    print(normalized_amplitude)

    for _season in list(ranges.keys()):
        plt.plot(ranges[_season], normalized_amplitude[_season],
                 label=_season, color=colors[_season])
    plt.xlabel('year (winter: year of January)')
    plt.ylabel('spot price: (max - min) / average')
    plt.xlim(2014, 2027)
    plt.ylim(0, 2)
    plt.legend(loc='upper left')
    plt.show()



def production_function_price(
     production:    pd.DataFrame,
     consumption:   pd.Series,
     price:         pd.Series,
     min_year:      int   = 2023,
     range_prices:  Tuple = [-20, 260] # euro/MWh
) -> None:

        _production = production [(production.index.year >= min_year)].\
            resample('h').mean().dropna()
        _consumption= consumption[(consumption.index.year>= min_year)].\
            resample('h').mean().dropna()
        _price = price[(price.index.year >= min_year)].resample('h').mean().dropna()

        _size  = 10
        _alpha =  0.05
        _labels= {'EnR_GW': "renewables", 'net_charge_GW': "net charge",
                  'Ech_physiques_GW': "inter-connect"}
        _colors= {'EnR_GW': 'green', 'net_charge_GW': 'blue',
                  'Ech_physiques_GW': 'red'}


        for _col in production.columns:
            plt.scatter(_price, _production[_col],
                        s=_size, alpha=_alpha,
                        color=_colors[_col], label=_labels[_col])
        plt.xlim(range_prices)
        # plt.ylim(bottom=   0)
        plt.xlabel("price [€/MWh]")
        plt.ylabel("production [GW]")
        _legend  = plt.legend(loc='upper right')
        for handle in _legend.legend_handles:
            handle.set_alpha(1)
        plt.show()


        for _col in production.columns:
            plt.scatter(_price, _production[_col] / _consumption * 100,
                        s=_size, alpha=_alpha,
                        color=_colors[_col], label=_labels[_col])
        plt.xlim(range_prices)
        # plt.ylim(bottom=   0)
        plt.xlabel("price [€/MWh]")
        plt.ylabel("fraction consumption [%]")
        _legend  = plt.legend(loc='upper right')
        for handle in _legend.legend_handles:
            handle.set_alpha(1)
        plt.show()



# -------------------------------------------------------
# thermosensitivity by région
# -------------------------------------------------------

def thermosensitivity_regions(df_consumption       : pd.DataFrame,
                              df_temperature       : pd.DataFrame,
                              threshold_winter_degC: float = 15.,
                              threshold_summer_degC: float = 20.) -> None:
    # df_consumption.drop(columns=['year', 'month', 'dateofyear', 'timeofday'],
    #                     inplace=True)
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
# thermosensitivity by time of day
# -------------------------------------------------------


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





def thermosensitivity_peak_hour(
     consumption      : pd.Series,
     temperature      : pd.Series,
     num_steps_per_day: int
)   -> None:

    _min_year = 2010

    _consumption = consumption[(consumption.index.year >= _min_year)].dropna()
    _temperature = temperature[(consumption.index.year >= _min_year)].dropna()


    # switch to local time (time changes at ends of March and October)
    _consumption.index = _consumption.index.tz_convert('Europe/Paris').sort_values()
    _temperature.index = _temperature.index.tz_convert('Europe/Paris').sort_values()
    name_tz = "local"
    # remove duplicates introduced by local time
    _consumption = _consumption[~_consumption.index.duplicated(keep="first")]
    _temperature = _temperature[~_temperature.index.duplicated(keep="first")]

    # _consumption.index = _consumption.index.tz_convert('utc')
    # _temperature.index = _temperature.index.tz_convert('utc')
    # name_tz = "UTC"


    dict_conso= {'all': _consumption}
    dict_T    = {'all': _temperature}
    dict_df   = {}

    dict_months = months_seasons
    # dict_months['spring'] = [ 4]
    # dict_months['autumn'] = [10]
    # dict_months['mid']    = dict_months['spring'] + dict_months['autumn']
    # del dict_months['spring'], dict_months['autumn']

    for season in list(dict_months.keys()):
        dict_conso[season]= _consumption[(_consumption.index.month.isin(dict_months[season]))]
        dict_T    [season]= _temperature[(_temperature.index.month.isin(dict_months[season]))]

    list_thresholds = list(np.arange(3, 26, 2.5))
    dict_thresholds = {
        'winter': list_thresholds[0:4],
        'mid'   : list_thresholds[2:7],
        'summer': list_thresholds[6:9],
        'all'   : list_thresholds[0:6]
    }
    dict_thresholds['spring'] = dict_thresholds['mid']
    dict_thresholds['autumn'] = dict_thresholds['mid']


    dict_colors = { # key: number of curves and of colors
        4: ['blue', 'skyblue', 'orange', 'red'],
        5: ['blue', 'skyblue', 'grey', 'orange', 'red'],
        6: ['blue', 'skyblue', 'lightgreen', 'gold', 'darkorange', 'red'],
        7: ['blue', 'skyblue', 'lightgreen', 'grey', 'gold', 'darkorange', 'red']
    }

    dict_slopes = {}
    for season in list(dict_conso.keys()):
        # initializing lists
        idx_T = dict_T[season] < dict_thresholds[season][0]
        count = [round(sum(idx_T) / num_steps_per_day)]  # steps -> days
        _list_series = [dict_conso[season][idx_T]]
        _cols        = [f"T < {dict_thresholds[season][0]} °C"]

        for _idx in range(0, len(dict_thresholds[season])-1):
            idx_T = (dict_T[season] >= dict_thresholds[season][_idx  ]) & \
                    (dict_T[season] <  dict_thresholds[season][_idx+1])
            count.append(round(sum(idx_T) / num_steps_per_day))  # steps -> days
            _list_series.append(dict_conso[season][idx_T])
            _cols.append(f"{dict_thresholds[season][_idx]} <= T < "
                         f"{dict_thresholds[season][_idx+1]} °C")


        idx_T = dict_T[season] >= dict_thresholds[season][-1]
        count.append(round(sum(idx_T) / num_steps_per_day))  # steps -> days
        _list_series.append(dict_conso[season][idx_T])
        _cols       .append(f"T >= {dict_thresholds[season][-1]} °C")

        # _list_series.append(dict_conso[season])
        # _cols       .append("summer")


        dict_df[season] = pd.DataFrame()
        for i, _series in enumerate(_list_series):
            _timeofday = _series.index.hour + _series.index.minute/60
            dict_df[season][i] = _series.groupby(_timeofday).mean()
        dict_df[season].columns  = _cols
        dict_df[season].loc[24.] = dict_df[season].loc[0.]

        print(f"{season:6s} count: {count}")
        # print(dict_df[season].round(1))

        _colors = dict_colors[dict_df[season].shape[1]]
        i = 0
        plt.plot(figsize=(10,6))
        for name, col in dict_df[season].T.iterrows():
            plt.plot(col.index, col.values, label=name, color=_colors[i])
            i += 1
        plt.xlabel(f"{name_tz} time of day")
        plt.ylabel(f"{season} consumption [GW]")
        # plt.title (title)
        plt.xlim( 0, 24)
        plt.xticks(range(0, 25, 4))
        plt.ylim(30, 82)
        plt.legend(loc='lower center' if season in ['all','winter'] else 'upper left')
        plt.show()


        # _stats = {'avg': dict_df[season].mean(0), 'std': dict_df[season].std(0),
        #           'amplitude': dict_df[season].max(0)-dict_df[season].min(0)}
        # print(season, "\n", pd.DataFrame(_stats).round(2))

        # plt.figure(figsize=(10,6))
        # plt.bar(_stats['amplitude'].index, _stats['amplitude'].values,color='skyblue')
        # plt.ylabel("winter consumption: max hour - min hour [GW]")
        # plt.show()

        plt.plot(figsize=(10,6))
        plt.hlines(0, 0, 24, color="black")
        i = 0
        ref_series = dict_df[season][list(dict_df[season].columns)[-3]]
        for name, col in dict_df[season].T.iterrows():
            # ref_series = col.mean(0)
            plt.plot(col.index, (col - ref_series).values,
                     label=name, color=_colors[i])
            i += 1

        plt.xlabel(f"{name_tz} time of day")
        plt.ylabel(f"{season} consumption [GW], ref.: "
                   f"{list(dict_df[season].columns)[-3]}")
        # plt.title (title)
        plt.xlim(  0, 24)
        plt.xticks(range(0, 25, 4))
        plt.ylim(-20, 20)
        plt.legend(loc='lower right', ncols=2)
            # if season in ['all','winter'] else 'upper left')
        plt.show()


        _idx   = dict_T    [season] <= 13
        _conso = dict_conso[season].loc[_idx]
        _T     = dict_T    [season].loc[_idx]
        # print(season, round(sum(_idx)/num_steps_per_day))

        if sum(_idx)/num_steps_per_day > 30:  # do only w/ enough data
            dict_slopes[season] = []
            for h in range(num_steps_per_day):
                _idx = (_T.index.hour == h//2) & (_T.index.minute == (h%2)*30)

                # linear regression to quantify thermosensitivity in winter
                model = LinearRegression()
                model.fit(_T.loc[_idx].to_frame(), _conso.loc[_idx])
                dict_slopes[season].append(-round(float(model.coef_[0]), 4))
            dict_slopes[season].append(dict_slopes[season][0])  # 24:00 == 0:00
            # print(season, dict_slopes[season])


    # plotting consumption
    plt.plot(figsize=(12,5))
    _dict_series_timeofday = dict()
    for season in dict_conso.keys():
        if season not in ['all']:
            _series_date = dict_conso[season]
            _timeofday = _series_date.index.hour + _series_date.index.minute/60
            _dict_series_timeofday[season] = _series_date.groupby(_timeofday).mean()
            _dict_series_timeofday[season].loc[24.] = _dict_series_timeofday[season].loc[0.]

            plt.plot(_dict_series_timeofday[season].index,
                     _dict_series_timeofday[season].values,
                     label=season, color=colors_seasons[season])
    plt.xlabel(f"{name_tz} time of day")
    plt.ylabel( "consumption [GW]")
    plt.xlim( 0, 24)
    plt.xticks(range(0, 25, 4))
    plt.legend(loc='lower right', ncols=2)
    plt.show()

    # plotting consumption w.r.t. summer
    plt.plot(figsize=(12,5))
    for season in dict_conso.keys():
        if season not in ['all']:
            plt.plot(_dict_series_timeofday[season].index,
                     (_dict_series_timeofday[season]-_dict_series_timeofday['summer']).values,
                     label=season, color=colors_seasons[season])
    plt.xlabel(f"{name_tz} time of day")
    plt.ylabel( "consumption compared to summer [GW]")
    plt.xlim( 0, 24)
    plt.xticks(range(0, 25, 4))
    plt.legend(loc='lower left', ncols=2)
    plt.show()


    # plotting thermosensitivity
    plt.plot(figsize=(12,5))
    for season in dict_slopes.keys():
        if season not in ['all']:
            plt.plot(np.arange(24.5, step=0.5), dict_slopes[season],
                     label=season, color=colors_seasons[season])
    plt.xlabel(f"{name_tz} time of day")
    plt.ylabel( "heating thermosensitivity [GW/K]")
    plt.xlim( 0, 24)
    plt.xticks(range(0, 25, 4))
    # if plt.ylim()[0] > 1.5:  # ensure zero is part of the axis
    #     plt.ylim(bottom = 1.5)
    plt.legend(loc='upper left')
    plt.show()



    # same T° range across seasons
    plt.plot(figsize=(10,6))

    for idx_threshold in [2, 3, 6]:
        _thresholds = [list_thresholds[idx_threshold],
                       list_thresholds[idx_threshold+1]]
        _list_possible_cols = [
            f'T >= {_thresholds[0]} °C',
            f'{_thresholds[0]} <= T < {_thresholds[1]} °C',
            f'T < {_thresholds[1]} °C'
        ]

        for season in dict_df.keys():
            if season not in ['all']:
                str_threshold = [e for e in _list_possible_cols
                                    if e in dict_df[season].columns]
                plt.plot(dict_df[season][str_threshold].index,
                         dict_df[season][str_threshold].values,
                         label = season, color=colors_seasons[season])

        str_threshold = _list_possible_cols[1]
        plt.xlabel(f"{name_tz} time of day")
        plt.ylabel("consumption [GW]")
        plt.title (str_threshold)
        plt.xlim( 0, 24)
        plt.xticks(range(0, 25, 4))
        plt.ylim(30, 80)
        plt.legend()
        plt.show()


# -------------------------------------------------------
# thermosensitivity by temperature
# -------------------------------------------------------


def threshold_temp_sensitivity(
    true_series         : pd.Series,
    dict_pred_series    : Dict[str, pd.Series],
    dict_baseline_series: Dict[str, pd.Series],
    dict_meta_series    : Dict[str, pd.Series],
    T_degC              : pd.Series,
    dates               : pd.DatetimeIndex,
    name_col            : str,
    thresholds          : Sequence[float | None],
    direction           : str,  # '<=', '>=' or '=='
    num_steps_per_day   : int,
    width               : float | pd.Timedelta,
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

    rows = dict()
    for _target in thresholds:
        if _target:
            _df = apply_threshold(df, name_col, _target, direction, width)
        else:
            _df = df.copy()

        # # Not enough variation -> undefined slope
        # if len(_df) < 10*num_steps_per_day or _df['T_degC'].var() == 0:
        #     rows[_target] = [np.nan] * len(columns_preds)
        #     continue


        slopes = []
        for _col in columns_preds:
            if len(_df[_col]) < 10:
                slopes.append(np.nan)
            else:
                slopes.append(round(LinearRegression().fit(
                    _df[['T_degC']], _df[_col]).coef_[0], 3))
                # print(_target, _col, slopes)

        rows[_target] = slopes

    _df = pd.DataFrame(rows).T
    _df.columns = columns_preds

    return _df


def thermosensitivity_per_temperature_by_season(
     consumption      : pd.Series,
     temperature      : pd.Series,
     thresholds_degC  : Sequence[float],
     num_steps_per_day: int,
     deltaT_K         : float = 1.5
     )   -> None:

    def _sensitivity(conso: pd.Series, temp: pd.Series)  -> pd.DataFrame:
        return threshold_temp_sensitivity(
                conso, {}, {}, {},
                temp.round(1), conso.index, name_col='T_degC',
                thresholds=thresholds_degC, direction='==',
                num_steps_per_day=num_steps_per_day, width = 1)

    sensitivity_df = pd.DataFrame()
    sensitivity_df["all"] = _sensitivity(consumption, temperature)

    for _season in list(months_seasons.keys()):
        # print(_state, sum(df_temperature[_state]))
        sensitivity_df[_season] = _sensitivity(
            consumption[consumption.index.month.isin(months_seasons[_season])],
            temperature[temperature.index.month.isin(months_seasons[_season])])

    plt.figure(figsize=(10,6))
    for _season in ["all"] + list(months_seasons.keys()):
        plt.plot(sensitivity_df[_season].index,
                 sensitivity_df[_season].rolling(15, min_periods=12).mean(),
                 color=colors_seasons[_season], label=_season)
    plt.xlabel("threshold T_avg [°C]")
    plt.ylabel("thermosensitivity [GW/K]")
    plt.ylim(bottom=-3.5)
    plt.legend()
    plt.show()



def thermosensitivity_per_temperature_hysteresis(
     consumption      : pd.Series,
     temperature      : pd.Series,
     thresholds_degC  : Sequence[float],
     num_steps_per_day: int,
     deltaT_K         : float = 1.
     )   -> None:

    def _sensitivity(conso: pd.Series, temp: pd.Series)  -> pd.DataFrame:
        return threshold_temp_sensitivity(
                conso, {}, {}, {},
                temp.round(1), conso.index, name_col='T_degC',
                thresholds=thresholds_degC, direction='==',
                num_steps_per_day=num_steps_per_day, width = 1)

    colors = {'all': 'black', 'is_warmer': 'orange',
              'is_colder': 'skyblue', 'is_stable': 'grey'}

    _consumption  =  consumption.resample('D').mean().dropna()
    common_indices= _consumption.index.intersection(temperature.index)
    _temperature  =  temperature.reindex(common_indices)
    _consumption  = _consumption.reindex(common_indices)


    df_temperature = _temperature.to_frame()
    df_temperature['yesterday'] = temperature.shift(1)
    df_temperature['delta_k'] = (df_temperature['Tavg_degC'] - df_temperature['yesterday'])

    df_temperature['is_warmer'] = (df_temperature['delta_k'] >  deltaT_K)
    df_temperature['is_colder'] = (df_temperature['delta_k'] < -deltaT_K)
    df_temperature['is_stable'] = (df_temperature['delta_k'] >= -deltaT_K) & \
                                  (df_temperature['delta_k'] <=  deltaT_K)


    sensitivity_df = pd.DataFrame()
    sensitivity_df["all"] = _sensitivity(_consumption, _temperature)

    for _state in ['is_warmer', 'is_colder', 'is_stable']:
        print(_state, sum(df_temperature[_state]))
        sensitivity_df[_state] = _sensitivity(
            _consumption[df_temperature[_state]],
            _temperature[df_temperature[_state]])

    plt.figure(figsize=(10,6))
    for _state in ['all', 'is_warmer', 'is_colder', 'is_stable']:
        plt.plot(sensitivity_df[_state].index,
                 sensitivity_df[_state].rolling(10, min_periods=7).mean(),
                 color=colors[_state], label=_state)
    plt.xlabel("threshold T_avg [°C]")
    plt.ylabel("thermosensitivity [GW/K]")
    plt.ylim(bottom=-3.5)
    plt.legend()
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
            num_steps_per_day=num_steps_per_day, width=2)

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



# -------------------------------------------------------
# thermosensitivity by date (long-term evoltions)
# -------------------------------------------------------


def thermosensitivity_per_date_discrete(
     consumption      : pd.Series,
     temperature      : pd.Series,
     ranges_years     : List[List[int]],
     num_steps_per_day: int,
     )   -> None:


    _consumption = consumption.resample('D').mean().dropna()
    _temperature = temperature.resample('D').mean().dropna()

    # use common indices

    common_indices = _consumption.index.intersection(_temperature.index)
    _temperature = _temperature.reindex(common_indices)
    _consumption = _consumption.reindex(common_indices)


    # statistics consumption and temperature
    consumption_per_range = [];  temperature_per_range = []
    DJU_per_range = []
    for _range in ranges_years:
        consumption_per_range.append(
            _consumption[(_consumption.index.year >= _range[0]) & \
                         (_consumption.index.year <= _range[1])])
        temperature_per_range.append(
            _temperature[(_temperature.index.year >= _range[0]) & \
                         (_temperature.index.year <= _range[1])])

    avg_consumption_per_range = [round(float(_series.mean()), 2)
                                 for _series in consumption_per_range]
    avg_temperature_per_range = [round(float(_series.mean()), 2)
                                 for _series in temperature_per_range]


    # thermosensitivity
    _list_DJU_range1 = [];  _list_DJU_range2 = []
    _list_sensitivity_range1 = [];  _list_avg_consumption_net_range1 = []
    _list_sensitivity_range2 = [];  _list_avg_consumption_net_range2 = []

    for threshold_degC in np.arange(15, 15.1, 1):
        # _temperature = temperature.resample('D').mean().dropna()
        _temperature_sat = (_temperature.clip(upper=threshold_degC) - threshold_degC)

        _DJU_per_range                = []; _sensitivity_per_range        = []
        _avg_consumption_net_per_range= []; _std_consumption_net_per_range= []
        for (i, _range) in enumerate(ranges_years):
            _temperature_in_range = _temperature_sat[
                            (_temperature_sat.index.year >= _range[0]) & \
                            (_temperature_sat.index.year <= _range[1])]

            _DJU_per_range.append(int(- _temperature_in_range.sum() / \
                                         (_range[1]-_range[0]+1)))

            _index_sat = _temperature_in_range.index[_temperature_in_range < 0]

            sensitivity_df = threshold_temp_sensitivity(
                consumption_per_range[i], {}, {}, {},
                _temperature_in_range, _index_sat, name_col='date',
                thresholds=[None], direction=None,
                num_steps_per_day=num_steps_per_day,
                width = pd.Timedelta(days=6*30)
                )

            _sensitivity_per_range.append(float(sensitivity_df['true'].mean()))

            _consumption_net = (consumption_per_range[i] - \
                        _sensitivity_per_range[i] * _temperature_sat).dropna()
            _avg_consumption_net_per_range.append(round(float(_consumption_net.mean()), 3))
            _std_consumption_net_per_range.append(round(float(_consumption_net.std()), 3))


        print(threshold_degC,
              _DJU_per_range, int(_DJU_per_range[1] - _DJU_per_range[0]),
              _sensitivity_per_range,
              round(_sensitivity_per_range[1] - _sensitivity_per_range[0], 3),
              _avg_consumption_net_per_range, _std_consumption_net_per_range)

        _list_DJU_range1.append(_DJU_per_range[0])
        _list_DJU_range2.append(_DJU_per_range[1])

        _list_sensitivity_range1.append(-_sensitivity_per_range[0])
        _list_sensitivity_range2.append(-_sensitivity_per_range[1])

        _list_avg_consumption_net_range1.append(_avg_consumption_net_per_range[0])
        _list_avg_consumption_net_range2.append(_avg_consumption_net_per_range[1])



    _delta = round(avg_temperature_per_range[1] - avg_temperature_per_range[0], 2)
    print("avg_temperature [°C]    ", avg_temperature_per_range, _delta)

    DJU_per_range = [int(np.mean(_list_DJU_range1)),
                     int(np.mean(_list_DJU_range2))]
    _delta = round(DJU_per_range[1] - DJU_per_range[0], 0)
    print("DJU                     ", DJU_per_range,
          _delta, f"{_delta/DJU_per_range[0] * 100:.1f}%")

    sensitivity_per_range = [round(float(np.mean(_list_sensitivity_range1)), 3),
                             round(float(np.mean(_list_sensitivity_range2)), 3)]
    _delta = round(sensitivity_per_range[1] - sensitivity_per_range[0], 3)
    print("sensitivity [GW/K]      ", sensitivity_per_range,
          _delta, f"{_delta/sensitivity_per_range[0] * 100:.1f}%")

    _delta = round(avg_consumption_per_range[1] - avg_consumption_per_range[0], 2)
    print("avg_consumption [GW]    ", avg_consumption_per_range,
          _delta, f"{_delta/avg_consumption_per_range[0] * 100:.1f}%")

    avg_consumption_net_per_range = [
        round(float(np.mean(_list_avg_consumption_net_range1)), 2),
        round(float(np.mean(_list_avg_consumption_net_range2)), 2)]
    _delta = round(avg_consumption_net_per_range[1] - avg_consumption_net_per_range[0], 2)
    print("avg net consumption [GW]", avg_consumption_net_per_range,
          _delta, f"{_delta/avg_consumption_net_per_range[0] * 100:.1f}%")


    # plotting ratios
    ratios = {
        "consumption, non-T": avg_consumption_net_per_range[1] / \
                              avg_consumption_net_per_range[0],
        "DJU15": DJU_per_range[1] / DJU_per_range[0],
        "sensitivity": sensitivity_per_range[1] / sensitivity_per_range[0],
        "consumption, T": (avg_consumption_per_range[1] - avg_consumption_net_per_range[1]) / \
                          (avg_consumption_per_range[0] - avg_consumption_net_per_range[0]),

        "consumption, all":  avg_consumption_per_range[1] / avg_consumption_per_range[0],
        }
    ratios = {k: (v - 1) * 100 for (k, v) in ratios.items()}

    plt.figure(figsize=(8, 6))
    plt.barh(list(ratios.keys()), list(ratios.values()), color='skyblue')
    plt.title(f"variation [%] between {ranges_years[0][0]}-{ranges_years[0][1]-2000} "
               f"and {ranges_years[1][0]}-{ranges_years[1][1]-2000}")
    for i, v in enumerate(ratios.values()):      # display values on bars
        plt.text(v, i, f" {v:.0f}%", color='black', va='center')
    plt.show()


    # plotting breakdown
    _consumption_thermo = (avg_consumption_per_range[1] - avg_consumption_net_per_range[1]) - \
                          (avg_consumption_per_range[0] - avg_consumption_net_per_range[0])
    _factor = _consumption_thermo / (ratios['DJU15'] + ratios['sensitivity'])

    breakdown_GW = {
        "non-T":avg_consumption_net_per_range[1]-avg_consumption_net_per_range[0],
        "DJU15":      round(ratios['DJU15']      * _factor, 2),
        "sensitivity":round(ratios['sensitivity']* _factor, 2)
        }
    print("breakdown [GW]:", breakdown_GW)

    breakdown_pc = {
        k: round(100 * v / (avg_consumption_per_range[1]-avg_consumption_per_range[0]), 1)
                 for (k, v) in breakdown_GW.items()}
    print("breakdown [%]:", breakdown_pc)

    # plt.figure(figsize=(8, 6))

    # left = 0
    # for name, value in breakdown.items():
    #     plt.barh(0, value, label=name, left=left, height=0.4)
    #     left += value
    # plt.ylim(-0.3, 0.3)
    # plt.legend()

    # plt.title(f"breakdown of the consumption decrease "
    #           f"between {ranges_years[0][0]}-{ranges_years[0][1]-2000} "
    #           f"and {ranges_years[1][0]}-{ranges_years[1][1]-2000} [GW]")
    # # for i, (k, v) in enumerate(breakdown.items()):      # display values on bars
    # #     plt.text(v, i, f"{k} {v:.1f} GW", color='black', va='center')
    # plt.show()


# -------------------------------------------------------
# production by price
# -------------------------------------------------------



def production_by_price(production: pd.DataFrame,
                        prices    : pd.DataFrame) -> None:
    _df_GW = production.copy()

    # interconnections
    _df_GW['export_Ech_comm_total_GW'] = 0.
    _df_GW['import_Ech_comm_total_GW'] = 0.
    for _col in [e for e in production.columns if ('Ech_comm' in e)]:
        # print(_col)
        _df_GW['export_'+_col]  = _df_GW[_col].clip(upper=0)
        _df_GW['import_'+_col]  = _df_GW[_col].clip(lower=0)

        _df_GW['export_Ech_comm_total_GW'] += _df_GW['export_'+_col]
        _df_GW['import_Ech_comm_total_GW'] += _df_GW['import_'+_col]


    _cols_GW = [e for e in _df_GW.columns if ('_GW' in e) \
                and not (('consumption' in e) or ('Prévision_J' in e))
                and e not in ['Eolien_terrestre_GW', 'Eolien_offshore_GW',
                              'EnR_GW', 'net_charge_GW',
                              'Ech_comm_Espagne']]
    # _cols_GW.append('consumption_GW')
    # print(_cols_GW)
    _prod    = _df_GW[_cols_GW]      .mean(0)
    _abs_prod= _df_GW[_cols_GW].abs().mean(0)
    _cost = _df_GW[_cols_GW].mul(prices, axis=0)
    _df_cost = pd.concat([_cost.mean(0).div(_prod) .round(1),
                         (_prod    *24*365.25/1000).round(1),
                         (_abs_prod*24*365.25/1000).round(1)
                         ], axis=1)
    _df_cost.columns= ["cost_euro_per_MWh", "prod_TWh_per_year", "abs_prod"]
    _df_cost.index  = [e[:-3] for e in _df_cost.index]
    _df_cost.rename(index={'Ech_physiques' : 'interconnexions',
                           'pompage_STEP'  : 'pompage STEP',
                           'turbinage_STEP': 'turbinage STEP'},
                    inplace=True)
    print("average price [€/MWh]:\n", _df_cost.sort_values(
        "cost_euro_per_MWh", ascending=False))
    # print(_df_cost.index)
    # _df_cost = _df_cost.reindex(['Charbon', 'Gaz', 'Fioul',
    #        'Hydraulique', 'lacs', 'turbinage STEP', 'fil de l\'eau', 'pompage STEP',
    #        'Consommation', 'Nucléaire', 'Eolien', 'Bioénergies', 'Solaire',
    #        'interconnexions', 'Ech_comm_Angleterre', 'Ech_comm_Suisse',
    #        'Ech_comm_AllemagneBelgique', 'Ech_comm_Italie', 'Ech_comm_Espagne',
    #        'Déstockage_batterie', 'Stockage_batterie'])


    # plot sources of production
    dict_colors = {
        # Renewables
        'Solaire':        'orange',
        'Eolien':         'greenyellow',
        'Bioénergies':    'tab:green',

        # hydroelectricity
        # 'Hydraulique':    'tab:blue',
        'fil de l\'eau':   'blue',
        'pompage STEP':   'deepskyblue',
        'lacs':           'cyan',
        'turbinage STEP': 'skyblue',

        # Storage / exchanges
        'Stockage_batterie':  'red',
        'Déstockage_batterie': 'darkorange',
        'interconnexions':    'fuchsia',
        'interconn All Belg': 'pink',

        # Low-carbon, non-renewable
        'Nucléaire':      'purple',

        # Demand
        'Consommation':   'navy',

        # Fossil fuels
        'Gaz':            'grey',
        'Fioul':          'saddlebrown',
        'Charbon':        'black',
    }

    plt.figure(figsize=(10, 6))
    # plt.scatter(_df_cost['power_GW'], _df_cost['cost_euro_per_MWh'])

    _indices_plot = [e for e in _df_cost.index
            if e not in [
                'Consommation', 'Hydraulique',
                'Nucléaire', 'interconnexions',
                'Stockage_batterie', 'Déstockage_batterie'] and
            'Ech_' not in e]

    for _idx in _indices_plot:
        plt.scatter(_df_cost.loc[_idx]['prod_TWh_per_year'],
                    _df_cost.loc[_idx]['cost_euro_per_MWh'],
                    label=_idx, s=120, color=dict_colors[_idx])
    plt.title("Prix de la production par filière, 2023–25")
    plt.xlabel('production [TWh/an]')
    plt.ylabel('prix moyen [€/MWh]')
    plt.xlim(-10,  60)
    plt.ylim( 40, 120)

    # Annotate each point with its key
    for _idx in _indices_plot:
        plt.annotate(_idx, (_df_cost.loc[_idx]['prod_TWh_per_year'] + 1.5,
                            _df_cost.loc[_idx]['cost_euro_per_MWh'] + 0.5))
    plt.show()




    # plot interconnections
    _df_interconnect = _df_cost.loc[[e for e in _df_cost.index
                                     if 'Ech_' in e]]

    _df_interconnect.index = _df_interconnect.index.\
                    str.replace('AllemagneBelgique', 'All Belg').\
                    str.replace('Ech_comm_', '')
                    # str.replace('Ech_physiques', 'total')
    print(_df_interconnect)

    values = {}
    for _direction in ['export', 'import']:
        _series = _df_interconnect.loc[_direction+'_total']
        values[_direction] = round(float(_series['prod_TWh_per_year'] * \
                                         _series['cost_euro_per_MWh']), 1)
    print("value [millions € per year]", values)

    _fraction_import = {}
    _fraction_import["energy"] = round(float(
        _df_interconnect.loc['import_total']['abs_prod'] / \
            (_df_interconnect.loc['export_total']['abs_prod'] + \
             _df_interconnect.loc['import_total']['abs_prod'])), 4)
    _fraction_import["value"] = round(abs(values['import']) / \
        (abs(values['export']) + values['import']), 4)
    print("proportion of imports:", _fraction_import)


    dict_colors_countries = {
        'total':     'black',
        'Angleterre':'darkorchid',
        'Suisse':    'chocolate',
        'Italie':    'cornflowerblue',
        'All Belg':  'grey',
        'Espagne':   'red',
    }

    plt.figure(figsize=(10, 6))
    # plt.scatter(_df_cost['power_GW'], _df_cost['cost_euro_per_MWh'])

    _indices_plot = list(dict_colors_countries.keys())

    for _idx in _indices_plot:
        if _df_interconnect.loc['export_'+_idx]['abs_prod'] > 2:
            plt.scatter(_df_interconnect.loc['export_'+_idx]['prod_TWh_per_year'],
                        _df_interconnect.loc['export_'+_idx]['cost_euro_per_MWh'],
                        label='export '+_idx, s=120, marker='o',
                        color=dict_colors_countries[_idx])

        if _df_interconnect.loc['import_'+_idx]['abs_prod'] > 2:
            plt.scatter(_df_interconnect.loc['import_'+_idx]['prod_TWh_per_year'],
                        _df_interconnect.loc['import_'+_idx]['cost_euro_per_MWh'],
                        label='import '+_idx, s=120, marker='s',
                        color=dict_colors_countries[_idx])
    plt.title("interconnexions, 2023–25")
    plt.xlabel('énergie échangée [TWh/an]')
    plt.ylabel('prix moyen [€/MWh]')
    plt.xlim(-25, 15)
    plt.ylim( 60, 95)

    # Annotate each point with its key
    for _idx in _indices_plot:
        x_off =  0.7
        y_off = -0.5 if _idx != 'All Belg' else -1.5
        if _df_interconnect.loc['export_'+_idx]['abs_prod'] > 2:
            plt.annotate('export '+_idx,
                (_df_interconnect.loc['export_'+_idx]['prod_TWh_per_year'] + x_off,
                 _df_interconnect.loc['export_'+_idx]['cost_euro_per_MWh'] + y_off))

        if _df_interconnect.loc['import_'+_idx]['abs_prod'] > 2:
            plt.annotate('import '+_idx,
                (_df_interconnect.loc['import_'+_idx]['prod_TWh_per_year'] + x_off,
                 _df_interconnect.loc['import_'+_idx]['cost_euro_per_MWh'] + y_off))
    plt.show()



# -------------------------------------------------------
# utils
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




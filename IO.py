###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# IO.py
# Load local (or download) CSV files and preprocess them
#
###############################################################################


import os, sys
import warnings

import requests
import zipfile

from   typing import List, Tuple, Dict, Optional

import json
import hashlib
import pickle

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt


import architecture, plots, plot_statistics  # constants



os.makedirs('data',  exist_ok=True)
os.makedirs('cache', exist_ok=True)


# -------------------------------------------------------
# consumption
# -------------------------------------------------------


def load_consumptions_recent(
    path_nation:str= 'data/eco2mix-national-tr.csv',
    url_nation: str= 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/eco2mix-national-tr/exports/csv?'
                       'timezone=UTC&use_labels=true&delimiter=%3B',

    path_region:str= 'data/eco2mix-regional-tr.csv',
    url_region: str= 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/eco2mix-regional-tr/exports/csv?'
                       'timezone=UTC&use_labels=true&delimiter=%3B',
    verbose:   int = 0) -> Tuple[pd.Series, pd.DataFrame]:

    # national data
    if os.path.exists(path_nation):
        df_nation = pd.read_csv(path_nation, sep=';')
    else:
        # Load from URL
        df_nation = pd.read_csv(url_nation, sep=';')
        df_nation.to_csv(path_nation, sep=';')

    df_nation['Date - Heure'] = pd.to_datetime(df_nation['Date - Heure'], utc=True)
    df_nation = df_nation.set_index('Date - Heure').sort_index()
    df_nation.index.name = "datetime_utc"

    #  convert to GW and remove rows of NA for the most recent time steps
    df_nation = df_nation['Consommation (MW)'].div(1000).round(3).dropna().squeeze()
    df_nation.name = 'consumption_GW'

    df_nation = df_nation.resample('30min').mean()  # to match the rest of the dataset


    # régional
    if os.path.exists(path_region):
        df_region = pd.read_csv(path_region, sep=';')
    else:
        # Load from URL
        df_region = pd.read_csv(url_region, sep=';')
        df_region.to_csv(path_region, sep=';')

    df_region['Date - Heure'] = pd.to_datetime(df_region['Date - Heure'], utc=True)
    df_region = df_region.set_index('Date - Heure').sort_index()
    df_region.index.name = "datetime_utc"

    df_region = df_region[['Région', 'Consommation (MW)']]

    #  convert to GW
    df_region = df_region.rename(columns={'Consommation (MW)': 'consumption_GW'})
    df_region['consumption_GW'] = (df_region['consumption_GW'] / 1000).round(3)

    df_region = df_region.pivot_table(
        index="datetime_utc", columns="Région", values='consumption_GW',
        aggfunc='mean').sort_index()

    df_region = df_region.resample('30min').mean()  # to match the rest of the dataset

    # remove rows with at least 2 NAs
    df_region = df_region.dropna(thresh=2)


    # Nouvelle-Aquitaine is missing lots of data
    # from 2025-01-01 to 2025-03-10 (when data are complete):
    #   (df_nation - df_region.sum(1, skipna=False)).dropna().mean() == 0.11

    # infer consumption as: whole minus sum of the other parts
    series_Aquitaine = df_nation - \
        df_region.drop(columns='Nouvelle-Aquitaine').sum(1, skipna=False) - 0.11
    df_region['Nouvelle-Aquitaine'] = \
        df_region['Nouvelle-Aquitaine'].fillna(series_Aquitaine)
    # print(pd.concat([df_nation.T, df_region.sum(1, skipna=False)], axis=1).dropna())


    return df_nation, df_region



# https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute/
# depth of historical data: 2012 to date (M-1)
# Last processing
#    January 13, 2026

def load_consumption(
        path   : str = 'data/consommation-quotidienne-brute.csv',
        url    : str = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/consommation-quotidienne-brute/exports/csv?'
                       'lang=en&timezone=Europe/Paris&use_labels=true&delimiter=%3B',
            # 'timezone=UTC' is not genuine UTC: it has duplicates
        df_recent: Optional[pd.Series] = None,
        verbose: int = 0) -> pd.DataFrame:

    # load local file if it exists, otherwise get it online
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
    else:
        # Load from URL
        df = pd.read_csv(url, sep=';')
        df.to_csv(path, sep=';')

    # /!\ at time change, `Date - Heure` is incorrect (same time, same tz)
    #       but `Heure` is monotonic (at +02:00?)
    # Date - Heure;Date;Heure; [...]
    #   version downloaded with `timezone=Europe/Paris`:
    # 2020-03-29T03:30:00+02:00;29/03/2020;03:30; [...]
    # 2020-03-29T03:00:00+02:00;29/03/2020;03:00; [...]
    # 2020-03-29T03:30:00+02:00;29/03/2020;02:30; [...]
    # 2020-03-29T03:00:00+02:00;29/03/2020;02:00; [...]
    #   version downloaded with `timezone=UTC`:
    # 2020-03-29T01:30:00+00:00;29/03/2020;03:30; [...]
    # 2020-03-29T01:00:00+00:00;29/03/2020;03:00; [...]
    # 2020-03-29T01:30:00+00:00;29/03/2020;02:30; [...]
    # 2020-03-29T01:00:00+00:00;29/03/2020;02:00; [...]


    # # attempt at using 'Date' + 'Heure' (no tz) instead of 'Date - Heure' (tz-aware)
    # #   offset by 1 hour 41% of the time!
    # df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'],
    #                                 format='%d/%m/%Y %H:%M')
    # df['Datetime'] = df['Datetime'].dt.tz_localize('+02:00')
    # # df['UTC Datetime'] = df['Datetime'].dt.tz_convert('UTC')

    # # testing
    # df['from_Heure'] = df['Datetime'].dt.tz_convert('utc')
    # df['Date - Heure'] = pd.to_datetime(df['Date - Heure'], utc=True)
    # df['from_datetime'] = df['Date - Heure'].dt.tz_convert('utc')
    # df['hour_difference'] = (df['from_Heure'] - df['from_datetime']).\
    #     dt.total_seconds() / 3600
    # print(df[['from_Heure', 'from_datetime', 'hour_difference']])

    # occurrences_df = df['hour_difference'].value_counts().reset_index()
    # # Rename columns for clarity
    # occurrences_df.columns = ['hour_difference', 'count_steps']
    # occurrences_df['count_years']= occurrences_df['count_steps'] / 48 / 365.25
    # occurrences_df['count_pc']= occurrences_df['count_steps'] / \
    #     occurrences_df['count_steps'].sum() * 100
    # print(occurrences_df)


    col_datetime = 'Date - Heure'  # tz-aware
    df[col_datetime] = pd.to_datetime(df[col_datetime], utc=True)  # tz='Europe/Paris')
    df = df.set_index(col_datetime).sort_index()
    # df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"

    df = df[['Consommation brute électricité (MW) - RTE']]
    df = df.rename(columns={
        "Consommation brute électricité (MW) - RTE": 'consumption_GW'
        })
    df['consumption_GW'] = df['consumption_GW']/1000

    # plots.data(df.resample('D').mean(),
    #           xlabel="date", ylabel="consumption (MW)")


    # concatenate more recent data
    if df_recent is not None:
        # print(df['consumption_GW'])
        # print(df_recent)
        df = df['consumption_GW'].combine_first(df_recent).to_frame()
        # df = df.resample('30min').mean()

    df['year']     = df.index.year
    df['month']    = df.index.month
    df['dateofyear']=df.index.map(lambda d: pd.Timestamp(
        year=2000, month=d.month, day=d.day))
    df['timeofday']= df.index.hour + df.index.minute/60


    if verbose >= 3:
        print(df.head())

        plots.data(df[~((df.index.month == 2) & (df.index.day == 29))]
                    .drop(columns=['year', 'month', 'timeofday'])\
                    .resample('D').mean()\
                    .groupby('dateofyear').mean().sort_index(),
                  xlabel="date", ylabel="consumption (GW)")

        _winter = df[df['month'].isin([12, 1, 2])].groupby('timeofday').mean()\
                  .rename(columns={"consumption_GW": "winter"})
        _summer = df[df['month'].isin([ 6, 7, 8])].groupby('timeofday').mean()\
                  .rename(columns={"consumption_GW": "summer"})
        plots.data(pd.concat([_winter, _summer], axis=1).sort_index()\
                  .drop(columns=['year','month', 'dateofyear']),
                  xlabel="time of day (UTC)", ylabel="consumption (GW)",
                  title ="seasonal consumption")

    return df


# https://odre.opendatasoft.com/explore/dataset/
#   consommation-quotidienne-brute-regionale/
# depth of historical data: 2013 to date (M-1)
# Last processing
#    January 13, 2026

def load_consumption_by_region(
        path   : str = 'data/consommation-quotidienne-brute-regionale.csv',
        url    : str = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/consommation-quotidienne-brute-regionale/exports/'
                       'csv?lang=en&timezone=Europe/Paris&use_labels=true&delimiter=%3B',
        cache_dir:str= 'cache',
        df_recent: Optional[pd.Series] = None,
        verbose: int = 0) -> Tuple[pd.DataFrame, List[str]]:

    # This input csv is an order of magnitude larger than all others combined
    #   so if only one csv is to be cached, this is the one


    # download csv if it does not exist locally
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(path, 'wb') as f:
            f.write(response.content)

        # df = pd.read_csv(url, sep=';')
        # df.to_csv(path, sep=';')
    # now, we are sure to have the csv locally

    # identify csv file by size and date
    _dict_csv  = {"file_size"        : os.path.getsize (path),
                  "modification_time": os.path.getmtime(path)}
    key_str    = json.dumps(_dict_csv, sort_keys=True)
    cache_key  = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"conso_region_{cache_key}.pkl")


    # either load pickle...
    if os.path.exists(cache_path):
        if verbose > 0:
            print(f"Loading consumption by region from: {cache_path}...")
        with open(cache_path, "rb") as f:
            (out, _names_clusters) = pickle.load(f)

    # ... or compute
    else:
        # load local file if it exists, otherwise get it online
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
        else:
            # Load from URL
            df = pd.read_csv(url, sep=';')
            df.to_csv(path, sep=';')

        # See comment in `load_consumption`
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'],
                                        format='%Y-%m-%d %H:%M')
        df['Datetime'] = df['Datetime'].dt.tz_localize('+02:00')
        # df['UTC Datetime'] = df['Datetime'].dt.tz_convert('UTC')

        # col_datetime = 'Date - Heure'  # given as pseudo-local
        # df[col_datetime] = pd.to_datetime(df[col_datetime], tz='Europe/Paris')
        df = df.set_index('Datetime').sort_index()
        # df.index.name = "datetime_utc"

        df = df[['Région', 'Consommation brute électricité (MW) - RTE']]
        df = df.rename(columns={
            "Consommation brute électricité (MW) - RTE": 'consumption_GW'
            })
        df['consumption_GW'] = df['consumption_GW']/1000

        # plots.data(df.resample('D').mean(),
        #           xlabel="date", ylabel="consumption (MW)")

        df['datetime_utc'] = df.index
        df = df.pivot_table(index="datetime_utc",columns="Région",values='consumption_GW',
                      aggfunc='mean').sort_index()

        # concatenate more recent data
        if df_recent is not None:
            # print(df)
            # print(df_recent)
            df = df.combine_first(df_recent)
            # df = df.resample('30min').mean()
            # print(df)


        out = pd.DataFrame()
        # print(list(df.columns))
        for (_cluster, _list) in CLUSTERS.items():
            # print(_cluster, _list)
            out[_cluster] = sum([df[_region] for _region in _list])

        del df  # we now use `out`

        _names_clusters = list(out.columns)
        out.columns = ["consumption_" + c + "_GW" for c in _names_clusters]

        out['year']     = out.index.year
        out['month']    = out.index.month
        out['dateofyear']=out.index.map(lambda d: pd.Timestamp(
            year=2000, month=d.month, day=d.day))
        out['timeofday']= out.index.hour + out.index.minute/60

        # Save pickle
        with open(cache_path, "wb") as f:
            pickle.dump((out, _names_clusters), f)
        if verbose > 0:
            print(f"Saved consumption by region to: {cache_path}")

        # print (out)


    if verbose >= 3:
        print(out.head())

        plots.data(out.drop(columns=['year', 'month', 'timeofday'])\
                    .resample('D').mean()\
                    .groupby('dateofyear').mean().sort_index(),
                  xlabel="date", ylabel="consumption (GW)")

        _winter = out[out['month'].isin([12, 1, 2])].groupby('timeofday').mean()\
                  .rename(columns={"consumption_GW": "winter"})
        _summer = out[out['month'].isin([ 6, 7, 8])].groupby('timeofday').mean()\
                  .rename(columns={"consumption_GW": "summer"})
        plots.data(pd.concat([_winter, _summer], axis=1).sort_index()\
                  .drop(columns=['year','month', 'dateofyear']),
                  xlabel="time of day (UTC)", ylabel="consumption (GW)",
                  title ="seasonal consumption")

    return out, _names_clusters



# -------------------------------------------------------
# temperature
# -------------------------------------------------------

# weights (electricity consumption per region):
#    https://www.data.gouv.fr/datasets/consommation-annuelle-brute-regionale/
#    latest update: November 10 2025
def load_weights(
        path   : str = 'data/consommation-annuelle-brute-regionale.csv',
        url    : str = 'https://www.data.gouv.fr/api/1/datasets/r/'
                       '20cbe478-4ee4-42e7-ad54-174f7e1f3a40',
        verbose: int = 0) -> [Dict[str, float], Dict[str, float]]:

    # load local file if it exists, otherwise get it online
    if os.path.exists(path):
        df_weigths = pd.read_csv(path, sep=';')
    else:
        # Load from URL
        df_weigths = pd.read_csv(url, sep=';')
        df_weigths.to_csv(path, sep=';')


    # print(df_weigths.columns)
    df_weigths = (
            df_weigths
            .groupby("Région")["Consommation brute électricité (GWh) - RTE"]
            .mean()
            .drop(index=['Corse'])
        ).T

    weights_regions = (df_weigths / df_weigths.sum()).round(5)
    weights_regions.columns = [SHORT_NAMES_REGIONS[normalize_name(r)]
                             for r in weights_regions.index]

    if not np.isclose(weights_regions.sum(), 1.):
        raise ValueError(f"Consumption weights do not sum to 100% "
                         f"({weights_regions.sum()*100}%)")


    # by cluster
    weights_clusters = dict()
    for (_cluster, _list) in CLUSTERS.items():
        # print(_cluster, _list)
        weights_clusters[_cluster] = \
            round(float(sum([df_weigths.loc[_region] for _region in _list]) \
                  / df_weigths.sum()), 5)

    if not np.isclose(sum(weights_clusters.values()), 1.):
        raise ValueError(f"Consumption weights do not sum to 100% "
                         f"({sum(weights_clusters.values())*100}%)")

    if verbose >= 2:
        print(weights_regions.to_dict())
        print(weights_clusters)

    return weights_regions.to_dict(), weights_clusters


# https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-regionale/
# Last processing
#   January 3, 2026 3:00 AM (data)

def load_temperature(
        path   : str,
        weights: Dict[str, float],
        url    : str = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/temperature-quotidienne-regionale/exports/csv?'
                       'timezone=UTC&use_labels=true&delimiter=%3B',
        # noise_std: float or Tuple[float] = 0.,# realistic forecast error
        verbose: int = 0) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # 4. Weighted aggregation helpers
    # def weighted_quantile(values, weights, q):
    #     sorter = np.argsort(values)
    #     values = values [sorter]
    #     weights= weights[sorter]
    #     cw     = np.cumsum(weights)
    #     return np.interp(q * cw[-1], cw, values)

    def weighted_mean(df, list_full: Optional[List[str]] = None, weights=weights):

        # print("df.columns:", list(df.columns))
        # print("long:",   list_full)

        if list_full is None:
            _list_short = list(df.columns)
        else:
            _list_short = [SHORT_NAMES_REGIONS[normalize_name(r)] for r in list_full]
        # print("short:", _list_short)
        # print("df[_list_short]:", df[_list_short])

        _dict_weights = {SHORT_NAMES_REGIONS[normalize_name(k)]: v
                         for (k, v) in weights.items()}

        # keep only relevant régions
        _dict_weights = {k: v for (k, v) in _dict_weights.items()
                         if k in _list_short}

        # normalize w/in the cluster
        _dict_weights = {k: v / sum(_dict_weights.values())
                             for (k, v) in _dict_weights.items()}
        # print("_dict_weights:", _dict_weights)


        is_good = df[_list_short].notna().all(axis=1)
        out     = (df[_list_short] * pd.Series(_dict_weights)).sum(axis=1)
        out[~is_good] = np.nan

        # print("out:", out)

        return out.round(2)

    # def weighted_q(df, q):
    #     return df.apply(
    #         lambda row: weighted_quantile(row.values, weights_aligned.values, q),
    #         axis=1
    #     ).round(2)


    # temperature data

    # load local file if it exists, otherwise get it online
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
    else:
        # Load from URL
        df = pd.read_csv(url, sep=';')
        df.to_csv(path, sep=';')


    if 'Date' not in df.columns:
        raise RuntimeError(f"No date column found in {path}")

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', utc=True)
    df = df.set_index('Date').sort_index()
    df.index.name = "date"

    df['Région'] = [SHORT_NAMES_REGIONS[normalize_name(r)]
                        for r in df['Région']]
    df.drop(columns=['Code INSEE région', 'ID'], errors="ignore", inplace=True)

    df = df.rename(columns={
        "TMin (°C)": 'Tmin_degC', "TMax (°C)": 'Tmax_degC', "TMoy (°C)": 'Tavg_degC'
        })

    # 2. Pivot by region
    def _pivot(col):
        return (
            df.pivot(index="date", columns="Région", values=col)
              .sort_index()
              .drop(columns=['corse'])
                  # Corsica is not in the consumption data:
                  #     electrically, it is Italian
        )

    df = df.reset_index()
    Tavg_full = _pivot("Tavg_degC")
    Tmin_full = _pivot("Tmin_degC")
    Tmax_full = _pivot("Tmax_degC")
    # print(Tavg_full)


    out = pd.DataFrame(index=Tavg_full.index)
    out["Tavg_degC"] = weighted_mean(Tavg_full, None, weights)
    out["Tmin_degC"] = weighted_mean(Tmin_full, None, weights)
    out["Tmax_degC"] = weighted_mean(Tmax_full, None, weights)


    # weighted averages by geographic cluster
    Tavg = pd.DataFrame()
    # print("weights:", weights)
    # print("CLUSTERS:", CLUSTERS)
    for (_cluster, _list_full) in CLUSTERS.items():
        Tavg[_cluster] = weighted_mean(Tavg_full, _list_full, weights)
    # print("Tavg:", Tavg)

    Tmin = pd.DataFrame()
    for (_cluster, _list_full) in CLUSTERS.items():
        Tmin[_cluster] = weighted_mean(Tmin_full, _list_full, weights)
    # print("Tmin:", Tmin)

    Tmax = pd.DataFrame()
    for (_cluster, _list_full) in CLUSTERS.items():
        Tmax[_cluster] = weighted_mean(Tmax_full, _list_full, weights)
    # print("Tmax:", Tmax)



    # print(Tavg)
    # Tavg['num_NAs'] = Tavg.isna().sum(axis=1)
    # print("Tavg\n", Tavg[Tavg['num_NAs'] > 0].head(10))
    # Tmax['num_NAs'] = Tmax.isna().sum(axis=1)
    # print("Tmax\n", Tmax[Tmax['num_NAs'] > 0].head(10))


    # # 3. Align weights
    # weights_aligned = weights.reindex(df['Région'])

    # if weights_aligned.isna().any():
    #     missing = weights_aligned[weights_aligned.isna()].index.tolist()
    #     raise ValueError(f"Missing weights for regions: {missing}")


    # 5. Build output features

    # Regional spread (heterogeneity)
    out["Tavg_region_spread_K"] = Tavg.max(axis=1) - Tavg.min(axis=1)


    for _cluster in CLUSTERS_SHORT.keys():
        # Averages per type of T°, per région
        out[f"Tavg_{_cluster}_degC"] = Tavg[_cluster]
        # out[f"Tmin_{_cluster}_degC"] = Tmin[_cluster]
        # out[f"Tmax_{_cluster}_degC"] = Tmax[_cluster]

        out[f"T_spread_{_cluster}_K"] = Tmax[_cluster] - Tmin[_cluster]

        # threshold relevant to heating, per région
        Tref_degC = 15
        out[f'Tavg_{_cluster}_inf_{Tref_degC}degC'] = \
            Tavg[_cluster].clip(upper=Tref_degC)

    # North-east: account for very cold days (even Tmax is low)
    Tref_cold_degC = 6
    out[f'Tmax_NE_inf_{Tref_cold_degC}degC']=Tmax['NE'].clip(upper=Tref_cold_degC)

    # south: account for air-conditioning (even Tmin is high)
    Tref_AC_degC = 18
    out[f'Tmin_S_sup_{Tref_AC_degC}degC']  = Tmin['S' ].clip(lower=Tref_AC_degC)


    # simple moving average (SMA) for Tavg, per région
    for _cluster in CLUSTERS_SHORT.keys():
        for duration_days in [3, 10]:
            out[f'Tavg_{_cluster}_SMA_{duration_days}days'] = Tavg[_cluster] \
                .rolling(duration_days, min_periods=int(duration_days*.8)).mean()

    # para-dates
    out['year']     = out.index.year
    out['month']    = out.index.month
    out['dateofyear']=out.index.map(lambda d: pd.Timestamp(
        year=2000, month=d.month, day=d.day))

    if verbose >= 1:
        print(f"[load_temperature] {len(Tavg.columns)} région clusters,"
              f"{len(out)} days")

    if verbose >= 2:
        print(out.drop(columns='dateofyear').head().to_string())

    #     plots.data(df, xlabel="date", ylabel="temperature (°C)")

    if verbose >= 3:
        plots.data(out.groupby('dateofyear').mean().sort_index()\
                    .drop(columns=['year','month']),
                  xlabel="date of year", ylabel="temperature (°C)",
                  title ="seasonal temperature")

    # print(list(out.columns))

    return out, Tavg_full, Tmin_full, Tmax_full



# https://odre.opendatasoft.com/explore/dataset/
#       rayonnement-solaire-vitesse-vent-tri-horaires-regionaux/
# Last processing
#    December 16, 2025 3:00 AM (metadata)
#    December  3, 2025 3:00 AM (data)

# BUG: The whole of September 2021 is missing

# def load_solar(path, verbose: int = 0):
#     df = pd.read_csv(path, sep=';')

#     # Detect datetime column (example name)
#     col = "Date"
#     df[col] = pd.to_datetime(df[col], utc=True)
#     df = df.set_index(col).sort_index()
#     df.index.name = "datetime_utc"

#     df.drop(columns=["Code INSEE région"], errors="ignore", inplace=True)

#     # National average (like temperature)
#     df = df.select_dtypes(include=[np.number]).groupby(df.index).mean()

#     # Rename explicitly

#     df['solar_kW_per_m2'] = df['Rayonnement solaire global (W/m2)'] / 1000
#     df.drop(columns=['Rayonnement solaire global (W/m2)'], inplace=True)
#     df = df.rename(columns={"Vitesse du vent à 100m (m/s)": 'wind_m_per_s'})

#     if verbose >= 2:
#         print(df.head())

#     return df





# -------------------------------------------------------
# Load misc. (nuclear, éco2mix)
# -------------------------------------------------------


# https://odre.opendatasoft.com/explore/dataset/production-nette-nucleaire/
# Last processing: December 15, 2025 (data + metadata)

def load_nuclear(
        path   : str = 'data/archives/production-nette-nucleaire.csv',
        url    : str = 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                       'datasets/production-nette-nucleaire/exports/csv?'
                       'timezone=UTC&use_labels=true&delimiter=%3B',
        verbose: int = 0) -> pd.Series:

    # load local file if it exists, otherwise get it online
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
    else:
        # Load from URL
        df = pd.read_csv(url, sep=';')
        df.to_csv(path, sep=';')

    df['datetime_utc'] = pd.to_datetime(df['Date heure'], utc=True)
    df = df.set_index('datetime_utc').sort_index()
        # was inverse inverse chronology
    df.index.name = "datetime_utc"

    df = df[['Production nette en GWh']]
    df = df.rename(columns={'Production nette en GWh': 'prod_nuclear_GW'})
        # hourly GWh equivalent to GW

    if verbose >= 3:
        df['year']     = df.index.year
        df['month']    = df.index.month
        df['dateofyear']=df.index.map(lambda d: pd.Timestamp(
            year=2000, month=d.month, day=d.day))
        df['timeofday']= df.index.hour + df.index.minute/60

        plt.figure(figsize=(10,6))
        df['prod_nuclear_GW'].rolling(24*365).mean().plot()
        plt.ylabel("nuclear production [GW], annual moving average")
        plt.xlabel("year")
        plt.show()

        plt.figure(figsize=(10,6))
        df.groupby('timeofday').mean()['prod_nuclear_GW'].plot()
        plt.ylabel("nuclear production [GW]")
        plt.xlabel('time of day (UTC)')
        plt.show()

        plt.figure(figsize=(10,6))
        df.groupby('dateofyear').mean() \
            .rolling(7).mean() \
            ['prod_nuclear_GW'].plot()
        plt.ylabel("nuclear production [GW], weekly moving average")
        plt.xlabel('dateofyear')
        plt.show()


    return df['prod_nuclear_GW']



# January 1st 2025 to January 26th 2026
def load_eco2mix(
    path_monthly:str= 'data/archives/eco2mix-national-cons-def.csv',
    url_monthly: str= 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                      'datasets/eco2mix-national-cons-def/exports/csv?'
                      'timezone=UTC&use_labels=true&delimiter=%3B',
    path_recent: str= 'data/eco2mix-national-tr.csv',
    url_recent:  str= 'https://odre.opendatasoft.com/api/explore/v2.1/catalog/'
                      'datasets/eco2mix-national-tr/exports/csv?'
                      'timezone=UTC&use_labels=true&delimiter=%3B',
    verbose:    int = 0) -> pd.DataFrame:


    # older data `monhtly`
    # load local files if they exist, otherwise get them online
    if os.path.exists(path_monthly):
        df_monthly = pd.read_csv(path_monthly, sep=';')
    else:
        # Load from URL
        df_monthly = pd.read_csv(url_monthly, sep=';')
        df_monthly.to_csv(path_monthly, sep=';')

    df_monthly['Date et Heure'] = pd.to_datetime(df_monthly['Date et Heure'], utc=True)
    df_monthly = df_monthly.set_index('Date et Heure').sort_index()
    df_monthly.index.name = "datetime_utc"

    # print("monthly:", df_monthly.shape, df_monthly.index.min(), df_monthly.index.max())
    # df_monthly.dropna(inplace=True)  # data are not really per 15 min but rather 30 min

    # analyze_datetime(df_monthly, freq="15min", name="eco2mix monthly")


    # recent (real time) data
    if os.path.exists(path_recent):
        df_recent = pd.read_csv(path_recent, sep=';')
    else:
        # Load from URL
        df_recent = pd.read_csv(url_recent, sep=';')
        df_recent.to_csv(path_recent, sep=';')


    df_recent['Date - Heure'] = pd.to_datetime(df_recent['Date - Heure'], utc=True)
    df_recent = df_recent.set_index('Date - Heure').sort_index()
    df_recent.index.name = "datetime_utc"

    df_recent.dropna(inplace=True)  # rows of NA for the most recent time steps


    # analyze_datetime(df_recent, freq="15min", name="eco2mix recent")



    # merge
    df = pd.concat([df_monthly, df_recent], axis=0)

    df.drop(columns=['Périmètre', 'Nature', 'Date', 'Heure'], inplace=True)

    # print("all:    ", df.shape, df.index.min(), df.index.max())

    # df = df[~df.index.duplicated(keep="first")]

    # cleaning names: eg "Bioénergies - Déchets (MW)" -> "Bioénergies_Déchets_MW"
    df.columns = df.columns.str.replace(r'[()\-.]', '', regex=True) \
                           .str.replace(' ', '_')

    # remove breakdowns
    df = df.loc[:, ~df.columns.str.contains(r'^Ech_comm_.*_MW$')]
    df = df.loc[:, ~df.columns.str.contains(r'^Fioul__.*_MW$')]
    df = df.loc[:, ~df.columns.str.contains(r'^Gaz__.*_MW$')]
    df = df.loc[:, ~df.columns.str.contains(r'^Bioénergies__.*_MW$')]
    df = df.loc[:, ~df.columns.str.contains(r'^Hydraulique__.*_MW$')]

    df =df.resample('30min').mean()  # older dat are on this freq

    # convert to GW
    df = (df / 1000).round(2)
    df.columns = df.columns.str.replace('MW', 'GW')


    if verbose >= 2:
        print("monthly:", df_monthly.shape, df_monthly.index.min(), df_monthly.index.max())
        print("recent:  ",df_recent .shape, df_recent .index.min(), df_recent .index.max())
        print("all:    ", df.shape, df.index.min(), df.index.max())

    # print(df)



    if verbose >= 3:
        print(df.mean(axis=0).round(2))

        # df['year']     = df.index.year
        # df['month']    = df.index.month
        # df['dateofyear']=df.index.map(lambda d: pd.Timestamp(
        #     year=2000, month=d.month, day=d.day))
        df['timeofday']= df.index.hour + df.index.minute/60

        plt.figure(figsize=(10,6))
        df[['Hydraulique_GW', 'Solaire_GW', 'Eolien_GW', #'Eolien_offshore_GW'
            'Ech_physiques_GW', 'Pompage_GW']].rolling(2*24*7*4).mean().plot()
        plt.ylabel("production [GW], weekly moving average")
        plt.xlabel("year")
        plt.legend()
        plt.show()

        y_lim_GW = [-10, 15]
        # plt.figure(figsize=(10,6))
        # df[['Hydraulique_GW', 'Solaire_GW', 'Eolien_GW', # 'Eolien_offshore_GW',
        #     'Ech_physiques_GW', 'Pompage_GW', 'timeofday']].groupby('timeofday').mean().plot()
        # plt.xlabel('time of day (UTC)')
        # plt.ylabel("production [GW]")
        # plt.ylim(y_lim_GW)
        # plt.legend()
        # plt.show()

        # seasons
        df_summer = df.loc[df.index.month.isin([6, 7, 8])]
        plt.figure(figsize=(10,6))
        df_summer[['Hydraulique_GW', 'Solaire_GW', 'Eolien_GW', #♦ 'Eolien_offshore_GW',
            'Ech_physiques_GW', 'Pompage_GW', 'timeofday']].groupby('timeofday').mean().plot()
        plt.xlabel('time of day (UTC)')
        plt.ylabel("summer production [GW]")
        plt.ylim(y_lim_GW)
        plt.legend(loc='upper left')
        plt.show()

        df_winter = df.loc[df.index.month.isin([12, 1, 2])]
        plt.figure(figsize=(10,6))
        df_winter[['Hydraulique_GW', 'Solaire_GW', 'Eolien_GW', #♦ 'Eolien_offshore_GW',
            'Ech_physiques_GW', 'Pompage_GW', 'timeofday']].groupby('timeofday').mean().plot()
        plt.xlabel('time of day (UTC)')
        plt.ylabel("winter production [GW]")
        plt.ylim(y_lim_GW)
        plt.legend(loc='upper left')
        plt.show()


        # as fraction of consumption
        df_norm_pc = df.div(df['Consommation_GW'], axis=0) * 100
        df_norm_pc.columns = df.columns.str.replace('GW', 'pc')
        df_norm_pc['timeofday'] = df['timeofday']
        print(df_norm_pc.mean(axis=0).round(2))

        plt.figure(figsize=(10,6))
        df_norm_pc[['Hydraulique_pc', 'Solaire_pc', 'Eolien_pc', #'Eolien_offshore_pc'
            'Ech_physiques_pc', 'Pompage_pc']].rolling(2*24*7*4).mean().plot()
        plt.ylabel("production [%], weekly moving average")
        plt.xlabel("year")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,6))
        df_norm_pc[['Hydraulique_pc', 'Solaire_pc', 'Eolien_pc', #♦ 'Eolien_offshore_pc',
            'Ech_physiques_pc', 'Pompage_pc', 'timeofday']].groupby('timeofday').mean().plot()
        plt.ylabel("production [%]")
        plt.xlabel('time of day (UTC)')
        plt.legend()
        plt.show()


    return df





# -------------------------------------------------------
# Load prices
# -------------------------------------------------------


# depth of historical data: 2015 to today

def load_price(
        path_csv:str = 'data/wholesale_electricity_price_hourly.csv',
        path_zip:str = 'data/european_wholesale_electricity_price_data_hourly.zip',
        url    : str = 'https://files.ember-energy.org/public-downloads/price/'
                       'outputs/european_wholesale_electricity_price_data_hourly.zip',
        verbose: int = 0) -> pd.DataFrame:

    # load local file if it exists, otherwise get it online
    if not os.path.exists(path_csv):   # we don't have the csv file locally
        if not os.path.exists(path_zip):  # nor the zip file locally
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes

            with open(path_zip, 'wb') as f:
                f.write(response.content)
        # we have the zip file locally

        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            # List all files in the zip archive
            file_list = zip_ref.namelist()
            # print("Files in the zip archive:", file_list)
            assert 'France.csv' in file_list, file_list

            zip_ref.extract('France.csv', StrPath=path_csv)
    # we have the csv file locally


    df = pd.read_csv(path_csv, sep=',')
    # print(df.columns)

    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], utc=True)
    df = df.set_index('Datetime (UTC)').sort_index()
    # df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime_utc"

    df = df[['Price (EUR/MWhe)']]
    df = df.rename(columns={'Price (EUR/MWhe)': 'price_euro_per_MWh'})

    df['year']     = df.index.year
    df['month']    = df.index.month
    df['dateofyear']=df.index.map(lambda d: pd.Timestamp(
        year=2000, month=d.month, day=d.day))
    df['timeofday']= df.index.hour + df.index.minute/60

    if verbose >= 3:
        # by day
        plots.data(df.drop(columns=['year', 'month', 'timeofday', 'dateofyear'])\
                     .rolling(91 * 24, center=True).mean(),
                   enforce_0_on_y = True,
                   xlabel="date", ylabel="price [€/MWh], trimester moving average")


        # by day of year
        rolling_df= df['price_euro_per_MWh'].rolling(window=7*24, center=True).mean()
        _df = pd.concat([df.drop(columns=['year', 'month', 'timeofday',
                                'price_euro_per_MWh']), rolling_df], axis=1)
        plots.data(_df[~((_df.index.month == 2) & (_df.index.day == 29))]\
                      .groupby('dateofyear').median().sort_index(),
                   enforce_0_on_y = True,
                   xlabel="date", ylabel="price [€/MWh], weekly moving median")


        # by time of day
        # plots.data(df.drop(columns=['year', 'month', 'dateofyear'])\
        #             .groupby('timeofday').median().sort_index(),
        #           xlabel="time of day (UTC)", ylabel="median price [€/MWh]")

            # seasonal effects
        _winter = df[df['month'].isin([12, 1, 2])].groupby('timeofday').median()\
                      .rename(columns={"price_euro_per_MWh": "winter"})
        _summer = df[df['month'].isin([ 6, 7, 8])].groupby('timeofday').median()\
                      .rename(columns={"price_euro_per_MWh": "summer"})
        plots.data(pd.concat([_winter, _summer], axis=1).sort_index()\
                     .drop(columns=['year', 'month', 'dateofyear']),
                  xlabel="time of day (UTC)", ylabel="median price [€/MWh]",
                  enforce_0_on_y = True,
                  title ="seasonal consumption")

    return df



# -------------------------------------------------------
# Load all data
# -------------------------------------------------------

def load_data(dict_input_csv_fnames: dict, cache_fname: str,
              num_steps_per_day: int, minutes_per_step: int, verbose: int = 0)\
            -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = {}

    (weights_by_region, weights_by_cluster) = load_weights(verbose=verbose)
    # print("weights_regions:", weights_regions)


    consumption_nation_recent, consumption_region_recent = load_consumptions_recent()

    # Load both CSVs
    for name, path in dict_input_csv_fnames.items():
        # if not os.path.exists(path):
        #     raise FileNotFoundError(f"Input file not found: {path}")


        if verbose >= 1:
            print(f"Loading {path}...")
        if name == 'consumption':
            dfs[name] = load_consumption(
                path, df_recent=consumption_nation_recent, verbose=verbose)

        if name == 'consumption_by_region':
            dfs[name], names_regions = load_consumption_by_region(
                path, df_recent=consumption_region_recent, verbose=verbose)

        elif name == 'temperature':
            dfs[name], Tavg_regions, Tmin_regions, Tmax_regions = \
                load_temperature(path, weights_by_region, verbose=verbose)

        # elif name == 'solar':
            # BUG: The whole of September 2021 is missing
            # dfs[name] = load_solar(path, verbose=verbose)

        elif name == 'price':
            dfs[name] = load_price(path, verbose=verbose)


    # check datetime indices (duplicates, missing)
    if verbose >= 3:
        analyze_datetime(dfs["consumption"],
                    freq=f"{minutes_per_step}min", name="consumption")
        analyze_datetime(dfs["consumption_by_region"],
                    freq=f"{minutes_per_step}min", name="consumption_by_region")
        analyze_datetime(dfs['temperature'], freq="D", name="temperature")
        # analyze_datetime(load_solar(path, verbose=verbose), freq="3h", name="solar")
        analyze_datetime(dfs['price'], freq="h", name="price")
        analyze_datetime(load_nuclear(), freq="h",   name="nuclear")
        analyze_datetime(load_eco2mix(), freq="30min", name="eco2mix")


    # print("dfs['temperature']", dfs['temperature'])
    # print("Tavg_regions", Tavg_regions)
    # print("Tmin_regions", Tmin_regions)
    # print("Tmax_regions", Tmax_regions)


    # plot statistics
    if verbose >= 3:
        # # plot_statistics.thermosensitivity_regions(
        # #     dfs['consumption_by_region'], dfs['temperature'])

        # plot_statistics.drift_with_time(
        #      dfs['consumption']['consumption_GW'],
        #      dfs['temperature']['Tavg_degC'],
        #      num_steps_per_day=num_steps_per_day
        # )

        plot_statistics.thermosensitivity_per_temperature_by_season(
             dfs['consumption']['consumption_GW'],
             dfs['temperature']['Tavg_degC'],
             thresholds_degC= np.arange(-1., 26.5, step=0.1),
             num_steps_per_day=num_steps_per_day
        )

        # plot_statistics.thermosensitivity_per_date_discrete(
        #      dfs['consumption']['consumption_GW'],
        #      dfs['temperature']['Tavg_degC'],
        #      ranges_years = [[2016, 2019], [2023, 2025]],
        #      num_steps_per_day=num_steps_per_day
        # )

        # plot_statistics.prices_per_season(
        #      dfs['price'][['price_euro_per_MWh']],
        # )

        # sys.exit()



    starts = {name: df.index.min() for name, df in dfs.items()}
    ends   = {name: df.index.max() for name, df in dfs.items()}
    dates_df = pd.DataFrame({
        "start": pd.Series(starts),
        "end":   pd.Series(ends),
    })

    # # Common range
    # starts = [df.index.min() for df in dfs.values()]
    # ends   = [df.index.max() for df in dfs.values()]

    common_start  = max(starts.values()); common_end = min(ends  .values())
    earliest_start= min(starts.values()); latest_end = max(ends  .values())

    if verbose >= 3:
        print(f"intersection start: {common_start  }, end: {common_end}")
        print(f"union        start: {earliest_start}, end: {latest_end}")

    # # Half‑hour index (padding will generate NAs which will be trimmed later)
    idx = pd.date_range(start= earliest_start,
                        end  = latest_end + pd.Timedelta(days=1), freq="30min")
    # idx = pd.date_range(start= common_start,
    #                     end  = common_end + pd.Timedelta(days=1), freq="30min")

    META_COLS = ["year","month","timeofday","dateofyear","dayofyear"]


    # Align
    aligned = []
    for name, df in dfs.items():

        # If there are duplicate timestamps
        # known issue: consumption has a date error on 5th Dec. in 2020, 22, 23
        if df.index.has_duplicates:
            warnings.warn(f"{name} has duplicates")
            # df.drop_duplicates(keep=False, inplace=True)
            # idx = idx.intersection(df.index)
            df = df.groupby(df.index).mean()  # collapse duplicates by averaging

        # Keep metadata ONLY for the consumption dataset
        if name != "consumption":
            df.drop(columns=[c for c in META_COLS if c in df.columns], inplace=True)

        d = df.reindex(idx)

        # # ---- Fill 24 hours of temperature when data are daily ----
        # is_daily = (
        #     (df.index.hour == 0).all()
        #     and (df.index.minute == 0).all()
        #     and (df.index.second == 0).all()
        # )
        # if is_daily:
        #     d = d.ffill(limit=48)   # 48×30min = 24 hours

        delta = df.index.to_series().diff().dropna().mode()[0]
        if delta == pd.Timedelta(days=1):
            d = d.ffill(limit=num_steps_per_day)
        elif delta == pd.Timedelta(hours=3):
            d = d.ffill(limit= 3*2)

        # d = d.add_prefix(f"{name}_")
        aligned.append(d)

    df_merged = pd.concat(aligned, axis=1).loc[:common_end]  # remove padding
    df_merged.index.name = "datetime_utc"

    if cache_fname is not None:
        df_merged.to_csv(cache_fname)

        if verbose >= 2:
            print(f"Saved merged dataset to {cache_fname}")
            print(df_merged.head())

    if verbose >= 3:
        plots.data(df_merged.drop(columns=['year', 'month', 'timeofday'])\
                    .resample('D').mean()\
                    .groupby('dateofyear').mean().sort_index(),
                  xlabel="date")

    # quantiles for input
    if verbose >= 2:
        quantiles_pc = [0.5, 1, 2, 5, 50, 95, 98, 99, 99.5]
        quantiles_df = df_merged[[
                'consumption_GW','Tmin_degC','Tavg_degC','Tmax_degC'
            ]].quantile([q/100 for q in quantiles_pc], axis=0)
        quantiles_df.index = [f'q{q}' for q in quantiles_pc]

        print(quantiles_df)

    return (df_merged, dates_df, weights_by_cluster)




# -------------------------------------------------------
# utils
# -------------------------------------------------------


def normalize_name(s: str) -> str:
    return (
        s.lower()
         .replace("’", "'")
         .replace("à", "a")
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("ë", "e")
         .replace("î", "i")
         .replace("ô", "o")
         .strip()
    )

SHORT_NAMES_REGIONS: Dict[str, str] = \
{'auvergne-rhone-alpes': "AuRA", 'bourgogne-franche-comte': "Bourg",
 'bretagne': "Bret", 'centre-val de loire': "Centre", 'corse': 'corse',
 'grand est': "Gd_Est", 'hauts-de-france': "HdF", 'normandie': 'Norm',
 'nouvelle-aquitaine': "Aqui", 'occitanie': "Occi", 'pays de la loire': "Loire",
 "provence-alpes-cote d'azur": "PACA", 'ile-de-france': "IdF"}

# based on winter and summer thermosensitivity
CLUSTERS = {
    "NE":  ['Hauts-de-France', 'Grand Est', 'Bourgogne-Franche-Comté',
           'Auvergne-Rhône-Alpes'],
    "W":   ['Bretagne', 'Normandie', 'Pays de la Loire', 'Nouvelle-Aquitaine'],
    "IdF": ['Île-de-France', 'Centre-Val de Loire'],
    "S":   ['Occitanie', 'Provence-Alpes-Côte d\'Azur']
   }
# CLUSTERS = {
#     "NE": ['Hauts-de-France', 'Grand Est', 'Bourgogne-Franche-Comté',
#            'Auvergne-Rhône-Alpes'],
#     "NW": ['Normandie', 'Bretagne', 'Pays de la Loire'],
#     "IdF":['Île-de-France', 'Centre-Val de Loire'],
#     "S":  ['Nouvelle-Aquitaine', 'Occitanie', 'Provence-Alpes-Côte d\'Azur']
#    }

CLUSTERS_SHORT = {_cluster: [SHORT_NAMES_REGIONS[normalize_name(v)] for v in _list]
                for (_cluster, _list) in CLUSTERS.items()}



def analyze_datetime(df, freq=None, name="dataset"):
    """
    Report duplicates and missing timestamps in a datetime-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    freq : str or None
        Expected frequency string ('30T', 'H', 'D', etc.).
        If None, missing-time analysis is skipped.
    name : str
        Dataset name for display.
    """
    print(f"\n=== Analyzing {name} ===")


    # 1. Check for duplicates
    if df.index.has_duplicates:
        dup_rows = df[df.index.duplicated(keep=False)].sort_index().index
        print(f"/!\ {len(dup_rows)} duplicates found:")
        print(dup_rows)

        # counts = df.index[df.index.duplicated()].value_counts()
        # print("\nDuplicate counts:")
        # print(counts)
    else:
        print("No duplicate timestamps.")


    # 2. Check for missing timestamps
    if freq is not None:
        # full expected index
        full_index = pd.date_range(
            start= df.index.min(),
            end  = df.index.max(),
            freq = freq,
            tz   = df.index.tz  # preserve timezone awareness
        )

        missing = full_index.difference(df.index)

        if len(missing) > 0:
            print(f"\n /!\ Missing {len(missing)} timestamps:")
            print(missing[:20])     # first 20 only
            if len(missing) > 20:
                print(f"... ({len(missing) - 20} more omitted)")
        else:
            print(f"No missing timestamps at freq = {freq}")



# ----------------------------------------------------------------------
# school holidays
# ----------------------------------------------------------------------

CANONICAL_HOLIDAYS = {
    "February": [
        "vacances d'hiver",
        "vacances de fevrier",
        "hiver",
    ],
    "easter": [
        "vacances de printemps",
        "vacances de paques",
        "printemps",
    ],
    "summer": [
        "vacances d'ete",
        "ete",
    ],
    "all_saints": [
        "vacances de la toussaint",
        "toussaint",
    ],
    "christmas": [
        "vacances de noel",
        "noel",
    ],
}

def school_holidays(
        fname1: str= 'data/fr-en-calendrier-scolaire.csv',
        url1  : str= 'https://data.education.gouv.fr/api/explore/v2.1/catalog/'
                     'datasets/fr-en-calendrier-scolaire/exports/csv?delimiter=;',
        fname2: str= 'data/vacances_scolaires_2015_2017.csv') -> pd.DataFrame:

    # load local file if it exists, otherwise get it online
    if os.path.exists(fname1):
        holidays = pd.read_csv(fname1, sep=";")
    else:
        # Load from URL
        holidays = pd.read_csv(url1, sep=';')
        holidays.to_csv(fname1, sep=';')


    # Keep only metropolitan France, which is A/B/C zones
    holidays = holidays[holidays['population' ] != "Enseignants"] # students or all
    holidays = holidays[holidays['description'].str.contains("Vacances")] # not "pont"
    holidays = holidays[holidays['zones'      ].str.contains("Zone")] # A, B or C
    holidays.drop(columns=['population', 'annee_scolaire'], inplace=True)
    # Deduplicate by zone and date range
    holidays = holidays.drop_duplicates(
        subset=["zones", "start_date", "end_date", 'description'],
        keep="first"
    )

    # Convert dates
    holidays["start_date"] = pd.to_datetime(holidays["start_date"])
    holidays["end_date"]   = pd.to_datetime(holidays["end_date"])


    # Complement: holidays 2015-17
    holidays_2015_2017 = pd.read_csv(fname2, sep=",", comment="#")
    for _date in ["start_date", "end_date"]:
        holidays_2015_2017[_date]= pd.to_datetime(holidays_2015_2017[_date],utc=True)

    holidays = pd.concat([holidays_2015_2017, holidays], axis=0)

    list_names = []
    for _, row in holidays.iterrows():
        name_norm = normalize_name(row['description'])

        # Find matching holiday type based on aliases
        matched = [
            htype
            for htype, aliases in CANONICAL_HOLIDAYS.items()
            if any(alias in name_norm for alias in aliases)
        ]
        if len(matched) != 1:
            raise ValueError(f"holiday name '{row['description']}' → {matched}")
        list_names += matched
    holidays['name'] = list_names
    holidays.drop(columns=['location', 'description'], inplace=True)

    # print(holidays.head())
    # print(holidays.tail())

    return holidays


def make_school_holidays_indicator(dates: pd.DatetimeIndex, verbose: int = 0) \
            -> Tuple[pd.Series, tuple]:
    """
    Returns a Series indexed like `dates`, with values in {0,1,2,3} equal to
    the number of French metropolitan zones (A,B,C) that are on holiday.
    """

    # Initialize counts at 0
    # zone_count = pd.Series(0, index=dates)

    holidays = school_holidays()
    if verbose >= 3:
        print(holidays.head())

    # Initialize output: one column per holiday type
    holiday_types = holidays["name"].unique()

    out = pd.DataFrame(
        0,
        index=dates,
        columns=[f"holiday_{h}" for h in holiday_types]
    )

    # For each holiday entry (zone-specific)
    for _, row in holidays.iterrows():
        start, end = row["start_date"], row["end_date"]
        htype = row["name"]   # canonical holiday type

        # Holidays are [start, end)
        mask = (dates >= start) & (dates < end)

        # Each zone contributes +1 to its holiday type
        out.loc[mask, f"holiday_{htype}"] += 1


    # drop holidays that turn out to be useless in LR and RF
    # out.drop(columns= ['holiday_all_saints','holiday_summer',
    #                    'holiday_February',  'holiday_easter'], inplace=True)

    if verbose >= 3:
        print(out.head())

    return out, [holidays["start_date"].min(), holidays["end_date"].max()]




# ----------------------------------------------------------------------
# print model parameters
# ----------------------------------------------------------------------

def print_model_summary(
    minutes_per_step,  num_steps_per_day: int,
    num_time_steps: int,
    cols_features: List[str],
    input_length: int, pred_length: int, valid_length: int, features_in_future:bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    warmup_steps: int,
    patience: int,  min_delta,
    model_dim: int,  num_layers: int,  num_heads: int, ffn_size: int,
    patch_length: int,  stride: int,  # num_patches: int,
    num_geo_blocks: int, geo_block_ratio: float,
    # quantile loss
    quantiles,
    lambda_cross:   float,
    lambda_coverage:float,
    lambda_deriv:   float,
    lambda_median:  float,
        # temperature-dependence (pinball loss, coverage penalty)
    saturation_cold_degC:float,
    threshold_cold_degC: float,
    lambda_cold:    float,
    # metamodel
    meta_epochs: int, meta_learning_rate: float,
    meta_weight_decay: float, meta_batch_size: int,
    meta_dropout: float, meta_num_cells, meta_patience: int, meta_factor: float
):
    # number of sliding windows
    num_samples = max(0, num_time_steps - (input_length + pred_length) + 1)

    # steps per epoch (optimizer updates)
    steps_per_epoch = np.ceil(num_samples / batch_size) if batch_size > 0 else 0

    num_patches = (input_length + features_in_future * pred_length - patch_length)\
            // stride + 1


    print("\n===== DATA =====")
    print(f"{'MINUTES_PER_STEP':17s} ={minutes_per_step:5n}")
    print(f"{'NUM_STEPS_PER_DAY':17s} ={num_steps_per_day:5n}")

    print("\n===== MODEL CONSTANTS =====")
    print(f"{'INPUT_LENGTH':17s} ={input_length:5n} half-hours"
          f" ={input_length/num_steps_per_day:5.1f} days")
    print(f"{'PRED_LENGTH' :17s} ={pred_length :5n} half-hours")
    print(f"{'VALID_LENGTH':17s} ={valid_length:5n} half-hours")
    print(f"{'FEATURES_IN_FUTURE':17s}= {features_in_future}")

    print("\n===== LOSSES =====")
    print(f"{'QUANTILES'   :17s} = {quantiles}")
    print(f"{'LAMBDA_CROSS':17s} ={lambda_cross:9.3f}")
    print(f"{'LAMBDA_COVERAGE':17s} ={lambda_coverage:9.3f}")
    print(f"{'LAMBDA_DERIV':17s} ={lambda_deriv:9.3f}")
    print(f"{'LAMBDA_MEDIAN':17s} ={lambda_median:9.3f}")
    print("\n  TEMPERATURE DEPENDENCE")
    print(f"{'SATURATION_COLD_DEGC':17s}={saturation_cold_degC:7.2f} °C")
    print(f"{'THRESHOLD_COLD_DEGC':17s}={threshold_cold_degC:8.2f} °C")
    print(f"{'LAMBDA_COLD':17s}  ={lambda_cold:9.3f}")


    print("\n===== TRAINING =====")
    print(f"{'BATCH_SIZE'  :17s} ={batch_size:5n}")
    print(f"{'EPOCHS'      :17s} ={epochs:5n}")
    print(f"time series length: {num_time_steps/24/2/365.25:.2f} years"
          f"= {num_samples/1000:n} samples =>  {steps_per_epoch:n} steps per epoch")
    print(f"{'LEARNING_RATE':17s} ={learning_rate*1e3:8.2f}e-3")
    print(f"{'WEIGHT_DECAY':17s} ={weight_decay*1e6:8.2f}e-6")
    print(f"{'DROPOUT'     :17s} ={dropout*100:7.1f}%")

    warmup_epochs = warmup_steps/steps_per_epoch \
            if steps_per_epoch > 0 else float("inf")
    print(f"{'WARMUP_STEPS':17s} ={warmup_steps:5n} steps "
          f"=  {warmup_epochs:.2f} epochs")

    print(f"{'PATIENCE'    :17s} ={patience:5n} epochs")
    print(f"{'MIN_DELTA'   :17s} ={int(round(min_delta*1000)):5n}e-3 = "
          f"{min_delta/patience*1000:.2f}e-3 per epoch")

    print("\n===== TRANSFORMER MODEL =====")
    print(f"{'MODEL_DIM'   :17s} ={model_dim:5n}")
    print(f"{'NUM_LAYERS'  :17s} ={num_layers:5n}")
    print(f"{'NUM_HEADS'   :17s} ={num_heads:5n} =>"
          f" {model_dim // num_heads} dims per head")
    print(f"{'FFN_SIZE'    :17s} ={ffn_size:5n}  (expansion factor)")

    print("\n===== PATCH EMBEDDING =====")
    print(f"{'PATCH_LENGTH':17s} ={patch_length:5n} half-hours")
    print(f"{'STRIDE'      :17s} ={stride:5n} half-hours"
          f" => NUM_PATCHES ={num_patches:5n}")

    block_sizes = architecture.block_sizes(
        num_patches, num_geo_blocks, geo_block_ratio)
    print(f"{'block_sizes'  :17s} = {block_sizes}")

    print("\n===== METAMODEL =====")
    print(f"{'META_EPOCHS'  :17s} ={meta_epochs:5n}")
    print(f"{'META_LR'      :17s} ={meta_learning_rate*1e3:8.2f}e-3")
    print(f"{'META_WEIGHT_DECAY':17s} ={meta_weight_decay*1e6:8.2f}e-6")
    print(f"{'META_BATCH_SIZE':17s} ={meta_batch_size:5n}")
    print(f"{'META_DROPOUT'  :17s} ={meta_dropout:8.2f}")
    print(f"{'META_NUM_CELLS':17s} = {meta_num_cells}")
    print(f"{'META_PATIENCE' :17s} ={meta_patience:5n}")
    print(f"{'META_FACTOR'   :17s} ={meta_factor:8.2f}")


    # ---- Approximate parameter count ----
    d = model_dim
    l = num_layers
    m = ffn_size
    p = num_patches
    pl= patch_length
    f = len(cols_features)
    h = pred_length
    n = max(1, num_samples)

    # Patch embedding
    patch_in = pl * f
    patch_embed_params = d * patch_in + d

    # Positional embedding
    pos_embed_params = p * d

    # Transformer layers
    attn_params = 4 * d * d
    ffn_params = 2 * d * d * m
    per_layer_params = attn_params + ffn_params
    encoder_params = l * per_layer_params

    # Output head
    out_head_params = d * h + h

    param_count = int((patch_embed_params + pos_embed_params + \
                       encoder_params + out_head_params)* 1.1)

    params_per_sample = param_count / n

    print("\n=== Approximate Model Capacity ===")
    print(f"{'parameters':17s}~ {param_count/1e6:.2f} million(s) =>"
          f" {params_per_sample:.1f} params per sample")
    print()

    # ---- Checks & warnings ----
    if not (100 <= steps_per_epoch <= 2000):
        print(f"/!\\ steps_per_epoch ({steps_per_epoch:n}) "
              f"but should be in [100, 2000]")

    if not (2 <= warmup_epochs <= 5):
        print(f"/!\\ WARMUP_STEPS ({warmup_steps}) / "
              f"steps_per_epoch ({steps_per_epoch:n})"
              f" = {warmup_epochs:.2f}, but should be in [2, 5]")

    if num_samples < 300:
        print(f"/!\\ Only {num_samples} training windows — "
              f"transformers typically require >= 300 for stable generalization.")

    reuse_factor = steps_per_epoch * batch_size / max(1, num_samples)
    if reuse_factor > 200:
        print(f"/!\\ Each window is reused {reuse_factor:.0f}× per epoch — "
              f"overfitting risk is very high (recommended < 50×).")
    elif reuse_factor > 100:
        print(f"/!\\ Each window is reused {reuse_factor:.0f}× per epoch — high.")

    if not (16 <= num_patches <= 64):
        print(f"/!\\ NUM_PATCHES ({num_patches}) should be between 16 and 64.")

    if dropout < 0.1:
        print(f"/!\\ DROPOUT ({100*dropout:.1f}%) should be in 10–20%.")

    msg = (f"/!\\ Model has {param_count/1e6:.2f} million parameters = "
           f"{params_per_sample:.0f} params/sample")
    if params_per_sample > 5000:
        print(f"{msg} — extremely high (>> 5000). Overfitting almost guaranteed.")
    elif params_per_sample > 1000:
        print(f"{msg} — high risk of overfitting (recommended < 1000).")
    elif params_per_sample > 300:
        print(f"{msg} — moderate capacity. Works only with strong regularization.")
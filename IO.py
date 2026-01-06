
# ----------------------------------------------------------------------
# Load local CSVs, detect datetime columns (handling Date + Heure, etc.),
# align on common time range, resample to 30 min, merge.
# ----------------------------------------------------------------------


import os, sys
from   typing import List, Tuple  # Dict  #, Optional

import numpy  as np
import pandas as pd


import architecture, plots


# https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute/
# depth of historical data: 2012 to date (M-1)
# Last processing
#    December  2, 2025 4:28 PM (metadata)
#    December  2, 2025 4:28 PM (data)

def load_consumption(path, verbose: int = 0):

    df = pd.read_csv(path, sep=';')

    if 'Date - Heure' not in df.columns:
        raise RuntimeError(f"No datetime-like column found in {path}")

    col = 'Date - Heure'
    # col = df.columns[cols.index(key)]
    df[col] = pd.to_datetime(df[col])
    df = df.set_index(col).sort_index()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "datetime"

    df = df[['Consommation brute électricité (MW) - RTE']]
    df = df.rename(columns={
        "Consommation brute électricité (MW) - RTE": 'consumption_GW'
        })
    df['consumption_GW'] = df['consumption_GW']/1000

    # plots.data(df.resample('D').mean(),
    #           xlabel="date", ylabel="consumption (MW)")

    df['year']     = df.index.year
    df['month']    = df.index.month
    df['dateofyear']=df.index.map(lambda d: pd.Timestamp(
        year=2000, month=d.month, day=d.day))
    df['timeofday']= df.index.hour + df.index.minute/60


    if verbose >= 3:
        print(df.head())

        plots.data(df.drop(columns=['year', 'month', 'timeofday'])\
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


# weights: https://www.data.gouv.fr/datasets/consommation-annuelle-brute-regionale/
#    latest update: November 10 2025
def load_weights(path, verbose: int = 0) -> pd.Series:
    # weights (electricity consumption per region)
    df_weigths = pd.read_csv(path, sep=';')
    # print(df_weigths.columns)
    df_weigths = (
            df_weigths
            .groupby("Région")["Consommation brute électricité (GWh) - RTE"]
            .mean()
        )
    weights = df_weigths / df_weigths.sum()

    weights.index = [_normalize_name(r) for r in weights.index]

    if not np.isclose(weights.sum(), 1.):
        raise ValueError(f"Temperature weights do not sum to 100% ({weights.sum()}%)")

    if verbose >= 2:  print(weights)

    return weights


# https://odre.opendatasoft.com/explore/dataset/temperature-quotidienne-regionale/
# Last processing
#   December 12, 2025 3:00 AM (metadata)
#   December  2, 2025 3:01 AM (data)

def load_temperature(path, weights,
                     # noise_std: float or Tuple[float] = 0.,# realistic forecast error
                     verbose: int = 0):
    # temperature data
    df      = pd.read_csv(path, sep=';')
    if 'Date' not in df.columns:
        raise RuntimeError(f"No date column found in {path}")

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', utc=True)
    df = df.set_index('Date').sort_index()
    df.index.name = "date"

    df['Région'] = [_normalize_name(r) for r in df['Région']]
    df.drop(columns=['Code INSEE région', 'ID'], errors="ignore", inplace=True)

    df = df.rename(columns={
        "TMin (°C)": 'Tmin_degC', "TMax (°C)": 'Tmax_degC', "TMoy (°C)": 'Tavg_degC'
        })

    # 2. Pivot by region
    def _pivot(col):
        return (
            df.pivot(index="date", columns="Région", values=col)
              .sort_index()
        )

    df = df.reset_index()
    Tavg = _pivot("Tavg_degC").drop(columns=['corse'])  # Corsica is an outlier
    Tmin = _pivot("Tmin_degC").drop(columns=['corse'])  #   warm and small
    Tmax = _pivot("Tmax_degC").drop(columns=['corse'])

    # Tavg['num_NAs'] = Tavg.isna().sum(axis=1)
    # print("Tavg\n", Tavg[Tavg['num_NAs'] > 0].head(10))
    # Tmax['num_NAs'] = Tmax.isna().sum(axis=1)
    # print("Tmax\n", Tmax[Tmax['num_NAs'] > 0].head(10))


    # 3. Align weights
    weights_aligned = weights.reindex(df['Région'])

    if weights_aligned.isna().any():
        missing = weights_aligned[weights_aligned.isna()].index.tolist()
        raise ValueError(f"Missing weights for regions: {missing}")


    # 4. Weighted aggregation helpers
    def weighted_quantile(values, weights, q):
        sorter = np.argsort(values)
        values = values [sorter]
        weights= weights[sorter]
        cw     = np.cumsum(weights)
        return np.interp(q * cw[-1], cw, values)

    def weighted_mean(df, weights=weights):
        is_good = df.notna().all(axis=1)
        out     = (df * weights).sum(axis=1)
        out[~is_good] = np.nan
        return out.round(2)

    def weighted_q(df, q):
        return df.apply(
            lambda row: weighted_quantile(row.values, weights_aligned.values, q),
            axis=1
        ).round(2)


    # 5. Build output features
    out = pd.DataFrame(index=Tavg.index)

    # Averages
    out["Tavg_degC"] = weighted_mean(Tavg)
    out["Tmin_degC"] = weighted_mean(Tmin)
    out["Tmax_degC"] = weighted_mean(Tmax)

    # Cold / hot tails
    quantile_pc: int = 25
    out['Tavg_q'+str(quantile_pc)   + '_degC'] = weighted_q(Tavg,  quantile_pc/100.)
    out['Tmin_q'+str(quantile_pc)   + '_degC'] = weighted_q(Tmin,  quantile_pc/100.)
    out['Tmax_q'+str(100-quantile_pc)+'_degC'] = weighted_q(Tmax,1-quantile_pc/100.)

    # Regional spread (heterogeneity)
    out["T_spread_K"] = Tavg.max(axis=1) - Tavg.min(axis=1)


    # for heating, we care only about T_avg <= 15 °C, very cold days: T_avg <= 2 °C
    # air-conditioning days: T_avg >= 22 °C
    for (T, name_T) in zip([Tavg, Tmin, Tmax],  ['avg', 'min', 'max']):
        for (Tref_degC, direction) in zip([15, 3, 22], ['low', 'low', 'high']):
            if direction == 'low':  # heating
                Tsat = T.clip(upper=Tref_degC)
                _direction_str = "inf"
            else:  # air-conditioning
                Tsat = T.clip(lower=Tref_degC)
                _direction_str = "sup"

            name_Tsat = 'T'+name_T+'_'+_direction_str+'_' + str(Tref_degC) + 'degC'
            out[name_Tsat] = weighted_mean(Tsat)

            # fraction of the population heating/cooling
            _frac_avg = (Tsat < Tref_degC)\
                .astype(int).mul(weights, axis=1).sum(axis=1)
            out['frac_'+name_Tsat]      = _frac_avg

            for duration_days in ([3, 10] if Tref_degC <= 15 else [10]):
                Tsat_SMA = Tsat.rolling(duration_days,
                                        min_periods=int(duration_days*.8)).mean()
                name_Tsat_SMA = name_Tsat + '_SMA_' + str(duration_days)+'days'

                out[name_Tsat_SMA]= weighted_mean(Tsat_SMA)
                out['frac_'+name_Tsat_SMA]= weighted_mean(Tsat_SMA).\
                    rolling(duration_days, min_periods=int(duration_days*.8)).mean()



    # drop temperatures that turn out to be useless in LR and RF
    # out.drop(columns=['Tavg_sat2_degC', 'Tavg_sat22_degC'], inplace=True)


    # para-dates
    out['year']     = out.index.year
    out['month']    = out.index.month
    out['dateofyear']=out.index.map(lambda d: pd.Timestamp(
        year=2000, month=d.month, day=d.day))

    if verbose >= 1:
        print(f"[load_temperature] {len(Tavg.columns)} régions, {len(out)} days")

    if verbose >= 2:
        print(out.drop(columns='dateofyear').head().to_string())

    #     plots.data(df, xlabel="date", ylabel="temperature (°C)")

    if verbose >= 3:
        plots.data(out.groupby('dateofyear').mean().sort_index()\
                    .drop(columns=['year','month']),
                  xlabel="date of year", ylabel="temperature (°C)",
                  title ="seasonal temperature")

    return out


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
#     df.index.name = "datetime"

#     df.drop(columns=["Code INSEE région"], errors="ignore", inplace=True)

#     # National average (like temperature)
#     df = df.select_dtypes(include=[np.number]).groupby(df.index).mean()

#     # Rename explicitly

#     df['solar_kW_per_m2'] = df['Rayonnement solaire global (W/m2)'] / 1000
#     df.drop(columns=['Rayonnement solaire global (W/m2)'], inplace=True)
#     df = df.rename(columns={"Vitesse du vent à 100m (m/s)": 'wind_m_per_s'})

#     if verbose >= 2:
#         print(df.head())

    return df


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

    # -------------------------------
    # 1. Check for duplicates
    # -------------------------------
    if df.index.has_duplicates:
        print("/!\ Duplicates found:")
        dup_rows = df[df.index.duplicated(keep=False)].sort_index().index
        print(dup_rows)

        # counts = df.index[df.index.duplicated()].value_counts()
        # print("\nDuplicate counts:")
        # print(counts)
    else:
        print("✓ No duplicate timestamps.")

    # -------------------------------
    # 2. Check for missing timestamps
    # -------------------------------
    if freq is not None:
        # full expected index
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq,
            tz=df.index.tz  # preserve timezone awareness
        )

        missing = full_index.difference(df.index)

        if len(missing) > 0:
            print(f"\n Missing {len(missing)} timestamps:")
            print(missing[:20])     # first 20 only
            if len(missing) > 20:
                print("... (more omitted)")
        else:
            print(f"No missing timestamps at freq = {freq}")


def load_data(dict_fnames: dict, cache_fname: str,
              num_steps_per_day: int, minutes_per_step: int, verbose: int = 0)\
            -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = {}

    weights = load_weights('data/consommation-annuelle-brute-regionale.csv', verbose)

    # Load both CSVs
    for name, path in dict_fnames.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")


        if verbose >= 1:
            print(f"Loading {path}...")
        if name == 'consumption':
            dfs[name] = load_consumption(path, verbose)
            if verbose >= 3:
                analyze_datetime(dfs["consumption"],
                                 freq=f"{minutes_per_step}T", name="consumption")
        elif name == 'temperature':
            dfs[name] = load_temperature(path, weights, verbose=verbose)
            if verbose >= 3:
                analyze_datetime(dfs["temperature"], freq="D",  name="temperature")
        # elif name == 'solar':
            # BUG: The whole of September 2021 is missing
            # dfs[name] = load_solar(path, verbose)
            # if verbose >= 3:
            #     analyze_datetime(dfs["solar"],       freq="3H", name="solar")

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

        # If there are duplicate timestamps, collapse them by averaging
        if df.index.has_duplicates:
            df = df.groupby(df.index).mean()  # TODO despicable

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
    df_merged.index.name = "datetime"

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
        quantiles_df = df_merged[['consumption_GW','Tmin_degC','Tavg_degC','Tmax_degC']]\
                .quantile([q/100 for q in quantiles_pc], axis=0)
        quantiles_df.index = [f'q{q}' for q in quantiles_pc]

        print(quantiles_df)


    return df_merged, dates_df



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

def _normalize_name(s: str) -> str:
    return (
        s.lower()
         .replace("’", "'")
         .replace("é", "e")
         .replace("è", "e")
         .replace("ê", "e")
         .replace("ë", "e")
         .replace("à", "a")
         .replace("ô", "o")
         .strip()
    )


def school_holidays(fname1: str='data/fr-en-calendrier-scolaire.csv',
                    fname2: str='data/vacances_scolaires_2015_2017.csv')->pd.DataFrame:
    # URL = 'https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/'
    # 'fr-en-calendrier-scolaire/exports/csv?delimiter=;'

    holidays = pd.read_csv(fname1, sep=";")

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
        holidays_2015_2017[_date] = pd.to_datetime(holidays_2015_2017[_date],utc=True)

    holidays = pd.concat([holidays_2015_2017, holidays], axis=0)

    list_names = []
    for _, row in holidays.iterrows():
        name_norm = _normalize_name(row['description'])

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
    feature_cols: List[str],
    input_length: int, pred_length: int, valid_length: int, features_in_future:bool,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    warmup_steps: int,
    patience: int,  min_delta,
    model_dim: int,  num_layers: int,  num_heads: int, ffn_size: int,
    patch_length: int,  stride: int,  num_patches: int,
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
    print(f"{'LAMBDA_CROSS':17s} ={lambda_cross:8.2f}")
    print(f"{'LAMBDA_COVERAGE':17s} ={lambda_coverage:8.2f}")
    print(f"{'LAMBDA_DERIV':17s} ={lambda_deriv:8.2f}")
    print(f"{'LAMBDA_MEDIAN':17s} ={lambda_median:8.2f}")
    print("\n  TEMPERATURE DEPENDENCE")
    print(f"{'SATURATION_COLD_DEGC':17s}={saturation_cold_degC:8.2f} °C")
    print(f"{'THRESHOLD_COLD_DEGC':17s}={threshold_cold_degC:8.2f} °C")
    print(f"{'LAMBDA_COLD':17s}  ={lambda_cold:8.2f}")


    print("\n===== TRAINING =====")
    print(f"{'BATCH_SIZE'  :17s} ={batch_size:5n}")
    print(f"{'EPOCHS'      :17s} ={epochs:5n}")
    print(f"time series length: {num_time_steps/24/2/365.25:.2f} years"
          f"= {num_samples/1000:n} samples =>  {steps_per_epoch:n} steps per epoch")
    print(f"{'LEARNING_RATE':17s} ={learning_rate*1e3:8.2f}e-3")
    print(f"{'WEIGHT_DECAY':17s} ={weight_decay*1e6:8.2f}e-6")
    print(f"{'DROPOUT'     :17s} ={dropout*100:5.0f}%")

    warmup_epochs = warmup_steps/steps_per_epoch if steps_per_epoch > 0 else float("inf")
    print(f"{'WARMUP_STEPS':17s} ={warmup_steps:5n} steps =  {warmup_epochs:.2f} epochs")

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
    f = len(feature_cols)
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

    param_count = int(
        (patch_embed_params + pos_embed_params + encoder_params + out_head_params) * 1.1
    )

    params_per_sample = param_count / n

    print("\n=== Approximate Model Capacity ===")
    print(f"{'parameters':17s}~ {param_count/1e6:.2f} million(s) =>"
          f" {params_per_sample:.1f} params per sample")
    print()

    # ---- Checks & warnings ----
    if not (100 <= steps_per_epoch <= 2000):
        print(f"/!\\ steps_per_epoch ({steps_per_epoch:n}) but should be in [100, 2000]")

    if not (2 <= warmup_epochs <= 5):
        print(f"/!\\ WARMUP_STEPS ({warmup_steps}) / steps_per_epoch ({steps_per_epoch:n})"
              f" = {warmup_epochs:.2f}, but should be in [2, 5]")

    if num_samples < 300:
        print(f"/!\\ Only {num_samples} training windows — transformers typically require "
              f">= 300 for stable generalization.")

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
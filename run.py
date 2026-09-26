###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# run.py
# Running the NNTQ and the metamodels
#
###############################################################################


import copy
import gc
import glob
# import sys
import inspect
import os
import warnings

import json
import hashlib
import pickle

from   typing   import Dict, Any, Optional, List, Tuple, Sequence

import time
from   datetime import datetime

import torch

import numpy  as np
import pandas as pd


import MC_search, Bayes_search, containers, architecture, \
    utils, baselines, IO, plot_statistics   # plots,
from   constants import Stage, Split, FORECAST_TZ

# system dimensions
# B = BATCH_SIZE
# L = INPUT_LENGTH
# H = prediction horizon  =  PRED_LENGTH
# V = validation horizon  = VALID_LENGTH
# Q = number of quantiles = len(quantiles)
# F = number of features
# R = number of régions (for consumption)


# ============================================================
# LOAD DATA FROM CSV AND CREATE DATAFRAME
# ============================================================

def load_and_create_df(dict_input_csv_fnames: Dict[str, str],
                       cache_fname        : str,  # pickle file
                       pred_length        : int,
                       num_steps_per_day  : int,
                       minutes_per_step   : int,
                       do_plot_statistics : Optional[bool] = None,
                       verbose            : int  = 0) \
        -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DatetimeIndex,
                 pd.Series, pd.Series, Dict[str, float], pd.DataFrame]:
        # (df, names_cols, dates, Tavg_full, holidays_full, weights_regions,
        #  dates_df)

    df, dates_df, weights_regions = utils.df_features(
            dict_input_csv_fnames, cache_fname, pred_length,
            num_steps_per_day, minutes_per_step, do_plot_statistics, verbose)

    # ---- Identify columns ----
    col_y_nation = "consumption_GW"

    # columns that are NOT model inputs even though they are numeric.
    # /!\ the wholesale price is the OUTPUT of the noon day-ahead auction this
    #     forecast feeds: using it (esp. its D+1 value, with features_in_future)
    #     would leak the target. It is loaded and plotted for statistics inside
    #     df_features()/load_data(), which already ran above, so dropping it here
    #     costs no statistics but removes both the leak and the range-clipping
    #     its NAs would cause in the dropna() below.
    cols_non_model = ["year", 'month', 'timeofday', 'price_euro_per_MWh']

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    cols_Y_regions = [c for c in numeric_cols
          if ('consumption' in c) and (c != col_y_nation) and \
              ('consumption_SMA' not in c)]

    # All features except the target, the non-model columns and the régions
    cols_features = [
        c for c in numeric_cols
        if c not in [col_y_nation] + cols_non_model + cols_Y_regions
    ]
    # print(len(cols_features), "cols_features:", cols_features)

    df_len_before = df.shape[0]

    if verbose >= 3:
        print("NA:", df[df.isna().any(axis=1)])

    # Remove every row containing any NA (no filling)
    df = df[[col_y_nation] + cols_Y_regions + cols_features].dropna()

    # print start and end dates
    dates_df.loc["df"]= [df.index.min().date(), df.index.max().date()]
    if verbose >= 1:
        # on a copy: dates_df enters the cache keys (NNTQ, baselines), which
        #   must depend neither on `verbose` nor on today's date
        _dates_df_print = dates_df.copy()
        _dates_df_print["days_ago"] = (pd.Timestamp(datetime.now()) - \
                                pd.to_datetime(_dates_df_print["end"])).dt.days
        _dates_df_print["days_ago"] = _dates_df_print["days_ago"]\
            .where(_dates_df_print["days_ago"] >= 0, 0).astype(int)

        print(_dates_df_print)

    drop = df_len_before - df.shape[0]
    if verbose >= 2:
        print(f"number of datetimes: {df_len_before} -> {df.shape[0]}, "
              f"drop by {drop} (= {drop/num_steps_per_day:.1f} days)")
    if verbose >= 3:
        print(f"df:  {df.shape} "
              f"({df.index.min()} -> {df.index.max()})")
        print("  NA:", df.index[df.isna().any(axis=1)].tolist())

    # Keep date separately for plotting later
    dates        = df.index
    Tavg_full    = df["Tavg_degC"]    # for plots and worst days
    holidays_full= df['is_holiday']   # for worst days

    df = df.reset_index(drop=True)


    return (df, {"features":  cols_features,  "Y_regions": cols_Y_regions,
                 "y_nation": [col_y_nation]},
            dates, Tavg_full, holidays_full, weights_regions, dates_df)




# ============================================================
# NORMALIZE PREDICTORS AND CREATE MODEL
# ============================================================

def normalize_features(df              : pd.DataFrame,
                       names_cols      : Dict[str, List[str]],
                       use_ML_features : bool,

                       weights_regions : Dict[str, float],
                       minutes_per_step: int,
                       dates           : pd.DatetimeIndex,
                       temperatures    : pd.DataFrame,
                       train_split     : float,
                       n_valid         : int,
                       input_length    : int,
                       pred_length     : int,
                       features_in_future:bool,
                       batch_size      : int,
                       forecast_hour   : int,
                       verbose         : int = 0):


    # print({k: len(w) for (k, w) in names_cols.items()})

    array = np.column_stack([
        df[names_cols['y_nation' ]].values.astype(np.float32),
        df[names_cols['Y_regions']].values.astype(np.float32),
        df[names_cols['features' ]].values.astype(np.float32),
        df[names_cols['ML_preds' ]].values.astype(np.float32)
    ])

    if verbose >= 3:
        print(f"array:{array.shape}")
        print("  NA:", np.where(np.isnan(array))[0])



    if verbose >= 1:
        num_steps_per_day = int(round(24*60/minutes_per_step))
        print(f"{len(array)/num_steps_per_day/365.25:.1f} years of data, "
              f"train + valid: {train_split/num_steps_per_day/365.25:.2f} yrs "
              # f" ({train_split_fraction*100:.1f}%), "
              # f"test: {test_months/num_steps_per_day/365.25:.2f} yrs "
              f"(switching to test {dates[train_split].date()})")
        print()


    # assert all(ts.hour == 12 for ts in train_dataset.forecast_origins)
    # assert len(set(test_results['predictions']['target_time'])) == \
    #    len(test_results['predictions'])



    data, X_test_scaled = architecture.make_X_and_y(
            array, dates, temperatures.to_numpy(), train_split, n_valid,
            names_cols, use_ML_features,
            weights_regions, minutes_per_step,
            input_length=input_length, pred_length=pred_length,
            features_in_future=features_in_future, batch_size=batch_size,
            forecast_hour=forecast_hour,
            verbose=verbose)


    if verbose >= 2:
        print(f"Train mean:{data.scaler_y_nation.mean_ [0]:6.2f} GW")
        print(f"Train std :{data.scaler_y_nation.scale_[0]:6.2f} GW")
        print(f"Valid mean:{data.valid.y_nation.mean():6.2f} GW")
        print(f"Test mean :{data.test .y_nation.mean():6.2f} GW")


    return data, X_test_scaled



# ============================================================
# RUNNING MODEL ONCE
# ============================================================



def postprocess(baseline_parameters   : Dict[str, Any],
                NNTQ_parameters       : Dict[str, Any],
                metamodel_parameters  : Dict[str, Any],
                num_features          : int,
                df_metrics            : pd.DataFrame(),
                quantile_delta_coverage:Dict[str, float],
                avg_weights_meta_NN   : Dict[str, float],
                avg_abs_worst_days_search:float,
                run_id                : int,
                verbose               : int   = 0,
                df_metrics_search     : Optional[pd.DataFrame] = None,
                quantile_delta_coverage_test: Optional[Dict[str, float]] = None
                ) -> [Dict[str, Any], [float, float]]:
    """csv row and losses of a run.
    `df_metrics`: metrics reported (test_*: second half of the test period);
    `df_metrics_search`, `quantile_delta_coverage`, `avg_abs_worst_days_search`:
    first half of the test period, the objectives of the searches (search_*,
    q10..q90, avg_abs_worst_days_search). Without `df_metrics_search`,
    loss_meta uses `df_metrics`."""

    # the dicts are modified below (x1e6, sequences flattened): work on copies,
    #   the caller's dicts (e.g. constants.NNTQ_PARAMETERS in 'once' mode) must
    #   stay usable for the next run
    baseline_parameters = copy.deepcopy(baseline_parameters)
    NNTQ_parameters     = copy.deepcopy(NNTQ_parameters)
    metamodel_parameters= copy.deepcopy(metamodel_parameters)

    _meta_run = avg_weights_meta_NN is not None

    def _flatten(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        out = {f"{prefix}_{model}_{metric}".replace(" ", "_"):
                   float(df.loc[model, metric])
               for model in df.index for metric in df.columns}
        # metamodels not run (do_metamodel=False): NaN, so that the csv keeps
        #   the same columns, in the same order
        if not _meta_run:
            for model in ['meta_LR', 'meta_NN']:
                for metric in df.columns:
                    out[f"{prefix}_{model}_{metric}"] = np.nan
        return out

    flat_metrics = _flatten(df_metrics, "test")                    # reported
    flat_metrics_search = _flatten(df_metrics_search, "search") \
        if df_metrics_search is not None else {}                   # objective
    if not _meta_run:
        avg_weights_meta_NN = {k: np.nan for k in ['NNTQ_q50', 'LR', 'RF', 'LGBM']}

    # learning_rate and weight_decay are small numbers, prone to round-off errors:
    #    save them multiplied by a million (and round to avoid 0.999999)
    for _name in ['learning_rate', 'weight_decay']:
        NNTQ_parameters     [_name]= round(NNTQ_parameters    [_name] * 1e6, 6)
        metamodel_parameters[_name]= round(metamodel_parameters[_name]* 1e6, 6)

    # flatten sequences
    _dict_quantiles = expand_sequence(name="quantiles",
           values=NNTQ_parameters["quantiles"],  length=5, prefix="")
    del NNTQ_parameters["quantiles"]
    NNTQ_parameters.update(_dict_quantiles)


    _dict_num_cells = expand_sequence(name="num_cells",
           values=metamodel_parameters["num_cells"], length=2, prefix="")
    del metamodel_parameters["num_cells"]
    metamodel_parameters.update(_dict_num_cells)

    _loss_NNTQ = round(loss_NNTQ(quantile_delta_coverage, avg_abs_worst_days_search,
                           verbose=verbose), 2)
    _loss_meta = round(loss_meta(
        {k.replace("search_", "test_", 1): v for k, v in flat_metrics_search.items()}
        if flat_metrics_search else flat_metrics, verbose=verbose), 5) \
        if _meta_run else np.nan

    # BUG: this does not do the job
    if baseline_parameters['RF']['max_features'] != 'sqrt':  # is number then
        baseline_parameters['RF']['max_features'] = \
            round(baseline_parameters['RF']['max_features'], 1)

    row = {
        "run"      : run_id,
        "timestamp": datetime.now(),   # Excel-compatible
            # input parameters
        **(flatten_dict(baseline_parameters, parent_key="")),
        **NNTQ_parameters,
        **{"metaNN_"+key: value for (key, value) in metamodel_parameters.items()},
        # output
        "num_features": num_features,
        **quantile_delta_coverage,
        **{"avg_weight_meta_NN_"+key: value
           for (key, value) in avg_weights_meta_NN.items()},
        **flat_metrics_search,   # bias, RMSE, MAE: objective (1st half of test)
        **flat_metrics,          # bias, RMSE, MAE: reported  (2nd half of test)
        **{"test_coverage_"+key: value
           for (key, value) in (quantile_delta_coverage_test or {}).items()},
        'avg_abs_worst_days_search': avg_abs_worst_days_search,
        'num_runs' : 1,
        "loss_NNTQ": _loss_NNTQ,
        "loss_meta": _loss_meta,
    }
    # print(row)

    return row, (_loss_NNTQ, _loss_meta)


def append_csv_row(df_row: pd.DataFrame, path: str,
                   float_format: str = "%.6f") -> None:
    """Append one row to a results csv, creating it (with header) if needed.
    Refuses to append a row whose columns differ from the file's header: pandas
    would otherwise append the values under the wrong columns, silently."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != list(df_row.columns):
            missing = [c for c in header if c not in df_row.columns]
            extra   = [c for c in df_row.columns if c not in header]
            raise ValueError(
                f"{path}: the new row does not have the columns of the file "
                f"(missing: {missing}, extra: {extra}"
                f"{', same names in another order' if not missing and not extra else ''}"
                f"). Rename the file to start a new one.")
    _new = not os.path.exists(path) or os.path.getsize(path) == 0
    df_row.to_csv(path, mode="a", header=_new, index=False,
                  float_format=float_format)


# Bayesian search on the metamodel: NNTQ variants
# ------------------------------------------------------------
# The NNTQ parameters are frozen, but one training is one random draw (loss_NNTQ
#   varies a lot with the seed). With N variants (N = number of meta runs per
#   trial), N+2 trainings with fixed seeds are cached once; the best and worst
#   are dropped and the middle N kept, so that the metamodel is not tuned to one
#   lucky or unlucky NNTQ. Variant 0 is the median, then alternately below /
#   above it.
SEED_NNTQ_VARIANTS = 1000   # seeds 1000, 1001...: independent of the trial numbers


def NNTQ_variants_paths(cache_dir: str, cache_key: str, num_variants: int
                        ) -> Tuple[List[str], str]:
    """Paths of the cached variants, and of their json summary."""
    return ([os.path.join(cache_dir, f"NNTQ_preds_{cache_key}_v{v}.pkl")
             for v in range(num_variants)],
            os.path.join(cache_dir, f"NNTQ_variants_{cache_key}.json"))


def NNTQ_variants_cached(cache_dir: str, cache_key: str, num_variants: int) -> bool:
    """True if all the variants exist and were built for this number of variants
    (the median and the dropped runs depend on it)."""
    paths, path_summary = NNTQ_variants_paths(cache_dir, cache_key, num_variants)
    if not all(os.path.exists(p) for p in paths + [path_summary]):
        return False
    with open(path_summary) as f:
        return len(json.load(f)["variants"]) == num_variants


def input_cache_fname(cache_dir: str, dict_input_csv_fnames: Dict[str, str]) -> str:
    """Pickle of the merged input data, keyed on the size and modification time
    of every file in the folder(s) of the input csv files (the loaders also
    read other files there: eco2mix, weights, school holidays, ...).
    (/!\\ was a fixed 'input_data.pkl': updated csv files were ignored once it
     existed)"""
    folders = sorted({os.path.dirname(p) or '.' for p in dict_input_csv_fnames.values()})
    files   = {os.path.relpath(f): (os.path.getsize(f), int(os.path.getmtime(f)))
               for d in folders for f in sorted(glob.glob(os.path.join(d, '*')))
               if os.path.isfile(f)}
    key_str = json.dumps({'inputs': dict_input_csv_fnames, 'files': files},
                         sort_keys=True)
    return os.path.join(cache_dir,
                        f"input_data_{hashlib.md5(key_str.encode()).hexdigest()}.pkl")


def remove_other_input_caches(cache_dir: str, keep: str) -> None:
    """Delete the input pickles of former data (and the former unkeyed one)."""
    for _path in glob.glob(os.path.join(cache_dir, 'input_data*.pkl')):
        if os.path.normcase(os.path.abspath(_path)) != \
           os.path.normcase(os.path.abspath(keep)):
            os.remove(_path)


def build_NNTQ_variants(train_NNTQ,
                        paths_variants: List[str],
                        cache_dir     : str,
                        cache_key     : str) -> None:
    """Train the NNTQ len(paths_variants)+2 times, drop the best and the worst,
    keep the others (sorted by loss_NNTQ) in `paths_variants`, variant 0 being
    the median.
    `train_NNTQ()` returns (data, quantile_delta_coverage, avg_abs_worst_days)."""
    num_drop  = 1
    num_seeds = len(paths_variants) + 2*num_drop

    results = []   # (loss_NNTQ, seed, path)
    for k in range(num_seeds):
        _seed = SEED_NNTQ_VARIANTS + k
        np.   random.seed(_seed)
        torch.manual_seed(_seed)
        _t0   = time.perf_counter()
        _out  = train_NNTQ()
        _loss = loss_NNTQ(_out[1], _out[2])
        _path = os.path.join(cache_dir, f"NNTQ_preds_{cache_key}_seed{_seed}.pkl")
        with open(_path, "wb") as f:   # on disk: one bundle in memory at a time
            pickle.dump((_out[0].for_cache(),) + tuple(_out[1:]), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        del _out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append((_loss, _seed, _path))
        print(f"NNTQ variants: seed {_seed} ({k+1}/{num_seeds}), "
              f"loss_NNTQ {_loss:.2f} ({time.perf_counter()-_t0:.0f} s)")

    results.sort()
    middle = results[num_drop:-num_drop]
    m      = len(middle) // 2
    order  = sorted(range(len(middle)), key=lambda j: (abs(j - m), j))
        # e.g. 5: [2, 1, 3, 0, 4] -> median, then alternately below / above
    for v, j in enumerate(order):
        os.replace(middle[j][2], paths_variants[v])
    for (_, _, _path) in results[:num_drop] + results[-num_drop:]:
        os.remove(_path)
    # variants of an earlier build with more variants (same key): obsolete
    _keep = {os.path.normcase(os.path.abspath(p)) for p in paths_variants}
    for _path in glob.glob(os.path.join(cache_dir, f"NNTQ_preds_{cache_key}_v*.pkl")):
        if os.path.normcase(os.path.abspath(_path)) not in _keep:
            os.remove(_path)

    summary = {"all"     : [{"seed": s, "loss_NNTQ": round(l, 2)}
                            for (l, s, _) in results],
               "variants": [{"variant": v, "seed": middle[j][1],
                             "loss_NNTQ": round(middle[j][0], 2)}
                            for v, j in enumerate(order)]}
    _, path_summary = NNTQ_variants_paths(cache_dir, cache_key, len(paths_variants))
    with open(path_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("NNTQ variants kept (variant: loss_NNTQ): " +
          ", ".join(f"{d['variant']}: {d['loss_NNTQ']:.2f}"
                    for d in summary["variants"]) +
          f"; dropped: {[round(l, 2) for (l, _, _) in results[:num_drop] + results[-num_drop:]]}")



def run_model_once(
        # configuration bundles
        baseline_parameters   : Dict[str, Dict[str, Any]],
        NNTQ_parameters       : Dict[str, Any],
        metamodel_NN_parameters:Dict[str, Any],
        dict_input_csv_fnames : Dict[str, str],

        # statistics of the dataset
        minutes_per_step      : int,
        train_split_fraction  : float,
        valid_ratio           : float,
        forecast_hour         : int,
        seed                  : int,

        force_calc_baselines  : bool,
        save_cache_baselines  : bool,
        save_cache_NNTQ       : bool,

        do_run_model          : bool,  # False: just statistics

        # XXX_EVERY (in epochs)
        validate_every        : int,
        display_every         : int,
        plot_conv_every       : int,
        run_id                : int,

        cache_dir             : str   = "cache",
        # trials_csv_path       : str   = 'parameter_search.csv',
        num_worst_days        : int   = 20,
        split_diagnostics     : Split = Split.test,

        do_plot_statistics    : Optional[bool] = None,
        do_metamodel          : bool  = True,
            # False: no metamodel (e.g. Bayesian search on NNTQ only: loss_NNTQ
            #   does not depend on it); meta columns of the csv are then NaN
        NNTQ_variant          : Optional[int] = None,
        num_NNTQ_variants     : Optional[int] = None,
            # NNTQ_variant not None: use cached NNTQ variant NNTQ_variant (out of
            #   num_NNTQ_variants, built on first use), e.g. run i of a
            #   metamodel Bayesian trial
        verbose               : int   = 0
    ) -> Tuple[containers.DatasetBundle, Dict[str, Any], pd.DataFrame, \
               Dict[str, float], Dict[str, float], float, float] | None:

    np.   random.seed(seed)
    torch.manual_seed(seed)

    if verbose > 0:
        print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()))
    if do_run_model:  # else: we do not use the GPU
        if torch.cuda.is_available():
            if verbose > 0:
                print(f"GPU: {torch.cuda.get_device_name(0)}, "
                      f"CUDA version: {torch.version.cuda}, "
                      f"CUDNN version: {torch.backends.cudnn.version()}")
        elif verbose > 0:
            print("CUDA unavailable")
            print()


    # load data from csv and create pd.DataFrame
    _cache_fname = input_cache_fname(cache_dir, dict_input_csv_fnames)
                # 'input_data_full.pkl' if do_plot_statistics else 'input_data.pkl')
    num_steps_per_day = int(round(24*60/minutes_per_step))
    (df, names_cols, dates, Tavg_full, holidays_full, weights_regions, dates_df) = \
        load_and_create_df(
            dict_input_csv_fnames, _cache_fname, NNTQ_parameters['pred_length'],
            num_steps_per_day, minutes_per_step, do_plot_statistics, verbose)
    if os.path.exists(_cache_fname):   # the current one exists: the others are obsolete
        remove_other_input_caches(cache_dir, keep=_cache_fname)
    # print("Tavg_full:", Tavg_full)

    # print(f"num cols: cols_Y_regions {len(cols_Y_regions)}, "
    #       f"cols_features {len(cols_features)}, df.shape {df.shape}")

    if not do_run_model:   # /!\ was `~do_run_model`: ~ is bitwise NOT,
        return             #     ~True == -2 and ~False == -1 are both truthy

    num_time_steps = df.shape[0]

    if verbose > 0:
        # keep only arguments expected by `IO.print_model_summary`
        valid_parameters = \
            inspect.signature(IO.print_model_summary).parameters.keys()
        filtered_parameters = {k: v for k, v in NNTQ_parameters.items()
                                       if k in valid_parameters}
        filtered_meta_parameters = {
            'meta_'+k : v for k, v in metamodel_NN_parameters.items()
                                       if 'meta_'+k in valid_parameters}

    if verbose >= 2:
        IO.print_model_summary(
                minutes_per_step, num_steps_per_day,
                num_time_steps, names_cols['features'],
                **filtered_parameters, **filtered_meta_parameters
        )

        # correlation matrix for temperatures
        # utils.temperature_correlation_matrix(df)


    # create baselines (linera regreassion, random forest, gradient boosting)
    if verbose > 0:
        print("Doing linear regression and random forest...")


    train_split = int(len(df) * train_split_fraction)
    test_steps  = len(df)-train_split
    n_valid     = int(train_split * valid_ratio)

    input_length = NNTQ_parameters['input_length']
    assert input_length + 60 < test_steps,\
        f"input_length ({input_length}) > test_steps ({test_steps}) - 60"

    # ML baselines
    dict_series_baselines_GW = baselines.create_baselines(df,
        names_cols, dates_df,
        baseline_parameters, train_split, n_valid,
        cache_dir, save_cache_baselines,
        force_calculation = force_calc_baselines,
        verbose           = verbose
    )
    # names_cols['ML_preds'] = [f"consumption_{name}"
    #                           for name in dict_series_baselines_GW.keys()]

    _old_shape = df.shape
    _df_ML = pd.DataFrame(dict_series_baselines_GW)
    _df_ML.columns = [f"consumption_{name}" for name in _df_ML.columns]
    names_cols['ML_preds'] = list(_df_ML.columns)
    df = pd.concat([df, _df_ML], axis=1, join='inner')
    if verbose > 0:
        print(f"ML models added to features: df.shape {_old_shape} -> {df.shape}")


    # if NNTQ_parameters['use_ML_features']:
    #     if verbose >0 :
    #         print(f"ML models added to features: df.shape {_old_shape} -> {df.shape}")
    # elif verbose >0 :
    #     print(f"ML models not added to features: df.shape {df.shape}")

    valid_length = NNTQ_parameters['valid_length']



    # ============================================================
    # NNTQ: Neural Network predicting Quantiles with Transformers
    # ============================================================

    # do not rerun same NNTQ all the time for Bayesian search focused on metamodel
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

        key_str    = json.dumps( {
            "train_split_fraction": train_split_fraction,
            "test_steps"   : test_steps,
            "forecast_hour": forecast_hour,
            "forecast_tz"  : FORECAST_TZ,   # /!\ origins were in UTC before
            # /!\ were missing: another validation split, or other regions or
            #     weights, reloaded a model trained with different ones
            "n_valid"      : n_valid,
            "weights_regions": weights_regions,
            "num_worst_days": num_worst_days,  # /!\ was missing: stale worst days
            "cols_features": names_cols['features'],
            "dates_df"     : dates_df.to_json(orient='index')} |
            {key: value for key, value in NNTQ_parameters.items() if key!='device'} |
            # with ML features, the NNTQ depends on the baselines as well
            ({"baseline_parameters": baseline_parameters}
                 if NNTQ_parameters['use_ML_features'] else {}),
                sort_keys=True, default=str)
        cache_key  = hashlib.md5(key_str.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"NNTQ_preds_{cache_key}.pkl")



    # NNTQ training, with the current random seed
    def _train_NNTQ():
        # Create splits
        _data, _ = normalize_features(df, names_cols,
            NNTQ_parameters['use_ML_features'], weights_regions,
            minutes_per_step, dates, Tavg_full,
            train_split, n_valid,
            NNTQ_parameters['input_length'],
            NNTQ_parameters['pred_length'],
            NNTQ_parameters['features_in_future'],
            NNTQ_parameters['batch_size'],
            forecast_hour,
            verbose)

        # Create model
        NNTQ_model = containers.NeuralNet(**NNTQ_parameters,
            # regional errors in national units: the sum is the national error
                                          regions_to_nation=
                                              _data.scaler_Y_regions.scale_ /
                                              _data.scaler_y_nation .scale_[0],
                                          len_train_data= len(_data.train.loader),
                                              # optimizer steps (batches) per epoch,
                                              # NOT time steps: the LR schedule
                                              # advances once per batch
                                          num_features  = _data.num_features,
                                          weights_regions= weights_regions)

        # run training, validation, test
        #   -> (data, quantile_delta_coverage, avg_abs_worst_days_test_NN_median)
        return NNTQ_model.run(
                _data, Tavg_full, holidays_full,
                minutes_per_step, validate_every, display_every, plot_conv_every,
                cache_dir, num_worst_days, verbose)

    # metamodel search: one of the cached NNTQ variants (built if missing)
    if NNTQ_variant is not None:
        assert cache_dir is not None, "NNTQ variants require a cache_dir"
        assert num_NNTQ_variants is not None and \
            0 <= NNTQ_variant < num_NNTQ_variants, (NNTQ_variant, num_NNTQ_variants)
        _paths_variants, _ = NNTQ_variants_paths(cache_dir, cache_key,
                                                 num_NNTQ_variants)
        if not NNTQ_variants_cached(cache_dir, cache_key, num_NNTQ_variants):
            print(f"Building {num_NNTQ_variants} NNTQ variants "
                  f"({num_NNTQ_variants + 2} trainings)...")
            build_NNTQ_variants(_train_NNTQ, _paths_variants, cache_dir, cache_key)
            np.   random.seed(seed)   # back to this run's seed (metamodel)
            torch.manual_seed(seed)
        cache_path = _paths_variants[NNTQ_variant]

    # either load...
    if cache_dir is not None and os.path.exists(cache_path):
        if verbose > 0:
            print(f"Loading NNTQ predictions from: {cache_path}...")
        with open(cache_path, "rb") as f:
            (data, quantile_delta_coverage,
             avg_abs_worst_days_test_NN_median) = pickle.load(f)

        # Without ML features the NNTQ does not depend on the baselines, so the
        #   cache key ignores them: the cached bundle holds the baseline
        #   predictions of whichever run created it. Replace them with the ones
        #   just computed (they vary in the metamodel Bayesian search).
        if not NNTQ_parameters['use_ML_features']:
            _df_ML_new = df[names_cols['ML_preds']].astype(np.float32)
            assert len(_df_ML_new) == len(dates), (len(_df_ML_new), len(dates))
            _df_ML_new.index = dates   # df has a positional index (reset_index)
            _df_ML_new.columns = [c.split('_')[1] for c in _df_ML_new.columns]
                # same naming as in architecture.make_X_and_y
            for _split in (data.train, data.valid, data.test, data.complete):
                _old = _split.dict_preds_ML
                assert list(_old.keys()) == list(_df_ML_new.columns), \
                    (list(_old.keys()), list(_df_ML_new.columns))
                _dates = list(next(iter(_old.values())).keys())
                _split.dict_preds_ML = {
                    name: series.astype(np.float64)
                    for name, series in _df_ML_new.loc[_dates].items()}

    # ... or compute
    else:
        (data, quantile_delta_coverage, avg_abs_worst_days_test_NN_median) = \
            _train_NNTQ()

        # Save pickle
        if cache_dir is not None and save_cache_NNTQ:
            with open(cache_path, "wb") as f:
                pickle.dump((data.for_cache(), quantile_delta_coverage,
                             avg_abs_worst_days_test_NN_median), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            if verbose > 0:
                print(f"Saved NNTQ predictions to: {cache_path}")


    # ============================================================
    # METAMODEL
    # ============================================================

    # same random state for the metamodels whether the NNTQ was just trained
    #   (which consumed random numbers) or loaded from the cache
    np.   random.seed(seed)
    torch.manual_seed(seed)

    # metamodel LR
    if do_metamodel:
        data.calculate_metamodel_LR(
            split_active = Split.valid, min_weight=0.15, verbose=verbose)

        if verbose > 0:
            print(f"weights_meta_LR [%]: "
              f"{ {k: round(v*100, 1) for k, v in data.weights_meta_LR.items()}}")
    # t_metamodel_end = time.perf_counter()
    # if verbose >= 2:
    #     print(f"metamodel_LR took: {time.perf_counter() - t_metamodel_start:.2f} s")


    # NN metamodel
    # ============================================================

    if do_metamodel:
        data.calculate_metamodel_NN(names_cols['features'], valid_length, Split.valid,
                                    metamodel_NN_parameters, verbose)
        avg_weights_meta_NN = data.avg_weights_meta_NN
    else:
        avg_weights_meta_NN = None   # postprocess writes NaN


    names_baseline= {}  # if you like it crowded: {'LGBM', 'LR', 'RF'}
    names_meta    = {'LR', 'NN'} if do_metamodel else set()

    if verbose > 0:
        data.train.compare_models(unit="GW", verbose=verbose)
        data.valid.compare_models(unit="GW", verbose=verbose)
    # test period in two: the first half is the objective of the searches,
    #   the second half is only reported (/!\ both used to be the whole period)
    _cut = containers.search_period_end(data.test)
    search_metrics = data.test.compare_models(unit="GW", period=(None, _cut))
    test_metrics   = data.test.compare_models(unit="GW", verbose=verbose,
                                              period=(_cut, None))
    quantile_delta_coverage_test = {
        f"q{int(100*tau)}": utils.quantile_coverage(
            containers._in_period(data.test.true_nation_GW, (_cut, None)),
            containers._in_period(data.test.dict_preds_NNTQ[f"q{int(100*tau)}"],
                                  (_cut, None))) - tau
        for tau in NNTQ_parameters['quantiles']}


    _plot_quantiles = ['q25', 'q50', 'q75']
    if verbose > 0:
        print("Plotting test results...")
    if verbose >= 3:
        data.train.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            # temperature_full=Tavg_full,
            num_steps_per_day=num_steps_per_day,
            quantiles=_plot_quantiles)
    if verbose > 0:
        data[split_diagnostics].plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            # temperature_full=Tavg_full,
            num_steps_per_day=num_steps_per_day,
            quantiles=_plot_quantiles)

        # plots.quantiles(
        #     data.test.true_nation_GW,
        #     data.test.dict_preds_NNTQ,
        #     q_low = "q10",
        #     q_med = "q50",
        #     q_high= "q90",
        #     baseline_series=data.test.dict_preds_ML,
        #     title = "Electricity consumption forecast (NN quantiles), test",
        #     dates = data.test.dates[-(8*num_steps_per_day):]
        # )


    if verbose >= 2:
        plot_statistics.thermosensitivity_per_time_of_day(
             data_split = data.complete,
             thresholds_degC = [('<=', 10), ('<=', 2), ('>=', 23)],
             ylim = [0, 2.5],
             num_steps_per_day=num_steps_per_day
        )

        plot_statistics.thermosensitivity_per_temperature_model(
             data_split = data.complete,
             thresholds_degC= np.arange(-1., 26.5, step=0.1),
             # np.arange(-1.2, 13+6, step=0.1),  np.arange(19-6, 26.7, step=0.1)],
             num_steps_per_day=num_steps_per_day
        )

        plot_statistics.thermosensitivity_per_temperature_model(
             data_split = data.train,
             thresholds_degC= np.arange(-1., 26.5, step=0.1),
             # np.arange(-1.2, 13+6, step=0.1),  np.arange(19-6, 26.7, step=0.1)],
             num_steps_per_day=num_steps_per_day
        )



    dict_row, (_loss_NNTQ, _loss_meta) = postprocess(
        baseline_parameters, NNTQ_parameters, metamodel_NN_parameters,
        len(names_cols['features']),
        test_metrics, quantile_delta_coverage, avg_weights_meta_NN,
        avg_abs_worst_days_test_NN_median, run_id, verbose,
        df_metrics_search=search_metrics,
        quantile_delta_coverage_test=quantile_delta_coverage_test)

    if torch.cuda.is_available():
        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    return data, dict_row, test_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
        (num_worst_days, avg_abs_worst_days_test_NN_median), (_loss_NNTQ, _loss_meta)





# ============================================================
# RUNNING MODEL ONCE, OR FOR A SEARCH (MC OR BAYES))
# ============================================================

def run_model(
        mode                : str,  # in ['once', 'random', 'Bayes_NNTQ', 'Bayes_meta',
                                    #     'statistics', 'stats_only', 'load_input']
        num_trials          : Optional[int],

        # configuration bundles
        baseline_parameters : Dict[str, Dict[str, Any]],
        NNTQ_parameters     : Dict[str, Any],
        metamodel_NN_parameters:Dict[str, Any],
        dict_input_csv_fnames: Dict[str, str],

        # statistics of the dataset
        minutes_per_step    : int,
        train_split_fraction: float,
        valid_ratio         : float,
        forecast_hour       : int,
        seed                : int,

        force_calc_baselines: bool,

        # XXX_EVERY (in epochs)
        validate_every      : Optional[int] = None,
        display_every       : Optional[int] = None,
        plot_conv_every     : Optional[int] = None,

        cache_dir           : str  = "cache",
        num_worst_days      : int  = 20,
        verbose             : Optional[int]  = 0
    ) -> None:
    # Tuple[Dict[str, Any], pd.DataFrame, \
    #            Dict[str, float], Dict[str, float], float, float]:


    if mode in ['once', 'load_input', 'stats_only', 'statistics']:
            # single model run (or none)
        if num_trials is not None and num_trials > 1:
            # /!\ was `num_trials in locals()`: tests the VALUE as a variable
            #     name, always False
            warnings.warn(f"num_runs ({num_trials}) will not be used")

        if 'stat' in mode:    # `stats_only` or `statistics`
            _split_diagnostics   = Split.complete
        else:
            _split_diagnostics   = Split.test


        _returned = \
                run_model_once(
                # configuration bundles
                baseline_parameters= baseline_parameters,
                NNTQ_parameters   = NNTQ_parameters,
                metamodel_NN_parameters= metamodel_NN_parameters,

                dict_input_csv_fnames= dict_input_csv_fnames,
                # trials_csv_path   = 'parameter_search_one-off.csv',

                # statistics of the dataset
                minutes_per_step  = minutes_per_step,
                train_split_fraction=train_split_fraction,
                valid_ratio       = valid_ratio,
                forecast_hour     = forecast_hour,
                seed              = seed,

                force_calc_baselines=force_calc_baselines,
                save_cache_baselines= True,
                save_cache_NNTQ     = True,

                do_run_model        = mode not in ['stats_only', 'load_input'],

                # XXX_EVERY (in epochs)
                validate_every    = validate_every,
                display_every     = display_every,
                plot_conv_every   = plot_conv_every,
                run_id            = 0,

                cache_dir         = cache_dir,
                split_diagnostics = _split_diagnostics,

                do_plot_statistics= 'stat' in mode,  # ['statistics',  'stats_only']
                verbose           = verbose
            )

        if mode in ['stats_only', 'load_input']: # no model
            assert _returned is None
            return

        # model ran
        assert _returned is not None
        data, dict_row, test_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
            (num_worst_days, avg_abs_worst_days_test_NN_median), \
            (_loss_NNTQ, _loss_meta) = _returned

        append_csv_row(pd.DataFrame([dict_row]), 'parameter_search_one-off.csv')

        if verbose > 0:
            print(f"loss_NNTQ = {_loss_NNTQ:.2f}, loss_meta = {_loss_meta:.2f}")

        num_steps_per_day = int(round(24*60/minutes_per_step))

        if 'stat' in mode:

            # names_baseline= {}  # if you like it crowded: {'LGBM', 'LR', 'RF'}
            # names_meta    = {'LR', 'NN'}
            # _plot_quantiles = ['q25', 'q50', 'q75']

            # # entire length of the data
            # data.complete.plots_diagnostics(
            #     names_baseline = names_baseline, names_meta = names_meta,
            #     # temperature_full=Tavg_full,
            #     num_steps_per_day=num_steps_per_day,
            #     quantiles=_plot_quantiles)

            plot_statistics.thermosensitivity_per_time_of_day(
                 data_split = data.train,
                 thresholds_degC = [('<=', 10), ('<=', 2), ('>=', 23)],
                 ylim = [0, 3.],
                 num_steps_per_day=num_steps_per_day
            )

            # plot_statistics.thermosensitivity_per_temperature_model(
            #      data_split = data.train,
            #      thresholds_degC= np.arange(-1., 26.5, step=0.1),
            #      # np.arange(-1.2, 13+6, step=0.1),  np.arange(19-6, 26.7, step=0.1)],
            #      num_steps_per_day=num_steps_per_day
            # )
            # # plot_statistics.drift_with_time(
            # #      dfs['consumption']['consumption_GW'],
            # #      dfs['temperature']["Tavg_degC"],
            # #      num_steps_per_day=num_steps_per_day
            # # )


    else:   # search for hyperparameters
        # the searches validate every epoch (early stopping) and display nothing
        #   (/!\ were `x in locals()`: tested the VALUE as a variable name, never
        #    true; display_every and plot_conv_every are not worth a warning)
        if validate_every is not None and validate_every != 1:
            warnings.warn(f"validate_every ({validate_every}) is not used: "
                          f"the searches validate every epoch")

        if mode in ['random', 'Monte Carlo', 'MC']:
            # /!\ no longer maintained
            parameter_search_function = MC_search.run_Monte_Carlo_search
            stage = Stage.all  # the only one implemented

        elif 'Bayes' in mode:  # works for `Bayes` and `Bayesian`
            parameter_search_function = Bayes_search.run_Bayes_search
            if 'NNTQ' in mode:
                stage = Stage.NNTQ
            elif 'meta' in mode: # works for `meta` and `metamodel`
                stage = Stage.meta
            elif 'all' in mode:
                stage = Stage.all
            else:
                raise ValueError(f"`{mode}` is not a valid mode")

        else:
            raise ValueError(f"`{mode}` is not a valid mode")

        parameter_search_function(
                stage               = stage,
                num_trials          = num_trials,
                trials_csv_path     = f'parameter_search_{stage.value}.csv',

                # configuration bundles
                base_baseline_params= baseline_parameters,
                base_NNTQ_params    = NNTQ_parameters,
                base_meta_NN_params = metamodel_NN_parameters,
                dict_input_csv_fnames= dict_input_csv_fnames,

                # statistics of the dataset
                minutes_per_step    = minutes_per_step,
                train_split_fraction= train_split_fraction,
                valid_ratio         = valid_ratio,
                forecast_hour       = forecast_hour,
                seed                = seed,
                force_calc_baselines= force_calc_baselines,

                cache_dir           = cache_dir,
                verbose             = verbose
            )




# -------------------------------------------------------
# losses (NNTQ and metamodel separately)
# -------------------------------------------------------

def expand_sequence(name: str, values: Sequence, length: int,
                    prefix: Optional[str]="", fill_value=np.nan) -> Dict[str, Any]:
    """
    Expand a list/tuple into fixed-length columns.
    Shorter lists are padded with fill_value.
    Longer lists raise an error (by default).
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be list or tuple, got {type(values)}")

    if len(values) > length:
        raise ValueError(f"{name} length {len(values)} > fixed length {length}")

    output = {}
    for i in range(length):
        output[f"{prefix}{name}_{i}"] = values[i] if i < len(values) else fill_value

    return output

def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dict using namespaced keys.
    Example:
      {"rf": {"n_estimators": 500}} →
      {"baseline__rf__n_estimators": 500}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def loss_NNTQ(
         quantile_delta_coverage: Dict[str, float],
         avg_abs_worst_days     : float,
             # in practice, should be specifically test_NN_median

         # constants:
         scale                  : float = 100.,   # multiplies everything
         weights_coverage       : list  = [7, 6, 5, 4, 3, 2, 1],  # from max to min
         weight_worst_days      : float = 0.02,

         quantile_weights       : Dict[str, float] = \
             {'q10': 2., 'q25': 1.5, 'q50': 1., 'q75': 1.5, 'q90': 2.,
              'bias': 1., 'spread': 2.},
                 # q10 off by 5% (10% -> 5%) is worse than w/ q50 (50% -> 45%)
                 # TODO quantiles are currently hard-coded
         verbose                : int  = 0
    ) -> float:
    _dict_coverage = quantile_delta_coverage.copy()
        # dict of quantiles: for each, measured - theoretical (e.g. 53% - 50% = 3%)


    # add spread: q90 - q10 (again, measured - theoretical)
    _spread = (_dict_coverage['q90'] - _dict_coverage['q10'] + \
               _dict_coverage['q75'] - _dict_coverage['q25']) / 2

    # if ('q10' in _dict_coverage.keys() and 'q90' in _dict_coverage.keys()):
    #     _spread = _dict_coverage['q90'] - _dict_coverage['q10']
    # elif ('q25' in _dict_coverage.keys() and 'q75' in _dict_coverage.keys()):
    #     _spread = _dict_coverage['q75'] - _dict_coverage['q25']
    # else:
    #     _spread = 0.5  # a.k.a. a lot

    _dict_coverage['spread'] = abs(_spread)
    # _dict_coverage['spread'] = (
    #     max(0, -_spread) * 1.  +
    #     max(0,  _spread) * 0.25
    # )  # asymmetrically penalizing narrow distributions (most common problem)
    # print(f"_spread = {_spread*100:.2f}% => "
    #       f"_dict_coverage['spread'] [%]: {_dict_coverage['spread']*100:.2f}")


    # add bias, i.e. signed error
    _bias_mean = np.mean(list(_dict_coverage.values()))
    # _dict_pc = {k : round(100*v, 2) for (k, v) in _dict_coverage.items()}
    # print(f"_bias_mean [%] = {_bias_mean*100:.2f} = mean({_dict_pc})")

    _dict_coverage['bias'] = float( _bias_mean)
    # _dict_coverage['bias'] = float (
    #     max(0,  _bias_mean) * 1.  +
    #     max(0, -_bias_mean) * 0.25
    # )  # asymmetrically penalizing upward drift (most common problem)
    # print(f"_dict_coverage['bias'] [%]: {_dict_coverage['bias']*100:.2f}")


    # two layers of weights
    _list_weighted_loss_coverage = [abs(gap) * quantile_weights[q]
                for q, gap in _dict_coverage.items()]

    _sum_weights_coverage   = sum(weights_coverage)
    _sorted_list = sorted(_list_weighted_loss_coverage, reverse=True)
    _loss_quantile_coverage = sum(loss * weight / _sum_weights_coverage
                    for loss, weight in zip(_sorted_list, weights_coverage))
                # was np.max (_list_weighted_loss_coverage)

    _loss = float(round(scale * (_loss_quantile_coverage + \
                                 avg_abs_worst_days * weight_worst_days), 3))

    if verbose > 0:
        print(f"loss_NNTQ = {_loss:.2f} = "
              f"w_sum({[round(e*scale, 2) for e in _list_weighted_loss_coverage]}) + "
              f"{avg_abs_worst_days:.2f} * {weight_worst_days * scale}")

    return _loss

def loss_meta(
         metrics        : Dict[str, float] | pd.DataFrame,

         # constants
         metric_weights : Dict[str, float] = \
             {'bias': 2., 'RMSE': 1., 'MAE': 1.},
                 # RMSE and MAE are variants of each other, bias is different

         model_weights  : Dict[str, float] = \
             {'NNTQ': 0., 'LR': 1., 'RF': 1., 'LGBM': 1.,
              'meta_LR': 1.5, 'meta_NN': 2.},
            # LR, RF and LGBM are just underlying models to the metamodels;
            #      improving them intrinsically is good, but secondary

         verbose        : int  = 0
    ) -> float:

    dict_by_metric = {k: [] for k in list(metric_weights.keys())}
    _list_models   = []  # models actually used

    if isinstance(metrics, dict):
        for key, value in metrics.items():
            parts  = key.replace("test_", "").split('_')
            model  = '_'.join(parts[:-1])  # Ex: "NNTQ", "meta_NN", etc.
            metric = parts[-1]             # Ex: "bias", "RMSE", etc.

            dict_by_metric[metric].append(model_weights[model] * abs(value))
            _list_models.append(model)

        for metric in dict_by_metric.keys():
            assert len(dict_by_metric[metric]) == len(model_weights)

    else:  # pd.df
        for metric in metrics.columns:
            dict_by_metric[metric] = [model_weights[model.replace(" ", "_")] * \
                    abs(metrics[metric].loc[model])
                        for model in metrics.index]
            _list_models = list(metrics.index)

    _sum_model_weights = np.sum([w for (m, w) in model_weights.items()
                                 if m in _list_models])  # only those actually used
    # _sum_model_weights = np.sum(list(model_weights.values()))

    dict_avg_metrics = {metric: round(float(np.sum(_list) / _sum_model_weights), 5)
                            for (metric, _list) in dict_by_metric.items()}

    weighted_list_metrics = [value * metric_weights[metric]
                        for (metric, value) in dict_avg_metrics.items()]

    _sum_metric_weights = np.sum(list(metric_weights.values()))
    avg_metric = round(np.sum(weighted_list_metrics) / _sum_metric_weights, 5)

    if verbose > 0:
        print(f"loss_meta: dict_avg_metrics = {dict_avg_metrics}")
        print(f"loss_meta: weighted_list_metrics = {weighted_list_metrics}")
        # print(avg_metric)

    return float(round(avg_metric, 4))



# -------------------------------------------------------
# modify existing csv life
# -------------------------------------------------------

# /!\ create a copy of the csv file before modifying it (just in case)


def recalculate_loss(csv_path: str,
                     verbose : int   = 0) -> None:
    """Recompute loss_NNTQ and loss_meta from the metric columns (after a
    change of the loss functions) and overwrite the csv.
    Only single-run rows are recomputed: in a multi-run row the metrics are
    those of the last run while the losses average all the runs (clean_avg),
    which the metrics cannot reproduce; those rows are left unchanged."""
    # Load the CSV file containing runs so far
    results_df = pd.read_csv(csv_path, index_col=False)

    # clean up dates
    results_df['timestamp'] = pd.to_datetime(
        results_df['timestamp'],
        errors   = 'coerce'
            # /!\ was also dayfirst=True: the csv timestamps are ISO
            #     (YYYY-MM-DD, as written by pandas), not day-first
    )

    # print(pd.concat([results_df[['timestamp']], dates], axis=1))

    _list_losses_NNTQ = []
    _list_losses_meta = []

    for index, row in results_df.iterrows():
        if row.get('num_runs', 1) > 1:   # losses averaged over the runs: keep
            _list_losses_NNTQ.append(row['loss_NNTQ'])
            _list_losses_meta.append(row['loss_meta'])
            continue
        # objective: first half of the test period (search_* columns)
        flat_metrics = {f"test_{m}_{k}": row[f"search_{m}_{k}"]
                        for m in ['NNTQ', 'LR', 'RF', 'LGBM', 'meta_LR', 'meta_NN']
                        for k in ['bias', 'RMSE', 'MAE']}

        quantile_delta_coverage = \
            row[['q10', 'q25', 'q50', 'q75', 'q90']].to_dict()

        avg_abs_worst_days_search = row['avg_abs_worst_days_search']

        _loss_NNTQ = loss_NNTQ(quantile_delta_coverage, avg_abs_worst_days_search,
                               verbose=verbose)
        _list_losses_NNTQ.append(_loss_NNTQ)

        _loss_meta = loss_meta(metrics = flat_metrics)
        _list_losses_meta.append(_loss_meta)

    # print(_list_losses_NNTQ, _list_losses_meta)

    results_df['loss_NNTQ'] = _list_losses_NNTQ
    results_df['loss_meta'] = _list_losses_meta

    results_df.to_csv(csv_path, index=False)


# recalculate_loss('parameter_search_NNTQ.csv')



def enforce_ranges(csv_path   : str,
                   dict_ranges: Dict[str, Tuple],
                   verbose    : int   = 0) -> None:

    # Load the CSV file containing runs so far
    results_df = pd.read_csv(csv_path, index_col=False)

    # clean up dates
    results_df['timestamp'] = pd.to_datetime(
        results_df['timestamp'],
        errors   = 'coerce'
            # /!\ was also dayfirst=True: the csv timestamps are ISO
            #     (YYYY-MM-DD, as written by pandas), not day-first
    )

    # Filter rows based on the ranges
    mask = pd.Series(True, index=results_df.index)
    for col_name, _range in dict_ranges.items():
        if col_name in results_df.columns:
            mask &= (results_df[col_name] >= _range[0]) & \
                    (results_df[col_name] <= _range[1])

    results_df = results_df[mask]

    results_df.to_csv(csv_path, index=False)

# enforce_ranges('parameter_search_NNTQ.csv',
#                {
#                    # 'ffn_size'  : [ 0,   4   ],
#                    # 'num_layers': [ 0,   3   ],
#                    # 'dropout'   : [ 0.,  0.15],
#                    # 'batch_size': [64, 128   ],
#                    # 'patience'  : [ 3,   6   ],
#                    # 'min_delta' : [ 0.02,0.04],
#                    'loss_NNTQ' : [25.,199.  ]
#                })

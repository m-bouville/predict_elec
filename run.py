import gc
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
from   constants import Stage

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
                       cache_fname        : str,
                       pred_length        : int,
                       num_steps_per_day  :int,
                       minutes_per_step   : int,
                       verbose            : int  = 0) \
        -> Tuple[pd.DataFrame, Dict[str, List[str]],
                 pd.Series, pd.Series, pd.Series, List[float]]:

    df, dates_df, weights_regions = utils.df_features(
            dict_input_csv_fnames, cache_fname, pred_length,
            num_steps_per_day, minutes_per_step, verbose)


    # ---- Identify columns ----
    col_y_nation = "consumption_GW"

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    cols_Y_regions = [c for c in numeric_cols
          if ('consumption' in c) and (c != col_y_nation) and \
              ('consumption_SMA' not in c)]

    # All features except the target and date are predictors
    cols_features = [
        c for c in numeric_cols
        if c not in ["year", 'month', 'timeofday', col_y_nation] + cols_Y_regions
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
        print(dates_df)

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


    return (df, {"features": cols_features,  "Y_regions": cols_Y_regions,
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
                avg_abs_worst_days_test:float,
                run_id                : int,
                verbose               : int   = 0
                ) -> [Dict[str, Any], [float, float]]:

    flat_metrics = {}
    for model in df_metrics.index:
        for metric in df_metrics.columns:
            key = f"test_{model}_{metric}".replace(" ", "_")
            flat_metrics[key] = float(df_metrics.loc[model, metric])

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

    _loss_NNTQ = round(loss_NNTQ(quantile_delta_coverage, avg_abs_worst_days_test,
                           verbose=verbose), 2)
    _loss_meta = round(loss_meta(flat_metrics, verbose=verbose), 4)

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
        **flat_metrics,   # bias, RMSE, MAE
        'avg_abs_worst_days_test': avg_abs_worst_days_test,
        'num_runs' : 1,
        "loss_NNTQ": _loss_NNTQ,
        "loss_meta": _loss_meta,
    }
    # print(row)

    return row, (_loss_NNTQ, _loss_meta)


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

        # XXX_EVERY (in epochs)
        validate_every        : int,
        display_every         : int,
        plot_conv_every       : int,
        run_id                : int,

        cache_dir             : str  = "cache",
        # trials_csv_path       : str  = 'parameter_search.csv',
        num_worst_days        : int  = 20,
        verbose               : int  = 0
    ) -> Tuple[Dict[str, Any], pd.DataFrame, \
               Dict[str, float], Dict[str, float], float, float]:

    np.   random.seed(seed)
    torch.manual_seed(seed)

    if verbose > 0:
        print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()))
    if torch.cuda.is_available():
        if verbose > 0:
            print(f"GPU: {torch.cuda.get_device_name(0)}, "
                  f"CUDA version: {torch.version.cuda}, "
                  f"CUDNN version: {torch.backends.cudnn.version()}")
    elif verbose > 0:
        print("CUDA unavailable")
        print()


    # load data from csv and create pd.DataFrame
    num_steps_per_day = int(round(24*60/minutes_per_step))
    (df, names_cols, dates, Tavg_full, holidays_full, weights_regions, dates_df) = \
        load_and_create_df(
            dict_input_csv_fnames, None, NNTQ_parameters['pred_length'],
            num_steps_per_day, minutes_per_step, verbose)

    # print(f"num cols: cols_Y_regions {len(cols_Y_regions)}, "
    #       f"cols_features {len(cols_features)}, df.shape {df.shape}")


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
            "cols_features": names_cols['features'],
            "dates_df"     : dates_df.to_json(orient='index')} |
            {key: value for key, value in NNTQ_parameters.items() if key!='device'},
                sort_keys=True)
        cache_key  = hashlib.md5(key_str.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"nntq_preds_{cache_key}.pkl")



    # either load...
    if cache_dir is not None and os.path.exists(cache_path):
        if verbose > 0:
            print(f"Loading NNTQ predictions from: {cache_path}...")
        with open(cache_path, "rb") as f:
            (data, quantile_delta_coverage,
             avg_abs_worst_days_test_NN_median) = pickle.load(f)

    # ... or compute
    else:
        # Create splits
        data, X_test_scaled = normalize_features(df, names_cols,
            NNTQ_parameters['use_ML_features'], weights_regions,
            minutes_per_step, dates, Tavg_full,
            train_split, n_valid,
            NNTQ_parameters['input_length'],
            NNTQ_parameters['pred_length'],
            NNTQ_parameters['features_in_future'],
            NNTQ_parameters['batch_size'],
            forecast_hour,
            verbose)

        del df  # pd.DataFrame no longer needed, we use np.ndarray now
        gc.collect()


        # print(f"X_test_scaled.shape {X_test_scaled.shape}")

        # Create model

        # # for NNTQ: features may include ML preds
        # _num_features = data.num_features + \
        #     NNTQ_parameters['use_ML_features'] * len(dict_series_baselines_GW.keys())

        NNTQ_model = containers.NeuralNet(**NNTQ_parameters,
                                          len_train_data= len(data.train),
                                          num_features  = data.num_features,
                                          weights_regions= weights_regions)

        # run training, validation, test
        (data, quantile_delta_coverage, avg_abs_worst_days_test_NN_median) = \
            NNTQ_model.run(
                data, Tavg_full, holidays_full,
                minutes_per_step, validate_every, display_every, plot_conv_every,
                cache_dir, num_worst_days, verbose)

        # Save pickle
        if cache_dir is not None and save_cache_NNTQ:
            with open(cache_path, "wb") as f:
                pickle.dump((data, quantile_delta_coverage,
                             avg_abs_worst_days_test_NN_median), f)
            if verbose > 0:
                print(f"Saved NNTQ predictions to: {cache_path}")


    # ============================================================
    # METAMODEL
    # ============================================================

    # metamodel LR
    data.calculate_metamodel_LR(
        split_active='valid', min_weight=0.15, verbose=verbose)

    if verbose > 0:
        print(f"weights_meta_LR [%]: "
          f"{ {k: round(v*100, 1) for k, v in data.weights_meta_LR.items()}}")
    # t_metamodel_end = time.perf_counter()
    # if verbose >= 2:
    #     print(f"metamodel_LR took: {time.perf_counter() - t_metamodel_start:.2f} s")


    # NN metamodel
    # ============================================================

    data.calculate_metamodel_NN(names_cols['features'], valid_length, 'valid',
                                metamodel_NN_parameters, verbose)
    avg_weights_meta_NN = data.avg_weights_meta_NN


    names_baseline= {}  # if you like it crowded: {'LGBM', 'LR', 'RF'}
    names_meta    = {'LR', 'NN'}

    if verbose > 0:
        data.train.compare_models(unit="GW", verbose=verbose)
        data.valid.compare_models(unit="GW", verbose=verbose)
    test_metrics = data.test .compare_models(unit="GW", verbose=verbose)


    _plot_quantiles = ['q25', 'q50', 'q75']
    if verbose > 0:
        print("Plotting test results...")
    if verbose >= 3:
        data.train.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            temperature_full=Tavg_full, num_steps_per_day=num_steps_per_day,
            quantiles=_plot_quantiles)
    if verbose > 0:
        data.test.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            temperature_full=Tavg_full, num_steps_per_day=num_steps_per_day,
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

        plot_statistics.thermosensitivity_per_temperature(
             data_split = data.complete,
             thresholds_degC= np.arange(-1., 26.5, step=0.1),
             # np.arange(-1.2, 13+6, step=0.1),  np.arange(19-6, 26.7, step=0.1)],
             num_steps_per_day=num_steps_per_day
        )

        plot_statistics.thermosensitivity_per_temperature(
             data_split = data.train,
             thresholds_degC= np.arange(-1., 26.5, step=0.1),
             # np.arange(-1.2, 13+6, step=0.1),  np.arange(19-6, 26.7, step=0.1)],
             num_steps_per_day=num_steps_per_day
        )




    # final cleanup to free pinned memory and intermediate arrays
    try:
        del data
    except NameError:
        pass


    dict_row, (_loss_NNTQ, _loss_meta) = postprocess(
        baseline_parameters, NNTQ_parameters, metamodel_NN_parameters,
        len(names_cols['features']),
        test_metrics, quantile_delta_coverage, avg_weights_meta_NN,
        avg_abs_worst_days_test_NN_median, run_id, verbose)

    if torch.cuda.is_available():
        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    return dict_row, test_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
        (num_worst_days, avg_abs_worst_days_test_NN_median), (_loss_NNTQ, _loss_meta)





# ============================================================
# RUNNING MODEL ONCE, OR FOR A SEARCH (MC OR BAYES))
# ============================================================

def run_model(
        mode                : str,  # in ['once', 'random', 'Bayes_NNTQ', 'Bayes_meta']
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
    ) -> Tuple[Dict[str, Any], pd.DataFrame, \
               Dict[str, float], Dict[str, float], float, float]:

    if mode == 'once':  # single run
        if num_trials in locals() and num_trials > 1:
            warnings.warn(f"num_runs ({num_trials}) will not be used")

        dict_row, test_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
            (num_worst_days, avg_abs_worst_days_test_NN_median), \
            (_loss_NNTQ, _loss_meta) = \
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

                # XXX_EVERY (in epochs)
                validate_every    = validate_every,
                display_every     = display_every,
                plot_conv_every   = plot_conv_every,
                run_id            = 0,

                cache_dir         = cache_dir,
                verbose           = verbose
            )

        df_row = pd.DataFrame([dict_row])
        df_row.to_csv(
            'parameter_search_one-off.csv',
            mode   = "a",
            header = not os.path.exists('parameter_search_one-off.csv'),
            index  = False,
            float_format="%.6f"
        )

        if verbose > 0:
            print(f"loss_NNTQ = {_loss_NNTQ:.2f}, loss_meta = {_loss_meta:.2f}")

    else:   # search for hyperparameters
        # no display => some arguments are not used
        if validate_every in locals() and validate_every > 0:
            warnings.warn(f"validate_every ({validate_every}) will not nbe used")
        if display_every in locals() and validate_every > 0:
            warnings.warn(f"display_every ({display_every}) will not nbe used")
        if plot_conv_every in locals() and validate_every > 0:
            warnings.warn(f"plot_conv_every ({plot_conv_every}) will not nbe used")

        if mode in ['random', 'Monte Carlo', 'MC']:
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
         weights_coverage       : list  = [4, 3, 2, 1, 0, 0, 0],  # from max to min
         weight_worst_days      : float = 0.02,

         quantile_weights       : Dict[str, float] = \
             {'q10': 2., 'q25': 1.5, 'q50': 1., 'q75': 1.5, 'q90': 2.,
              'bias': 1., 'spread': 2.},
                 # q10 off by 5% (10% -> 5%) is worse than w/ q50 (50% -> 45%)
                 # TODO quantiles are currently hard-coded
         verbose                : int  = 0
    ) -> float:
    _quantile_delta_coverage = quantile_delta_coverage.copy()
        # dict of quantiles: for each, measured - theoretical (e.g. 53% - 50% = 3%)

    # add spread: q90 - q10 (again, measured - theoretical)
    _quantile_delta_coverage['spread'] = max(0, _quantile_delta_coverage['q90'] - \
                                                _quantile_delta_coverage['q10'])

    # add bias, i.e. signed error
    _bias_mean = np.mean(list(_quantile_delta_coverage.values()))
    _bias_penalty = (  # asymmetrically penalizing 'everything above the truth'
        max(0,  _bias_mean) * 1.  +
        max(0, -_bias_mean) * 0.1
    )
    _quantile_delta_coverage['bias']= float(_bias_penalty)

    # two layers of weights
    _list_weighted_loss_coverage = [abs(gap) * quantile_weights[q]
                for q, gap in _quantile_delta_coverage.items()]

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
              'meta_LR': 2., 'meta_NN': 4.},
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
    # Load the CSV file containing runs so far
    results_df = pd.read_csv(csv_path, index_col=False)

    # clean up dates
    results_df['timestamp'] = pd.to_datetime(
        results_df['timestamp'],
        errors   = 'coerce',
        infer_datetime_format=True,
        dayfirst = True
    )

    # print(pd.concat([results_df[['timestamp']], dates], axis=1))

    _list_losses_NNTQ = []
    _list_losses_meta = []

    for index, row in results_df.iterrows():
        flat_metrics = (row \
        [['test_NN_bias',     'test_NN_RMSE',     'test_NN_MAE',
          'test_LR_bias',     'test_LR_RMSE',     'test_LR_MAE',
          'test_RF_bias',     'test_RF_RMSE',     'test_RF_MAE',
          'test_GB_bias',     'test_GB_RMSE',     'test_GB_MAE',
          'test_meta_LR_bias','test_meta_LR_RMSE','test_meta_LR_MAE',
          'test_meta_NN_bias','test_meta_NN_RMSE','test_meta_NN_MAE']]).to_dict()

        quantile_delta_coverage = \
            row[['q10', 'q25', 'q50', 'q75', 'q90']].to_dict()

        avg_abs_worst_days_test = row[['avg_abs_worst_days_test']].iloc[0]

        _loss_NNTQ = loss_NNTQ(quantile_delta_coverage, avg_abs_worst_days_test,
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
        errors   = 'coerce',
        infer_datetime_format=True,
        dayfirst = True
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

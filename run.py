import gc
# import sys
import inspect
import os

from   typing   import Dict, Any, Optional, List, Tuple, Sequence

import time
from   datetime import datetime

import torch

import numpy  as np
import pandas as pd


import containers, architecture, utils, LR_RF, IO, plots  # losses, metamodel,


# system dimensions
# B = BATCH_SIZE
# L = INPUT_LENGTH
# H = prediction horizon  =  PRED_LENGTH
# V = validation horizon  = VALID_LENGTH
# Q = number of quantiles = len(quantiles)
# F = number of features



# ============================================================
# LOAD DATA FROM CSV AND CREATE DATAFRAME
# ============================================================

def load_and_create_df(dict_fnames     : Dict[str, str],
                       cache_fname     : str,
                       pred_length     : int,
                       num_steps_per_day:int,
                       minutes_per_step: int,
                       verbose         : int  = 0) \
        -> Tuple[pd.DataFrame, str, List[str], pd.Series, pd.Series, pd.Series]:

    df, dates_df = utils.df_features(dict_fnames, cache_fname, pred_length,
                            num_steps_per_day, minutes_per_step, verbose)



    # ---- Identify columns ----
    target_col = "consumption_GW"

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    # All features EXCEPT the target and date are predictors
    feature_cols = [
        c for c in numeric_cols
        if c not in ("year", 'month', 'timeofday', target_col)
    ]


    df_len_before = df.shape[0]

    if verbose >= 3:
        print("NA:", df[df.isna().any(axis=1)])

    # Remove every row containing any NA (no filling)
    df = df[feature_cols + [target_col]].dropna()

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


    return df, target_col, feature_cols, dates, Tavg_full, holidays_full




# ============================================================
# NORMALIZE PREDICTORS AND CREATE MODEL
# ============================================================

def normalize_features(df            : pd.DataFrame,
                       target_col    : str,
                       feature_cols  : List[str],
                       minutes_per_step: int,
                       dates         : pd.DatetimeIndex,
                       temperatures  : pd.DataFrame,
                       train_split   : float,
                       n_valid       : int,
                       test_months   : int,
                       input_length  : int,
                       pred_length   : int,
                       features_in_future:bool,
                       batch_size    : int,
                       forecast_hour : int,
                       verbose       : int = 0):


    array = np.column_stack([
        df[target_col]  .values.astype(np.float32),
        df[feature_cols].values.astype(np.float32)
    ])

    if verbose >= 3:
        print(f"array:{array.shape}")
        print("  NA:", np.where(np.isnan(array))[0])


    del df  # pd.DataFrame no longer needed, we use np.ndarray now



    if verbose >= 1:
        num_steps_per_day = int(round(24*60/minutes_per_step))
        print(f"{len(array)/num_steps_per_day/365.25:.1f} years of data, "
              f"train: {train_split/num_steps_per_day/365.25:.2f} yrs"
              # f" ({train_split_fraction*100:.1f}%), "
              f"test: {test_months/num_steps_per_day/365.25:.2f} yrs"
              f" (switching  {dates[train_split].date()})")
        print()
    assert input_length + 60 < test_months,\
        f"input_length ({input_length}) > test_months ({test_months}) - 60"


    # assert all(ts.hour == 12 for ts in train_dataset.forecast_origins)
    # assert len(set(test_results['predictions']['target_time'])) == \
    #    len(test_results['predictions'])



    data, X_test_scaled = architecture.make_X_and_y(
            array, dates, temperatures.to_numpy(), train_split, n_valid,
            feature_cols, target_col, minutes_per_step,
            input_length=input_length, pred_length=pred_length,
            features_in_future=features_in_future, batch_size=batch_size,
            forecast_hour=forecast_hour,
            verbose=verbose)


    if verbose >= 2:
        print(f"Train mean:{data.scaler_y.mean_ [0]:6.2f} GW")
        print(f"Train std :{data.scaler_y.scale_[0]:6.2f} GW")
        print(f"Valid mean:{data.valid.y_dev.mean():6.2f} GW")
        print(f"Test mean :{data.test .y_dev.mean():6.2f} GW")


    del array

    gc.collect()
    torch.cuda.empty_cache()

    return data, X_test_scaled



# ============================================================
# TRAINING LOOP
# ============================================================

def training_loop(data          : containers.DatasetBundle,
                  NNTQ_model    : containers.NeuralNet,
                  validate_every: int,
                  display_every : int,
                  plot_conv_every:int,
                  verbose       : int = 0):

    num_epochs: int = NNTQ_model.epochs

    t_epoch_loop_start = time.perf_counter()

    list_of_min_losses= (9.999, 9.999, 9.999)
    list_of_lists     = ([], [], [], [])

    # first_step = True

    for epoch in range(num_epochs):

        # Training
        t_train_start = time.perf_counter()

        train_loss_quantile_h_scaled, dict_train_loss_quantile_h = \
            architecture.subset_evolution_torch(
                NNTQ_model, data.train.loader)  #, data.train.Tavg_degC)

        train_loss_quantile_h_scaled= \
            train_loss_quantile_h_scaled.detach().cpu().numpy()
        dict_train_loss_quantile_h  = {k: v.detach().cpu().numpy()
                           for k, v in dict_train_loss_quantile_h.items()}


        if verbose >= 2:
            print(f"training took:   {time.perf_counter() - t_train_start:.2f} s")


        # validation
        if ((epoch+1) % validate_every == 0) | (epoch == 0):

            t_valid_start     = time.perf_counter()
            valid_loss_quantile_h_scaled, dict_valid_loss_quantile_h = \
                architecture.subset_evolution_numpy(
                    NNTQ_model, data.valid.loader)  #, data.valid.Tavg_degC)

            if verbose >= 2:
                print(f"validation took: {time.perf_counter()-t_valid_start:.2f} s")


        # display evolution of losses
        (list_of_min_losses, list_of_lists) = \
            utils.display_evolution(
                epoch, t_epoch_loop_start,
                train_loss_quantile_h_scaled.mean(),
                valid_loss_quantile_h_scaled.mean(),
                list_of_min_losses, list_of_lists,
                num_epochs, display_every, plot_conv_every,
                NNTQ_model.min_delta, verbose)

        # plotting convergence
        if ((epoch+1 == plot_conv_every) | ((epoch+1) % plot_conv_every == 0))\
                & (epoch < num_epochs-2) & verbose > 0:
            plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                              list_of_lists[2], list_of_lists[3],
                              partial=True, verbose=verbose)

        # Check for early stopping
        if NNTQ_model.early_stopping(valid_loss_quantile_h_scaled.mean()):
            if verbose > 0:
                print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        torch.cuda.empty_cache()

    return (list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
            dict_valid_loss_quantile_h)



def post_process_run(baseline_parameters   : Dict[str, Any],
                     NNTQ_parameters       : Dict[str, Any],
                     metamodel_parameters  : Dict[str, Any],
                     df_metrics            : pd.DataFrame(),
                     quantile_delta_coverage:Dict[str, float],
                     avg_weights_meta_NN   : Dict[str, float],
                     avg_abs_worst_days_train:float,
                     run_id                : int,
                     csv_path              : str  = 'parameter_search.csv'
                     ) -> [Dict[str, Any], float]:
    flat_metrics = {}
    for model in df_metrics.index:
        for metric in df_metrics.columns:
            key = f"test_{model}_{metric}".replace(" ", "_")
            flat_metrics[key] = float(df_metrics.loc[model, metric])

    # learning_rate and weight_decay are so small
    NNTQ_parameters    ['learning_rate' ]= NNTQ_parameters  ['learning_rate'] * 1e6
    metamodel_parameters['learning_rate']=metamodel_parameters['learning_rate']*1e6

    NNTQ_parameters    ['weight_decay' ]= NNTQ_parameters  ['weight_decay'] * 1e6
    metamodel_parameters['weight_decay']=metamodel_parameters['weight_decay']*1e6

    # flatten sequences
    _dict_quantiles = expand_sequence(name="quantiles",
           values=NNTQ_parameters["quantiles"],  length=5, prefix="")
    del NNTQ_parameters["quantiles"]
    NNTQ_parameters.update(_dict_quantiles)


    _dict_num_cells = expand_sequence(name="num_cells",
           values=metamodel_parameters["num_cells"], length=2, prefix="")
    del metamodel_parameters["num_cells"]
    metamodel_parameters.update(_dict_num_cells)

    _overall_loss = overall_loss(
        flat_metrics, quantile_delta_coverage, avg_abs_worst_days_train)

    row = {
        "run"      : run_id,
        "timestamp": datetime.now(),   # Excel-compatible
        **(flatten_dict(baseline_parameters, parent_key="")),
        **NNTQ_parameters,
        **{"metaNN_"+key: value for (key, value) in metamodel_parameters.items()},
        **quantile_delta_coverage,
        **{"avg_weight_meta_NN_"+key: value
           for (key, value) in avg_weights_meta_NN.items()},
        **flat_metrics,
        'avg_abs_worst_days_train': avg_abs_worst_days_train,
        "overall_loss": _overall_loss
    }
    # print(row)

    df_row = pd.DataFrame([row])
    df_row.to_csv(
        csv_path,
        mode   = "a",
        header = not os.path.exists(csv_path),
        index  = False,
        float_format="%.6f"
    )

    return row, _overall_loss


def run_model(
        # configuration bundles
        baseline_cfg      : Dict[str, Dict[str, Any]],
        NNTQ_parameters   : Dict[str, Any],
        metamodel_NN_parameters:Dict[str, Any],
        dict_fnames       : Dict[str, str],
        # statistics of the dataset
        minutes_per_step  : int,
        train_split_fraction:float,
        val_ratio         : float,
        forecast_hour     : int,
        seed              : int,
        force_calc_baselines:bool,
        # XXX_EVERY (in epochs)
        validate_every    : int,
        display_every     : int,
        plot_conv_every   : int,
        run_id            : int,

        cache_fname       : str  = "cache",
        csv_path          : str  = 'parameter_search.csv',
        num_worst_days    : int  = 40,
        verbose           : int  = 0
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

        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
    elif verbose > 0:
        print("CUDA unavailable")
    print()


    # load data from csv and create pd.DataFrame
    num_steps_per_day = int(round(24*60/minutes_per_step))
    (df, target_col, feature_cols, dates, Tavg_full, holidays_full) = \
        load_and_create_df(
            dict_fnames, cache_fname, NNTQ_parameters['pred_length'],
            num_steps_per_day, minutes_per_step, verbose)


    num_time_steps = df.shape[0]

    if verbose > 0:
        # keep only arguments expected by `IO.print_model_summary`
        valid_parameters = inspect.signature(IO.print_model_summary).parameters.keys()
        filtered_parameters = {k: v for k, v in NNTQ_parameters.items()
                                       if k in valid_parameters}
        filtered_meta_parameters = {
            'meta_'+k : v for k, v in metamodel_NN_parameters.items()
                                       if 'meta_'+k in valid_parameters}

        IO.print_model_summary(
                minutes_per_step, num_steps_per_day,
                num_time_steps, feature_cols,
                **filtered_parameters, **filtered_meta_parameters
        )

        # correlation matrix for temperatures
        # utils.temperature_correlation_matrix(df)


    # create baselines (linera regreassion, random forest, gradient boosting)
    if verbose > 0:
        print("Doing linear regression and random forest...")


    train_split = int(len(df) * train_split_fraction)
    test_months = len(df)-train_split
    n_valid     = int(train_split * val_ratio)

    (df, feature_cols) = LR_RF.create_baselines(df, target_col, feature_cols,
        baseline_cfg, train_split, n_valid,
        cache_dir = "cache",
        force_calculation = force_calc_baselines,
        verbose           = verbose
    )
        # df now has new columns (baseline predictions)
        #    but fewer columns overall (some features were removed by lasso)


    # Create splits
    data, X_test_scaled = normalize_features(
        df, target_col, feature_cols, minutes_per_step, dates, Tavg_full,
        train_split, n_valid, test_months,
        NNTQ_parameters['input_length'],
        NNTQ_parameters['pred_length'],
        NNTQ_parameters['features_in_future'],
        NNTQ_parameters['batch_size'],
        forecast_hour,
        verbose)


    # Create model
    NNTQ_model = containers.NeuralNet(**NNTQ_parameters,
                                      len_train_data= len(data.train),
                                      num_features  = data.num_features)


    # loop for training and validation
    if verbose > 0:
        print("Starting training...")

    list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
            dict_valid_loss_quantile_h = training_loop(data,
                NNTQ_model, validate_every, display_every, plot_conv_every, verbose)



    # plotting convergence for entire training
    if verbose > 0:
        plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                                   list_of_lists[2], list_of_lists[3],
                                   partial=False, verbose=verbose)

        plots.loss_per_horizon(dict({"total": valid_loss_quantile_h_scaled}, \
                               **dict_valid_loss_quantile_h), minutes_per_step,
                               "validation loss")


    # test loss
    test_loss_quantile_h_scaled, dict_test_loss_quantile_h = \
       architecture.subset_evolution_numpy(
           NNTQ_model, data.test.loader)  #, data.test.Tavg_degC)

    if verbose >= 3:
        print(pd.DataFrame(dict({"total": test_loss_quantile_h_scaled}, \
                                **dict_test_loss_quantile_h)
                           ).round(2).to_string())
    if verbose > 0:
        plots.loss_per_horizon(dict({"total": test_loss_quantile_h_scaled}, \
                                     **dict_test_loss_quantile_h), minutes_per_step,
                               "test loss")


    t_metamodel_start = time.perf_counter()

    data.predictions_day_ahead(
            NNTQ_model.model, data.scaler_y,
            feature_cols = feature_cols,
            device       = NNTQ_model.device,
            input_length = NNTQ_model.input_length,
            pred_length  = NNTQ_model.pred_length,
            valid_length = NNTQ_model.valid_length,
            minutes_per_step=minutes_per_step,
            quantiles    = NNTQ_model.quantiles
            )

    # metamodel LR
    data.calculate_metamodel_LR(
        split_active='valid', min_weight=0.15, verbose=verbose)

    if verbose > 0:
        print(f"weights_meta_LR [%]: "
          f"{ {k: round(v*100, 1) for k, v in data.weights_meta_LR.items()}}")
    t_metamodel_end = time.perf_counter()
    if verbose >= 2:
        print(f"metamodel_LR took: {time.perf_counter() - t_metamodel_start:.2f} s")


    top_bad_days_train_df, avg_abs_diff_train = data.train.worst_days_by_loss(
        temperature_full = Tavg_full,
        holidays_full    = holidays_full,
        num_steps_per_day= num_steps_per_day,
        top_n            = num_worst_days,
        verbose          = verbose
    )
    if verbose >= 3:
        print(top_bad_days_train_df.to_string())




    # ============================================================
    # TEST PREDICTIONS
    # ============================================================

    if verbose > 0:
        print("\nStarting test ...")  #"(baseline: {name_baseline})...")

        print("\nTesting quantiles")

    quantile_delta_coverage = {}
    for tau in NNTQ_model.quantiles:
        key = f"q{int(100*tau)}"
        cov = utils.quantile_coverage(data.test.true_GW,
                                      data.test.dict_preds_NN[key])
        quantile_delta_coverage[key] = cov-tau

        if verbose > 0:
            print(f"Coverage {key}:{cov*100:5.1f}%, i.e."
                  f"{(cov-tau)*100:5.1f}%pt off{tau*100:3n}% target")

    if verbose > 0:
        print()


    # rows = []
    # rows.append(utils.index_summary("test_dates", data.test.dates, None))

    # rows.append({"series": "origin_times",
    #         "start": data.test.origin_times[0].date(),
    #         "end":   data.test.origin_times[1].date(),
    #         "n": None, "n_common": None, "start_diff": None, "end_diff": None})

    # common_idx = data.test.true_GW.index
    # common_idx = common_idx.intersection(data.test.dict_preds_NN['q50'].index)
    # rows.append(utils.index_summary(
    #     "true",  data.test.true_GW             .index, common_idx))
    # rows.append(utils.index_summary(
    #     "nn_q50",data.test.dict_preds_NN["q50"].index,common_idx))

    # for _name in data.test.dict_preds_ML:
    #     _baseline_test_idx = data.test.dict_preds_ML[_name].index
    #     common_idx = common_idx.intersection(_baseline_test_idx)
    #     rows.append(utils.index_summary(_name, _baseline_test_idx, common_idx))

    # rows.append(utils.index_summary("common", common_idx, common_idx))
    # if verbose >= 3:
    #     print(pd.DataFrame(rows).set_index("series"))
    #     print()

    # assert len(common_idx)>0, "No common timestamps between truth and predictions!"

    # print(data.test.true_GW.index, common_idx)

    # true_test_GW = data.test.true_GW.loc[common_idx]
    # for k in data.test.dict_preds_NN:
    #     data.test.dict_preds_NN[k] = data.test.dict_preds_NN[k].loc[common_idx]
    # for k in data.test.dict_preds_ML:
    #     data.test.dict_preds_ML[k] = data.test.dict_preds_ML[k].loc[common_idx]




    # ============================================================
    # METAMODEL
    # ============================================================


    # NN metamodel
    # ============================================================

    data.calculate_metamodel_NN(feature_cols, NNTQ_model.valid_length, 'valid',
                                metamodel_NN_parameters, verbose)
    avg_weights_meta_NN = data.avg_weights_meta_NN


    names_baseline= {'GB', 'LR', 'RF'}
    names_meta    = {'LR', 'NN'}

    if verbose > 0:
        data.train.compare_models(unit="GW", verbose=verbose)
        data.valid.compare_models(unit="GW", verbose=verbose)
    test_metrics = data.test .compare_models(unit="GW", verbose=verbose)


    if verbose > 0:
        print("Plotting test results...")
    if verbose >= 3:
        data.train.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            temperature_full=Tavg_full, num_steps_per_day=num_steps_per_day)

    if verbose > 0:
        data.test.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            temperature_full=Tavg_full, num_steps_per_day=num_steps_per_day)

        plots.quantiles(
            data.test.true_GW,
            data.test.dict_preds_NN,
            q_low = "q10",
            q_med = "q50",
            q_high= "q90",
            baseline_series=data.test.dict_preds_ML,
            title = "Electricity consumption forecast (NN quantiles), test",
            dates = data.test.dates[-(8*num_steps_per_day):]
        )


    # final cleanup to free pinned memory and intermediate arrays
    try:
        del data
    except NameError:
        pass

    if torch.cuda.is_available():
        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()

    dict_row, overall_loss = post_process_run(
        baseline_cfg, NNTQ_parameters, metamodel_NN_parameters,
        test_metrics, quantile_delta_coverage, avg_weights_meta_NN,
        avg_abs_diff_train, run_id, csv_path)

    return dict_row, test_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
        (num_worst_days, avg_abs_diff_train), overall_loss



# -------------------------------------------------------
# overall_all loss
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

def overall_loss(
         flat_metrics           : Dict[str, float],
         quantile_delta_coverage: Dict[str, float],
         avg_abs_worst_days     : float,

         # constants
         max_quantile_delta_coverage: float = 0.25,
                 # prevents one quantile from dominating completely

         weight_coverage        : float   = 5.,
                 # coverage and bias, MAE have different scales

         weight_worst_days      : float   = 0.2,
                # was not saved initially: start slow

         quantile_weights       : Dict[str, float] = \
             {'q10': 2., 'q25': 1.5, 'q50': 1., 'q75': 1.5,'q90': 2.},
                 # q10 off by 5% is worse than w/ q50: 10% -> 5% vs. 50% -> 45%

         metric_weights         : Dict[str, float] = \
             {'bias': 2., 'RMSE': 1., 'MAE': 1.},
                 # RMSE and MAE are variants of each other, bias is different

         model_weights         : Dict[str, float] = \
             {'NN': 2., 'LR': 1., 'RF': 1., 'GB': 1.,
              'meta_LR'     : 3., 'meta_NN'     : 4.},
            # LR, RF and LGBM are just underlying models to the metamodels;
            #      improving them intrinsically is good, but secondary
    ) -> float:

    _loss_quantile_coverage = \
        np.max([min(abs(gap), max_quantile_delta_coverage) * quantile_weights[q]
                    for q, gap in quantile_delta_coverage.items()])
                # was np.mean

    dict_by_metric = {k: [] for k in list(metric_weights.keys())}

    for key, value in flat_metrics.items():
        parts  = key.replace("test_", "").split('_')
        model  = '_'.join(parts[:-1])  # Ex: "NN", "meta_NN", etc.
        metric = parts[-1]             # Ex: "bias", "RMSE", etc.

        dict_by_metric[metric].append(model_weights[model] * abs(value))

    for metric in dict_by_metric.keys():
        assert len(dict_by_metric[metric]) == len(model_weights)

    dict_avg_metrics = {metric: np.sum(_list) / np.sum(list(model_weights.values()))
                            for (metric, _list) in dict_by_metric.items()}

    avg_metric = np.sum([value * metric_weights[metric]
                        for (metric, value) in dict_avg_metrics.items()]) / \
                    np.sum((list(metric_weights.values())))
    # print(avg_metrics)

    return float(round(_loss_quantile_coverage * weight_coverage + \
                       avg_metric + \
                       avg_abs_worst_days * weight_worst_days, 4))


def recalculate_loss(csv_path       : str   = 'parameter_search.csv',
                     weight_coverage: float = 10.) -> None:
    # Load the CSV file containing MC runs
    results_df = pd.read_csv(csv_path, index_col=False)

    # clean up dates
    results_df['timestamp'] = pd.to_datetime(
        results_df['timestamp'],
        errors   = 'coerce',
        infer_datetime_format=True,
        dayfirst = True
    )

    # print(pd.concat([results_df[['timestamp']], dates], axis=1))

    _list_overall_losses = []
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

        _overall_loss = overall_loss(flat_metrics, quantile_delta_coverage, weight_coverage)
        _list_overall_losses.append(_overall_loss)

    # print(_list_overall_losses)

    results_df['overall_loss'] = _list_overall_losses
    results_df.to_csv(csv_path, index=False)


# recalculate_loss()


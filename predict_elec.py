import gc
# import sys
import inspect

from   typing import List, Tuple, Dict   #, Optional, Any

# from datetime import datetime
import time

import torch

import numpy  as np
import pandas as pd


# from constants import * # annoying Spyder complains "may be defined in constants"
from   constants import (SYSTEM_SIZE, SEED, TRAIN_SPLIT_FRACTION, VAL_RATIO,
           VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           VERBOSE, DICT_FNAMES, CACHE_FNAME, BASELINE_CFG,
           FORECAST_HOUR, MINUTES_PER_STEP, NUM_STEPS_PER_DAY,
           NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS
           )

import containers, architecture, utils, LR_RF, IO, plots  # losses, metamodel,


# system dimensions
# B = BATCH_SIZE
# L = INPUT_LENGTH
# H = prediction horizon  =  PRED_LENGTH
# V = validation horizon  = VALID_LENGTH
# Q = number of quantiles = len(quantiles)
# F = number of features


# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.
# TODO make future T° noisy to mimic the uncertainty of forecasts
# BUG RF and Boosting generalize poorly
# TODO make the metamodel reduce the bias
# TODO save RF and GB pickles separately
# [done] use lasso with LR to select features, and use only these with other models
# TODO have separate public holidays, as with the school holidays
# GB complains about pd vs. np
# TODO MC hyperparameter search
# TODO preparation: in `predict_elec.py`
#   - create functions
#   - make main call these functions




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

    if VERBOSE >= 3:
        print("NA:", df[df.isna().any(axis=1)])

    # Remove every row containing any NA (no filling)
    df = df[feature_cols + [target_col]].dropna()

    # print start and end dates
    dates_df.loc["df"]= [df.index.min().date(), df.index.max().date()]
    if VERBOSE >= 1:
        print(dates_df)

    drop = df_len_before - df.shape[0]
    if VERBOSE >= 2:
        print(f"number of datetimes: {df_len_before} -> {df.shape[0]}, "
              f"drop by {drop} (= {drop/NUM_STEPS_PER_DAY:.1f} days)")
    if VERBOSE >= 3:
        print(f"df:  {df.shape} "
              f"({df.index.min()} -> {df.index.max()})")
        print("  NA:", df.index[df.isna().any(axis=1)].tolist())

    # Keep date separately for plotting later
    dates        = df.index
    Tavg_full    = df["Tavg_degC"]    # for plots and worst days
    holidays_full= df['is_holiday']   # for worst days

    df = df.reset_index(drop=True)


    num_time_steps = df.shape[0]



    if VERBOSE >= 1:
        # keep only arguments expected by `IO.print_model_summary`
        valid_parameters = inspect.signature(IO.print_model_summary).parameters.keys()
        filtered_parameters = {k: v for k, v in NNTQ_PARAMETERS.items()
                                       if k in valid_parameters}
        filtered_meta_parameters = {
            'meta_'+k : v for k, v in METAMODEL_NN_PARAMETERS.items()
                                       if 'meta_'+k in valid_parameters}

        IO.print_model_summary(
                minutes_per_step, num_steps_per_day,
                num_time_steps, feature_cols,
                **filtered_parameters, **filtered_meta_parameters
        )


    # correlation matrix for temperatures
    # utils.temperature_correlation_matrix(df)

    if VERBOSE >= 1:
        print("Doing linear regression and random forest...")

    return df, target_col, feature_cols, dates, Tavg_full, holidays_full





# ============================================================
# NORMALIZE PREDICTORS AND CREATE MODEL
# ============================================================

def normalize_features(df            : pd.DataFrame,
                       target_col    : str,
                       feature_cols  : List[str],
                       train_split   : float,
                       n_valid       : int,
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

    if VERBOSE >= 3:
        print(f"array:{array.shape}")
        print("  NA:", np.where(np.isnan(array))[0])


    del df  # pd.DataFrame no longer needed, we use np.ndarray now



    if VERBOSE >= 1:
        print(f"{len(array)/NUM_STEPS_PER_DAY/365.25:.1f} years of data, "
              f"train: {TRAIN_SPLIT/NUM_STEPS_PER_DAY/365.25:.2f} yrs"
              f" ({TRAIN_SPLIT_FRACTION*100:.1f}%), "
              f"test: {test_months/NUM_STEPS_PER_DAY/365.25:.2f} yrs"
              f" (switching  {dates[TRAIN_SPLIT].date()})")
        print()
    assert input_length + 60 < test_months,\
        f"input_length ({input_length}) > test_months ({test_months}) - 60"


    # assert all(ts.hour == 12 for ts in train_dataset.forecast_origins)
    # assert len(set(test_results['predictions']['target_time'])) == \
    #    len(test_results['predictions'])



    data, X_test_scaled = architecture.make_X_and_y(
            array, dates, train_split, n_valid,
            feature_cols, target_col,
            input_length=input_length, pred_length=pred_length,
            features_in_future=features_in_future, batch_size=batch_size,
            forecast_hour=forecast_hour,
            verbose=verbose)


    if VERBOSE >= 2:
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

def training_loop(model         : containers.NeuralNet,
                  display_every : int,
                  plot_conv_every:int,
                  verbose       : int = 0):

    num_epochs: int = NNTQ_PARAMETERS['epochs']

    t_epoch_loop_start = time.perf_counter()

    list_of_min_losses= (9.999, 9.999, 9.999)
    list_of_lists     = ([], [], [], [])

    # first_step = True

    for epoch in range(num_epochs):

        # Training
        t_train_start = time.perf_counter()

        train_loss_quantile_h_scaled, dict_train_loss_quantile_h = \
            architecture.subset_evolution_torch(model, data.train.loader)

        train_loss_quantile_h_scaled= \
            train_loss_quantile_h_scaled.detach().cpu().numpy()
        dict_train_loss_quantile_h  = {k: v.detach().cpu().numpy()
                           for k, v in dict_train_loss_quantile_h.items()}


        if verbose >= 2:
            print(f"training took:   {time.perf_counter() - t_train_start:.2f} s")


        # validation
        if ((epoch+1) % VALIDATE_EVERY == 0) | (epoch == 0):

            t_valid_start     = time.perf_counter()
            valid_loss_quantile_h_scaled, dict_valid_loss_quantile_h = \
                architecture.subset_evolution_numpy(model, data.train.loader)

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
                NNTQ_PARAMETERS['min_delta'], verbose)

        # plotting convergence
        if ((epoch+1 == PLOT_CONV_EVERY) | ((epoch+1) % PLOT_CONV_EVERY == 0))\
                & (epoch < num_epochs-2):
            plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                              list_of_lists[2], list_of_lists[3],
                              partial=True, verbose=verbose)

        # Check for early stopping
        if NNTQ_model.early_stopping(valid_loss_quantile_h_scaled.mean()):
            if VERBOSE >= 1:
                print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        torch.cuda.empty_cache()

    return (list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
            dict_valid_loss_quantile_h)




if __name__ == "__main__":

    np.   random.seed(SEED)
    torch.manual_seed(SEED)


    if VERBOSE >= 1:
        print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()))
    if torch.cuda.is_available():
        if VERBOSE >= 1:
            print(f"GPU: {torch.cuda.get_device_name(0)}, "
                  f"CUDA version: {torch.version.cuda}, "
                  f"CUDNN version: {torch.backends.cudnn.version()}")

        # clear VRAM
        gc.collect()
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())
    elif VERBOSE >= 1:
        print("CUDA unavailable")
    print()


    # load data from csv and create pd.DataFrame
    (df, target_col, feature_cols, dates, Tavg_full, holidays_full) = \
        load_and_create_df(
            DICT_FNAMES, CACHE_FNAME, NNTQ_PARAMETERS['pred_length'],
            NUM_STEPS_PER_DAY, MINUTES_PER_STEP, VERBOSE)


    TRAIN_SPLIT = int(len(df) * TRAIN_SPLIT_FRACTION)
    test_months = len(df)-TRAIN_SPLIT
    n_valid     = int(TRAIN_SPLIT * VAL_RATIO)


    # create baselines (linera regreassion, random forest, gradient boosting)
    (df, feature_cols) = LR_RF.create_baselines(df, target_col, feature_cols,
        BASELINE_CFG, SYSTEM_SIZE, TRAIN_SPLIT, n_valid,
        cache_dir = "cache",
        force_calculation = VERBOSE >= 2,  #SYSTEM_SIZE == 'DEBUG',
        verbose           = VERBOSE
    )
        # df now has new columns (baseline predictions)
        #    but fewer columns overall (some features were removed by lasso)


    # Create splits
    data, X_test_scaled = normalize_features(
        df, target_col, feature_cols,
        TRAIN_SPLIT, n_valid,
        NNTQ_PARAMETERS['input_length'],
        NNTQ_PARAMETERS['pred_length'],
        NNTQ_PARAMETERS['features_in_future'],
        NNTQ_PARAMETERS['batch_size'],
        FORECAST_HOUR,
        VERBOSE)


    # Create model
    NNTQ_model = containers.NeuralNet(**NNTQ_PARAMETERS,
                                      len_train_data= len(data.train),
                                      num_features  = data.num_features)


    # loop for training and validation
    if VERBOSE >= 1:
        print("Starting training...")

    list_of_min_losses, list_of_lists, valid_loss_quantile_h_scaled, \
            dict_valid_loss_quantile_h = training_loop(
                NNTQ_model, DISPLAY_EVERY, PLOT_CONV_EVERY, VERBOSE)



    # plotting convergence for entire training
    plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                               list_of_lists[2], list_of_lists[3],
                               partial=False, verbose=VERBOSE)

    plots.loss_per_horizon(dict({"total": valid_loss_quantile_h_scaled}, \
                           **dict_valid_loss_quantile_h), MINUTES_PER_STEP,
                           "validation loss")


    # test loss
    test_loss_quantile_h_scaled, dict_test_loss_quantile_h = \
       architecture.subset_evolution_numpy(NNTQ_model, data.test.loader)

    if VERBOSE >= 3:
        print(pd.DataFrame(dict({"total": test_loss_quantile_h_scaled}, \
                                **dict_test_loss_quantile_h)
                           ).round(2).to_string())
    plots.loss_per_horizon(dict({"total": test_loss_quantile_h_scaled}, \
                                 **dict_test_loss_quantile_h), MINUTES_PER_STEP,
                           "test loss")


    t_metamodel_start = time.perf_counter()

    data.predictions_day_ahead(
            NNTQ_model.model, data.scaler_y,
            feature_cols = feature_cols,
            device       = NNTQ_model.device,
            input_length = NNTQ_model.input_length,
            pred_length  = NNTQ_model.pred_length,
            valid_length = NNTQ_model.valid_length,
            minutes_per_step=MINUTES_PER_STEP,
            quantiles    = NNTQ_model.quantiles
            )

    # metamodel LR
    data.calculate_metamodel_LR(
        split_active='valid', min_weight=0.15, verbose=VERBOSE)

    if VERBOSE >= 1:
        print(f"weights_meta_LR [%]: "
          f"{ {k: round(v*100, 1) for k, v in data.weights_meta_LR.items()}}")
    t_metamodel_end = time.perf_counter()
    if VERBOSE >= 2:
        print(f"metamodel_LR took: {time.perf_counter() - t_metamodel_start:.2f} s")




    if VERBOSE >= 3:
        top_bad_days_train = data.train.worst_days_by_loss(
            temperature_full = Tavg_full,
            holidays_full    = holidays_full,
            num_steps_per_day= NUM_STEPS_PER_DAY,
            top_n            = 40,
        )
        print(top_bad_days_train.to_string())




    # ============================================================
    # TEST PREDICTIONS
    # ============================================================

    if VERBOSE >= 1:
        print("\nStarting test ...")  #"(baseline: {name_baseline})...")

    # print("true_test_GW:\n",      true_test_GW.head())
    # print("dict_pred_test_GW:\n", {_name: _test.head() \
    #                     for (_name, _test) in dict_pred_test_GW.items()})
    # print("dict_baseline_test_GW:\n", {_name: _test.head() \
    #                     for (_name, _test) in dict_baseline_test_GW.items()})

    if VERBOSE >= 1:
        print("\nTesting quantiles")
        for tau in NNTQ_model.quantiles:
            key = f"q{int(100*tau)}"
            cov = utils.quantile_coverage(data.test.true_GW,
                                          data.test.dict_preds_NN[key])
            print(f"Coverage {key}:{cov*100:5.1f}%, i.e."
                  f"{(cov-tau)*100:5.1f}%pt off{tau*100:3n}% target")
        print()


    rows = []
    rows.append(utils.index_summary("test_dates", data.test.dates, None))

    rows.append({"series": "origin_times",
            "start": data.test.origin_times[0].date(),
            "end":   data.test.origin_times[1].date(),
            "n": None, "n_common": None, "start_diff": None, "end_diff": None})

    common_idx = data.test.true_GW.index
    common_idx = common_idx.intersection(data.test.dict_preds_NN['q50'].index)
    rows.append(utils.index_summary(
        "true",  data.test.true_GW             .index, common_idx))
    rows.append(utils.index_summary(
        "nn_q50",data.test.dict_preds_NN["q50"].index,common_idx))

    for _name in data.test.dict_preds_ML:
        _baseline_test_idx = data.test.dict_preds_ML[_name].index
        common_idx = common_idx.intersection(_baseline_test_idx)
        rows.append(utils.index_summary(_name, _baseline_test_idx, common_idx))

    rows.append(utils.index_summary("common", common_idx, common_idx))
    if VERBOSE >= 3:
        print(pd.DataFrame(rows).set_index("series"))
        print()

    assert len(common_idx)>0, "No common timestamps between truth and predictions!"


    true_test_GW = data.test.true_GW.loc[common_idx]
    for k in data.test.dict_preds_NN:
        data.test.dict_preds_NN[k] = data.test.dict_preds_NN[k].loc[common_idx]
    for k in data.test.dict_preds_ML:
        data.test.dict_preds_ML[k] = data.test.dict_preds_ML[k].loc[common_idx]




    # ============================================================
    # METAMODEL
    # ============================================================


    # NN metamodel
    # ============================================================

    meta_model = containers.NeuralNet(
        dropout     = METAMODEL_NN_PARAMETERS['dropout'],
        num_cells   = METAMODEL_NN_PARAMETERS['num_cells'],
        epochs      = METAMODEL_NN_PARAMETERS['epochs'],
        learning_rate=METAMODEL_NN_PARAMETERS['learning_rate'],
        weight_decay= METAMODEL_NN_PARAMETERS['weight_decay'],
        patience    = METAMODEL_NN_PARAMETERS['patience'],
        factor      = METAMODEL_NN_PARAMETERS['factor'],
        batch_size  = METAMODEL_NN_PARAMETERS['batch_size'],
        device      = NNTQ_model.device,
    )

    data.calculate_metamodel_NN(
            feature_cols, NNTQ_model.valid_length, 'valid', meta_model, VERBOSE)


    names_baseline= {'GB', 'LR', 'RF'}
    names_meta    = {'LR', 'NN'}

    if VERBOSE >= 1:
        print("\nTraining metrics [GW]:")
        data.train.compare_models(unit="GW", verbose=VERBOSE)

        print("\nValidation metrics [GW]:")
        data.valid.compare_models(unit="GW", verbose=VERBOSE)

        print("\nTesting metrics [GW]:")
        data.test .compare_models(unit="GW", verbose=VERBOSE)


        print("Plotting test results...")
    if VERBOSE >= 3:
        data.train.plots_diagnostics(
            names_baseline = names_baseline, names_meta = names_meta,
            temperature_full=Tavg_full, num_steps_per_day=NUM_STEPS_PER_DAY)

    data.test.plots_diagnostics(
        names_baseline = names_baseline, names_meta = names_meta,
        temperature_full=Tavg_full, num_steps_per_day=NUM_STEPS_PER_DAY)



    plots.quantiles(
        data.test.true_GW,
        data.test.dict_preds_NN,
        q_low = "q10",
        q_med = "q50",
        q_high= "q90",
        baseline_series=data.test.dict_preds_ML,
        title = "Electricity consumption forecast (NN quantiles), test",
        dates = data.test.dates[-(8*NUM_STEPS_PER_DAY):]
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

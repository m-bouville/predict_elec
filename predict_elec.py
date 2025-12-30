import gc
import sys
# from   typing import Dict  # List

# from datetime import datetime
import time

import torch
# import torch.nn as nn

import numpy  as np
import pandas as pd


# from constants import * # Spyder annoyingly complains: "may be defined in constants"
from   constants import (SYSTEM_SIZE, SEED, TRAIN_SPLIT_FRACTION, VAL_RATIO,
           INPUT_LENGTH, PRED_LENGTH, VALID_LENGTH,
           BATCH_SIZE, EPOCHS, MODEL_DIM, NUM_HEADS, FFN_SIZE,
           NUM_LAYERS, PATCH_LEN, STRIDE, FEATURES_IN_FUTURE,
           LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV, LAMBDA_MEDIAN,
           SMOOTHING_CROSS, QUANTILES, NUM_GEO_BLOCKS, GEO_BLOCK_RATIO,
           LEARNING_RATE, WEIGHT_DECAY, DROPOUT, WARMUP_STEPS, PATIENCE,
           MIN_DELTA, VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           VERBOSE, DICT_FNAMES, CACHE_FNAME, BASELINE_CFG,
           FORECAST_HOUR, MINUTES_PER_STEP, NUM_STEPS_PER_DAY,
           META_EPOCHS, META_LR, META_WEIGHT_DECAY, META_BATCH_SIZE,
           META_DROPOUT, META_NUM_CELLS, META_PATIENCE, META_FACTOR)

import architecture, utils, LR_RF, IO, plots  # losses, metamodel,


# system dimensions
# B = BATCH_SIZE
# L = INPUT_LENGTH
# H = prediction horizon  =  PRED_LENGTH
# V = validation horizon  = VALID_LENGTH
# Q = number of quantiles = len(quantiles)
# F = number of features


# TODO make future T° noisy to mimic the uncertainty of forecasts
# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.
# TODO Boosting




if __name__ == "__main__":

    np.   random.seed(SEED)
    torch.manual_seed(SEED)


    if VERBOSE >= 1:
        print(time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



    # ============================================================
    # 2. LOAD
    # ============================================================

    df, dates_df = utils.df_features(DICT_FNAMES, CACHE_FNAME, PRED_LENGTH,
                            NUM_STEPS_PER_DAY, MINUTES_PER_STEP, VERBOSE)
        # 2 if SYSTEM_SIZE == 'DEBUG' else 1)



    # ---- Identify columns ----
    target_col = "consumption_GW"

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()


    # All features EXCEPT the target and date are predictors
    feature_cols = [
        c for c in numeric_cols
        if c not in ("year", 'month', 'timeofday', target_col)
    ]
    # ['year', 'month', 'timeofday', 'Tmin_degC', 'Tmax_degC', 'Tavg_degC']
    # feature_cols = ['timeofday', 'Tavg_degC']


    df_len_before = df.shape[0]
    # df_2021 = df[df.index.year==2021]
    # print(df[target_col  ][df[target_col  ].isna()])
    # print(df_2021[feature_cols][df_2021[feature_cols].isna().any(axis=1)].to_string())

    if VERBOSE >= 3:
        print("NA:", df[df.isna().any(axis=1)])

    # Remove every row containing any NA (no filling)
    df = df[feature_cols + [target_col]].dropna()
    # print(dates[:10])

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

    # print(df.head())

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,6))
    # plt.plot(dates[dates.year==2021], df['consumption_GW'].loc[dates.year==2021])
    # plt.show()



    TRAIN_SPLIT = int(len(df) * TRAIN_SPLIT_FRACTION)
    test_months = len(df)-TRAIN_SPLIT
    n_valid     = int(TRAIN_SPLIT * VAL_RATIO)

    num_time_steps = df.shape[0]

    NUM_PATCHES = (INPUT_LENGTH + FEATURES_IN_FUTURE*PRED_LENGTH - PATCH_LEN) \
                    // STRIDE + 1


    if VERBOSE >= 1:
        IO.print_model_summary(
                MINUTES_PER_STEP, NUM_STEPS_PER_DAY,
                num_time_steps, feature_cols,
                INPUT_LENGTH, PRED_LENGTH, VALID_LENGTH, FEATURES_IN_FUTURE,
                BATCH_SIZE, EPOCHS,
                LEARNING_RATE, WEIGHT_DECAY, DROPOUT, WARMUP_STEPS,
                PATIENCE, MIN_DELTA,
                MODEL_DIM, NUM_LAYERS, NUM_HEADS, FFN_SIZE,
                PATCH_LEN, STRIDE, NUM_PATCHES,
                NUM_GEO_BLOCKS, GEO_BLOCK_RATIO,
                QUANTILES,
                LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV, LAMBDA_MEDIAN,
                META_EPOCHS, META_LR, META_WEIGHT_DECAY, META_BATCH_SIZE,
                META_DROPOUT, META_NUM_CELLS, META_PATIENCE, META_FACTOR
        )


# correlation matrix for temperatures
# utils.temperature_correlation_matrix(df)

if VERBOSE >= 1:
    print("Doing linear regression and random forest...")



t_start = time.perf_counter()
rf_params = dict(
        df            = df,
        target_col    = target_col,
        feature_cols  = feature_cols,
        train_end     = TRAIN_SPLIT-n_valid,
        val_end       = TRAIN_SPLIT,
        models_cfg    = BASELINE_CFG,
        verbose       = VERBOSE
    )

cache_id = {
    "system_size":  SYSTEM_SIZE,
    "target":       target_col,
    "feature_cols": feature_cols,
    'train_end':    TRAIN_SPLIT-n_valid,
    'val_end':      TRAIN_SPLIT,
    # "split": "v1",   # optional: data split identifier
}

baseline_features_GW, baseline_models = \
    LR_RF.load_or_compute_regression_and_forest(
        compute_kwargs  = rf_params,
        cache_dir       = "cache",
        cache_id_dict   = cache_id,
        force_calculation=VERBOSE >= 2,  #SYSTEM_SIZE == 'DEBUG',
        verbose         = VERBOSE
    )
if VERBOSE >= 1:
    print(f"LR + RF took: {time.perf_counter() - t_start:.2f} s")



# reset random number generation because sklearn changed it
np.   random.seed(SEED)
torch.manual_seed(SEED)

# Add features
baseline_idx = dict()
for name, series in baseline_features_GW.items():
    col_name     = f"consumption_{name}"
    df[col_name] = series
    feature_cols.append(col_name)
    baseline_idx[name] = feature_cols.index(col_name)
    # print(f"{name}:{series.shape}")
    # print("  NA:", np.where(np.isnan(series))[0])
# feature_cols.append('consumption_regression')
# print(df['consumption_regression'].head(20))
if VERBOSE >= 3:
    print(f"baseline_idx: {baseline_idx}")



# ---- Construct final matrix: target first, then features ----

if VERBOSE >= 2:
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print("Using target:  ", target_col)


# median = dict_pred_series_GW.get('q50')
# print(f"nn:    {median.shape} ({median.index.min()} -> {median.index.max()})")
# print("  NA:", median.index[median.isna()].tolist())
# missing_indices = train_dates.difference(median.index).tolist()
# date_str = [dt.strftime('%Y-%m-%d') for dt in missing_indices]
# # Count occurrences per date
# from collections import Counter
# date_counts = Counter(date_str)
# print(f"  {len(missing_indices)} missing indices "
#       f"({len(missing_indices)/NUM_STEPS_PER_DAY:.1f} days): "
#       # f"{[e.strftime('%Y-%m-%d %H:%M') for e in missing_indices[14*48:]]}")
#       f"{date_counts}")


series = np.column_stack([
    df[target_col]  .values.astype(np.float32),
    df[feature_cols].values.astype(np.float32)
])

if VERBOSE >= 3:
    print(f"series:{series.shape}")
    print("  NA:", np.where(np.isnan(series))[0])


NUM_FEATURES   = len(feature_cols)  # /!\ y should not be included
num_time_steps = series.shape[0]

del df




# ============================================================
# 3. TRAIN/TEST SPLIT (time-respecting)
# ============================================================

if VERBOSE >= 1:
    print(f"{len(series)/NUM_STEPS_PER_DAY/365.25:.1f} years of data, "
          f"train: {TRAIN_SPLIT/NUM_STEPS_PER_DAY/365.25:.2f} yrs"
          f" ({TRAIN_SPLIT_FRACTION*100:.1f}%), "
          f"test: {test_months/NUM_STEPS_PER_DAY/365.25:.2f} yrs"
          f" (switching at {dates[TRAIN_SPLIT]})")
    print()
assert INPUT_LENGTH + 60 < test_months,\
    f"INPUT_LENGTH ({INPUT_LENGTH}) > test_months ({test_months}) - 60"


# assert all(ts.hour == 12 for ts in train_dataset.forecast_origins)
# assert len(set(test_results['predictions']['target_time'])) == \
#    len(test_results['predictions'])


# ============================================================
# 4. NORMALIZE PREDICTORS ONLY (not the log-target)
# ============================================================

data, X_test_scaled = architecture.make_X_and_y(
        series, dates, TRAIN_SPLIT, n_valid,
        feature_cols, target_col,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        features_in_future=FEATURES_IN_FUTURE, batch_size=BATCH_SIZE,
        forecast_hour=FORECAST_HOUR,
        verbose=VERBOSE)

if VERBOSE >= 2:
    print(f"Train mean:{data.scaler_y.mean_ [0]:6.2f} GW")
    print(f"Train std :{data.scaler_y.scale_[0]:6.2f} GW")
    print(f"Valid mean:{data.valid.y_dev.mean():6.2f} GW")
    print(f"Test mean :{data.test .y_dev.mean():6.2f} GW")


# Free heavy CPU-side arrays that are no longer needed.
# This reduces pinned memory and helps CUDA reclaim blocks.
try:
    del series
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()



model = architecture.TimeSeriesTransformer(
    num_features= NUM_FEATURES,
    dim_model   = MODEL_DIM,
    nhead       = NUM_HEADS,
    num_layers  = NUM_LAYERS,
    input_length= INPUT_LENGTH,
    pred_length = PRED_LENGTH,
    patch_length= PATCH_LEN,
    stride      = STRIDE,
    features_in_future=FEATURES_IN_FUTURE,
    dropout     = DROPOUT,
    ffn_mult    = FFN_SIZE,
    num_quantiles=len(QUANTILES),
    num_geo_blocks=NUM_GEO_BLOCKS,
    geo_block_ratio=GEO_BLOCK_RATIO
)
# model = torch.compile(model)  # Speedup: 1.5× to 2× on NVIDIA GPUs.
model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode    = "min",
#     factor  = SCHED_FACTOR,
#     patience= SCHED_PATIENCE,
#     min_lr  = 1e-6,
#     # verbose = VERBOSE >= 1
# )

def my_lr_warmup_cosine(step):
    return architecture.lr_warmup_cosine(step, WARMUP_STEPS, EPOCHS, len(data.train))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=my_lr_warmup_cosine
)
# print("Initial LR:", scheduler.get_last_lr())


# Example usage in your training loop:
# Initialize early stopping
early_stopping = architecture.EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)


# ============================================================
# 6. TRAINING LOOP (modified to include auxiliary feature loss)
# ============================================================

if VERBOSE >= 1:
    print("Starting training...")


list_of_min_losses= (9.999, 9.999, 9.999)
list_of_lists     = ([], [], [], [])

amp_scaler = torch.amp.GradScaler(device=device)

# first_step = True


t_epoch_loop_start = time.perf_counter()
for epoch in range(EPOCHS):

    # Training
    t_train_start = time.perf_counter()

    train_loss_quantile_h_scaled, dict_train_loss_quantile_h = \
        architecture.subset_evolution_torch(
            model, amp_scaler, optimizer, scheduler,
            data.train.loader, data.train.dates,
            # constants
            device, VALID_LENGTH, QUANTILES,
            LAMBDA_CROSS, LAMBDA_COVERAGE,
            LAMBDA_DERIV, LAMBDA_MEDIAN, SMOOTHING_CROSS
        )
    train_loss_quantile_h_scaled= train_loss_quantile_h_scaled.detach().cpu().numpy()
    dict_train_loss_quantile_h  = {k: v.detach().cpu().numpy()
                                   for k, v in dict_train_loss_quantile_h.items()}

    # print(f"train_loss_quantile_scaled = {train_loss_quantile_scaled} "
    #       f"meta_train_loss_quantile_scaled = {meta_train_loss_quantile_scaled}")


    if VERBOSE >= 2:
        print(f"training took:   {time.perf_counter() - t_train_start:.2f} s")


    # validation
    if ((epoch+1) % VALIDATE_EVERY == 0) | (epoch == 0):

        t_valid_start     = time.perf_counter()
        valid_loss_quantile_h_scaled, dict_valid_loss_quantile_h = \
            architecture.subset_evolution_numpy(
                model, data.valid.loader, data.valid.dates,
                # constants
                device, VALID_LENGTH, QUANTILES,
                LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV,
                LAMBDA_MEDIAN, SMOOTHING_CROSS
            )
        # print("valid_loss_quantile_scaled:", valid_loss_quantile_scaled,
        #       "meta_valid_loss_quantile_scaled:", meta_valid_loss_quantile_scaled)

        if VERBOSE >= 2:
            print(f"validation took: {time.perf_counter() - t_valid_start:.2f} s")


    # display evolution of losses
    (list_of_min_losses, list_of_lists) = \
        utils.display_evolution(
            epoch, t_epoch_loop_start,
            train_loss_quantile_h_scaled.mean(),
            valid_loss_quantile_h_scaled.mean(),
            list_of_min_losses, list_of_lists,
            EPOCHS, DISPLAY_EVERY, PLOT_CONV_EVERY, MIN_DELTA, VERBOSE)

    # plotting convergence
    if ((epoch+1 == PLOT_CONV_EVERY) | ((epoch+1) % PLOT_CONV_EVERY == 0))\
            & (epoch < EPOCHS-2):
        plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                          list_of_lists[2], list_of_lists[3],
                          partial=True, verbose=VERBOSE)

    # Check for early stopping
    if early_stopping(valid_loss_quantile_h_scaled.mean()):
        if VERBOSE >= 1:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        break

    torch.cuda.empty_cache()

# plotting convergence for entire training
plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                           list_of_lists[2], list_of_lists[3],
                           partial=False, verbose=VERBOSE)

plots.loss_per_horizon(dict({"total": valid_loss_quantile_h_scaled}, \
                       **dict_valid_loss_quantile_h), MINUTES_PER_STEP,
                       "validation loss")


# test loss
test_loss_quantile_h_scaled, dict_test_loss_quantile_h = \
   architecture.subset_evolution_numpy(
        model, data.test.loader, data.test.dates,
        # constants
        device, VALID_LENGTH, QUANTILES,
        LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV,
        LAMBDA_MEDIAN, SMOOTHING_CROSS
    )

if VERBOSE >= 3:
    print(pd.DataFrame(dict({"total": test_loss_quantile_h_scaled}, \
                            **dict_test_loss_quantile_h)
                       ).round(2).to_string())
plots.loss_per_horizon(dict({"total": test_loss_quantile_h_scaled}, \
                             **dict_test_loss_quantile_h), MINUTES_PER_STEP,
                       "test loss")


t_metamodel_start = time.perf_counter()

data.predictions_day_ahead(
        model, data.scaler_y,
        feature_cols = feature_cols,
        device       = device,
        input_length = INPUT_LENGTH,
        pred_length  = PRED_LENGTH,
        valid_length = VALID_LENGTH,
        minutes_per_step=MINUTES_PER_STEP,
        quantiles    = QUANTILES
        )

# metamodel LR
data.calculate_metamodel_LR(min_weight=0.15, verbose=VERBOSE)

if VERBOSE >= 1:
    print(f"weights_meta_LR [%]: "
      f"{ {name: round(value*100, 1) for name, value in data.weights_meta_LR.items()}}")
t_metamodel_end = time.perf_counter()
# TODO very slow b/c of pandas, concat, etc. for alignment
if VERBOSE >= 2:
    print(f"metamodel_LR took: {time.perf_counter() - t_metamodel_start:.2f} s")




if VERBOSE >= 3:

    print("\nBad days during training")
    top_bad_days_train = data.train.worst_days_by_loss(
        temperature_full = Tavg_full,
        holidays_full    = holidays_full,
        num_steps_per_day= NUM_STEPS_PER_DAY,
        top_n            = 30,
    )
    print(top_bad_days_train.to_string())




# ============================================================
# 7. TEST PREDICTIONS
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
    for tau in QUANTILES:
        key = f"q{int(100*tau)}"
        cov = utils.quantile_coverage(data.test.true_GW, data.test.dict_preds_NN[key])
        print(f"Coverage {key}:{cov*100:5.1f}%, i.e."
              f"{(cov-tau)*100:5.1f}%pt off{tau*100:3n}% target")
    print()

name_baseline =  'gb' if 'gb' in data.test.dict_preds_ML else \
                ('rf' if 'rf' in data.test.dict_preds_ML else \
                ('lr' if 'lr' in data.test.dict_preds_ML else None))
name_meta     = 'meta_LR'


rows = []
rows.append(utils.index_summary("test_dates", data.test.dates, None))

rows.append({"series": "origin_times",
        "start": data.test.origin_times[0].date(), "end": data.test.origin_times[1].date(),
        "n": None, "n_common": None, "start_diff": None, "end_diff": None})

common_idx = data.test.true_GW.index
common_idx = common_idx.intersection(data.test.dict_preds_NN['q50'].index)
rows.append(utils.index_summary("true",  data.test.true_GW            .index, common_idx))
rows.append(utils.index_summary("nn_q50",data.test.dict_preds_NN["q50"].index, common_idx))

for _name in ['lr', 'rf', 'gb']:
    if _name in data.test.dict_preds_ML:  # keep only those we trained
        _baseline_test_idx = data.test.dict_preds_ML[_name].index
        common_idx = common_idx.intersection(_baseline_test_idx)
        rows.append(utils.index_summary(_name, _baseline_test_idx, common_idx))

rows.append(utils.index_summary("common", common_idx, common_idx))
if VERBOSE >= 3:
    print(pd.DataFrame(rows).set_index("series"))
    print()

assert len(common_idx) > 0, "No common timestamps between truth and predictions!"


true_test_GW = data.test.true_GW.loc[common_idx]
for k in data.test.dict_preds_NN:
    data.test.dict_preds_NN[k] = data.test.dict_preds_NN[k].loc[common_idx]
for k in data.test.dict_preds_ML:
    data.test.dict_preds_ML[k] = data.test.dict_preds_ML[k].loc[common_idx]





# ============================================================
# 8. METAMODEL
# ============================================================


# NN metamodel
# ============================================================

data.calculate_metamodel_NN(
        feature_cols, VALID_LENGTH,
        #constants
        META_DROPOUT, META_NUM_CELLS, META_EPOCHS,
        META_LR, META_WEIGHT_DECAY, META_PATIENCE, META_FACTOR, META_BATCH_SIZE, device)

if VERBOSE >= 1:
    print("\nTraining metrics [GW]:")
    data.train.compare_models(unit="GW", verbose=VERBOSE)

    print("\nvalidation metrics [GW]:")
    data.valid.compare_models(unit="GW", verbose=VERBOSE)

    print("\nTesting metrics [GW]:")
    data.test .compare_models(unit="GW", verbose=VERBOSE)


    print("Plotting test results...")
if VERBOSE >= 2:
    data.train.plots_diagnostics(
        name_baseline, name_meta, Tavg_full, NUM_STEPS_PER_DAY)

data.test.plots_diagnostics(name_baseline, name_meta, Tavg_full, NUM_STEPS_PER_DAY)



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

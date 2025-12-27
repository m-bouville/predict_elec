import gc
# import sys
# from   typing import Dict  # List

# from datetime import datetime
import time

import torch
# import torch.nn as nn

import numpy  as np
import pandas as pd


# from   constants import *  # Spyder annoyingly complains: "may be defined in constants"
from   constants import (SYSTEM_SIZE, SEED, TRAIN_SPLIT_FRACTION, VAL_RATIO,
           INPUT_LENGTH,PRED_LENGTH, BATCH_SIZE, EPOCHS, MODEL_DIM, NUM_HEADS, FFN_SIZE,
           NUM_LAYERS, PATCH_LEN, STRIDE, FEATURES_IN_FUTURE,
           LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV, LAMBDA_MEDIAN,
           SMOOTHING_CROSS, QUANTILES, NUM_GEO_BLOCKS, GEO_BLOCK_RATIO,
           LEARNING_RATE, WEIGHT_DECAY, DROPOUT, WARMUP_STEPS, PATIENCE,
           MIN_DELTA, VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           VERBOSE, DICT_FNAMES, CACHE_FNAME, BASELINE_CFG,
           FORECAST_HOUR, MINUTES_PER_STEP, NUM_STEPS_PER_DAY,
           META_EPOCHS, META_LR, META_WEIGHT_DECAY, META_BATCH_SIZE,
           META_DROPOUT, META_NUM_CELLS, META_PATIENCE, META_FACTOR)

import architecture, utils, metamodel, LR_RF, IO, plots  # losses,


# system dimensions
# B = batch size
# L = input length
# H = prediction horizon  = PRED_LENGTH
# Q = number of quantiles = len(quantiles)


# [done] account for known future: sines, WE, T° (as forecast: _add noise_), etc.
# TODO make future T° noisy
# TODO trained metamodel (NN + RF)
#   - [done] Linear meta-learner per horizon (fast, interpretable)
#   - Small MLP or ridge regression trained on validation OOF predictions
# TODO Add public holidays to features
# TODO make PRED_LENGTH == 36h and validate h+12 to h+36
# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.





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
    dates     = df.index
    Tavg_full = df["Tavg_degC"]   # for plots and worst days

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
                INPUT_LENGTH, PRED_LENGTH, FEATURES_IN_FUTURE,
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


[train_loader, valid_loader, test_loader], [train_dates, valid_dates, test_dates],\
        scaler_y, [X_GW, y_GW], \
        [X_train_GW, y_train_GW, train_dataset_scaled], \
        [X_valid_GW, y_valid_GW, valid_dataset_scaled], \
        [X_test_GW,  y_test_GW,  test_dataset_scaled ], X_test_scaled =\
    architecture.make_X_and_y(
        series, dates, TRAIN_SPLIT, n_valid,
        feature_cols, target_col,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        features_in_future=FEATURES_IN_FUTURE, batch_size=BATCH_SIZE,
        forecast_hour=FORECAST_HOUR,
        verbose=VERBOSE)

if VERBOSE >= 2:
    print(f"Train mean:{scaler_y.mean_ [0]:6.2f} GW")
    print(f"Train std :{scaler_y.scale_[0]:6.2f} GW")
    print(f"Valid mean:{y_valid_GW.mean() :6.2f} GW")
    print(f"Test mean :{y_test_GW.mean()  :6.2f} GW")


# Free heavy CPU-side arrays that are no longer needed.
# This reduces pinned memory and helps CUDA reclaim blocks.
try:
    del series
    # del train_scaled, valid_scaled
    # del X_train, X_valid, y_train, y_valid
    # del train_data, valid_data, test_data
    # dataset objects keep what's needed; DataLoader will read from them
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()



model = architecture.TimeSeriesTransformer(
    num_features= NUM_FEATURES,
    dim_model   = MODEL_DIM,
    nhead       = NUM_HEADS,
    num_layers  = NUM_LAYERS,
    input_len   = INPUT_LENGTH,
    pred_len    = PRED_LENGTH,
    patch_len   = PATCH_LEN,
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
    return architecture.lr_warmup_cosine(step, WARMUP_STEPS, EPOCHS, len(train_loader))
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



def subset_predictions_day_ahead(X_subset_GW, subset_loader):
    return utils.subset_predictions_day_ahead(
            X_subset_GW, subset_loader, model, scaler_y,
            feature_cols = feature_cols,
            device       = device,
            input_length = INPUT_LENGTH,
            pred_length  = PRED_LENGTH,
            quantiles    = QUANTILES,
            minutes_per_step=MINUTES_PER_STEP)

t_epoch_loop_start = time.perf_counter()
for epoch in range(EPOCHS):

    # Training
    t_train_start = time.perf_counter()

    train_loss_quantile_scaled = architecture.subset_evolution_torch(
            model, amp_scaler, optimizer, scheduler,
            train_loader, train_dates, # scaler_y,
            # constants
            device, # INPUT_LENGTH, PRED_LENGTH,
            QUANTILES,
            LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV, LAMBDA_MEDIAN, SMOOTHING_CROSS
        )
    # print(f"train_loss_quantile_scaled = {train_loss_quantile_scaled} "
    #       f"meta_train_loss_quantile_scaled = {meta_train_loss_quantile_scaled}")


    if VERBOSE >= 2:
        print(f"training took:   {time.perf_counter() - t_train_start:.2f} s")


    # validation
    if ((epoch+1) % VALIDATE_EVERY == 0) | (epoch == 0):

        t_valid_start     = time.perf_counter()
        valid_loss_quantile_scaled = architecture.subset_evolution_numpy(
                model, valid_loader, valid_dates,  # scaler_y,
                # constants
                device, # INPUT_LENGTH, PRED_LENGTH,
                QUANTILES, LAMBDA_CROSS, LAMBDA_COVERAGE, LAMBDA_DERIV,
                LAMBDA_MEDIAN, SMOOTHING_CROSS
            )
        # print("valid_loss_quantile_scaled:", valid_loss_quantile_scaled,
        #       "meta_valid_loss_quantile_scaled:", meta_valid_loss_quantile_scaled)

        if VERBOSE >= 2:
            print(f"validation took: {time.perf_counter() - t_valid_start:.2f} s")


    # display evoltion of losses
    (list_of_min_losses, list_of_lists) = \
        utils.display_evolution(
            epoch, t_epoch_loop_start,
            train_loss_quantile_scaled, valid_loss_quantile_scaled,
            list_of_min_losses, list_of_lists,
            EPOCHS, DISPLAY_EVERY, PLOT_CONV_EVERY, MIN_DELTA, VERBOSE)

    # plotting convergence
    if ((epoch+1 == PLOT_CONV_EVERY) | ((epoch+1) % PLOT_CONV_EVERY == 0))\
            & (epoch < EPOCHS-2):
        plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                          list_of_lists[2], list_of_lists[3],
                          partial=True, verbose=VERBOSE)

    # Check for early stopping
    if early_stopping(valid_loss_quantile_scaled):
        if VERBOSE >= 1:
            print(f"Early stopping triggered at epoch {epoch+1}.")
        break

    torch.cuda.empty_cache()

# plotting convergence for entire training
plots.convergence_quantile(list_of_lists[0], list_of_lists[1],
                            list_of_lists[2], list_of_lists[3],
                            partial=False, verbose=VERBOSE)


t_metamodel_start = time.perf_counter()
# small LR for meta-model weights

_baselines = pd.concat([
         baseline_features_GW['lr'],
         baseline_features_GW['rf']], axis=1)
_baselines.columns = ['lr', 'rf']
_baselines_train = _baselines[:TRAIN_SPLIT][:-n_valid]
_baselines_train.index   = train_dates


# preparating training
true_train_GW, dict_pred_train_GW, dict_baseline_train_GW = \
    subset_predictions_day_ahead(X_train_GW, train_loader)
# print(_baselines_train)
y_true_train = pd.Series(y_train_GW, name='y', index=train_dates)
# print("true values [GW] at these times", y_true_train[missing_indices].round(2))

df_train = pd.concat([
        y_true_train,
        _baselines_train,
        dict_pred_train_GW['q50']
    ], axis=1, join="inner").astype(np.float32)
# print(f"df_train:    {df_train.shape} ({df_train.index.min()} -> {df_train.index.max()})")
df_train.columns = ['y', 'lr', 'rf', 'nn']
input_train = df_train[['lr', 'rf', 'nn']]  #.to_numpy()
y_train = df_train[['y']].squeeze()  #.to_numpy()


# preparating validation
true_valid_GW, dict_pred_valid_GW, dict_baseline_valid_GW = \
    subset_predictions_day_ahead(X_valid_GW, valid_loader)
_baselines_valid = _baselines[:TRAIN_SPLIT][-n_valid:]
_baselines_valid.index = valid_dates
input_valid = pd.concat([_baselines_valid, dict_pred_valid_GW['q50']],
                     axis=1, join="inner").astype('float32')
input_valid.columns = ['lr', 'rf', 'nn']


# preparating testing
true_test_GW, dict_pred_test_GW, dict_baseline_test_GW = \
    subset_predictions_day_ahead(X_test_GW, test_loader)
_baselines_test = _baselines[TRAIN_SPLIT:]
_baselines_test.index = test_dates
input_test = pd.concat([_baselines_test, dict_pred_test_GW['q50']],
                     axis=1, join="inner").astype('float32')
input_test.columns = ['lr', 'rf', 'nn']


# metamodel
weights_meta, pred_meta1_train, pred_meta1_valid, pred_meta1_test = \
    metamodel.weights_LR_metamodel(input_train, y_train, input_valid, input_test,
                            verbose=VERBOSE)
if VERBOSE >= 1:
    print(f"weights_meta [%]: "
          f"{ {name: round(value*100, 1) for name, value in weights_meta.items()}}")
t_metamodel_end = time.perf_counter()
# TODO very slow b/c of pandas, concat, etc. for alignment
if VERBOSE >= 2:
    print(f"metamodel  took: {time.perf_counter() - t_metamodel_start:.2f} s")




# if VERBOSE >= 2:
#     y_valid_agg_scaled, y_valid_pred_agg_scaled, _, has_pred =\
#         utils.aggregate_day_ahead(
#             model        = model,
#             loader       = valid_loader,
#             dates        = valid_dates,
#             scaler_y     = scaler_y,
#             baseline_idx = baseline_idx,
#             device       = device,
#             input_length = INPUT_LENGTH,
#             weights_meta = WEIGHTS_META,
#             quantiles    = QUANTILES,
#         )
#     assert has_pred.dtype == bool, has_pred.dtype
#     assert has_pred.shape[0] == len(valid_dates), \
#         f"has_pred.shape[0] ({has_pred.shape[0]}) != len(dates) ({len(dates)})"

#     # window-aligned dates (prediction at t = i + input_length)
#     valid_dates_win = valid_dates #[INPUT_LENGTH : INPUT_LENGTH + len(valid_loader.dataset)]

#     dates_masked  = valid_dates_win        [has_pred]
#     y_true_masked = y_valid_agg_scaled     [has_pred]
#     y_pred_masked = y_valid_pred_agg_scaled[has_pred]

#     top_bad_days = utils.worst_days_by_loss(
#         dates     = dates_masked,
#         y_true    = y_true_masked,
#         y_pred    = y_pred_masked,
#         quantiles = QUANTILES,
#         temperature=Tavg_full[TRAIN_SPLIT-n_valid : TRAIN_SPLIT][has_pred],
#         num_steps_per_day=NUM_STEPS_PER_DAY,
#         top_n     = 25,
#     )

#     print(top_bad_days.to_string())


#     # import matplotlib.pyplot as plt
#     # day = top_bad_days.iloc[0]["date"]

#     # q50_idx = QUANTILES.index(0.5)

#     # day = top_bad_days.iloc[0]["date"]
#     # mask_day = dates_masked.normalize() == day

#     # # inverse-scale for plotting
#     # y_true_day_GW = scaler_y.inverse_transform(
#     #     y_true_masked.reshape(-1, 1)
#     # ).ravel()

#     # y_pred_day_GW = scaler_y.inverse_transform(
#     #     y_pred_masked[:, q50_idx].reshape(-1, 1)
#     # ).ravel()

#     # plt.figure(figsize=(10, 4))
#     # plt.plot(dates_masked, y_true_day_GW, label="true")
#     # plt.plot(dates_masked, y_pred_day_GW, label="pred (q50)")
#     # plt.legend()
#     # plt.title(f"Worst training day: {day}")
#     # plt.show()


# ============================================================
# 7. TEST PREDICTIONS
# ============================================================

if VERBOSE >= 1:
    print("\nStarting test ...")  #"(baseline: {name_baseline})...")


true_test_GW, dict_pred_test_GW, dict_baseline_test_GW = \
    subset_predictions_day_ahead(X_test_GW, test_loader)

# print("true_test_GW:\n",      true_test_GW.head())
# print("dict_pred_test_GW:\n", {_name: _test.head() \
#                     for (_name, _test) in dict_pred_test_GW.items()})
# print("dict_baseline_test_GW:\n", {_name: _test.head() \
#                     for (_name, _test) in dict_baseline_test_GW.items()})

if VERBOSE >= 1:
    print("\nTesting quantiles")
    for tau in QUANTILES:
        key = f"q{int(100*tau)}"
        cov = utils.quantile_coverage(true_test_GW, dict_pred_test_GW[key])
        print(f"Coverage {key}:{cov*100:5.1f}%, i.e."
              f"{(cov-tau)*100:5.1f}%pt off{tau*100:3n}% target")
    print()

name_baseline =  'rf' if 'rf' in dict_baseline_test_GW else \
                ('lr' if 'lr' in dict_baseline_test_GW else None)

common_idx = true_test_GW.index.intersection(dict_pred_test_GW['q50'].index)

for _name in ['rf']:  # 'lr',
    if _name in dict_baseline_test_GW:  # keep only those we trained
        common_idx = common_idx.intersection(dict_baseline_test_GW[_name].index)
# common_idx = (
#     true_test.index
#     .intersection(dict_pred_test['median'].index)
#     .intersection(dict_baseline_test['rf'].index)
# )

# assert len(common_idx) > 0, "No common timestamps between truth and predictions!"
# TODO reinstate assert


true_test_GW = true_test_GW.loc[common_idx]
for k in dict_pred_test_GW:
    dict_pred_test_GW    [k] = dict_pred_test_GW    [k].loc[common_idx]
for k in dict_baseline_test_GW:
    dict_baseline_test_GW[k] = dict_baseline_test_GW[k].loc[common_idx]





# ============================================================
# 8. METAMODEL
# ============================================================


# NN metamodel
# ============================================================

pred_meta2_train, pred_meta2_valid, pred_meta2_test = \
    metamodel.metamodel_NN(
        [dict_pred_train_GW, dict_pred_valid_GW, dict_pred_test_GW],
         [dict_baseline_train_GW,dict_baseline_valid_GW,dict_baseline_test_GW],
         [X_train_GW,  X_valid_GW, X_test_GW],
         [y_train_GW,  y_valid_GW, y_test_GW],
         [train_dates, valid_dates,test_dates],
        feature_cols, PRED_LENGTH,
        #constants
        META_DROPOUT, META_NUM_CELLS, META_EPOCHS,
        META_LR, META_WEIGHT_DECAY,
        META_PATIENCE, META_FACTOR,
        META_BATCH_SIZE, device)

if VERBOSE >= 1:
    print("\nTraining metrics [GW]:")
    utils.compare_models(true_train_GW, dict_pred_train_GW, dict_baseline_train_GW,
                         [pred_meta1_train, pred_meta2_train],
                         subset="train", unit="GW", verbose=VERBOSE)

    print("\nvalidation metrics [GW]:")
    utils.compare_models(true_valid_GW, dict_pred_valid_GW, dict_baseline_valid_GW,
                         [pred_meta1_valid, pred_meta2_valid],
                         subset="valid", unit="GW", verbose=VERBOSE)

    print("\nTesting metrics [GW]:")
    utils.compare_models(true_test_GW, dict_pred_test_GW,
                         dict_baseline_test_GW, [pred_meta1_test, pred_meta2_test],
                         subset="test", unit="GW", verbose=VERBOSE)

    print("Plotting test results...")

# print(true_test_GW)
# print(dict_pred_test_GW['q50'])
# print(dict_baseline_test_GW['rf'])
# print(meta_test_GW)

plots.all_tests(true_test_GW, {'q50': dict_pred_test_GW['q50']},
                dict_baseline_test_GW, pred_meta2_test, name_baseline,
                Tavg_full.reindex(true_test_GW.index),
                NUM_STEPS_PER_DAY)



# plots.plot_quantile_fan(
#     true_series_GW,
#     dict_pred_series_GW,
#     q_low = "q10",
#     q_med = "q50",
#     q_high= "q90",
#     baseline_series=dict_baseline_series_GW,
#     title="Electricity consumption forecast (NN quantiles)"
# )


# final cleanup to free pinned memory and intermediate arrays
try:
    del valid_loader, test_loader, test_dataset_scaled
    # del train_dataset, valid_dataset
    del true_train_GW, dict_pred_train_GW, pred_meta1_train, pred_meta2_train
    del true_valid_GW, dict_pred_valid_GW, pred_meta1_valid, pred_meta2_valid
    del true_test_GW,  dict_pred_test_GW,  pred_meta1_test,  pred_meta2_test
    # del baseline_losses_quantile_scaled,
except NameError:
    pass



if torch.cuda.is_available():
    # clear VRAM
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.synchronize()

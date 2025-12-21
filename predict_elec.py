import gc
# from datetime import datetime
import time
# from   typing import Dict  # List

import torch
# import torch.nn as nn

import numpy  as np
import pandas as pd


# from   constants import *  # Spyder annoyingly complains: "may be defined in constants"
from   constants import (SYSTEM_SIZE, SEED, TRAIN_SPLIT_FRACTION, VAL_RATIO, INPUT_LENGTH,
           PRED_LENGTH, BATCH_SIZE, EPOCHS, MODEL_DIM, NUM_HEADS, FFN_SIZE,
           NUM_LAYERS, PATCH_LEN, STRIDE, LAMBDA_CROSS, LAMBDA_COVERAGE,
           LAMBDA_DERIV, QUANTILES, NUM_GEO_BLOCKS, GEO_BLOCK_RATIO,
           LEARNING_RATE, WEIGHT_DECAY, DROPOUT, WARMUP_STEPS, PATIENCE,
           MIN_DELTA, VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY, WEIGHTS_META,
           VERBOSE, DICT_FNAMES, OUTPUT_FNAME, BASELINE_CFG,
           DAY_AHEAD, FORECAST_HOUR)
# import day_ahead
import architecture, utils, IO, plots


# system dimensions
# B = batch size
# L = input length
# H = prediction horizon
# Q = number of quantiles (len(quantiles))

# TODO look at tutorial on transformers in pytorch
# TODO account for known future: sines, WE, T° (as forecast: _add noise_), etc.
#       (difficulty: medi-high, 2–4 days)
# TODO  - scenarios, distribution of possible outcomes
# TODO? compare per-horizon error vs LR
# TODO? use a single-contiguous sliding forecast: Only advance by PRED_LENGTH each time
# [??] random forest
#  - overfitting? final model on train + validation?
#  - TODO? OOF/VC
# TODO metamodel (NN + RF)
#  - [done] static metamodel
#  - TODO trained metamodel (difficulty: medium, 1–2 days)
#    . Linear meta-learner per horizon (fast, interpretable)
#    . Small MLP or ridge regression trained on validation OOF predictions
#  - TODO Avoid Using Only the First Horizon
# BUG? leak future information?
# TODO move scheduler.step() into the loop
# validate_with_aggregation: Validation is significantly more expensive than needed
# [done] remove SMA and diff from LR and RF?
# Add features:
#  - public holidays,
#  - [done] Xmas, sun
# [done] Add abilities to spot outliers
# TODO add constant saying there are 48 data points per day






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

    df, dates_df = utils.df_features(DICT_FNAMES, OUTPUT_FNAME,
                           verbose = VERBOSE)  # 2 if SYSTEM_SIZE == 'DEBUG' else 1)



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



    # Remove every row containing any NA (no filling)
    df = df[feature_cols + [target_col]].dropna()
    # print(dates[:10])

    # Keep date separately for plotting later
    dates = df.index

    df = df.reset_index(drop=True)


    TRAIN_SPLIT = int(len(df) * TRAIN_SPLIT_FRACTION)
    test_months = len(df)-TRAIN_SPLIT
    n_valid     = int(TRAIN_SPLIT * VAL_RATIO)

    num_time_steps = df.shape[0]

    NUM_PATCHES = (INPUT_LENGTH - PATCH_LEN) // STRIDE + 1

    if VERBOSE >= 1:
        IO.print_model_summary(
            num_time_steps   = num_time_steps,
            feature_cols     = feature_cols,
            input_length     = INPUT_LENGTH,
            pred_length      = PRED_LENGTH,
            batch_size       = BATCH_SIZE,
            epochs           = EPOCHS,
            learning_rate    = LEARNING_RATE,
            weight_decay     = WEIGHT_DECAY,
            dropout          = DROPOUT,
            warmup_steps     = WARMUP_STEPS,
            patience         = PATIENCE,
            min_delta        = MIN_DELTA,
            model_dim        = MODEL_DIM,
            num_layers       = NUM_LAYERS,
            num_heads        = NUM_HEADS,
            ffn_size         = FFN_SIZE,
            patch_len        = PATCH_LEN,
            stride           = STRIDE,
            num_patches      = NUM_PATCHES,
            quantiles        = QUANTILES,
            lambda_cross     = LAMBDA_CROSS,
            lambda_coverage=LAMBDA_COVERAGE,
            lambda_deriv     = LAMBDA_DERIV
        )


# correlation matrix for temperatures
# utils.temperature_correlation_matrix(df)

if VERBOSE >= 1:
    print("Doing linear regression and random forest...")

t_start = time.perf_counter()
rf_params = dict(
        df          = df,
        target_col  = target_col,
        feature_cols= feature_cols,
        train_end   = TRAIN_SPLIT-n_valid,
        val_end     = TRAIN_SPLIT,
        models_cfg  = BASELINE_CFG,
        quantiles   = QUANTILES,
        lambda_cross= LAMBDA_CROSS,
        lambda_coverage=LAMBDA_COVERAGE,
        lambda_deriv= LAMBDA_DERIV,
        verbose     = VERBOSE
    )

cache_id = {
    "system_size":  SYSTEM_SIZE,
    "target":       target_col,
    "feature_cols": feature_cols,
    'train_end':    TRAIN_SPLIT-n_valid,
    'val_end':      TRAIN_SPLIT,
    # "split": "v1",   # optional: data split identifier
}

baseline_features_GW, baseline_models, baseline_losses_GW = \
    utils.load_or_compute_rf_predictions(
        compute_kwargs = rf_params,
        cache_dir      = "output",
        cache_id_dict  = cache_id,
    )
if VERBOSE >= 1:
    print(f"LR + RF took: {time.perf_counter() - t_start:.2f} s")


# # df['consumption_regression'], _, lr_losses
# baseline_features_GW, baseline_models, baseline_losses_GW = utils.regression_and_forest(
#     df          = df,
#     target_col  = target_col,
#     feature_cols= feature_cols,
#     train_end   = TRAIN_SPLIT-n_valid,
#     val_end     = TRAIN_SPLIT,
#     models_cfg  = baseline_cfg,
#     quantiles   = QUANTILES,
#     lambda_cross= LAMBDA_CROSS,
#     lambda_coverage=LAMBDA_COVERAGE,
#     lambda_deriv= LAMBDA_DERIV,
#     verbose     = VERBOSE
# )


# reset random number generation because sklearn changed it
np.   random.seed(SEED)
torch.manual_seed(SEED)

# Add features
for name, series in baseline_features_GW.items():
    col_name     = f"consumption_{name}"
    df[col_name] = series
    feature_cols.append(col_name)
# feature_cols.append('consumption_regression')
# print(df['consumption_regression'].head(20))

# ---- Construct final matrix: target first, then features ----

if VERBOSE >= 1:
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print("Using target:  ", target_col)


series = np.column_stack([
    df[target_col]  .values.astype(np.float32),
    df[feature_cols].values.astype(np.float32)
])

Tavg_full = df["Tavg_degC"].values   # for worst days

NUM_FEATURES   = len(feature_cols)  # /!\ y should not be included
num_time_steps = series.shape[0]

del df


block_sizes = architecture.geometric_block_sizes(
    NUM_PATCHES, NUM_GEO_BLOCKS, GEO_BLOCK_RATIO)
if VERBOSE >= 1:
    print(f"{'block_sizes'  :17s} = {block_sizes}")




# ============================================================
# 3. TRAIN/TEST SPLIT (time-respecting)
# ============================================================

if VERBOSE >= 1:
    print(f"{len(series)/24/2/365.25:.1f} years of data, "
          f"train: {TRAIN_SPLIT/24/2/365.25:.2f} yrs ({TRAIN_SPLIT_FRACTION*100:.1f}%), "
          f"test: {test_months/24/2/365.25:.2f} yrs"
          f" (switching at {dates[TRAIN_SPLIT]})")
    print()
assert INPUT_LENGTH + 60 < test_months,\
    f"INPUT_LENGTH ({INPUT_LENGTH}) > test_months ({test_months}) - 60"


# assert all(ts.hour == 12 for ts in train_dataset.forecast_origins)
# assert len(set(test_results['predictions']['target_time'])) == len(test_results['predictions'])


# ============================================================
# 4. NORMALIZE PREDICTORS ONLY (not the log-target)
# ============================================================


[train_loader, valid_loader, test_loader], [train_dates, valid_dates, test_dates ],\
        scaler_y, X_test_GW, y_test_GW, test_dataset_scaled, X_test_scaled =\
    architecture.make_X_and_y(
        series, dates, TRAIN_SPLIT, n_valid,
        feature_cols, target_col,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH, batch_size=BATCH_SIZE,
        do_day_ahead=DAY_AHEAD, forecast_hour=FORECAST_HOUR,
        verbose=VERBOSE)



# scale loss for LR
sigma_y = float(scaler_y.scale_[0])   # scale to normalize y (use for loss)
if VERBOSE >= 1:
    print(f"sigma_y = {sigma_y:.1f} GW")
baseline_losses_scaled = {
    name: {k: v / sigma_y**2 for k, v in d.items()}
    for name, d in baseline_losses_GW.items()
}
# lr_losses = {_key: lr_losses[_key] / sigma_y**2 for _key in lr_losses}


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

baseline_idx = dict()
for _name in BASELINE_CFG:
    baseline_idx[_name] = feature_cols.index(f"consumption_{_name}")


min_train_loss_scaled     = 9.999; min_valid_loss_scaled     = 9.999
meta_min_train_loss_scaled= 9.999; meta_min_valid_loss_scaled= 9.999
min_loss_display_scaled   = 9.999

list_train_loss_scaled     = []; list_min_train_loss_scaled     = []
list_valid_loss_scaled     = []; list_min_valid_loss_scaled     = []
list_meta_train_loss_scaled= []; list_meta_min_train_loss_scaled= []
list_meta_valid_loss_scaled= []; list_meta_min_valid_loss_scaled= []

amp_scaler = torch.amp.GradScaler(device=device)

# first_step = True


t_epoch_start = time.perf_counter()
for epoch in range(EPOCHS):

    model.train()
    train_loss_scaled     = 0.; valid_loss_scaled     = 0.
    meta_train_loss_scaled= 0.; meta_valid_loss_scaled= 0.

    # for batch_idx, (x_scaled, y_scaled, origins) in enumerate(train_loader):
    for batch_idx, (x_scaled, y_scaled, idx, origin_unix) in enumerate(train_loader):
        x_scaled_dev = x_scaled.to(device)
        y_scaled_dev = y_scaled.to(device)

        origins = [pd.Timestamp(t, unit='s') for t in origin_unix.tolist()]
        # print(batch_idx, x_scaled, y_scaled, origins[0], "to", origins[-1])

        # optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type): # mixed precision
            pred_scaled_dev = model(x_scaled_dev)
            loss_scaled_dev = architecture.loss_wrapper_quantile_torch(
                pred_scaled_dev, y_scaled_dev, quantiles=QUANTILES,
                lambda_cross=LAMBDA_CROSS, lambda_coverage=LAMBDA_COVERAGE,
                lambda_deriv=LAMBDA_DERIV)

        amp_scaler.scale(loss_scaled_dev).backward()       # full precision
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        amp_scaler.step(optimizer)
        amp_scaler.update()

        train_loss_scaled += loss_scaled_dev.item()

        with torch.no_grad():  # monitoring only
            loss_meta_scaled = architecture.compute_meta_loss(
                pred_scaled_dev, x_scaled_dev, y_scaled_dev, baseline_idx,
                WEIGHTS_META, quantiles=QUANTILES, lambda_cross=LAMBDA_CROSS,
                lambda_coverage=LAMBDA_COVERAGE, lambda_deriv=LAMBDA_DERIV)
        meta_train_loss_scaled += loss_meta_scaled.item()

    scheduler.step()
    train_loss_scaled     /= len(train_loader)
    meta_train_loss_scaled/= len(train_loader)

    # validation
    model.eval()

    if ((epoch+1) % VALIDATE_EVERY == 0) | (epoch == 0):
        # results = daf.validate_day_ahead(
        #     model    = model,
        #     dataset  = valid_dataset,
        #     scaler_y = scaler_y,
        #     device   = device,
        #     quantiles= QUANTILES,
        #     verbose  = VERBOSE
        # )

        # valid_loss_scaled = results['overall_metrics']['mae'] / sigma_y
        # meta_valid_loss_scaled = 0.

        valid_loss_scaled, meta_valid_loss_scaled =\
            utils.validate_day_ahead(
                model        = model,
                valid_loader = valid_loader,
                valid_dates  = valid_dates,
                scaler_y     = scaler_y,
                baseline_idx = baseline_idx,
                device       = device,
                input_length = INPUT_LENGTH,
                pred_length  = PRED_LENGTH,
                # incr_steps   = INCR_STEPS_TEST,
                weights_meta = WEIGHTS_META,
                quantiles    = QUANTILES,
                lambda_cross = LAMBDA_CROSS,
                lambda_coverage=LAMBDA_COVERAGE,
                lambda_deriv = LAMBDA_DERIV
            )
    # print("valid_loss_scaled:", valid_loss_scaled,
    #       "meta_valid_loss_scaled:", meta_valid_loss_scaled)


    if ((epoch+1) % DISPLAY_EVERY == 0) | (epoch == 0):
        # comparing latest loss to lowest so far
        if valid_loss_scaled <= min_loss_display_scaled - MIN_DELTA:
            is_better = '**'
        elif valid_loss_scaled <= min_loss_display_scaled:
            is_better = '*'
        else:
            is_better = ''

        min_loss_display_scaled = min_valid_loss_scaled

        t_epoch = time.perf_counter() - t_epoch_start

        if VERBOSE >= 1:
            print(f"{epoch+1:3n} /{EPOCHS:3n} ={(epoch+1)/EPOCHS*100:3.0f}%,"
                  f"{t_epoch/60*(EPOCHS/(epoch+1)-1)+.5:3.0f} min left, "
                  f"loss (1e-3): "
                  f"train{train_loss_scaled*1000:5.0f} (best{min_train_loss_scaled*1000:5.0f}), "
                  f"valid{valid_loss_scaled*1000:5.0f} ({    min_valid_loss_scaled*1000:5.0f})"
                  f" {is_better}")

    min_train_loss_scaled = min(min_train_loss_scaled, train_loss_scaled)
    list_train_loss_scaled    .append(train_loss_scaled)
    list_min_train_loss_scaled.append(min_train_loss_scaled)

    min_valid_loss_scaled = min(min_valid_loss_scaled, valid_loss_scaled)
    list_valid_loss_scaled    .append(valid_loss_scaled)
    list_min_valid_loss_scaled.append(min_valid_loss_scaled)

    # metamodel
    meta_min_train_loss_scaled = min(meta_min_train_loss_scaled, meta_train_loss_scaled)
    list_meta_train_loss_scaled    .append(meta_train_loss_scaled)
    list_meta_min_train_loss_scaled.append(meta_min_train_loss_scaled)

    meta_min_valid_loss_scaled = min(meta_min_valid_loss_scaled, meta_valid_loss_scaled)
    list_meta_valid_loss_scaled    .append(meta_valid_loss_scaled)
    list_meta_min_valid_loss_scaled.append(meta_min_valid_loss_scaled)


    if ((epoch+1 == PLOT_CONV_EVERY) | ((epoch+1) % PLOT_CONV_EVERY == 0))\
            & (epoch < EPOCHS-2):
        plots.convergence(list_train_loss_scaled, list_min_train_loss_scaled,
                          list_valid_loss_scaled, list_min_valid_loss_scaled,
                          None,  # baseline_losses_scaled,
                          None, None, None, None,
                          # list_meta_train_loss_scaled, list_meta_min_train_loss_scaled,
                          # list_meta_valid_loss_scaled, list_meta_min_valid_loss_scaled,
                          partial=True, verbose=VERBOSE)

    # Check for early stopping
    if early_stopping(valid_loss_scaled):
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break


    torch.cuda.empty_cache()


plots.convergence(list_train_loss_scaled, list_min_train_loss_scaled,
                  list_valid_loss_scaled, list_min_valid_loss_scaled,
                  None,  #baseline_losses_scaled,
                  None, None, None, None,
                  # list_meta_train_loss_scaled, list_meta_min_train_loss_scaled,
                  # list_meta_valid_loss_scaled, list_meta_min_valid_loss_scaled,
                  partial=False, verbose=VERBOSE)

if VERBOSE >= 2:
    y_valid_agg_scaled, y_valid_pred_agg_scaled, _, has_pred =\
        utils.aggregate_over_windows(
            model        = model,
            loader       = valid_loader,
            dates        = valid_dates,
            scaler_y     = scaler_y,
            baseline_idx = baseline_idx,
            device       = device,
            input_length = INPUT_LENGTH,
            weights_meta = WEIGHTS_META,
            quantiles    = QUANTILES,
        )
    assert has_pred.dtype == bool, has_pred.dtype
    assert has_pred.shape[0] == len(valid_dates), \
        f"has_pred.shape[0] ({has_pred.shape[0]}) != len(dates) ({len(dates)})"

    # window-aligned dates (prediction at t = i + input_length)
    valid_dates_win = valid_dates #[INPUT_LENGTH : INPUT_LENGTH + len(valid_loader.dataset)]

    dates_masked  = valid_dates_win        [has_pred]
    y_true_masked = y_valid_agg_scaled     [has_pred]
    y_pred_masked = y_valid_pred_agg_scaled[has_pred]

    top_bad_days = utils.worst_days_by_loss(
        dates     = dates_masked,
        y_true    = y_true_masked,
        y_pred    = y_pred_masked,
        quantiles = QUANTILES,
        temperature=Tavg_full[TRAIN_SPLIT-n_valid : TRAIN_SPLIT][has_pred],
        top_n     = 25,
    )

    print(top_bad_days.to_string())


    # import matplotlib.pyplot as plt
    # day = top_bad_days.iloc[0]["date"]

    # q50_idx = QUANTILES.index(0.5)

    # day = top_bad_days.iloc[0]["date"]
    # mask_day = dates_masked.normalize() == day

    # # inverse-scale for plotting
    # y_true_day_GW = scaler_y.inverse_transform(
    #     y_true_masked.reshape(-1, 1)
    # ).ravel()

    # y_pred_day_GW = scaler_y.inverse_transform(
    #     y_pred_masked[:, q50_idx].reshape(-1, 1)
    # ).ravel()

    # plt.figure(figsize=(10, 4))
    # plt.plot(dates_masked, y_true_day_GW, label="true")
    # plt.plot(dates_masked, y_pred_day_GW, label="pred (q50)")
    # plt.legend()
    # plt.title(f"Worst training day: {day}")
    # plt.show()


# ============================================================
# 7. TEST PREDICTIONS
# ============================================================

if VERBOSE >= 1:
    print("Starting test ...")  #"(baseline: {name_baseline})...")

# test_results = daf.validate_day_ahead(
#     model   = model,
#     dataset = test_dataset,
#     scaler_y= scaler_y,
#     device  = device,
#     quantiles=QUANTILES,
#     verbose = VERBOSE
# )

# pred_df = test_results['predictions']

# true_series_GW = pred_df.set_index('target_time')['y_true']

# dict_pred_series_GW = {
#     f'q{int(100*q)}': pred_df.set_index('target_time')[f'q{int(100*q)}']
#     for q in QUANTILES
# }


true_series_GW, dict_pred_series_GW, dict_baseline_series_GW = \
    utils.test_predictions_day_ahead(
        X_test_GW, y_test_GW, test_loader, model, scaler_y,
        num_test_windows=len(test_dataset_scaled),
        feature_cols = feature_cols,
        test_dates   = test_dates,
        device       = device,
        input_length = INPUT_LENGTH,
        pred_length  = PRED_LENGTH,
        # incr_steps   = INCR_STEPS_TEST,
        quantiles    = QUANTILES)

# print("true_series_GW:\n",      true_series_GW.head())
# print("dict_pred_series_GW:\n", {_name: _series.head() \
#                     for (_name, _series) in dict_pred_series_GW.items()})
# print("dict_baseline_series_GW:\n", {_name: _series.head() \
#                     for (_name, _series) in dict_baseline_series_GW.items()})


name_baseline = 'rf' if 'rf' in dict_baseline_series_GW else 'lr'

common_idx = true_series_GW.index.intersection(dict_pred_series_GW['q50'].index)

for _name in ['rf']:  # 'lr',
    if _name in dict_baseline_series_GW:  # keep only those we trained
        common_idx = common_idx.intersection(dict_baseline_series_GW[_name].index)
# common_idx = (
#     true_series.index
#     .intersection(dict_pred_series['median'].index)
#     .intersection(dict_baseline_series['rf'].index)
# )

# assert len(common_idx) > 0, "No common timestamps between truth and predictions!"
# TODO reinstate assert

if VERBOSE >= 1:
    print()
    for tau in QUANTILES:
        key = f"q{int(100*tau)}"
        cov = utils.quantile_coverage(true_series_GW, dict_pred_series_GW[key])
        print(f"Coverage {key}:{cov*100:5.1f}% (target{tau*100:3n}%)")


true_series_GW = true_series_GW.loc[common_idx]
for k in dict_pred_series_GW:
    dict_pred_series_GW    [k] = dict_pred_series_GW    [k].loc[common_idx]
for k in dict_baseline_series_GW:
    dict_baseline_series_GW[k] = dict_baseline_series_GW[k].loc[common_idx]

# assert true_series.index.equals(dict_pred_series['median'].index),\
#     (true_series.index, dict_pred_series['median'].index)
# assert true_series.index.equals(dict_baseline_series['rf'].index),\
#     (true_series.index, dict_baseline_series['rf'].index)


# ============================================================
# 9. FINAL OUT-OF-SAMPLE FORECAST (beyond last available data)
# ============================================================

# future dates: add months to last observed date
last_date = true_series_GW.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(minutes=30),
    periods=PRED_LENGTH,
    freq="30min"
)

with torch.no_grad():
    last_window = X_test_scaled[-INPUT_LENGTH:]         # shape (L, F)
    last_window = torch.tensor(last_window).unsqueeze(0).to(device)

    pred = model(last_window)                         # (1, H, Q)
    pred = pred.squeeze(0).cpu().numpy()       # (H, Q)

    # Inverse scaling (works column-wise)
    future_pred = scaler_y.inverse_transform(pred)     # (H, Q)

    # Build one Series per quantile
    future_series = {
        f"q{int(100*tau)}": pd.Series(
            future_pred[:, i],
            index=future_dates
        )
        for i, tau in enumerate(QUANTILES)
    }

    # free temporaries
    try:
        del pred, last_window, future_pred
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()




# ============================================================
# 8. PLOT ROLLING RESULTS (all test windows)
# ============================================================
# print(len(true_series), len(lr_series))
# assert len(true_series) == len(lr_series),\
#     f"{len(true_series)} != {len(lr_series)}"
# assert true_series.index.equals(pred_series.index)
# assert true_series.index.equals(lr_series.index)

utils.compare_models(true_series_GW, dict_pred_series_GW, dict_baseline_series_GW,
                     WEIGHTS_META, unit="GW", verbose=VERBOSE)

if VERBOSE >= 1:
    print("Plotting test results...")

plots.all_tests(true_series_GW, {'q50': dict_pred_series_GW['q50']},
                dict_baseline_series_GW, future_series, name_baseline)



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
    del true_series_GW, dict_pred_series_GW, baseline_losses_scaled, future_series
except NameError:
    pass



if torch.cuda.is_available():
    # clear VRAM
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.synchronize()

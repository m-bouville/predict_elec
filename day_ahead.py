"""
Day-Ahead Market Forecasting Module

Implements the European day-ahead market pattern:
- Forecasts made at noon (12:00) each day
- Predicts 48 half-hourly values from h+12 to h+36 hours ahead
- Each timestep predicted exactly once (no aggregation across forecast origins)
"""

from   typing import Dict, Tuple #, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# from   datetime import time


# ==============================================================================
# 1. FORECAST ORIGIN IDENTIFICATION
# ==============================================================================

# def get_day_ahead_forecast_origins(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
#     """
#     Identify forecast origins: noon (12:00) of each day.

#     Parameters
#     ----------
#     dates : pd.DatetimeIndex
#         All available timestamps in dataset

#     Returns
#     -------
#     pd.DatetimeIndex
#         Timestamps at 12:00 UTC for each day
#     """
#     # Filter to noon times only
#     noon_mask = (dates.hour == 12) & (dates.minute == 0)
#     origins   = dates[noon_mask]

#     return origins


# def create_day_ahead_dataset(
#     data:  np.ndarray,
#     dates: pd.DatetimeIndex,
#     input_length:  int,
#     pred_length:   int,
#     forecast_hour: int,
#     target_index : int
# ) -> Tuple[List[np.ndarray], List[np.ndarray], List[pd.Timestamp], List[pd.DatetimeIndex]]:
#     """
#     Create dataset following day-ahead market pattern.

#     Parameters
#     ----------
#     data : np.ndarray
#         Shape (T, F) - time series with features
#     dates : pd.DatetimeIndex
#         Timestamps for each row in data
#     input_length : int
#         Number of historical half-hours to use as input
#     pred_length : int
#         Number of future half-hours to predict (default 48 = 24 hours)
#     forecast_hour : int
#         Hour of day when forecast is made (default 12 = noon)
#     target_index : int
#         Column index of target variable

#     Returns
#     -------
#     X_list : List[np.ndarray]
#         Input windows, each shape (input_length, F)
#     y_list : List[np.ndarray]
#         Target windows, each shape (pred_length,)
#     origin_list : List[pd.Timestamp]
#         Forecast origin timestamp for each sample
#     target_dates_list : List[pd.DatetimeIndex]
#         Target timestamps for each sample's predictions
#     """

#     assert isinstance(dates, pd.DatetimeIndex), type(dates)

#     X_list      = []
#     y_list      = []
#     origin_list = []
#     target_dates_list = []

#     # Find all valid forecast origins (noon times with sufficient history and future)
#     for i, date in enumerate(dates):

#         # Must be at forecast hour
#         if date.hour != forecast_hour or date.minute != 0:
#             continue

#         # Must have enough history
#         if i < input_length:
#             continue

#         # Must have enough future data
#         if i + pred_length >= len(data):
#             continue

#         # print ("kept:", i, date)

#         # Extract input window (history before forecast origin)
#         X = data[i - input_length : i]  # (input_length, F)
#         X = np.delete(X, target_index, axis=1)

#         # Extract target (future consumption only, column 0)
#         y = data[i : i + pred_length, target_index]  # (pred_length,)

#         # Get target timestamps
#         target_dates = dates[i : i + pred_length]

#         X_list.append(X)
#         y_list.append(y)
#         origin_list.append(date)
#         target_dates_list.append(target_dates)

#     # print("len(X_list) =", len(X_list), "len(y_list) =", len(y_list),
#     #       "len(origin_list) =", len(origin_list),
#     #       "len(target_dates_list) =", len(target_dates_list))

#     return (
#         torch.tensor(self.X_list[idx]),
#         torch.tensor(self.y_list[idx]).unsqueeze(-1),
#         int(self.origin_list[idx].timestamp())
#     )

#     # return X_list, y_list, origin_list, target_dates_list


# ==============================================================================
# 2. DATASET CLASS FOR DAY-AHEAD FORECASTING
# ==============================================================================

class DayAheadDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for day-ahead market forecasting.

    Each sample represents one forecast made at noon, predicting the next
    48 half-hourly values.
    """

    def __init__(
        self,
        data:  np.ndarray,
        dates: pd.DatetimeIndex,
        input_length : int,
        pred_length  : int,
        forecast_hour: int,
        target_index : int
    ):
        """
        Parameters
        ----------
        data : np.ndarray
            Shape (T, F) where F includes target in column 0
        dates : pd.DatetimeIndex
            Timestamps for each row
        input_length : int
            Number of historical half-hours
        pred_length : int
            Number of future half-hours to predict
        forecast_hour : int
            Hour when forecast is made
        target_index : int
            Column index of target variable
        """
        self.data         = data.astype(np.float32)
        self.dates        = dates
        self.input_length = input_length
        self.pred_length  = pred_length
        self.forecast_hour= forecast_hour
        self.target_index = target_index

        # Pre-compute valid forecast indices
        self.valid_indices   = []
        self.forecast_origins= []

        for i, date in enumerate(dates):
            if (date.hour   == forecast_hour and
                date.minute == 0 and
                i >= input_length and
                i + pred_length < len(data)):

                self.valid_indices   .append(i)
                self.forecast_origins.append(date)

        # print("forecast_origins:", type(self.forecast_origins[0]))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns
        -------
        x : torch.Tensor
            Input features, shape (input_length, F)
        y : torch.Tensor
            Target values, shape (pred_length, 1)
        forecast_origin : pd.Timestamp
            When this forecast was made
        """
        i = self.valid_indices[idx]

        # Input: all features from history
        X = self.data[i - self.input_length : i]  # (input_length, F)
        X = np.delete(X, self.target_index, axis=1)

        # Target: only consumption, future values
        y = self.data[i : i + self.pred_length, self.target_index]


        # origin = self.forecast_origins[idx]
        # print(f"Type: {type(origin)}, Value: {origin}")
        # origin_int = int(origin.timestamp())
        # print(f"After conversion: {type(origin_int)}, Value: {origin_int}")

        return (
            torch.tensor(X),
            torch.tensor(y).unsqueeze(-1),  # (pred_length, 1)
            idx,
            int(self.forecast_origins[idx].timestamp()) # pd.Timestamp != batchable
        )


# ==============================================================================
# 3. VALIDATION FOR DAY-AHEAD PATTERN
# ==============================================================================

@torch.no_grad()
def validate_day_ahead(
    model    : nn.Module,
    dataset  : DayAheadDataset,
    scaler_y,
    device   : torch.device,
    quantiles: Tuple[float, ...],
    verbose  : int = 0
) -> Dict[str, any]:
    """
    Validate model on day-ahead forecasting task.

    Returns per-horizon metrics without any aggregation across forecast origins.

    Parameters
    ----------
    model : torch.nn.Module
        Trained quantile forecasting model
    dataset : DayAheadDataset
        Validation dataset
    scaler_y : StandardScaler
        Scaler for inverse transforming predictions
    device : torch.device
        Device for computation
    quantiles : Tuple[float, ...]
        Quantile levels being predicted
    verbose : int
        Verbosity level

    Returns
    -------
    dict
        Contains:
        - 'predictions': DataFrame with columns [origin, target_time, y_true, q50, ...]
        - 'metrics_by_horizon': Dict mapping horizon -> metrics
        - 'overall_metrics': Overall performance metrics
    """
    model.eval()

    all_origins     = []
    all_target_times= []
    all_y_true      = []
    all_y_pred      = []  # List of arrays, each (pred_length, num_quantiles)


    for idx in range(len(dataset)):
        x, y, origin = dataset[idx]

        # Move to device and add batch dimension
        x = x.unsqueeze(0).to(device)  # (1, L, F)

        # Predict
        pred_scaled = model(x)  # (1, H, Q)
        pred_scaled = pred_scaled.squeeze(0).cpu().numpy()  # (H, Q)

        # Inverse scale
        pred = scaler_y.inverse_transform(pred_scaled)  # (H, Q)
        y_true = scaler_y.inverse_transform(
            y.numpy().reshape(-1, 1)
        ).ravel()  # (H,)

        # Get target timestamps
        start_idx = dataset.valid_indices[idx]
        target_times = dataset.dates[start_idx : start_idx + dataset.pred_length]

        # Store
        all_origins.extend([origin] * len(y_true))
        all_target_times.extend(target_times)
        all_y_true.extend(y_true)
        all_y_pred.append(pred)



    # Concatenate all predictions
    all_y_pred = np.vstack(all_y_pred)  # (N_samples * pred_length, Q)
    all_y_true = np.array(all_y_true)

    # Create DataFrame
    pred_df = pd.DataFrame({
        'origin': all_origins,
        'target_time': all_target_times,
        'y_true': all_y_true
    })

    # Add quantile columns
    for i, q in enumerate(quantiles):
        pred_df[f'q{int(100*q)}'] = all_y_pred[:, i]

    # Compute overall metrics
    q50_idx = quantiles.index(0.5)
    overall_metrics = {
        'rmse': np.sqrt(np.mean((all_y_true - all_y_pred[:, q50_idx])**2)),
        'mae':  np.mean(np.abs  (all_y_true - all_y_pred[:, q50_idx])),
        'bias': np.mean(all_y_pred[:, q50_idx] - all_y_true),
        'mape': np.mean(np.abs((all_y_true - all_y_pred[:, q50_idx]) / all_y_true)) * 100
    }

    # Compute per-horizon metrics
    pred_df['horizon'] = pred_df.groupby('origin').cumcount()
    metrics_by_horizon = {}

    for h in range(dataset.pred_length):
        h_data = pred_df[pred_df['horizon'] == h]
        y_true_h = h_data['y_true'].values
        y_pred_h = h_data['q50'].values

        metrics_by_horizon[h] = {
            'rmse': np.sqrt(np.mean((y_true_h - y_pred_h)**2)),
            'mae':  np.mean(np.abs  (y_true_h - y_pred_h)),
            'bias': np.mean(y_pred_h - y_true_h)
        }

    if verbose >= 1:
        print("\n=== Day-Ahead Validation Results ===")
        print(f"Number of forecasts: {len(dataset)}")
        print(f"Overall RMSE: {overall_metrics['rmse']:.2f} GW")
        print(f"Overall MAE:  {overall_metrics['mae']:.2f} GW")
        print(f"Overall Bias: {overall_metrics['bias']:.2f} GW")

        # Show metrics at key horizons
        key_horizons = [0, 11, 23, 35, 47]  # Start, 6h, 12h, 18h, 24h
        print("\nMetrics by horizon:")
        print(f"{'Horizon':>8} {'Hours':>7} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
        for h in key_horizons:
            if h < len(metrics_by_horizon):
                m = metrics_by_horizon[h]
                print(f"{h:8d} {h/2:7.1f}h {m['rmse']:8.2f} {m['mae']:8.2f} {m['bias']:8.2f}")

    return {
        'predictions':        pred_df,
        'metrics_by_horizon': metrics_by_horizon,
        'overall_metrics':    overall_metrics
    }


# ==============================================================================
# 4. TRAINING MODIFICATIONS
# ==============================================================================

def compute_day_ahead_loss(
    model  : torch.nn.Module,
    batch  : Tuple,
    loss_fn: callable,
    device : torch.device
) -> torch.Tensor:
    """
    Compute loss for a batch in day-ahead format.

    Parameters
    ----------
    model : torch.nn.Module
        Forecasting model
    batch : Tuple
        (x, y, origins) from DayAheadDataset
    loss_fn : callable
        Loss function (e.g., quantile loss)
    device : torch.device
        Computation device

    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    x, y, _ = batch
    x = x.to(device)
    y = y.to(device)

    pred = model(x)  # (B, H, Q)
    loss = loss_fn(pred, y)

    return loss


# ==============================================================================
# 5. INTEGRATION HELPER
# ==============================================================================

def create_day_ahead_splits(
    data : np.ndarray,
    dates: pd.DatetimeIndex,
    train_end_date: pd.Timestamp,
    val_end_date  : pd.Timestamp,
    input_length  : int,
    pred_length   : int = 48,
    forecast_hour : int = 12
) -> Tuple[DayAheadDataset, DayAheadDataset, DayAheadDataset]:
    """
    Create train/val/test splits for day-ahead forecasting.

    Parameters
    ----------
    data : np.ndarray
        Full dataset, shape (T, F)
    dates : pd.DatetimeIndex
        Timestamps for each row
    train_end_date : pd.Timestamp
        Last date in training set
    val_end_date : pd.Timestamp
        Last date in validation set
    input_length : int
        Historical window length
    pred_length : int
        Forecast horizon (48 = 24 hours)
    forecast_hour : int
        Hour of forecast (12 = noon)

    Returns
    -------
    train_dataset, val_dataset, test_dataset : DayAheadDataset
    """
    # Split data by date
    train_mask= dates <  train_end_date
    val_mask  =(dates >= train_end_date) & (dates < val_end_date)
    test_mask = dates >= val_end_date

    train_data = data [train_mask]
    train_dates= dates[train_mask]

    val_data  = data [train_mask | val_mask]  # Include history
    val_dates = dates[train_mask | val_mask]

    test_data = data  # Full dataset for test (needs history)
    test_dates= dates # BUG?

    # Create datasets
    train_dataset = DayAheadDataset(
        train_data, train_dates, input_length, pred_length, forecast_hour
    )

    val_dataset = DayAheadDataset(
        val_data,  val_dates, input_length, pred_length, forecast_hour
    )

    test_dataset = DayAheadDataset(
        test_data, test_dates, input_length, pred_length, forecast_hour
    )

    return train_dataset, val_dataset, test_dataset


# ==============================================================================
# 6. REPORTING
# ==============================================================================

def plot_horizon_degradation(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    save_path: str = None
):
    """
    Plot how forecast error increases with horizon.

    Parameters
    ----------
    metrics_by_horizon : Dict
        Mapping from horizon index to metrics dict
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    horizons = sorted(metrics_by_horizon.keys())
    rmse = [metrics_by_horizon[h]['rmse'] for h in horizons]
    mae  = [metrics_by_horizon[h]['mae']  for h in horizons]
    bias = [metrics_by_horizon[h]['bias'] for h in horizons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # RMSE
    axes[0].plot([h/2 for h in horizons], rmse, 'o-')
    axes[0].set_xlabel('Forecast Horizon (hours)')
    axes[0].set_ylabel('RMSE (GW)')
    axes[0].set_title ('RMSE vs Horizon')
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot([h/2 for h in horizons], mae, 'o-', color='orange')
    axes[1].set_xlabel('Forecast Horizon (hours)')
    axes[1].set_ylabel('MAE (GW)')
    axes[1].set_title('MAE vs Horizon')
    axes[1].grid(True, alpha=0.3)

    # Bias
    axes[2].plot([h/2 for h in horizons], bias, 'o-', color='red')
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Forecast Horizon (hours)')
    axes[2].set_ylabel('Bias (GW)')
    axes[2].set_title('Bias vs Horizon')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def print_comparison_to_baseline(
    day_ahead_metrics: Dict,
    baseline_metrics : Dict,
    baseline_name    : str = 'LR'
):
    """
    Compare day-ahead forecasting to baseline models.

    Parameters
    ----------
    day_ahead_metrics : Dict
        Results from validate_day_ahead()
    baseline_metrics : Dict
        Baseline model metrics
    baseline_name : str
        Name of baseline model
    """
    print(f"\n=== Comparison: Day-Ahead NN vs {baseline_name} ===")
    print(f"{'Metric':12} {'NN':>10} {baseline_name:>10} {'Improvement':>12}")
    print("-" * 50)

    for metric in ['rmse', 'mae', 'bias']:
        nn_val = day_ahead_metrics['overall_metrics'][metric]
        base_val = baseline_metrics.get(metric, np.nan)

        if not np.isnan(base_val) and base_val != 0:
            improvement = (base_val - nn_val) / base_val * 100
            sign = '+' if improvement > 0 else ''
            print(f"{metric.upper():12} {nn_val:10.2f} {base_val:10.2f} {sign}{improvement:11.1f}%")
        else:
            print(f"{metric.upper():12} {nn_val:10.2f} {base_val:10.2f} {'N/A':>12}")

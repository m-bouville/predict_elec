

# --------------------------------------------------------
# Plots
# --------------------------------------------------------

from   typing import Dict, Tuple, Optional  # List

import matplotlib.pyplot as plt
import pandas as pd  # for types




def data(df, xlabel=None, ylabel=None, title=None) -> None:
    print(f"Plotting {title}..." if title is not None else "Plotting...")
    df.plot(figsize=(10,6))
    if xlabel is not None:  plt.xlabel(xlabel)
    if ylabel is not None:  plt.ylabel(ylabel)
    if title  is not None:  plt.title (title)
    plt.legend()
    plt.show()



# --------------------------------------------------------
# Convergence
# --------------------------------------------------------

def convergence(list_train_loss: list, list_min_train_loss: list,
                list_valid_loss: list, list_min_valid_loss: list,
                baseline_losses: Dict[str, Dict[str, float]],
                list_meta_train_loss: list, list_meta_min_train_loss: list,
                list_meta_valid_loss: list, list_meta_min_valid_loss: list,
                partial: bool = False, verbose:int = 0) -> None:
    assert len(list_train_loss) == len(list_min_train_loss) ==\
           len(list_valid_loss) == len(list_min_valid_loss), \
        f"len(list_train_loss) ({len(list_train_loss)}), "\
        f"len(list_min_train_loss) ({len(list_min_train_loss)})"\
        f"len(list_valid_loss) ({len(list_valid_loss)}), "\
        f"len(list_min_valid_loss) ({len(list_min_valid_loss)}) must be equal)"
    length = len(list_train_loss)

    if verbose >= 1:
        if not partial:
            print("Training done. Plotting convergence...")
        else:
            print("Plotting convergence for training so far...")


    plt.figure(figsize=(10,6))

    # neural net
    plt.plot(range(1, length+1), list_train_loss    [:], label="train",    color="black")
    plt.plot(range(1, length+1), list_min_train_loss[:], label="train min",color="black",alpha=0.4)
    plt.plot(range(1, length+1), list_valid_loss    [:], label="valid",    color="red")
    plt.plot(range(1, length+1), list_min_valid_loss[:], label="valid min",color="red", alpha=0.4)

    # LR, RF
    if baseline_losses is not None:
        shape = {'lr': ',',     'rf': 'o'}
        for name, d in baseline_losses.items():
            plt.scatter(1, d['train'], label=f"train {name}", color="green",
                        marker=shape[name], alpha=0.6)
            plt.scatter(1, d['valid'], label=f"valid {name}", color="green", marker=shape[name])

    # metamodel
    if list_meta_train_loss is not None:
        plt.plot(range(1, length+1), list_meta_train_loss    [:], label="train meta",    color="blue")
    if list_meta_min_train_loss is not None:
        plt.plot(range(1, length+1), list_meta_min_train_loss[:], label="train meta min",color="blue", alpha=0.4)
    if list_meta_valid_loss is not None:
        plt.plot(range(1, length+1), list_meta_valid_loss    [:], label="valid meta",    color="magenta")
    if list_meta_min_valid_loss is not None:
        plt.plot(range(1, length+1), list_meta_min_valid_loss[:], label="valid meta min",color="magenta", alpha=0.4)


    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.ylabel(("(partial) " if partial else "") + "training loss (log)")

    # legend (two columns)
    handles, labels = plt.gca().get_legend_handles_labels()
        # Split into train vs valid groups by label name
    train_items = [(h, l) for h, l in zip(handles, labels) if "train" in l]
    valid_items = [(h, l) for h, l in zip(handles, labels) if "valid" in l]
        # Interleave into two columns: left=train, right=valid
    ordered = train_items + valid_items
    handles2, labels2 = zip(*ordered)
    plt.legend(handles2, labels2, ncol=2)

    plt.show()



# --------------------------------------------------------
# Plotting the test set
# --------------------------------------------------------

def _apply_range(series: pd.Series,
                date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
                ):
    if date_range is None:
        return series
    start, end = date_range
    return series.loc[start:end]

def _apply_moving_average(series: pd.Series, ma: Optional[int]):
    if ma is None:
        return series
    return series.rolling(ma, min_periods=1).mean()

def _apply_groupby(series: pd.Series, col: Optional[str]):
    if col is None:
        return series
    if col in ['year', 'month', 'day', 'hour', 'minute', 'dayofyear']:
        return series.groupby(getattr(series.index, col)).mean()
    if col == 'timeofday':
        timeofday = series.index.hour + series.index.minute/60
        return series.groupby(timeofday).mean()
    if col == 'dayofweek':
        dayofweek = series.index.dayofweek + (series.index.hour + series.index.minute/60) / 24
        return series.groupby(dayofweek).mean()
    raise ValueError(f"Invalid column: {col}.")



def test(true_series: pd.Series, dict_pred_series: Dict[str, pd.Series],
         baseline_series: pd.Series or None, future_series: pd.Series,
         name_baseline: str or None,
         xlabel: str = "date", ylabel: str = "consumption [GW]", title=None,
         ylim: [float, float] or None = None,
         date_range:     Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
         moving_average: Optional[int] = None,
         groupby:        Optional[str] = None) -> None:

    # preparation: range and/or SME
    _true_series     = _apply_groupby(_apply_range(_apply_moving_average(
        true_series, moving_average),     date_range), groupby)
    _baseline_series = _apply_groupby(_apply_range(_apply_moving_average(
        baseline_series, moving_average), date_range), groupby) \
            if name_baseline is not None else None

    _dict_pred_series = {}
    for name, series in dict_pred_series.items():
        s = _apply_moving_average(series, moving_average)
        s = _apply_range  (s, date_range)
        s = _apply_groupby(s, groupby)
        _dict_pred_series[name] = s

    _future_series = future_series # SMA would make little sense
    # if future_series is not None:
    #     _future_series = _apply_moving_average(future_series, moving_average)
    # else:
    #     _future_series = None


    plt.figure(figsize=(10,6))

    if _true_series is not None:
        plt.plot(_true_series.index, _true_series.values, label="actual", color="black")

    for name, series in _dict_pred_series.items():
        plt.plot(series.index, series.values,
                 color="red",
                 alpha=0.7,
                 label=f"forecast NN ({name})") # if k == 0 else None)


    if name_baseline is not None:
        plt.plot(_baseline_series.index, _baseline_series.values,
                 label=f"forecast {name_baseline}", color="green")

    if _future_series is not None:
        plt.plot(_future_series.index,_future_series.values, label="future", color="blue")

    if ylim   is not None:  plt.ylim  (ylim)
    if xlabel is not None:  plt.xlabel(xlabel)
    if ylabel is not None:  plt.ylabel(ylabel)
    if title  is not None:  plt.title (title)
    plt.legend()
    plt.show()



def all_tests(true_series:         pd.Series,
              dict_pred_series:    Dict[str, pd.Series],
              dict_baseline_series:Dict[str, pd.Series],
              future_series:       pd.Series,
              name_baseline:       str,
              days_zoom:           int | Tuple[int] = [8, 61],
              ylim:  Tuple[Tuple[float, float], Tuple[float, float]] = [[35, 55], [-4, 4]]
             ) -> None:
    # print("latest date:")
    # print("true:", true_series.index[-1])
    # print("pred:", max([s.index[-1] for s in dict_pred_series]))
    # print("regr:", lr_series.index[-1])

    # test(true_series, dict_pred_series, lr_series, future_series, name_baseline)

    baseline_series = dict_baseline_series[name_baseline] \
        if name_baseline is not None else None
    SMA_consumption = [None, 2*24]
    SMA_residual    = [2*2,  2*24]

    for _idx, _zoom in enumerate(days_zoom):
        # zoom
        zoom_horizon= pd.Timedelta(days=_zoom)
        zoom_end    = true_series.index[-1]
        zoom_start  = zoom_end - zoom_horizon


        # plot consumption
        _SMA    = SMA_consumption[_idx]
        _SMA_str = f" (SMA {_SMA/2} hrs)" if _SMA is not None else ""
        test(true_series, dict_pred_series, baseline_series, None,
             name_baseline, ylabel=f"consumption{_SMA_str} [GW]",
             ylim=ylim[0], date_range=[zoom_start, zoom_end], moving_average=_SMA)

        # compute and plot residuals
        _SMA    = SMA_residual[_idx]
        _SMA_str = f" (SMA {_SMA/2} hrs)" if _SMA is not None else ""

        residual_true_series     = true_series    - true_series
        residual_baseline_series = baseline_series- true_series \
            if name_baseline is not None else None
        dict_residual_pred_series= {name: series - true_series\
            for name, series in dict_pred_series.items()}

        test(residual_true_series, dict_residual_pred_series, residual_baseline_series, None,
             name_baseline, ylabel=f"consumption difference{_SMA_str} [GW]",
             ylim=ylim[1], date_range=[zoom_start, zoom_end], moving_average=_SMA)


    # plot consumption by hour, over a day
    _zoom = max(days_zoom)
    zoom_horizon= pd.Timedelta(days=_zoom)
    zoom_end    = true_series.index[-1]
    zoom_start  = zoom_end - zoom_horizon

    test(true_series, dict_pred_series, baseline_series, None,
         name_baseline, xlabel="time of day", ylabel="consumption [GW]",
         ylim=ylim[0], date_range=[zoom_start, zoom_end], groupby='timeofday')
    test(residual_true_series, dict_residual_pred_series, residual_baseline_series, None,
         name_baseline, xlabel="time of day", ylabel="consumption difference [GW]",
         ylim=ylim[1], date_range=[zoom_start, zoom_end], groupby='timeofday')


    # plot consumption over a week
    test(true_series, dict_pred_series, baseline_series, None,
         name_baseline, xlabel="day of week", ylabel="consumption [GW]",
         ylim=ylim[0], date_range=[zoom_start, zoom_end], groupby='dayofweek')
    test(residual_true_series, dict_residual_pred_series, residual_baseline_series, None,
         name_baseline, xlabel="day of week", ylabel="consumption difference [GW]",
         ylim=ylim[1], date_range=[zoom_start, zoom_end], groupby='dayofweek',
         moving_average=SMA_residual[0])  # smoothing a little





def plot_quantile_fan(
    true_series: pd.Series,
    pred_quantiles: Dict[str, pd.Series],
    *,
    q_low:  str = "q10",
    q_med:  str = "q50",
    q_high: str = "q90",
    baseline_series: Optional[Dict[str, pd.Series]] = None,
    title:  str = "Forecast with uncertainty",
    ylabel: str = "consumption [GW]",
    figsize=(10, 6),
) -> None:
    """
    Plot actual series, median forecast, and quantile uncertainty band.

    Parameters
    ----------
    true_series:
        Ground truth time series.
    pred_quantiles:
        Dict like {"q10": Series, "q50": Series, "q90": Series, ...}
        All series must share the same datetime index.
    q_low, q_med, q_high:
        Keys in pred_quantiles used for the fan and median.
    baseline_series:
        Optional dict of baseline Series (e.g. {"rf": Series}).
    """

    # ------------------------
    # Align indices safely
    # ------------------------
    idx = true_series.index
    idx = idx.intersection(pred_quantiles[q_med].index)

    if q_low in pred_quantiles:
        idx = idx.intersection(pred_quantiles[q_low].index)
    if q_high in pred_quantiles:
        idx = idx.intersection(pred_quantiles[q_high].index)

    true = true_series.loc[idx]
    qmed = pred_quantiles[q_med].loc[idx]

    qlo = pred_quantiles[q_low] .loc[idx] if q_low  in pred_quantiles else None
    qhi = pred_quantiles[q_high].loc[idx] if q_high in pred_quantiles else None

    # ------------------------
    # Plot
    # ------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        true,
        color="black",
        linewidth=1.5,
        label="actual"
    )

    ax.plot(
        qmed,
        color="red",
        linewidth=1.5,
        label=f"NN median ({q_med})"
    )

    if qlo is not None and qhi is not None:
        ax.fill_between(
            idx,
            qlo,
            qhi,
            color="red",
            alpha=0.20,
            label=f"NN {q_low}–{q_high}"
        )

    # ------------------------
    # Optional baselines
    # ------------------------
    if baseline_series is not None:
        for name, series in baseline_series.items():
            common = idx.intersection(series.index)
            ax.plot(
                series.loc[common],
                linewidth=1.2,
                label=name
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

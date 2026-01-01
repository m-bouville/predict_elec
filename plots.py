

# --------------------------------------------------------
# Plots
# --------------------------------------------------------

from   typing import Dict, Tuple,Sequence, Optional  # List

import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

import numpy  as np
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

def convergence_quantile(list_train_loss: list, list_min_train_loss: list,
                list_valid_loss: list, list_min_valid_loss: list,
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
            print("\nTraining done. Plotting convergence...")
        else:
            print("Plotting convergence for training so far...")


    plt.figure(figsize=(10,6))

    # neural net
    x = range(1, length+1)
    plt.plot(x, list_train_loss    [:], label="train",    color="black")
    plt.plot(x, list_min_train_loss[:], label="train min",color="black",alpha=0.4)
    plt.plot(x, list_valid_loss    [:], label="valid",    color="red")
    plt.plot(x, list_min_valid_loss[:], label="valid min",color="red",  alpha=0.4)

    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.ylabel(("(partial) " if partial else "") + "training quantile loss (log)")

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


def loss_per_horizon(dict_evolution_loss: Dict[str, np.ndarray],
                     minutes_per_step   : int,
                     title: Optional[str] = str):
    hours_per_step = minutes_per_step/60.
    x = np.arange(0, len(next(iter(dict_evolution_loss.values()))) * hours_per_step,
                  step=hours_per_step)

    for key, loss_array in dict_evolution_loss.items():
        plt.plot(x, loss_array, label=key)

    plt.xlabel("Horizon")
    plt.ylabel("Loss")
    plt.ylim(bottom=0.)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()


# --------------------------------------------------------
# SMA, groupby
# --------------------------------------------------------

def _apply_range(series: pd.Series,
                date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
                ) -> pd.Series:
    if date_range is None:
        return series
    start, end = date_range
    return series.loc[start:end]

def _apply_moving_average(series: pd.Series, ma: Optional[int] = None) -> pd.Series:
    if ma is None:
        return series
    return series.rolling(ma, min_periods=max(ma//2, 1), center=True).mean()

def _apply_groupby(series: pd.Series, col: Optional[str] = None) -> pd.Series:
    if col is None:
        return series
    if col in ['year', 'month', 'day', 'hour', 'minute', 'dayofyear']:
        return series.groupby(getattr(series.index, col)).mean()
    if col == 'timeofday':
        timeofday = series.index.hour + series.index.minute/60
        return series.groupby(timeofday).mean()
    if col == 'dayofweek':
        dayofweek = series.index.dayofweek + \
            (series.index.hour + series.index.minute/60) / 24
        return series.groupby(dayofweek).mean()
    if col == 'dateofyear':
        dateofyear = series.index.map(lambda d: pd.Timestamp(
            year=2000, month=d.month, day=d.day))
        return series.groupby(dateofyear).mean()
    raise ValueError(f"Invalid column: {col}.")



# --------------------------------------------------------
# Individual plots
# --------------------------------------------------------

_color_baseline:Dict[str, str] = {'LR': 'seagreen', 'RF': 'chartreuse', 'GB': 'lime'}
_color_meta:    Dict[str, str] = {'LR': 'blue', 'NN': 'deepskyblue'}
_display_name_baseline:Dict[str, str] = \
        {'LR': 'lin. reg.', 'RF': 'random forest', 'GB': 'gradient boost'}

def curves( true_series        : pd.Series,
            dict_pred_series   : Dict[str, pd.Series],
            dict_baseline_series:Dict[str, pd.Series],
            dict_meta_series   : Dict[str, pd.Series],
            xlabel: str = "date", ylabel: str = "consumption [GW]", title=None,
            ylim: [float, float] or None = None,
            date_range:     Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
            moving_average: Optional[int] = None,
            groupby:        Optional[str] = None) -> None:

    # preparation: range and/or SME
    _true_series    = _apply_range(_apply_groupby(_apply_moving_average(
        true_series,    moving_average), groupby), date_range)

    _dict_baseline_series = {}
    for name, series in dict_baseline_series.items():
        s = _apply_moving_average(series, moving_average)
        s = _apply_range  (s, date_range)
        s = _apply_groupby(s, groupby)
        _dict_baseline_series[name] = s

    _dict_meta_series = {}
    for name, series in dict_meta_series.items():
        s = _apply_moving_average(series, moving_average)
        s = _apply_range  (s, date_range)
        s = _apply_groupby(s, groupby)
        _dict_meta_series[name] = s

    _dict_pred_series = {}
    for name, series in dict_pred_series.items():
        s = _apply_moving_average(series, moving_average)
        s = _apply_range  (s, date_range)
        s = _apply_groupby(s, groupby)
        _dict_pred_series[name] = s


    plt.figure(figsize=(10,6))

    if _true_series is not None:
        plt.plot(_true_series.index, _true_series.values, label="actual", color="black")

    for quantile, series in _dict_pred_series.items():
        plt.plot(series.index, series.values,
                 color="red",  alpha=0.7,  label=f"NN ({quantile})")

    for name, series in _dict_baseline_series.items():
        plt.plot(series.index, series.values,
                 color=_color_baseline[name],  alpha=0.7,  label=name)

    for name, series in _dict_meta_series.items():
        plt.plot(series.index, series.values,
                 color=_color_meta[name],  alpha=0.7,  label=f"meta {name}")

    if ylim   is not None:  plt.ylim  (ylim)
    if xlabel is not None:  plt.xlabel(xlabel)
    if ylabel is not None:  plt.ylabel(ylabel)
    if title  is not None:  plt.title (title)

    if groupby == 'dateofyear':
        # Format x-axis to show only day and month (e.g., "01 Jan")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.legend()
    plt.show()



def scatter(
         true_series        : pd.Series,
         dict_pred_series   : Dict[str, pd.Series],
         dict_baseline_series:Dict[str, pd.Series],
         dict_meta_series   : Dict[str, pd.Series],
         x_axis_series      : pd.Series,
         resample:       Optional[str] = None,
         xlabel: str = "date", ylabel: str = "consumption [GW]", title=None,
         xlim: [float, float] or None = None, ylim: [float, float] or None = None,
         alpha: Optional[float] = 1.) -> None:

    # resample
    if resample is not None:
        if true_series is not None:
            true_series = true_series.resample(resample).mean()

        dict_baseline_series = {
            name: s.resample(resample).mean()
            for name, s in dict_baseline_series.items()
        }

        dict_meta_series = {
            name: s.resample(resample).mean()
            for name, s in dict_meta_series.items()
        }

        dict_pred_series = {
            name: s.resample(resample).mean()
            for name, s in dict_pred_series.items()
        }

        x_axis_series = x_axis_series.resample(resample).mean()

    # find common index
    common_idx = x_axis_series.index

    if true_series is not None:
        common_idx = common_idx.intersection(true_series.index)
    for s in dict_baseline_series.values():
        common_idx = common_idx.intersection(s.index)
    for s in dict_pred_series.values():
        common_idx = common_idx.intersection(s.index)
    for s in dict_meta_series.values():
        common_idx = common_idx.intersection(s.index)


    # --- align everything ---
    x_axis_series = x_axis_series.loc[common_idx]


    # plotting
    plt.figure(figsize=(10,6))

    if true_series is not None:
        plt.scatter(x_axis_series, true_series.loc[common_idx].values,
                    label="actual", color="black", alpha=alpha)

    for quantile, series in dict_pred_series.items():
        plt.scatter(x_axis_series, series.loc[common_idx].values,
                 color="red",              alpha=alpha, label=f"NN ({quantile})")

    for name, series in dict_baseline_series.items():
        plt.scatter(x_axis_series, series.loc[common_idx].values,
                 color=_color_baseline[name],alpha=0.7,  label=name)

    for name, series in dict_meta_series.items():
        plt.scatter(x_axis_series, series.loc[common_idx].values,
                 color=_color_meta[name],    alpha=0.7,  label=f"meta {name}")

    if xlim   is not None:  plt.xlim  (xlim)
    if ylim   is not None:  plt.ylim  (ylim)
    if xlabel is not None:  plt.xlabel(xlabel)
    if ylabel is not None:  plt.ylabel(ylabel)
    if title  is not None:  plt.title (title)

    plt.legend()
    plt.show()



# --------------------------------------------------------
# Plotting many plots
# --------------------------------------------------------

def diagnostics(name:                str,
                true_series:         pd.Series,
                dict_pred_series:    Dict[str, pd.Series],
                dict_baseline_series:Dict[str, pd.Series],
                dict_meta_series:    Dict[str, pd.Series],
                names_baseline:      Optional[Sequence[str]],
                names_meta:          Optional[Sequence[str]],
                Tavg:                Optional[pd.Series],
                num_steps_per_day:   int) -> None:
    # print("latest date:")
    # print("true:", true_series.index[-1])
    # print("pred:", max([s.index[-1] for s in dict_pred_series]))
    # print("regr:", lr_series.index[-1])

    # test(true_series, dict_pred_series, lr_series, future_series, name_baseline)

    dict_baseline_series= {_name: dict_baseline_series.get(_name)
            for _name in names_baseline if _name in dict_baseline_series.keys()}
    dict_meta_series    = {_name: dict_meta_series.get(_name)
            for _name in names_meta     if _name in dict_meta_series    .keys()}

    # SMA_consumption = [None, num_steps_per_day]
    # SMA_residual    = [2*2,  num_steps_per_day]
    # days_zoom:           int | Tuple[int] = [8, 61]

    ylim: Tuple[Tuple[float, float], Tuple[float, float]] = [[40, 60], [-1.5, 2.5]]

    # for _idx, _zoom in enumerate(days_zoom):
    #     # zoom
    #     zoom_horizon= pd.Timedelta(days=_zoom)
    #     zoom_end    = true_series.index[-1]
    #     zoom_start  = zoom_end - zoom_horizon


    #     # plot consumption
    #     _SMA    = SMA_consumption[_idx]
    #     _SMA_str = f" (SMA {_SMA/2} hrs)" if _SMA is not None else ""
    #     test(true_series, dict_pred_series, baseline_series, meta_series,
    #          _name_baseline, ylabel=f"consumption{_SMA_str} [GW]",
    #          ylim=ylim[0], date_range=[zoom_start, zoom_end], moving_average=_SMA)

    #     # compute and plot residuals
    #     _SMA    = SMA_residual[_idx]
    #     _SMA_str = f" (SMA {_SMA/2} hrs)" if _SMA is not None else ""

    residual_true_series     = true_series    - true_series
    dict_residual_baseline_series= {_name: dict_baseline_series[_name] - true_series
                                 for _name in dict_baseline_series.keys()}
    dict_residual_meta_series    = {_name: dict_meta_series    [_name] - true_series
                                 for _name in dict_meta_series.keys()}

    dict_residual_pred_series= {_name: _series - true_series\
        for _name, _series in dict_pred_series.items()}

    _ylabel_consumption= "consumption [GW]"
    _ylabel_residual   = "consumption difference [GW]"
    _title             = name


    # plot seasonal consumption by date, SMA 1 week
    curves(true_series, dict_pred_series, dict_baseline_series, dict_meta_series,
         xlabel="date of year", ylabel=_ylabel_consumption, title=_title,
         ylim=[ylim[0][0], ylim[0][1]+15], date_range=None,
         moving_average=num_steps_per_day*7, groupby='dateofyear')
    curves(residual_true_series, dict_residual_pred_series,
         dict_residual_baseline_series, dict_residual_meta_series,
         xlabel="date of year", ylabel=_ylabel_residual, title=_title,
         ylim=ylim[1], date_range=None, # [zoom_start, zoom_end]
         moving_average=num_steps_per_day*7, groupby='dateofyear')


    # plot consumption by hour, over a day
    curves(true_series, dict_pred_series, dict_baseline_series, dict_meta_series,
         xlabel="time of day", ylabel=_ylabel_consumption, title=_title,
         ylim=ylim[0], date_range=None, groupby='timeofday')
    curves(residual_true_series, dict_residual_pred_series,
         dict_residual_baseline_series, dict_residual_meta_series,
         xlabel="time of day", ylabel=_ylabel_residual, title=_title,
         ylim=ylim[1], date_range=None, groupby='timeofday')


    # plot consumption over a week
    curves(true_series, dict_pred_series, dict_baseline_series, dict_meta_series,
         xlabel="day of week (0 is Monday)", ylabel=_ylabel_consumption, title=_title,
         ylim=ylim[0], date_range=None, groupby='dayofweek',
         moving_average=num_steps_per_day//2)
    curves(residual_true_series, dict_residual_pred_series,
         dict_residual_baseline_series, dict_residual_meta_series,
         xlabel="day of week (0 is Monday)", ylabel=_ylabel_residual, title=_title,
         ylim=ylim[1], date_range=None, groupby='dayofweek',
         moving_average=num_steps_per_day//2)  # smoothing a little


    # as a function of temperature
    if Tavg is not None:
        Tavg_winter = Tavg[(Tavg.index.month <= 5) | (Tavg.index.month >= 9)]
        scatter(true_series, dict_pred_series, dict_baseline_series, dict_meta_series,
            Tavg_winter,
            ylim=[ylim[0][0]-5, ylim[0][1]+22],
            xlim=[-5, 20], resample='D', alpha=0.3,
            xlabel="(winter) average temperature [°C]",
            ylabel=_ylabel_consumption, title=_title)
        scatter(residual_true_series, dict_residual_pred_series,
                     dict_residual_baseline_series, dict_residual_meta_series,
            Tavg_winter,
            xlim=[-5, 20], ylim=[-6,10], resample='D', alpha=0.3,
            xlabel="(winter) average temperature [°C]",
            ylabel=_ylabel_residual,  title=_title)

        Tavg_summer = Tavg[(Tavg.index.month >= 5) & (Tavg.index.month <= 9)]
        scatter(true_series, dict_pred_series, dict_baseline_series, dict_meta_series,
            Tavg_summer,
            ylim=[ylim[0][0]-10, ylim[0][1]-5],
            xlim=[15, 30], resample='D', alpha=0.3,
            xlabel="(summer) average temperature [°C]",
            ylabel=_ylabel_consumption,  title=_title)
        scatter(residual_true_series, dict_residual_pred_series,
                     dict_residual_baseline_series, dict_residual_meta_series,
            Tavg_summer,
            xlim=[15, 30], ylim=[-4, 6], resample='D', alpha=0.3,
            xlabel="(summer) average temperature [°C]",
            ylabel=_ylabel_residual,  title=_title)




def quantiles(
    true_series: pd.Series,
    pred_quantiles: Dict[str, pd.Series],
    *,
    q_low:  str = "q10",
    q_med:  str = "q50",
    q_high: str = "q90",
    baseline_series: Optional[Dict[str, pd.Series]] = None,
    title:  str = "Forecast with uncertainty",
    ylabel: str = "consumption [GW]",
    dates: Optional[pd.DatetimeIndex] = None,
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
        Optional dict of baseline Series (e.g. {'RF': Series}).
    """

    # ------------------------
    # Align indices safely
    # ------------------------
    idx = true_series.index
    idx = idx.intersection(pred_quantiles[q_med].index)

    if dates is not None:
        idx = idx.intersection(dates)
    if q_low in pred_quantiles:
        idx = idx.intersection(pred_quantiles[q_low].index)
    if q_high in pred_quantiles:
        idx = idx.intersection(pred_quantiles[q_high].index)

    true = true_series.loc[idx]
    qmed = pred_quantiles[q_med] .loc[idx]

    qlo  = pred_quantiles[q_low] .loc[idx] if q_low  in pred_quantiles else None
    qhi  = pred_quantiles[q_high].loc[idx] if q_high in pred_quantiles else None

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

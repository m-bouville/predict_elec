"""
Tests for the plotting code (``plots``, ``plot_statistics``, ``plot_optuna``).

* B8: every curve of a figure (actual, NNTQ quantiles, baselines, metamodels)
  goes through exactly the same MA -> range -> groupby pipeline (the actual
  series used to be processed as MA -> groupby -> range);
* calendar groupings (time of day, day of week) in French local time;
* robustness: single-quantile curves, Series input, legends, default title;
* ``plot_statistics``: ``drift_with_time`` runs, a 0 degC threshold is a threshold;
* ``plot_optuna``: parameter-importance table.

``plots`` only needs matplotlib/numpy/pandas; the other tests need torch (via
``constants``) and, for plot_optuna, optuna.
"""
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")            # headless: no display, no files
import matplotlib.pyplot as plt

import plots

# plots call plt.show(), a no-op (with a warning) under the Agg backend
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# B8: one shared pipeline for all curves
# ---------------------------------------------------------------------------
@pytest.fixture
def hourly_series():
    # Two years of hourly data spanning several calendars so that grouping by
    # day-of-year and slicing by date interact non-trivially.
    idx = pd.date_range("2021-01-01", "2022-12-31 23:00", freq="h", tz="UTC")
    # a smooth seasonal signal + noise, so means are well defined
    doy = idx.dayofyear.to_numpy()
    values = 50 + 10 * np.sin(2 * np.pi * doy / 365.0) + np.arange(len(idx)) * 1e-4
    return pd.Series(values, index=idx, name="consumption")


def test_prepare_series_matches_ma_then_range_then_groupby(hourly_series):
    """_prepare_series is exactly MA -> range -> groupby, in that order."""
    ma = 24
    date_range = (pd.Timestamp("2021-03-01", tz="UTC"),
                  pd.Timestamp("2022-09-30", tz="UTC"))
    groupby = "dateofyear"

    got = plots._prepare_series(hourly_series, ma, date_range, groupby)

    expected = plots._apply_moving_average(hourly_series, ma)
    expected = plots._apply_range(expected, date_range)
    expected = plots._apply_groupby(expected, groupby)

    pd.testing.assert_series_equal(got, expected)


def test_all_curve_types_share_one_pipeline(hourly_series, monkeypatch):
    """
    Actual, NNTQ, baseline and metamodel curves must all be transformed
    identically.  We capture what ``curves`` hands to ``plt.plot`` and assert
    every captured y-array equals the shared-pipeline output for that series.
    """
    ma = 48
    date_range = (pd.Timestamp("2021-02-01", tz="UTC"),
                  pd.Timestamp("2022-11-30", tz="UTC"))
    groupby = "dateofyear"

    plotted = {}

    # Keep the real figure/axes (curves() calls xlabel/legend/...), but capture
    # the y-array handed to each plot() call and skip the actual drawing.
    def _record(x, y, *a, **k):
        plotted[k.get("label")] = np.asarray(y)
    monkeypatch.setattr(plots.plt, "plot", _record)
    monkeypatch.setattr(plots.plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plots.plt, "savefig", lambda *a, **k: None)

    true_series = hourly_series
    pred = {"q50": hourly_series * 1.01}
    baselines = {"LR": hourly_series * 0.98}
    metas = {"NN": hourly_series * 1.02}

    plots.curves(true_series, pred, baselines, metas,
                 date_range=date_range, moving_average=ma, groupby=groupby)

    def shared(s):
        return plots._prepare_series(s, ma, date_range, groupby).values

    # the q50 quantile is drawn by curves() with the label "NNTQ (median)"
    np.testing.assert_allclose(plotted["actual"], shared(true_series))
    np.testing.assert_allclose(plotted["NNTQ (median)"], shared(pred["q50"]))
    np.testing.assert_allclose(plotted["LR"], shared(baselines["LR"]))
    np.testing.assert_allclose(plotted["meta NN"], shared(metas["NN"]))


# ---------------------------------------------------------------------------
# calendar groupings in local time
# ---------------------------------------------------------------------------
def test_time_of_day_grouping_in_local_time():
    # a peak at 19:00 Paris time, in winter (18:00 UTC) and summer (17:00 UTC)
    idx = pd.date_range("2022-01-01", "2022-12-31 23:30", freq="30min", tz="UTC")
    local = idx.tz_convert("Europe/Paris")
    s = pd.Series(((local.hour == 19) & (local.minute == 0)).astype(float), index=idx)
    by_time = plots._apply_groupby(s, 'timeofday')
    assert by_time.idxmax() == 19.0 and by_time.max() == pytest.approx(1.0)
    by_day = plots._apply_groupby(s, 'dayofweek')
    assert np.allclose(np.sort(by_day[by_day > 0].index % 1), 19 / 24)


# ---------------------------------------------------------------------------
# robustness of the plotting functions
# ---------------------------------------------------------------------------
def test_loss_per_horizon_default_title():
    plt.figure()
    plots.loss_per_horizon({'a': np.ones(4)}, 30)
    assert "class" not in plt.gca().get_title()


def test_data_accepts_series_and_single_curve_has_no_legend():
    s = pd.Series(np.arange(5.), name="x")
    plots.data(s)                                        # used to IndexError
    assert plt.gca().get_legend() is None
    plots.data(pd.DataFrame({"a": np.arange(5.), "b": np.arange(5.)}))
    assert plt.gca().get_legend() is not None


def test_scatter_has_legend_with_two_clouds():
    idx = pd.date_range("2022-01-01", periods=20, freq="D", tz="UTC")
    true = pd.Series(np.arange(20.), index=idx)
    x = pd.Series(np.arange(20.) + 1, index=idx)
    plots.scatter(true, {"q50": true * 1.1}, None, None, x_axis_series=x)
    assert plt.gca().get_legend() is not None


def test_curves_with_a_single_non_median_quantile():
    idx = pd.date_range("2022-01-01", periods=48 * 10, freq="30min", tz="UTC")
    true = pd.Series(np.sin(np.arange(len(idx)) / 10), index=idx)
    plots.curves(true, {"q10": true - 1}, {}, {})       # used to KeyError 'q50'
    labels = [l.get_label() for l in plt.gca().get_lines()]
    assert "NNTQ (q10)" in labels


# ---------------------------------------------------------------------------
# plot_statistics
# ---------------------------------------------------------------------------
def test_drift_with_time_runs():
    pytest.importorskip("torch", reason="plot_statistics imports constants -> torch")
    import plot_statistics
    idx = pd.date_range("2016-01-01", "2020-12-31 23:30", freq="30min", tz="UTC")
    t = np.arange(len(idx))
    temp = pd.Series(12 + 8 * np.sin(2 * np.pi * t / (48 * 365)), index=idx,
                     name="Tavg_degC")
    cons = 55 - 1.5 * (temp - 12) + 0.001 * t / 48
    plot_statistics.drift_with_time(cons, temp, 48)     # used to UnboundLocalError


def test_threshold_zero_degC_is_a_threshold():
    pytest.importorskip("torch", reason="plot_statistics imports constants -> torch")
    import plot_statistics
    idx = pd.date_range("2022-01-01", periods=48 * 200, freq="30min", tz="UTC")
    T = pd.Series(np.linspace(-10, 10, len(idx)), index=idx)
    y = T ** 3                                           # slope 3T^2: ~0 at 0 degC
    out = plot_statistics.threshold_temp_sensitivity(
        y, {'q50': y}, {}, {}, T, idx, 'T_degC', [-1., 0., 1.], '==', 48, 0.5)
    slopes = out.iloc[:, 0]
    # at 0 degC the slope is local (small), as at +-1 degC, not the global fit
    assert abs(slopes.loc[0.]) < 5 and abs(slopes.loc[0.]) <= abs(slopes.loc[1.]) + 1


# ---------------------------------------------------------------------------
# plot_optuna: parameter importance
# ---------------------------------------------------------------------------
def test_plot_optuna_importance_table(capsys):
    pytest.importorskip("torch", reason="Bayes_search imports run -> torch")
    optuna = pytest.importorskip("optuna")
    import Bayes_search as bs
    from constants import Stage
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        trial.suggest_int("LR_max_iter", 2000, 2000)
        if x > .5:                                       # conditional parameter
            trial.suggest_float("cond", 0, 1)
        return x
    study.optimize(objective, n_trials=40)
    bs.plot_optuna(study, Stage.meta, list_parameters_hist=["x"],
                   num_best_runs_params=5, num_best_runs_hist=10)
    out = capsys.readouterr().out
    table = out.split("Parameter Importance")[1].split("shape:")[0]
    names = [l.split()[0] for l in table.strip().splitlines()[2:]]
    assert "median" not in names and "LR_max_iter" not in names
    assert "cond" in names
    cond_line = [l for l in table.splitlines() if l.startswith("cond")][0]
    assert "NaN" not in cond_line.split()[1]


# ---------------------------------------------------------------------------
# date_of_year: vectorized, same result as the former per-row map
# ---------------------------------------------------------------------------
def test_date_of_year_matches_per_row_map():
    idx = pd.date_range("2015-12-30", "2017-03-02", freq="30min", tz="UTC")
    ref = idx.map(lambda d: pd.Timestamp(year=2000, month=d.month, day=d.day))
    # pandas < 3 keeps the tz of `idx` through map (2000-mm-dd 00:00 UTC), pandas 3
    #   does not: compare the dates themselves
    if ref.tz is not None:
        ref = ref.tz_localize(None)
    got = plots.date_of_year(idx)
    assert (got.as_unit('ns') == ref.as_unit('ns')).all()
    assert pd.Timestamp("2000-02-29") in got                 # leap day kept
    assert got.tz is None


def test_date_of_year_is_fast():
    import time
    idx = pd.date_range("2012-01-01", "2026-01-01", freq="30min", tz="UTC")
    t0 = time.perf_counter()
    plots.date_of_year(idx)
    assert time.perf_counter() - t0 < 0.5      # the per-row map took ~1.4 s


# ---------------------------------------------------------------------------
# to_local_time: rows and values converted together
# ---------------------------------------------------------------------------
def test_to_local_time_keeps_values_with_their_timestamps():
    idx = pd.date_range("2024-03-30 22:00", periods=8, freq="h", tz="UTC")
    s = pd.Series(np.arange(8.), index=idx)
    shuffled = s.iloc[[3, 0, 7, 1, 6, 2, 5, 4]]              # unsorted input
    local = plots.to_local_time(shuffled)
    assert str(local.index.tz) == "Europe/Paris" and local.index.is_monotonic_increasing
    # each value still at its instant (the former index-only sort broke this)
    pd.testing.assert_series_equal(local.tz_convert("UTC"), s, check_freq=False)
    # the DST jump (31/03 02:00 -> 03:00) creates no duplicate
    assert local.index.is_unique and local.index[4].hour == 4

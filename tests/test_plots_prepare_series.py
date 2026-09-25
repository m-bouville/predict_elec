"""
Regression tests for the shared curve-preparation pipeline in ``plots``.

Fix B8: every curve of a figure (actual, NNTQ quantiles, baselines, metamodels)
must be processed through exactly the same transform order.  The actual series
used to be processed as MA -> groupby -> range while the predictions used
MA -> range -> groupby; the two orders disagree whenever ``date_range`` and
``groupby`` are combined, so the "actual" line could no longer be compared to
the predictions on the same figure.

``plots`` only depends on matplotlib/numpy/pandas, so these tests run
everywhere.
"""
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")            # headless: no display, no files

import plots


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

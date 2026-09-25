"""
Tests for two fixes that live in torch-importing modules:

* fix 6 -- ``containers.DataSplit.__post_init__`` now actually runs its length
  and column-count checks (they were dead because ``dict_preds_ML`` was treated
  as an array and ``X_columns`` was never set).
* fix 1 -- ``run.load_and_create_df`` must exclude ``price_euro_per_MWh`` from
  the model feature set (it is kept for statistics, plotted earlier) and must
  NOT let the price column's NaNs clip the training date range.

Both modules import torch, so the file skips without it.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="containers/run pull in torch")

import containers
from constants import Split


# ---------------------------------------------------------------------------
# fix 6: DataSplit consistency checks
# ---------------------------------------------------------------------------
def _make_split(n=10, f=4, cols=None, preds=None):
    return containers.DataSplit(
        Split.train, "training",
        np.arange(n),
        np.zeros((n, f)),                       # X
        np.zeros(n),                            # y_nation
        np.zeros((n, 1)),                       # Y_regions
        pd.date_range("2022-01-01", periods=n, freq="D"),  # dates
        np.zeros(n),                            # Tavg_degC
        X_columns=cols,
        dict_preds_ML=preds,
    )


def test_matching_lengths_and_columns_pass():
    cols = [f"f{i}" for i in range(4)]
    split = _make_split(n=10, f=4, cols=cols)
    assert split.X.shape == (10, 4)


def test_column_count_mismatch_is_caught():
    """X has 4 columns but only 3 names -> the (previously dead) assert fires."""
    with pytest.raises(AssertionError):
        _make_split(n=10, f=4, cols=["a", "b", "c"])


def test_dict_preds_ml_length_mismatch_is_caught():
    """
    dict_preds_ML is a dict of {model: {date: value}}; a wrong length must be
    detected.  Before fix 6 the code did ``dict_preds_ML.shape`` and crashed
    with AttributeError instead of validating.
    """
    bad_preds = {"LR": {d: 0.0 for d in range(3)}}   # length 3 != n=10
    with pytest.raises(AssertionError):
        _make_split(n=10, f=4, cols=[f"f{i}" for i in range(4)], preds=bad_preds)


# ---------------------------------------------------------------------------
# fix 1: price excluded from features, not used to clip the range
# ---------------------------------------------------------------------------
def _synthetic_features_frame():
    """A tiny feature frame as ``utils.df_features`` would return it."""
    idx = pd.date_range("2022-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["consumption_GW"] = np.arange(6, dtype=float)
    df["consumption_NE_GW"] = np.arange(6, dtype=float) * 0.5   # a region
    df["consumption_SMA_1wk_GW"] = 1.0                          # an SMA feature
    df["Tavg_degC"] = 10.0
    df["is_holiday"] = 0
    df["sin_24h"] = 0.3
    df["year"] = 2022
    df["month"] = 1
    df["timeofday"] = 0.0
    df["price_euro_per_MWh"] = 50.0
    df.loc[idx[2], "price_euro_per_MWh"] = np.nan   # price gap in the MIDDLE
    return df


def test_price_excluded_and_does_not_clip_range(monkeypatch):
    import run

    df = _synthetic_features_frame()
    dates_df = pd.DataFrame(columns=["start", "end"])
    monkeypatch.setattr(run.utils, "df_features",
                        lambda *a, **k: (df.copy(), dates_df, [1.0]))

    (out_df, cols, dates, Tavg_full, holidays_full,
     weights, _dates_df) = run.load_and_create_df(
        {"dummy": "x"}, "cache", pred_length=1, num_steps_per_day=48,
        minutes_per_step=30, verbose=0)

    # price is not a model feature and not mistaken for a region
    assert "price_euro_per_MWh" not in cols["features"]
    assert "price_euro_per_MWh" not in cols["Y_regions"]
    # the consumption SMA is retained as a feature
    assert "consumption_SMA_1wk_GW" in cols["features"]
    # the region column is picked up
    assert cols["Y_regions"] == ["consumption_NE_GW"]
    # calendar bookkeeping columns are excluded
    for c in ("year", "month", "timeofday"):
        assert c not in cols["features"]

    # crucially: the row where ONLY the price is NaN must survive, because price
    # is no longer part of the dropna subset.  All 6 rows are kept.
    assert len(out_df) == 6
    assert len(dates) == 6


def test_feature_nan_still_drops_its_row(monkeypatch):
    """A NaN in a real *feature* must still drop that row (dropna still works)."""
    import run

    df = _synthetic_features_frame()
    df.loc[df.index[4], "Tavg_degC"] = np.nan     # real feature gap
    dates_df = pd.DataFrame(columns=["start", "end"])
    monkeypatch.setattr(run.utils, "df_features",
                        lambda *a, **k: (df.copy(), dates_df, [1.0]))

    out_df, cols, dates, *_ = run.load_and_create_df(
        {"dummy": "x"}, "cache", pred_length=1, num_steps_per_day=48,
        minutes_per_step=30, verbose=0)

    assert len(out_df) == 5        # the Tavg-NaN row is gone


# ---------------------------------------------------------------------------
# DatasetBundle: items() and [] give the same four splits
# ---------------------------------------------------------------------------
def test_dataset_bundle_items_and_getitem():
    import types
    splits = {s: types.SimpleNamespace(name=s.name) for s in
              (Split.train, Split.valid, Split.test, Split.complete)}
    bundle = containers.DatasetBundle.__new__(containers.DatasetBundle)
    bundle.train, bundle.valid = splits[Split.train], splits[Split.valid]
    bundle.test,  bundle.complete = splits[Split.test], splits[Split.complete]
    assert dict(bundle.items()) == splits
    for s, obj in splits.items():
        assert bundle[s] is obj
    with pytest.raises(KeyError):
        bundle["train"]

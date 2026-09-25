"""
Tests for the feature engineering in ``utils``.

* ``df_features_calendar`` -- fix 9: the ``cos`` line was dedented out of the
  daily loop, so only ``cos_24h`` survived; and the yearly terms were cos-only,
  which cannot tell spring from autumn.
* ``df_features_past_consumption`` -- no leak of future consumption.
  (Open item 3, SMA windows twice their named size: test_open_bugs.py.)

``utils`` imports ``IO`` -> ``architecture`` -> ``torch`` at import time, so the
whole module needs torch to import.  It also needs the ``holidays`` package.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="utils imports torch via IO/architecture")
pytest.importorskip("holidays", reason="calendar features need the holidays package")

import utils


@pytest.fixture
def one_year_halfhourly():
    return pd.date_range("2022-01-01", "2022-12-31 23:30", freq="30min", tz="UTC")


def test_all_daily_cosine_features_exist(one_year_halfhourly):
    """
    Fix 9 regression: every daily period must have BOTH a sin and a cos column.
    Before the fix only ``cos_24h`` was created (the line sat outside the loop).
    """
    df = utils.df_features_calendar(one_year_halfhourly)
    for h in (6, 8, 12, 24):
        assert f"sin_{h}h" in df.columns
        assert f"cos_{h}h" in df.columns, (
            f"cos_{h}h missing -- the cos line is outside the daily loop again")


def test_sin_cos_are_a_valid_pair(one_year_halfhourly):
    """sin^2 + cos^2 == 1 for each daily harmonic (they are the same angle)."""
    df = utils.df_features_calendar(one_year_halfhourly)
    for h in (6, 8, 12, 24):
        r2 = df[f"sin_{h}h"] ** 2 + df[f"cos_{h}h"] ** 2
        np.testing.assert_allclose(r2.to_numpy(), 1.0, atol=1e-9)


def test_yearly_features_break_spring_autumn_symmetry(one_year_halfhourly):
    """
    A cosine-only yearly feature is symmetric about 1 January, so a spring day
    and its autumn mirror map to the same value -- the model cannot separate
    e.g. April from September.  ``sin_12mo`` must break that symmetry.
    """
    df = utils.df_features_calendar(one_year_halfhourly)
    assert "sin_12mo" in df.columns

    # Pick two days symmetric around the winter solstice-ish reference: their
    # day-of-year cosines are (nearly) equal, their sines opposite in sign.
    apr = df.loc["2022-04-15 12:00"]
    sep = df.loc["2022-09-15 12:00"]

    # cosine terms are close (the symmetry that caused the ambiguity)
    assert abs(apr["cos_12mo"] - sep["cos_12mo"]) < 0.15
    # the fundamental sine cleanly separates the two halves of the year
    assert np.sign(apr["sin_12mo"]) != np.sign(sep["sin_12mo"])
    assert abs(apr["sin_12mo"] - sep["sin_12mo"]) > 0.5


def test_calendar_frame_is_aligned_and_finite(one_year_halfhourly):
    df = utils.df_features_calendar(one_year_halfhourly)
    assert len(df) == len(one_year_halfhourly)
    assert df.index.equals(one_year_halfhourly)
    assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()


# ---------------------------------------------------------------------------
# no leak of future consumption through the past-consumption features
# ---------------------------------------------------------------------------
def test_past_consumption_features_only_use_data_lag_steps_old():
    """With features_in_future, X reaches origin + pred_length - 1: the
    consumption features there must only use consumption up to origin - 1,
    i.e. a feature at t may depend on consumption at t - lag or earlier only."""
    num_steps_per_day, lag = 48, 72
    idx = pd.date_range("2022-01-01", periods=num_steps_per_day * 400,
                        freq="30min", tz="UTC")
    consumption = pd.Series(np.random.default_rng(0).normal(50, 5, len(idx)),
                            index=idx)
    k = num_steps_per_day * 380                      # perturbed step
    perturbed = consumption.copy()
    perturbed.iloc[k] += 1000.

    ref = utils.df_features_past_consumption(consumption, lag, num_steps_per_day)
    new = utils.df_features_past_consumption(perturbed,   lag, num_steps_per_day)
    # before k + lag: unchanged (no feature sees the future value)
    pd.testing.assert_frame_equal(ref.iloc[:k + lag], new.iloc[:k + lag])
    # from k + lag on, the value enters the moving averages
    assert (new.iloc[k + lag] - ref.iloc[k + lag]).abs().min() > 0.


def test_lag_is_the_prediction_length(monkeypatch):
    """run.load_and_create_df must pass pred_length as the lag, so that the
    future part of X (pred_length steps) never sees consumption after the
    origin."""
    import run
    captured = {}

    def fake_df_features(dict_fnames, cache_fname, lag, *a, **k):
        captured['lag'] = lag
        raise RuntimeError("stop")

    monkeypatch.setattr(run.utils, "df_features", fake_df_features)
    with pytest.raises(RuntimeError, match="stop"):
        run.load_and_create_df({}, "cache", pred_length=72, num_steps_per_day=48,
                               minutes_per_step=30)
    assert captured['lag'] == 72

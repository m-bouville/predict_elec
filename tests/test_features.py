"""
Tests for the feature engineering in ``utils``.

* ``df_features_calendar`` -- in Paris local time; fix 9: the ``cos`` line was dedented out of the
  daily loop, so only ``cos_24h`` survived; and the yearly terms were cos-only,
  which cannot tell spring from autumn.
* ``df_features_past_consumption`` -- no leak of future consumption; SMA
  windows of their named size (/!\ they were twice as long).

``utils`` imports ``IO`` -> ``architecture`` -> ``torch`` at import time, so the
whole module needs torch to import.  It also needs the ``holidays`` package.
"""
import warnings

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


# ---------------------------------------------------------------------------
# public holidays after 2026
# ---------------------------------------------------------------------------
def test_holidays_after_2026():
    pytest.importorskip("holidays")
    import utils
    dates = pd.date_range("2026-12-30", "2028-01-02", freq="30min", tz="UTC")
    df = utils.df_features_calendar(dates)
    for day in ("2027-01-01", "2027-07-14", "2027-12-25"):
        assert df.loc[day + " 12:00", "is_holiday"] == 1, day
    assert df.loc["2027-01-04 12:00", "is_holiday"] == 0


# ---------------------------------------------------------------------------
# school holidays: after the last date of the calendar, unknown (NaN), not 0
# ---------------------------------------------------------------------------
def _toy_calendar():
    return pd.DataFrame({
        "start_date": pd.to_datetime(["2026-07-03T22:00:00Z", "2026-10-16T22:00:00Z",
                                      "2027-07-02T22:00:00Z"]),
        "end_date":   pd.to_datetime(["2026-08-31T22:00:00Z", "2026-11-01T23:00:00Z",
                                      "2027-07-02T22:00:00Z"]),   # summer: start only
        "zones": ["Zone A"] * 3,
        "name":  ["summer", "all_saints", "summer"]})


def test_school_holidays_unknown_after_calendar(monkeypatch):
    import IO
    monkeypatch.setattr(IO, "school_holidays", _toy_calendar)
    dates = pd.date_range("2026-07-01", "2027-07-10", freq="30min", tz="UTC")
    with pytest.warns(UserWarning, match="school holidays unknown after"):
        out, _ = IO.make_school_holidays_indicator(dates)
    assert out.loc["2026-08-01 12:00", "holiday_summer"] == 1      # holiday
    assert out.loc["2026-09-15 12:00", "holiday_summer"] == 0      # known: none
    assert out.loc["2027-06-01 12:00"].eq(0).all()                 # still known
    assert out.loc["2027-07-05 12:00"].isna().all()                # unknown
    assert out.loc["2027-07-02 21:30"].notna().all()


def test_school_holidays_no_nan_within_calendar(monkeypatch):
    import IO
    monkeypatch.setattr(IO, "school_holidays", _toy_calendar)
    dates = pd.date_range("2026-07-01", "2027-06-30", freq="30min", tz="UTC")
    with warnings.catch_warnings():
        warnings.simplefilter("error")                             # no warning
        out, _ = IO.make_school_holidays_indicator(dates)
    assert out.notna().all().all()


def test_school_holidays_real_calendar():
    """With the project's csv (skipped when absent): known summer 2026,
    unknown after the last published date."""
    import os, IO
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, "data", "fr-en-calendrier-scolaire.csv")):
        pytest.skip("school calendar csv not found")
    cwd = os.getcwd()
    os.chdir(root)                                   # the loader uses data/...
    try:
        dates = pd.date_range("2026-06-01", "2027-12-31", freq="30min", tz="UTC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, (start, end) = IO.make_school_holidays_indicator(dates)
    finally:
        os.chdir(cwd)
    assert out.loc["2026-08-01 12:00", "holiday_summer"] == 3
    assert out.loc["2027-08-01 12:00"].isna().all()


# ---------------------------------------------------------------------------
# calendar features in Paris local time
# ---------------------------------------------------------------------------
def test_calendar_features_in_paris_time():
    """The same local time gives the same features in winter and in summer
    (/!\\ they were computed on the UTC dates: 1 h apart between CET and
    CEST); holidays cover the Paris day."""
    named = {"mon_0730_winter": "2022-01-10 06:30", "mon_0730_summer": "2022-07-11 05:30",
             "tue_0030_winter": "2022-01-10 23:30",
             "fri_1700_winter": "2022-01-14 16:00", "fri_1700_summer": "2022-07-15 15:00",
             "apr30_2330":      "2022-04-30 21:30", "may1_0000": "2022-04-30 22:00",
             "may1_2330":       "2022-05-01 21:30", "may2_0000": "2022-05-01 22:00"}
    stamps = pd.DatetimeIndex(sorted(named.values()), tz="UTC")   # sorted index
    df = utils.df_features_calendar(stamps)
    row = {k: df.loc[pd.Timestamp(v, tz="UTC")] for k, v in named.items()}

    for col in ['sin_24h', 'cos_24h', 'sin_12h', 'cos_6h', 'is_Monday',
                'is_morning_peak', 'is_evening_peak', 'is_evening']:
        assert row["mon_0730_winter"][col] == pytest.approx(row["mon_0730_summer"][col]), col
    assert row["mon_0730_winter"]['is_morning_peak'] == 1
    assert row["mon_0730_winter"]['is_Monday'] == 1
    assert row["mon_0730_winter"]['sin_24h'] == pytest.approx(np.sin(2*np.pi * 7.5/24))
    assert row["fri_1700_winter"]['is_weekend'] == row["fri_1700_summer"]['is_weekend'] == 1
    assert [row[k]['is_holiday'] for k in
            ("apr30_2330", "may1_0000", "may1_2330", "may2_0000")] == [0, 1, 1, 0]
    assert row["tue_0030_winter"]['is_evening'] == 1
    assert row["tue_0030_winter"]['is_Tuesday'] == 1


# ---------------------------------------------------------------------------
# is_holiday: vectorized, same result as np.isin on the (Paris) dates
# ---------------------------------------------------------------------------
def test_is_holiday_matches_isin_on_dates():
    import holidays
    dates = pd.date_range("2012-01-01", "2026-12-31 23:30", freq="30min", tz="UTC")
    df = utils.df_features_calendar(dates)
    days = set(holidays.France(years=range(2012, 2028)).keys())
    ref = np.isin(dates.tz_convert("Europe/Paris").date, list(days)).astype(np.int16)
    np.testing.assert_array_equal(df['is_holiday'].to_numpy(), ref)
    assert df['is_holiday'].dtype == np.int16


# ---------------------------------------------------------------------------
# IO.add_calendar_columns: the bookkeeping columns, in one place
# ---------------------------------------------------------------------------
def test_add_calendar_columns():
    import IO, plots
    idx = pd.date_range("2024-02-28 22:00", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame({"x": np.arange(6.)}, index=idx)
    out = IO.add_calendar_columns(df)
    assert out is df                                        # in place
    assert df['year'].tolist() == [2024] * 6
    assert df['month'].tolist() == [2] * 4 + [2] * 2        # 22:00..00:30 UTC
    assert df['dateofyear'].equals(pd.Series(plots.date_of_year(idx), index=idx,
                                             name='dateofyear'))
    assert df['timeofday'].tolist() == [22., 22.5, 23., 23.5, 0., 0.5]
    df2 = IO.add_calendar_columns(pd.DataFrame(index=idx), timeofday=False)
    assert list(df2.columns) == ['year', 'month', 'dateofyear']


# ---------------------------------------------------------------------------
# SMA windows of the named size
# ---------------------------------------------------------------------------
def test_sma_window_matches_its_name():
    """
    ``consumption_SMA_1wk_GW`` averages over exactly ONE week: one week after a
    lagged step from 0 to 1 the trailing mean is 1 (/!\\ the window was two
    weeks: ~0.5), and 2 days before it is 5/7.
    """
    pytest.importorskip("torch", reason="utils imports torch via IO/architecture")
    pytest.importorskip("holidays")
    import utils

    num_steps_per_day = 48
    lag = num_steps_per_day
    idx = pd.date_range("2022-01-01", periods=num_steps_per_day * 400,
                        freq="30min", tz="UTC")
    step_at = num_steps_per_day * 100
    values = np.where(np.arange(len(idx)) >= step_at, 1.0, 0.0)
    consumption = pd.Series(values, index=idx)

    df = utils.df_features_past_consumption(consumption, lag, num_steps_per_day)
    one_week_after = idx[step_at + lag + num_steps_per_day * 7 + 2]
    assert df["consumption_SMA_1wk_GW"].loc[one_week_after] == \
        pytest.approx(1.0, abs=0.05)
    five_days_after = idx[step_at + lag + num_steps_per_day * 5]
    assert df["consumption_SMA_1wk_GW"].loc[five_days_after] == \
        pytest.approx(5 / 7, abs=0.01)
    for weeks in (2, 4):
        after = idx[step_at + lag + num_steps_per_day * 7 * weeks + 2]
        assert df[f"consumption_SMA_{weeks}wk_GW"].loc[after] == pytest.approx(1.)


def test_sma_needs_80pc_of_its_window():
    """An SMA value needs 80% of its window: none before 269 of the 336
    half-hours of a week, none over a 2-day gap (71%), one over a 1-day gap."""
    pytest.importorskip("torch", reason="utils imports torch via IO/architecture")
    pytest.importorskip("holidays")
    import utils

    steps, lag = 48, 48
    idx = pd.date_range("2022-01-01", periods=steps * 60, freq="30min", tz="UTC")
    consumption = pd.Series(1., index=idx)
    sma = utils.df_features_past_consumption(consumption, lag, steps)["consumption_SMA_1wk_GW"]
    assert idx.get_loc(sma.first_valid_index()) == lag + int(round(steps * 7 * .8)) - 1

    for gap_days, valid in [(2, False), (1, True)]:
        c = consumption.copy()
        start = steps * 30
        c.iloc[start: start + steps * gap_days] = np.nan
        sma = utils.df_features_past_consumption(c, lag, steps)["consumption_SMA_1wk_GW"]
        after_gap = idx[start + steps * gap_days + lag + steps]   # window holds the gap
        assert bool(np.isfinite(sma.loc[after_gap])) is valid, gap_days

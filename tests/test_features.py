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
    """With the project's csv files (skipped when either is absent): known
    summer 2026, unknown (NaN) from the last published date on, known before."""
    import os, IO
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fname in ("fr-en-calendrier-scolaire.csv", "vacances_scolaires_2015_2017.csv"):
        if not os.path.exists(os.path.join(root, "data", fname)):
            pytest.skip(f"school calendar csv not found: data/{fname}")
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
    # the returned end is the last published date (known_until: here the start
    #   of the next summer holidays, whose end is not published yet)
    assert dates[0] < end < dates[-1]
    assert out.loc[dates >= end].isna().all().all()
    assert out.loc[dates <  end].notna().all().all()


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
    idx = pd.date_range("2022-01-01", periods=num_steps_per_day * 470,
                        freq="30min", tz="UTC")     # 52 weeks after the step
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
    for weeks in (2, 4, 52):
        after = idx[step_at + lag + num_steps_per_day * 7 * weeks + 2]
        assert df[f"consumption_SMA_{weeks}wk_GW"].loc[after] == pytest.approx(1.)
        # one day before the window is past the step: 1/(7 * weeks) still at 0
        before = idx[step_at + lag + num_steps_per_day * (7 * weeks - 1)]
        assert df[f"consumption_SMA_{weeks}wk_GW"].loc[before] == \
            pytest.approx(1 - 1 / (7 * weeks), abs=2e-3), weeks


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


# ---------------------------------------------------------------------------
# school holidays: toy calendar in the real files' format, two zones
# ---------------------------------------------------------------------------
CALENDAR_HEADER = "description;population;start_date;end_date;location;zones;annee_scolaire"
CALENDAR_ROWS = [   # dates as served: UTC stamps of the Paris midnights
    "Vacances de la Toussaint;-;2025-10-17T22:00:00+00:00;2025-11-02T23:00:00+00:00;Lyon;Zone A;2025-2026",
    "Vacances de la Toussaint;-;2025-10-17T22:00:00+00:00;2025-11-02T23:00:00+00:00;Dijon;Zone A;2025-2026",
    "Vacances de la Toussaint;Élèves;2025-10-17T22:00:00+00:00;2025-11-02T23:00:00+00:00;Rennes;Zone B;2025-2026",
    "Vacances de la Toussaint;Enseignants;2025-10-16T22:00:00+00:00;2025-11-03T23:00:00+00:00;Lyon;Zone A;2025-2026",
    "Vacances d'Hiver;-;2026-02-06T23:00:00+00:00;2026-02-22T23:00:00+00:00;Lyon;Zone A;2025-2026",
    "Vacances d'Hiver;-;2026-02-13T23:00:00+00:00;2026-03-01T23:00:00+00:00;Rennes;Zone B;2025-2026",
    "Vacances de Noël;-;2025-12-19T23:00:00+00:00;2026-01-04T23:00:00+00:00;Corse;Corse;2025-2026",
    "Pont de l'Ascension;-;2026-05-13T22:00:00+00:00;2026-05-17T22:00:00+00:00;Lyon;Zone A;2025-2026",
    "Début des Vacances d'Été;-;2026-07-03T22:00:00+00:00;2026-07-03T22:00:00+00:00;Lyon;Zone A;2025-2026",
]
OLD_CALENDAR = "start_date,end_date,zones,description\n# ==== 2014-2015 ====\n" \
               "2014-10-18,2014-11-02,A,Toussaint\n"


@pytest.fixture
def toy_school_calendar(tmp_path, monkeypatch):
    """IO.school_holidays reading toy csv files written under tmp_path."""
    import IO
    fname1, fname2 = tmp_path / "calendrier.csv", tmp_path / "old.csv"
    fname1.write_text("﻿" + CALENDAR_HEADER + "\r\n" + "\r\n".join(CALENDAR_ROWS)
                      + "\r\n", encoding="utf-8")
    fname2.write_text(OLD_CALENDAR, encoding="utf-8")
    real = IO.school_holidays
    monkeypatch.setattr(IO, "school_holidays",
                        lambda: real(fname1=str(fname1), url1="-", fname2=str(fname2)))
    return IO


def _paris(stamp):
    return pd.Timestamp(stamp, tz="Europe/Paris").tz_convert("UTC")


def test_school_holidays_two_zones(toy_school_calendar):
    """Per holiday type, the number of zones on holiday (+1 per zone: A and B
    together count 2; a location listed twice counts once; teachers, 'pont'
    and non-metropolitan rows ignored). Holidays are [start, end) at Paris
    midnights: 0 at start - 30 min, on at start, still on at end - 30 min,
    off at end."""
    IO = toy_school_calendar
    dates = pd.date_range("2025-09-01", "2026-06-30 23:30", freq="30min", tz="UTC")
    out, _ = IO.make_school_holidays_indicator(dates)
    assert set(out.columns) == {"holiday_all_saints", "holiday_February",
                                "holiday_summer"}
    half = pd.Timedelta("30min")

    ts = out["holiday_all_saints"]
    start, end = _paris("2025-10-18 00:00"), _paris("2025-11-03 00:00")  # CEST, CET
    assert [ts[start - half], ts[start], ts[end - half], ts[end]] == [0, 2, 2, 0]

    feb = out["holiday_February"]
    for stamp, before, after in [("2026-02-07", 0, 1),     # A starts
                                 ("2026-02-14", 1, 2),     # B joins
                                 ("2026-02-23", 2, 1),     # A back to school
                                 ("2026-03-02", 1, 0)]:    # B back to school
        t = _paris(stamp + " 00:00")
        assert (feb[t - half], feb[t]) == (before, after), stamp

    assert out.loc[_paris("2026-05-15 12:00")].eq(0).all()          # 'pont'
    assert out.loc[_paris("2026-01-01 12:00")].eq(0).all()          # Corse only
    assert out["holiday_all_saints"].max() == 2                     # no double count


def test_school_holidays_unknown_after_known_until(toy_school_calendar):
    """From the last date of the calendar (known_until: here the start of the
    summer holidays, whose end is not published) on, every column is NaN."""
    IO = toy_school_calendar
    dates = pd.date_range("2026-06-01", "2026-07-10", freq="30min", tz="UTC")
    with pytest.warns(UserWarning, match="school holidays unknown after"):
        out, (first, last) = IO.make_school_holidays_indicator(dates)
    known_until = _paris("2026-07-04 00:00")
    assert last == known_until
    assert first == pd.Timestamp("2014-10-18", tz="UTC")           # the old file
    assert out.loc[dates >= known_until].isna().all().all()
    assert out.loc[dates <  known_until].eq(0).all().all()


# ---------------------------------------------------------------------------
# df_features: assembly of the feature frame
# ---------------------------------------------------------------------------
def test_df_features_assembly(monkeypatch):
    """Rows before the documented start (2014-09-15 UTC, return to class) keep
    their calendar features but have NaN input data (dropped later with the
    NaN rows); the bookkeeping columns are removed; no column name twice;
    dates_df gets the school-calendar row and dates."""
    import IO
    idx = pd.date_range("2014-09-10", "2014-09-25 23:30", freq="30min", tz="UTC")
    df = pd.DataFrame({"consumption_GW": 50. + np.sin(np.arange(len(idx)) / 10),
                       "Tavg_degC": 12.}, index=idx)
    IO.add_calendar_columns(df)
    dates_df = pd.DataFrame({"start": [idx[0]] * 2, "end": [idx[-1]] * 2},
                            index=["consumption", "temperature"])
    monkeypatch.setattr(IO, "load_data", lambda *a, **k: (df.copy(), dates_df.copy(),
                                                           {"NE": 1.}))
    monkeypatch.setattr(IO, "school_holidays", _toy_calendar)     # all known, 0

    out, dates_out, weights = utils.df_features({}, None, lag=48, num_steps_per_day=48,
                                                minutes_per_step=30)

    assert out.index.equals(idx)
    assert out.columns.is_unique
    assert not {"year", "month", "timeofday", "dateofyear"} & set(out.columns)
    first = pd.Timestamp("2014-09-15", tz="UTC")
    early = out.index < first
    assert out.loc[early, ["consumption_GW", "Tavg_degC"]].isna().all().all()
    assert out.loc[~early, ["consumption_GW", "Tavg_degC"]].notna().all().all()
    assert out.loc[early, "sin_24h"].notna().all()                  # calendar kept
    for col in ("holiday_summer", "is_holiday", "lockdown", "consumption_SMA_1wk_GW",
                "date"):
        assert col in out.columns, col
    assert "school_holidays" in dates_out.index
    assert dates_out.loc["consumption", "start"] == pd.Timestamp("2014-09-10").date()
    assert weights == {"NE": 1.}


# ---------------------------------------------------------------------------
# covid lockdown flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("start, end", [("2020-03-17", "2020-05-11"),
                                        ("2020-10-30", "2020-12-15"),
                                        ("2021-04-03", "2021-05-03")])
def test_lockdown_periods_in_paris_time(start, end):
    """lockdown = 1 from the Paris midnight starting `start` to the Paris
    midnight starting `end`, both included (label slice: the first half-hour
    of `end` is flagged), 0 outside."""
    dates = pd.date_range("2020-01-01", "2021-12-31 23:30", freq="30min", tz="UTC")
    flag = utils.df_features_calendar(dates)["lockdown"]
    half = pd.Timedelta("30min")
    t0, t1 = _paris(start + " 00:00"), _paris(end + " 00:00")
    assert (flag[t0 - half], flag[t0], flag[t1], flag[t1 + half]) == (0, 1, 1, 0)
    assert flag[t0:t1].eq(1).all()


def test_lockdown_total_duration():
    """Exactly the three periods: (55 + 46 + 30) Paris days, less the 2
    half-hours of the spring-forward night (2020-03-29, in the first period),
    plus the first half-hour of each end day."""
    dates = pd.date_range("2019-12-31", "2022-01-01", freq="30min", tz="UTC")
    flag = utils.df_features_calendar(dates)["lockdown"]
    assert int(flag.sum()) == (55 + 46 + 30) * 48 - 2 + 3

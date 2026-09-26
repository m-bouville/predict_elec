"""
Tests for ``IO.load_data`` beyond the time zones (those: test_io_timezone.py).

* statistics mode: eco2mix is parsed (and its figures drawn) once, whether the
  input data come from the pickle or not, and at verbose 3 (/!\ it was parsed
  and plotted a second time at the end of the statistics block, and parsed a
  third time for the verbose-3 checks);
* real-time consumption (quarter-hourly) -> half-hourly: the :00 / :30 value,
  as in the historical files (/!\ was the mean of :00 and :15);
* the data end at the end of the model inputs: neither price nor eco2mix
  (statistics only) truncate them, and the last temperature day is whole;
* ``_read_or_download``: a missing csv is downloaded as served, then read like
  a local one (/!\ the first run used to read the URL with other options, e.g.
  eco2mix without na_values='ND', and saved an extra index column); a failed
  download leaves no file; no loader reads a URL directly any more.

``IO`` imports ``architecture`` -> ``torch``: the file skips without torch.
"""
import os
import pickle
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="IO imports torch via architecture")

import IO


def _stub_module(names):
    """A module whose listed functions do nothing."""
    return types.SimpleNamespace(**{n: (lambda *a, **k: None) for n in names})


@pytest.mark.parametrize("from_cache", [True, False])
def test_statistics_parse_eco2mix_once(tmp_path, monkeypatch, from_cache):
    idx = pd.date_range("2023-01-01", "2025-12-31 23:30", freq="30min", tz="UTC")
    df_merged = pd.DataFrame({"consumption_GW": 50., "Tavg_degC": 10.,
                              "price_euro_per_MWh": 80.}, index=idx)
    IO.add_calendar_columns(df_merged)
    df_eco2mix = pd.DataFrame({"EnR_GW": 1., "net_charge_GW": 1.,
                               "Ech_physiques_GW": 1., "Taux_de_CO2_g/kWh": 1.},
                              index=idx)
    cache = tmp_path / "input.pkl"
    if from_cache:
        with open(cache, "wb") as f:
            pickle.dump((df_merged, df_eco2mix, pd.DataFrame(),
                         {"x": idx[0]}, {"x": idx[-1]}, {}), f)
    else:   # computed: a single input, the consumption (with the other columns)
        monkeypatch.setattr(IO, "load_weights", lambda **k: ({}, {}))
        monkeypatch.setattr(IO, "load_consumptions_recent", lambda: (None, None))
        monkeypatch.setattr(IO, "load_consumption",
                            lambda *a, **k: df_merged.copy())

    calls = []
    monkeypatch.setattr(IO, "load_eco2mix",
                        lambda **k: calls.append(k) or df_eco2mix)
    monkeypatch.setattr(IO, "load_temperature_world", lambda *a, **k: None)
    monkeypatch.setattr(IO.plots, "data", lambda *a, **k: None)
    monkeypatch.setattr(IO, "plot_statistics", _stub_module([
        "production_by_price", "thermosensitivity_per_temperature_by_season",
        "thermosensitivity_per_date_discrete", "thermosensitivity_peak_hour",
        "production_function_price"]))

    IO.load_data({} if from_cache else {"consumption": "unused.csv"},
                 str(cache), 48, 30, do_plot_statistics=True)

    assert len(calls) == 1 and calls[0]["do_plot_statistics"] is True
    assert cache.exists()


@pytest.mark.filterwarnings("ignore")
def test_statistics_verbose_3_parse_eco2mix_once(tmp_path, monkeypatch, capsys):
    """Every input computed (none from a pickle), verbose 3: eco2mix parsed
    once, and the verbose-3 check of its dates uses that frame."""
    idx = pd.date_range("2023-01-01", "2024-12-31 23:30", freq="30min", tz="UTC")
    conso = pd.DataFrame({"consumption_GW": 50.}, index=idx)
    IO.add_calendar_columns(conso)
    regions = pd.DataFrame({"NE": 30., "S": 20.}, index=idx)
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D", tz="UTC")
    temps = pd.DataFrame({"Tmin_degC": 5., "Tavg_degC": 10., "Tmax_degC": 15.},
                         index=days)
    hours = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    price = pd.DataFrame({"price_euro_per_MWh": 80.}, index=hours)
    df_eco2mix = pd.DataFrame({"EnR_GW": 1., "net_charge_GW": 1.,
                               "Ech_physiques_GW": 1., "Taux_de_CO2_g/kWh": 1.},
                              index=idx)

    monkeypatch.setattr(IO, "load_weights", lambda **k: ({}, {}))
    monkeypatch.setattr(IO, "load_consumptions_recent", lambda: (None, None))
    monkeypatch.setattr(IO, "load_consumption", lambda *a, **k: conso.copy())
    monkeypatch.setattr(IO, "load_consumption_by_region",
                        lambda *a, **k: (regions.copy(), None))
    monkeypatch.setattr(IO, "load_temperature",
                        lambda *a, **k: (temps.copy(), None, None, None))
    monkeypatch.setattr(IO, "load_price", lambda **k: price.copy())
    monkeypatch.setattr(IO, "load_nuclear", lambda *a, **k: price.copy())
    calls, analyzed = [], []
    monkeypatch.setattr(IO, "load_eco2mix",
                        lambda **k: calls.append(k) or df_eco2mix)
    real_analyze = IO.analyze_datetime
    monkeypatch.setattr(IO, "analyze_datetime", lambda df, **k:
                        analyzed.append(k.get("name")) or real_analyze(df, **k))
    monkeypatch.setattr(IO, "load_temperature_world", lambda *a, **k: None)
    monkeypatch.setattr(IO.plots, "data", lambda *a, **k: None)
    monkeypatch.setattr(IO, "plot_statistics", _stub_module([
        "prices_per_season", "production_by_price",
        "thermosensitivity_per_temperature_by_season",
        "thermosensitivity_per_date_discrete", "thermosensitivity_peak_hour",
        "production_function_price"]))

    IO.load_data({"consumption": "-", "consumption_by_region": "-",
                  "temperature": "-", "price": "-"},
                 str(tmp_path / "input.pkl"), 48, 30,
                 do_plot_statistics=True, verbose=3)

    assert len(calls) == 1 and calls[0]["do_plot_statistics"] is True
    assert "eco2mix" in analyzed


@pytest.mark.parametrize("temperature_last_day, expected_end", [
    ("2025-06-30", "2025-06-30 21:30"),   # whole last Paris day (CEST: 22:00 UTC)
    ("2025-07-31", "2025-07-15 23:30"),   # consumption ends first
    ("2025-03-30", "2025-03-30 21:30"),   # 23-h day: 23:30 CEST (/!\ was 00:30 of D+1)
    ("2025-01-31", "2025-01-31 22:30"),   # winter: 23:30 CET
])
def test_data_end_at_the_end_of_the_model_inputs(tmp_path, temperature_last_day,
                                                 expected_end):
    """(/!\\ the end was the earliest of ALL sources: a lagging price or eco2mix
    file truncated the data, and the last temperature day was cut to its
    first row)"""
    idx = pd.date_range("2025-01-01", "2025-08-31 23:30", freq="30min", tz="UTC")
    df_merged = pd.DataFrame({"consumption_GW": 50.}, index=idx)
    t_end = pd.Timestamp(temperature_last_day, tz="Europe/Paris").tz_convert("UTC")
    ends = {"consumption":           pd.Timestamp("2025-07-15 23:30", tz="UTC"),
            "consumption_by_region": pd.Timestamp("2025-08-31 23:30", tz="UTC"),
            "temperature":           t_end,          # daily: start of its last day
            "price":                 pd.Timestamp("2025-03-01 00:00", tz="UTC"),
            "eco2mix":               pd.Timestamp("2025-02-01 00:00", tz="UTC")}
    starts = {k: idx[0] for k in ends}
    cache = tmp_path / "input.pkl"
    with open(cache, "wb") as f:
        pickle.dump((df_merged, None, pd.DataFrame(), starts, ends, {}), f)

    out, _, _ = IO.load_data({}, str(cache), 48, 30, do_plot_statistics=False)

    assert out.index.max() == pd.Timestamp(expected_end, tz="UTC")


def test_data_end_on_the_fall_back_day(tmp_path):
    """Last temperature day = the 25-h day: its last half-hour is 23:30 CET
    (/!\ was + 24 h: 22:30 CET, an hour lost)."""
    idx = pd.date_range("2025-10-01", "2025-11-30 23:30", freq="30min", tz="UTC")
    t_end = pd.Timestamp("2025-10-26", tz="Europe/Paris").tz_convert("UTC")
    ends = {"consumption": idx[-1], "temperature": t_end}
    cache = tmp_path / "input.pkl"
    with open(cache, "wb") as f:
        pickle.dump((pd.DataFrame({"consumption_GW": 50.}, index=idx), None,
                     pd.DataFrame(), {k: idx[0] for k in ends}, ends, {}), f)
    out, _, _ = IO.load_data({}, str(cache), 48, 30, do_plot_statistics=False)
    assert out.index.max() == pd.Timestamp("2025-10-26 22:30", tz="UTC")


def test_daily_temperature_covers_every_half_hour_of_its_local_day(monkeypatch):
    """Each half-hour gets its Paris day's temperature, the 50 of the fall-back
    day and the 46 of the spring-forward day included (/!\ ffill(limit=48):
    the 50th half-hour of the fall-back day was NaN, its row then dropped)."""
    idx = pd.date_range("2025-03-25", "2025-11-05", freq="30min", tz="UTC")
    conso = pd.DataFrame({"consumption_GW": 50.}, index=idx)
    days = pd.date_range("2025-03-25", "2025-11-05", freq="D", tz="Europe/Paris")
    temps = pd.DataFrame({"Tavg_degC": np.arange(len(days), dtype=float)}, index=days)
    monkeypatch.setattr(IO, "load_weights", lambda **k: ({}, {}))
    monkeypatch.setattr(IO, "load_consumptions_recent", lambda: (None, None))
    monkeypatch.setattr(IO, "load_consumption", lambda *a, **k: conso.copy())
    monkeypatch.setattr(IO, "load_temperature",
                        lambda *a, **k: (temps.copy(), None, None, None))
    monkeypatch.setattr(IO, "load_eco2mix", lambda **k: conso[[]])

    out, _, _ = IO.load_data({"consumption": "-", "temperature": "-"}, None, 48, 30,
                             do_plot_statistics=False)

    local_day = out.index.tz_convert("Europe/Paris").normalize()
    expected  = temps["Tavg_degC"].reindex(local_day).to_numpy()
    np.testing.assert_array_equal(out["Tavg_degC"].to_numpy(), expected)
    for day, n in [("2025-03-30", 46), ("2025-10-26", 50)]:
        rows = out[local_day == pd.Timestamp(day, tz="Europe/Paris")]
        assert len(rows) == n and rows["Tavg_degC"].notna().all()


def test_real_time_consumption_takes_the_half_hour_value(tmp_path):
    """National (sample of the real file) and regional real-time data: the
    half-hourly value is the reading at :00 / :30, not the mean with :15."""
    stamps = pd.date_range("2026-07-01 00:00", periods=8, freq="15min", tz="UTC")
    rows = [f"{t.isoformat()};{r};{1000 * (k + 1) + 10 * i}"
            for i, t in enumerate(stamps)
            for k, r in enumerate(["Nouvelle-Aquitaine", "Bretagne", "Occitanie"])]
    regional = tmp_path / "regional.csv"
    regional.write_text("Date - Heure;Région;Consommation (MW)\n" + "\n".join(rows) + "\n",
                        encoding="utf-8")   # as RTE's file (Windows default: cp1252)

    nation, regions = IO.load_consumptions_recent(
        path_nation=os.path.join(DATA, "eco2mix-national-tr_sample.csv"),
        url_nation="-", path_region=str(regional), url_region="-")

    # sample: 46807 (00:00), 46358 (00:15), 45014 (00:30), ... MW, at +02:00
    assert nation.iloc[:2].tolist() == pytest.approx([46.807, 45.014])
    assert (nation.index.minute % 30 == 0).all()
    assert regions["Bretagne"].tolist() == pytest.approx([2.0, 2.02, 2.04, 2.06])
    assert regions["Occitanie"].iloc[0] == pytest.approx(3.0)

    # a missing :00 reading stays missing (/!\ .first() took the :15 one)
    rows = [r for r in rows if not (r.startswith(stamps[2].isoformat())
                                    and ";Bretagne;" in r)]
    regional.write_text("Date - Heure;Région;Consommation (MW)\n" + "\n".join(rows)
                        + "\n", encoding="utf-8")
    _, regions = IO.load_consumptions_recent(
        path_nation=os.path.join(DATA, "eco2mix-national-tr_sample.csv"),
        url_nation="-", path_region=str(regional), url_region="-")
    assert regions["Bretagne"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(regions["Bretagne"].iloc[1])        # 00:30 missing (not 00:45)


def test_real_time_national_missing_half_hour_is_not_the_quarter_after(tmp_path):
    """National file without its 00:30 row: no value from 00:45 in its place."""
    with open(os.path.join(DATA, "eco2mix-national-tr_sample.csv"), "rb") as f:
        lines = f.read().split(b"\r\n")
    path = tmp_path / "nation.csv"
    path.write_bytes(b"\r\n".join(l for l in lines if b"T00:30:00+02:00" not in l))
    regional = tmp_path / "regional.csv"
    regional.write_text("Date - Heure;Région;Consommation (MW)\n"
                        + "".join(f"2026-07-01T00:00:00+00:00;{r};1000\n" for r in
                                  ["Nouvelle-Aquitaine", "Bretagne", "Occitanie"]),
                        encoding="utf-8")

    nation, _ = IO.load_consumptions_recent(
        path_nation=str(path), url_nation="-", path_region=str(regional),
        url_region="-")

    assert nation.iloc[0] == pytest.approx(46.807)
    assert 43.599 not in nation.round(3).tolist()        # the 00:45 reading
    assert (nation.index.minute % 30 == 0).all()


# ---------------------------------------------------------------------------
# _read_or_download
# ---------------------------------------------------------------------------
CSV = "Date;Value\n2024-01-01;1\n2024-01-02;ND\n"


class _Response:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status = content, status
    def __enter__(self):  return self
    def __exit__(self, *a): return False
    def raise_for_status(self):
        if self.status >= 400:
            raise IO.requests.HTTPError(f"{self.status}")
    def iter_content(self, chunk_size):
        for i in range(0, len(self.content), 7):     # several chunks
            yield self.content[i:i + 7]


@pytest.fixture
def server(monkeypatch):
    """requests.get replaced: serves `server.files[url]`, counts the calls."""
    state = types.SimpleNamespace(files={}, calls=[], status=200)

    def _get(url, **kwargs):
        state.calls.append(url)
        return _Response(state.files.get(url, b""), state.status)
    monkeypatch.setattr(IO.requests, "get", _get)
    return state


def test_download_then_read_like_a_local_file(tmp_path, server):
    server.files["http://x/a.csv"] = CSV.encode()
    path = str(tmp_path / "sub" / "a.csv")          # directory created
    first  = IO._read_or_download(path, "http://x/a.csv", sep=';', na_values='ND')
    second = IO._read_or_download(path, "http://x/a.csv", sep=';', na_values='ND')
    assert server.calls == ["http://x/a.csv"]       # downloaded once
    assert open(path, "rb").read() == CSV.encode()  # saved as served
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["Date", "Value"] # no 'Unnamed: 0'
    assert first["Value"].isna().tolist() == [False, True]   # 'ND' -> NaN


def test_failed_download_leaves_no_file(tmp_path, server):
    server.status = 503
    path = tmp_path / "a.csv"
    with pytest.raises(IO.requests.HTTPError):
        IO._read_or_download(str(path), "http://x/a.csv", sep=';')
    assert list(tmp_path.iterdir()) == []           # neither a.csv nor a.csv.part


def test_empty_download_leaves_no_file(tmp_path, server):
    """HTTP 200 with no body: an error, and nothing saved (/!\ an empty a.csv
    was kept, and every later run failed on it)."""
    path = tmp_path / "a.csv"
    with pytest.raises(RuntimeError, match="empty"):
        IO._read_or_download(str(path), "http://x/a.csv", sep=';')
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_leaves_no_file(tmp_path, monkeypatch):
    """The connection drops after the first chunk: neither a.csv nor a.csv.part."""
    class _Broken(_Response):
        def iter_content(self, chunk_size):
            yield self.content[:7]
            raise IO.requests.ConnectionError("dropped")
    monkeypatch.setattr(IO.requests, "get", lambda url, **k: _Broken(CSV.encode()))
    path = tmp_path / "a.csv"
    with pytest.raises(IO.requests.ConnectionError):
        IO._read_or_download(str(path), "http://x/a.csv", sep=';')
    assert list(tmp_path.iterdir()) == []


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def test_eco2mix_first_download_reads_ND(tmp_path, server):
    """Samples of the real files (the 2012 rows have 'ND'): the first run
    (download) and the next ones (local files) give the same frame.
    (/!\\ the first run crashed: 'ND' strings without na_values)"""
    for name in ("cons-def", "tr"):
        with open(os.path.join(DATA, f"eco2mix-national-{name}_sample.csv"), "rb") as f:
            server.files[f"http://x/{name}"] = f.read()
    kwargs = dict(path_monthly=str(tmp_path / "def.csv"), url_monthly="http://x/cons-def",
                  path_recent =str(tmp_path / "tr.csv"),  url_recent ="http://x/tr")
    df_first  = IO.load_eco2mix(**kwargs)
    df_second = IO.load_eco2mix(**kwargs)   # from the local files
    assert len(server.calls) == 2
    pd.testing.assert_frame_equal(df_first, df_second)
    assert "Consommation_GW" in df_first.columns and df_first["Consommation_GW"].notna().any()


def test_no_loader_reads_a_url_directly():
    import pathlib, re
    src = (pathlib.Path(IO.__file__)).read_text(encoding="utf-8")
    # the only exception: the world temperature file is parsed from the URL and
    #   cached in another format (not a copy of the download)
    offenders = [m.start() for m in re.finditer(r"read_csv\(\s*url", src)]
    assert len(offenders) == 1
    assert src.rfind("def ", 0, offenders[0]) == src.find("def load_temperature_world")
    assert "requests.get(" not in src.replace("requests.get(url, stream=True", "")


def test_temperature_world_first_run_and_cache_agree(tmp_path, monkeypatch):
    """The Berkeley Earth file is parsed from the URL (whitespace-separated,
    '%' comments) and cached in its own format.
    (/!\\ delim_whitespace=True: TypeError on the first run with pandas 3)"""
    monkeypatch.setattr(IO.plots, "finish", lambda: None)
    lines = ["% Berkeley Earth, header", "%"] + [
        f"  {y} {m:2d}  {0.01*(y-1960):.3f} 0.1  NaN  NaN  NaN NaN NaN NaN NaN NaN"
        for y in range(1960, 1990) for m in range(1, 13)]
    source = tmp_path / "Complete_TAVG_complete.txt"
    source.write_text("\n".join(lines) + "\n")
    path = str(tmp_path / "world.csv")
    first  = IO.load_temperature_world(path=path, url=str(source))   # parsed
    second = IO.load_temperature_world(path=path, url="unused")      # cached csv
    assert len(first) == 30 * 12
    np.testing.assert_allclose(first["monthly_diff_K"].to_numpy(),
                               second["monthly_diff_K"].to_numpy())
    np.testing.assert_allclose(first.index.to_numpy(), second.index.to_numpy())


# ---------------------------------------------------------------------------
# load_consumption: history (monthly file) + recent real-time data
# ---------------------------------------------------------------------------
# Sample of the real history file (UTF-8 with BOM, CRLF, ';'): the Paris day
#   2025-10-26 (fall-back) as RTE serves it, i.e. 48 rows, the two CEST
#   half-hours 02:00 and 02:30 (00:00 and 00:30 UTC) absent; and 2025-03-30
#   00:00-04:30 local, where the non-existent 02:00 / 02:30 rows carry the same
#   UTC stamps as 03:00 / 03:30 (duplicates).
HISTORY = os.path.join(DATA, "consommation-quotidienne-brute_sample.csv")
FALL_BACK_DAY = pd.Timestamp("2025-10-26", tz="Europe/Paris")
FALL_BACK_GAP = pd.DatetimeIndex(["2025-10-26 00:00", "2025-10-26 00:30"], tz="UTC")


def _history_without_spring_rows(tmp_path):
    """The sample restricted to the fall-back day (no duplicate stamps)."""
    with open(HISTORY, "rb") as f:
        lines = f.read().split(b"\r\n")
    path = tmp_path / "history.csv"
    path.write_bytes(b"\r\n".join(l for l in lines if b";30/03/2025;" not in l))
    return str(path)


def _recent_series(start="2025-10-25 22:00", end="2025-10-27 06:00"):
    """As load_consumptions_recent returns it: genuine UTC half-hours, GW;
    values (100 + k/1000) that cannot be mistaken for the history (30-50 GW)."""
    idx = pd.date_range(start, end, freq="30min", tz="UTC", name="datetime_utc")
    return pd.Series(100. + np.arange(len(idx)) / 1000., index=idx,
                     name="consumption_GW")


def _rows_of_paris_day(df, day):
    return df[df.index.tz_convert("Europe/Paris").normalize() == day]


def test_consumption_history_alone_on_the_fall_back_day(tmp_path):
    """History alone: MW -> GW, the UTC stamps of the file; the fall-back Paris
    day has 48 rows, 00:00 and 00:30 UTC missing (the file has one row per
    local wall-clock half-hour: the first 02:00-03:00 CEST is not in it)."""
    df = IO.load_consumption(_history_without_spring_rows(tmp_path), url="-")
    day = _rows_of_paris_day(df, FALL_BACK_DAY)
    assert len(day) == 48 and day.index.is_unique
    assert not FALL_BACK_GAP.isin(df.index).any()
    assert df.loc["2025-10-26 22:30", "consumption_GW"] == pytest.approx(48.436)
    assert df.loc["2025-10-25 22:00", "consumption_GW"] == pytest.approx(46.384)
    assert {"year", "month", "dateofyear", "timeofday"} <= set(df.columns)


def test_consumption_history_then_recent(tmp_path):
    """History + real-time data: the history is kept where both exist, the
    real-time data fill its gaps (the two fall-back half-hours) and continue
    after its end. The fall-back Paris day then has its 50 UTC half-hours,
    each once, all known."""
    path   = _history_without_spring_rows(tmp_path)
    hist   = IO.load_consumption(path, url="-")["consumption_GW"]
    recent = _recent_series()
    df     = IO.load_consumption(path, url="-", df_recent=recent)

    assert df.index.is_unique and df.index.is_monotonic_increasing
    # overlap: history preferred
    pd.testing.assert_series_equal(df["consumption_GW"].reindex(hist.index), hist,
                                   check_names=False)
    # gaps of the history: real-time values
    np.testing.assert_allclose(df.loc[FALL_BACK_GAP, "consumption_GW"],
                               recent[FALL_BACK_GAP])
    # after the history ends: real-time values
    after = recent.index[recent.index > hist.index.max()]
    assert len(after) > 0
    np.testing.assert_allclose(df.loc[after, "consumption_GW"], recent[after])
    # the 25-hour day: 50 unique UTC half-hours, all known
    day = _rows_of_paris_day(df, FALL_BACK_DAY)
    assert len(day) == 50 and day.index.is_unique
    assert day["consumption_GW"].notna().all()
    assert (day.index[1:] - day.index[:-1] == pd.Timedelta("30min")).all()
    # calendar columns recomputed on the merged index
    assert df["timeofday"].isna().sum() == 0


def test_consumption_duplicates_kept_then_averaged_by_load_data(monkeypatch):
    """Duplicate stamps (the non-existent spring-forward local hour): kept as
    they are by load_consumption (without real-time data), then collapsed by
    load_data into their mean, with a warning, as its comment documents."""
    df = IO.load_consumption(HISTORY, url="-")
    dup = df.index[df.index.duplicated()]
    assert dup.equals(pd.DatetimeIndex(["2025-03-30 01:00", "2025-03-30 01:30"],
                                       tz="UTC", name="datetime_utc"))
    assert sorted(df.loc["2025-03-30 01:30", "consumption_GW"]) == \
        pytest.approx([46.395, 47.832])

    monkeypatch.setattr(IO, "load_weights", lambda **k: ({}, {}))
    monkeypatch.setattr(IO, "load_consumptions_recent", lambda: (None, None))
    monkeypatch.setattr(IO, "load_eco2mix", lambda **k: df[[]].iloc[:1])
    with pytest.warns(UserWarning, match="consumption has duplicates"):
        out, _, _ = IO.load_data({"consumption": HISTORY}, None, 48, 30,
                                 do_plot_statistics=False)
    assert out.index.is_unique
    assert out.loc["2025-03-30 01:30", "consumption_GW"] == \
        pytest.approx((46.395 + 47.832) / 2)
    assert out.loc["2025-03-30 01:00", "consumption_GW"] == pytest.approx(48.398)
    assert out.loc["2025-03-30 00:30", "consumption_GW"] == pytest.approx(47.832)


# ---------------------------------------------------------------------------
# load_weights and load_temperature
# ---------------------------------------------------------------------------
REGIONS = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne",
           "Centre-Val de Loire", "Grand Est", "Hauts-de-France", "Normandie",
           "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire",
           "Provence-Alpes-Côte d'Azur", "Île-de-France"]           # no Corse
CONSO_GWh = dict(zip(REGIONS, [60e3, 20e3, 22e3, 18e3, 42e3, 48e3, 26e3,
                               40e3, 38e3, 27e3, 40e3, 67e3]))


def _write_weights_csv(tmp_path):
    """Annual regional consumption, two years (the mean is used), Corse
    included (dropped by the loader)."""
    lines = ["Année;Code INSEE région;Région;Consommation brute électricité (GWh) - RTE"]
    for year, factor in [(2023, 0.9), (2024, 1.1)]:
        for k, (region, gwh) in enumerate(list(CONSO_GWh.items()) + [("Corse", 2e3)]):
            lines.append(f"{year};{k};{region};{gwh * factor:.1f}")
    path = tmp_path / "weights.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_load_weights_per_region_and_per_cluster(tmp_path):
    """Region weight = its mean annual consumption / the total without Corse
    (sum 1); cluster weight = the sum of its regions'; the clusters partition
    the 12 regions."""
    w_regions, w_clusters = IO.load_weights(path=_write_weights_csv(tmp_path))
    total = sum(CONSO_GWh.values())
    assert set(w_regions) == set(REGIONS)                   # Corse dropped
    for region, gwh in CONSO_GWh.items():
        assert w_regions[region] == pytest.approx(gwh / total, abs=1e-5)
    assert sum(w_regions.values()) == pytest.approx(1., abs=1e-4)

    assert set(w_clusters) == set(IO.CLUSTERS)
    in_clusters = [r for regions in IO.CLUSTERS.values() for r in regions]
    assert sorted(in_clusters) == sorted(REGIONS)           # each region once
    for cluster, regions in IO.CLUSTERS.items():
        assert w_clusters[cluster] == pytest.approx(
            sum(CONSO_GWh[r] for r in regions) / total, abs=1e-5)
        assert w_clusters[cluster] == pytest.approx(
            sum(w_regions[r] for r in regions), abs=1e-4)
    assert sum(w_clusters.values()) == pytest.approx(1., abs=1e-4)


DAYS = pd.date_range("2025-01-01", periods=14, freq="D")
BASE = dict(zip(REGIONS + ["Corse"], [2., 4., 9., 6., 1., 3., 7., 10., 12., 8.,
                                      14., 5., 40.]))          # Corse: outlier


def _tavg(region, day_idx):
    return BASE[region] + day_idx + (0.3 if day_idx % 2 else 0.)


def _write_temperature_csv(tmp_path, drop=()):
    """Daily regional temperatures in the real file's format; `drop`: (region
    or None for all regions, day index) rows left out."""
    lines = ["ID;Date;Code INSEE région;Région;TMin (°C);TMax (°C);TMoy (°C)"]
    for d, day in enumerate(DAYS):
        for k, region in enumerate(BASE):
            if (region, d) in drop or (None, d) in drop:
                continue
            t = _tavg(region, d)
            lines.append(f"{day:%Y-%m-%d}-{k};{day:%Y-%m-%d};{k};{region};"
                         f"{t - 3:.2f};{t + 4:.2f};{t:.2f}")
    path = tmp_path / "temperature.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _weights():
    total = sum(CONSO_GWh.values())
    return {r: v / total for r, v in CONSO_GWh.items()}


def _expected_mean(regions, d, weights=None):
    weights = weights or _weights()
    w = np.array([weights[r] for r in regions])
    return float(np.dot(w / w.sum(), [_tavg(r, d) for r in regions]))


def test_temperature_region_weighted_means(tmp_path):
    """National Tavg: the consumption-weighted mean of the 12 regions; cluster
    Tavg: the same with the weights renormalised within the cluster; Corsica
    (40 degC here) plays no part; Tmin / Tmax the same way."""
    out, Tavg_full, _, _ = IO.load_temperature(_write_temperature_csv(tmp_path),
                                               _weights())
    assert "corse" not in Tavg_full.columns and Tavg_full.shape[1] == 12
    assert out.index.tz is not None and str(out.index.tz) == "Europe/Paris"
    for d in range(len(DAYS)):
        row = out.iloc[d]
        assert row["Tavg_degC"] == pytest.approx(_expected_mean(REGIONS, d), abs=0.006)
        assert row["Tmin_degC"] == pytest.approx(_expected_mean(REGIONS, d) - 3, abs=0.006)
        for cluster, regions in IO.CLUSTERS.items():
            assert row[f"Tavg_{cluster}_degC"] == pytest.approx(
                _expected_mean(regions, d), abs=0.006), (cluster, d)
            assert row[f"T_spread_{cluster}_K"] == pytest.approx(7., abs=0.02)


def test_temperature_missing_region_gives_nan(tmp_path):
    """A region missing on a day: the national value and its cluster's are NaN
    (no mean over the others); the other clusters are unaffected."""
    out, _, _, _ = IO.load_temperature(
        _write_temperature_csv(tmp_path, drop={("Bretagne", 5)}), _weights())
    row = out.iloc[5]
    assert np.isnan(row["Tavg_degC"]) and np.isnan(row["Tavg_W_degC"])
    assert np.isnan(row["T_spread_W_K"])
    for cluster in ("NE", "IdF", "S"):
        assert row[f"Tavg_{cluster}_degC"] == pytest.approx(
            _expected_mean(IO.CLUSTERS[cluster], 5), abs=0.006)
    assert out["Tavg_degC"].drop(out.index[5]).notna().all()


def test_temperature_sma_windows_count_rows(tmp_path):
    """SMA_3days = rolling(3 rows, min_periods=2), SMA_10days = rolling(10
    rows, min_periods=8). They count ROWS: across a day absent from the file,
    SMA_3days averages 3 rows spread over 4 calendar days (open bug: see
    test_open_bugs_C.py); a NaN value inside the window is skipped."""
    out, _, _, _ = IO.load_temperature(
        _write_temperature_csv(tmp_path, drop={(None, 9), ("Bretagne", 5)}),
        _weights())
    assert len(out) == len(DAYS) - 1                        # day 9: no row
    ne, w = out["Tavg_NE_degC"], out["Tavg_W_degC"]
    sma3, sma10 = out["Tavg_NE_SMA_3days"], out["Tavg_NE_SMA_10days"]
    # min_periods
    assert np.isnan(sma3.iloc[0]) and sma3.iloc[1] == pytest.approx(ne.iloc[:2].mean())
    assert sma10.iloc[:7].isna().all() and sma10.iloc[7] == pytest.approx(ne.iloc[:8].mean())
    # across the missing day 9: rows of days 7, 8 and 10 (4 calendar days)
    day10 = pd.Timestamp(DAYS[10], tz="Europe/Paris")
    assert sma3.loc[day10] == pytest.approx(ne.loc[[pd.Timestamp(DAYS[d], tz="Europe/Paris")
                                                    for d in (7, 8, 10)]].mean())
    # NaN inside the window (W on day 5): mean of the other 2 rows
    assert out["Tavg_W_SMA_3days"].iloc[5] == pytest.approx(w.iloc[3:5].mean())
    assert out["Tavg_W_SMA_3days"].iloc[6] == pytest.approx(w.iloc[[4, 6]].mean())


def test_weights_then_temperature(tmp_path):
    """The weights of load_weights are keyed as load_temperature needs them."""
    w_regions, _ = IO.load_weights(path=_write_weights_csv(tmp_path))
    out, _, _, _ = IO.load_temperature(_write_temperature_csv(tmp_path), w_regions)
    assert out["Tavg_degC"].iloc[0] == pytest.approx(_expected_mean(REGIONS, 0),
                                                     abs=0.01)

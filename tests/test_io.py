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

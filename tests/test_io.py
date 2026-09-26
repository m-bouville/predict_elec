"""
Tests for ``IO.load_data`` beyond the time zones (those: test_io_timezone.py).

* statistics mode: eco2mix is parsed (and its figures drawn) once, whether the
  input data come from the pickle or not (/!\ it was parsed and plotted a
  second time at the end of the statistics block);
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

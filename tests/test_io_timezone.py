"""
Regression tests for ``IO.load_consumption_by_region`` (fix B5).

The regional CSV is Paris wall-clock time.  The old code localised it to a
constant ``+02:00`` (permanent summer offset), which shifted every WINTER
timestamp by one hour relative to the UTC national series and the features.
The fix localises to ``Europe/Paris`` and drops the two DST-transition
wall-clock hours per year (as ``NaT``) instead of mislabelling them.

``IO`` imports ``architecture`` -> ``torch`` at import time, so the file skips
without torch.
"""
import pandas as pd
import pytest

pytest.importorskip("torch", reason="IO imports torch via architecture")

import IO


ALL_REGIONS = [r for regions in IO.CLUSTERS.values() for r in regions]


def _write_regional_csv(path, wallclock_times):
    """Emit a ';'-separated CSV with every region present at each timestamp."""
    rows = []
    for t in wallclock_times:
        date, heure = t.split(" ")
        for region in ALL_REGIONS:
            rows.append({
                "Date": date,
                "Heure": heure,
                "Région": region,
                "Consommation brute électricité (MW) - RTE": 1000.0,
            })
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


def test_localised_to_paris_not_fixed_offset(tmp_path):
    """
    A winter noon and a summer noon must get the correct Paris offsets
    (+01:00 and +02:00), proving the timezone is no longer a constant +02:00.
    """
    csv = tmp_path / "regional.csv"
    _write_regional_csv(csv, ["2022-01-15 12:00", "2022-07-15 12:00"])

    out, names = IO.load_consumption_by_region(
        path=str(csv), cache_dir=str(tmp_path), verbose=0)

    assert str(out.index.tz) == "Europe/Paris"

    winter = pd.Timestamp("2022-01-15 12:00", tz="Europe/Paris")
    summer = pd.Timestamp("2022-07-15 12:00", tz="Europe/Paris")
    assert winter in out.index      # same instant: 11:00 UTC (a fixed +02:00
    assert summer in out.index      #   gave 10:00 UTC for the winter noon)
    # the output itself: 12:00 local for both, winter UTC+1, summer UTC+2
    assert (out.index.hour == 12).all()
    offsets = sorted(t.utcoffset() for t in out.index)
    assert offsets == [pd.Timedelta(hours=1), pd.Timedelta(hours=2)]
    assert sorted(out.index.tz_convert("UTC").hour) == [10, 11]


def test_nonexistent_spring_forward_hour_is_dropped(tmp_path):
    """
    Wall-clock 02:30 on 2022-03-27 does not exist in Europe/Paris (clocks jump
    02:00 -> 03:00).  It must be dropped, not shifted onto a real instant.
    """
    csv = tmp_path / "regional.csv"
    _write_regional_csv(csv, ["2022-03-27 01:30",   # exists (before jump)
                              "2022-03-27 02:30",   # does NOT exist
                              "2022-03-27 03:30"])  # exists (after jump)

    out, _ = IO.load_consumption_by_region(
        path=str(csv), cache_dir=str(tmp_path), verbose=0)

    local = out.index  # already Europe/Paris
    hours = {(ts.hour, ts.minute) for ts in local}
    assert (1, 30) in hours
    assert (3, 30) in hours
    assert (2, 30) not in hours          # the nonexistent hour was dropped
    assert len(out) == 2


def test_cache_key_versioned_by_timezone(tmp_path):
    """A pickle written by the old code (key without "tz", fixed +02:00
    offsets) must NOT be reloaded: plant one under the old key and check that
    the csv is parsed again."""
    import hashlib, json, os, pickle
    csv = tmp_path / "regional.csv"
    _write_regional_csv(csv, ["2022-01-15 12:00", "2022-01-15 12:30"])

    old_key = hashlib.md5(json.dumps(
        {"file_size": os.path.getsize(csv),
         "modification_time": os.path.getmtime(csv), "recent": ""},
        sort_keys=True).encode()).hexdigest()
    stale = pd.DataFrame({"stale": [1.]},
                         index=pd.DatetimeIndex(["2000-01-01"], tz="UTC"))
    with open(tmp_path / f"conso_region_{old_key}.pkl", "wb") as f:
        pickle.dump((stale, ["stale"]), f)

    out, names = IO.load_consumption_by_region(path=str(csv),
                                               cache_dir=str(tmp_path))
    assert "stale" not in out.columns and names != ["stale"]
    assert pd.Timestamp("2022-01-15 12:00", tz="Europe/Paris") in out.index


def test_cache_is_reloaded_without_parsing(tmp_path, monkeypatch):
    """Second call: loaded from the pickle (the csv is not read again) and
    identical."""
    csv = tmp_path / "regional.csv"
    _write_regional_csv(csv, ["2022-01-15 12:00", "2022-01-15 12:30"])

    out1, _ = IO.load_consumption_by_region(path=str(csv), cache_dir=str(tmp_path))
    assert list(tmp_path.glob("conso_region_*.pkl"))

    monkeypatch.setattr(IO.pd, "read_csv",
                        lambda *a, **k: pytest.fail("csv parsed again"))
    out2, _ = IO.load_consumption_by_region(path=str(csv), cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(out1, out2)

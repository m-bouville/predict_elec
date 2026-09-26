"""
Tests for ``run`` (outside the model itself) and ``predict_elec``:

* ``postprocess`` builds the csv row without modifying its input dicts;
* ``load_and_create_df``: ``dates_df``, part of the cache keys, depends
  neither on ``verbose`` nor on today's date;
* ``run_model``: warning when ``num_trials`` would not be used, and in a
  search when ``validate_every`` is not 1 (searches validate every epoch);
* ``recalculate_loss``: single-run rows recomputed (current column names),
  multi-run rows (averaged losses) left unchanged;
* ``input_cache_fname``: the input pickle is keyed on the data files; the
  obsolete pickles are removed;
* ``enforce_ranges`` (csv maintenance) works with pandas >= 3;
* ``append_csv_row`` refuses a row whose columns differ from the file;
* ``predict_elec``: the 'statistics' split and the RUN_FAST parameters are
  passed on, and the constants are not modified.

``run`` imports torch, so the file skips without it.
"""
import copy
import re

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="run imports torch")

import constants


# ---------------------------------------------------------------------------
# postprocess: csv row, inputs unchanged
# ---------------------------------------------------------------------------
def _postprocess_row(nntq=None, meta=None, base=None):
    import run
    models = ['NNTQ', 'LR', 'RF', 'LGBM', 'meta LR', 'meta NN']
    metrics = pd.DataFrame(np.ones((6, 3)), index=models,
                           columns=['bias', 'RMSE', 'MAE'])
    cov = {'q10': .01, 'q25': -.02, 'q50': 0., 'q75': .03, 'q90': -.01}
    weights = pd.Series([.4, .2, .2, .2], index=['NNTQ_q50', 'LR', 'RF', 'LGBM'])
    return run.postprocess(
        base if base is not None else copy.deepcopy(constants.BASELINES_PARAMETERS),
        nntq if nntq is not None else copy.deepcopy(constants.NNTQ_PARAMETERS),
        meta if meta is not None else copy.deepcopy(constants.METAMODEL_NN_PARAMETERS),
        60, metrics, cov, weights, 2.5, 0)


def test_postprocess_leaves_parameters_unchanged():
    nntq = copy.deepcopy(constants.NNTQ_PARAMETERS)
    meta = copy.deepcopy(constants.METAMODEL_NN_PARAMETERS)
    base = copy.deepcopy(constants.BASELINES_PARAMETERS)
    ref = copy.deepcopy((nntq, meta, base))
    row, _ = _postprocess_row(nntq, meta, base)
    assert (nntq, meta, base) == ref
    # the csv row still gets the x1e6 values and the flattened sequences
    assert row['learning_rate'] == pytest.approx(nntq['learning_rate'] * 1e6)
    assert 'quantiles' not in row and row['quantiles_2'] == 0.5
    assert 'metaNN_num_cells' not in row and 'metaNN_num_cells_0' in row


# ---------------------------------------------------------------------------
# dates_df (part of the cache keys) independent of verbose and of today
# ---------------------------------------------------------------------------
def test_dates_df_independent_of_verbose(monkeypatch, capsys):
    import run
    idx = pd.date_range("2022-01-01", periods=6, freq="30min", tz="UTC")
    df = pd.DataFrame({"consumption_GW": np.arange(6.), "Tavg_degC": 10.,
                       "is_holiday": 0, "sin_24h": .3}, index=idx)

    def fake_df_features(*a, **k):
        return (df.copy(), pd.DataFrame({"start": ["2022-01-01"],
                                         "end": ["2022-01-02"]}, index=["x"]),
                {})
    monkeypatch.setattr(run.utils, "df_features", fake_df_features)
    out = {v: run.load_and_create_df({}, "cache", 1, 48, 30, verbose=v)[-1]
           for v in (0, 1)}
    pd.testing.assert_frame_equal(out[0], out[1])
    assert "days_ago" not in out[1].columns
    assert "days_ago" in capsys.readouterr().out          # still printed


# ---------------------------------------------------------------------------
# run_model: num_trials warning in single-run modes
# ---------------------------------------------------------------------------
def test_num_trials_warning_fires(monkeypatch):
    import run

    class Stop(Exception):
        pass

    def stop(**kw):
        raise Stop
    monkeypatch.setattr(run, "run_model_once", stop)
    with pytest.warns(UserWarning, match="will not be used"), pytest.raises(Stop):
        run.run_model(
            mode='once', num_trials=3,
            baseline_parameters=constants.BASELINES_PARAMETERS,
            NNTQ_parameters=dict(constants.NNTQ_PARAMETERS),
            metamodel_NN_parameters=dict(constants.METAMODEL_NN_PARAMETERS),
            dict_input_csv_fnames={}, minutes_per_step=30,
            train_split_fraction=.8, valid_ratio=.25, forecast_hour=12, seed=0,
            force_calc_baselines=False,
            validate_every=1, display_every=1, plot_conv_every=1)


@pytest.mark.parametrize("validate_every, warns", [(2, True), (1, False)])
def test_search_warns_when_validate_every_is_not_used(monkeypatch, validate_every,
                                                      warns):
    """(/!\\ the checks were `x in locals()`, never true)"""
    import warnings as _w
    import run
    monkeypatch.setattr(run.Bayes_search, "run_Bayes_search", lambda **kw: None)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        run.run_model(
            mode='Bayes_NNTQ', num_trials=3,
            baseline_parameters=constants.BASELINES_PARAMETERS,
            NNTQ_parameters=dict(constants.NNTQ_PARAMETERS),
            metamodel_NN_parameters=dict(constants.METAMODEL_NN_PARAMETERS),
            dict_input_csv_fnames={}, minutes_per_step=30,
            train_split_fraction=.8, valid_ratio=.25, forecast_hour=12, seed=0,
            force_calc_baselines=False,
            validate_every=validate_every, display_every=5, plot_conv_every=5)
    messages = [str(w.message) for w in caught if "validate_every" in str(w.message)]
    assert bool(messages) is warns, messages


# ---------------------------------------------------------------------------
# recalculate_loss: current column names, multi-run rows kept
# ---------------------------------------------------------------------------
def test_recalculate_loss(tmp_path):
    import run
    metrics = {f"test_{m}_{k}": 1. for m in ['NNTQ', 'LR', 'RF', 'LGBM',
                                              'meta_LR', 'meta_NN']
               for k in ['bias', 'RMSE', 'MAE']}
    cov = {'q10': .01, 'q25': -.02, 'q50': 0., 'q75': .03, 'q90': -.01}
    rows = [dict(timestamp="2026-09-26 10:00:00", **metrics, **cov,
                 avg_abs_worst_days_test=1.5, num_runs=n,
                 loss_NNTQ=99., loss_meta=99.) for n in (1, 5)]
    csv = tmp_path / "search.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    run.recalculate_loss(str(csv))        # /!\ used to KeyError ('test_NN_bias')

    df = pd.read_csv(csv)
    assert df.loc[0, 'loss_NNTQ'] == pytest.approx(run.loss_NNTQ(cov, 1.5))
    assert df.loc[0, 'loss_meta'] == pytest.approx(run.loss_meta(metrics))
    assert df.loc[1, ['loss_NNTQ', 'loss_meta']].tolist() == [99., 99.]  # averaged


# ---------------------------------------------------------------------------
# input pickle keyed on the data files
# ---------------------------------------------------------------------------
def test_input_cache_fname_follows_the_data(tmp_path, monkeypatch):
    import os, time
    import run
    monkeypatch.chdir(tmp_path)
    os.makedirs("data"); os.makedirs("cache")
    open("data/a.csv", "w").write("x\n1\n")
    open("data/eco2mix.csv", "w").write("y\n1\n")      # read by a loader, not listed
    inputs = {"consumption": "data/a.csv"}

    name = run.input_cache_fname("cache", inputs)
    assert name == run.input_cache_fname("cache", inputs)          # stable
    assert os.path.basename(name).startswith("input_data_")

    open("data/eco2mix.csv", "a").write("2\n")                    # other file changed
    assert run.input_cache_fname("cache", inputs) != name
    name2 = run.input_cache_fname("cache", inputs)
    t = time.time() + 100
    os.utime("data/a.csv", (t, t))                                 # newer download
    assert run.input_cache_fname("cache", inputs) != name2

    # obsolete pickles (and the former unkeyed one) removed, current one kept
    current = run.input_cache_fname("cache", inputs)
    for f in (current, name, "cache/input_data.pkl", "cache/NNTQ_preds_x.pkl"):
        open(f, "wb").write(b"0")
    run.remove_other_input_caches("cache", keep=current)
    assert sorted(os.listdir("cache")) == sorted([os.path.basename(current),
                                                  "NNTQ_preds_x.pkl"])


# ---------------------------------------------------------------------------
# enforce_ranges with pandas >= 3 (infer_datetime_format removed)
# ---------------------------------------------------------------------------
def test_enforce_ranges(tmp_path):
    import run
    csv = tmp_path / "s.csv"
    pd.DataFrame({"timestamp": ["2026-09-25 10:00:00"] * 3,
                  "a": [1., 5., 9.]}).to_csv(csv, index=False)
    run.enforce_ranges(str(csv), {"a": (2., 10.)})
    assert pd.read_csv(csv)["a"].tolist() == [5., 9.]


# ---------------------------------------------------------------------------
# predict_elec passes its settings, constants untouched
# ---------------------------------------------------------------------------
def test_predict_elec_passes_its_settings(monkeypatch):
    import os, run
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predict_elec.py")
    src = open(path).read()
    src, n1 = re.subn(r"MODE = '[A-Za-z_]+'", "MODE = 'statistics'", src, count=1)
    src, n2 = re.subn(r"RUN_FAST\s*=\s*(True|False)", "RUN_FAST = True", src, count=1)
    assert n1 == n2 == 1, "MODE / RUN_FAST not found in predict_elec.py"

    captured = {}
    monkeypatch.setattr(run, "run_model", lambda **kw: captured.update(kw))
    ref = copy.deepcopy(constants.NNTQ_PARAMETERS)
    exec(compile(src, path, "exec"), {"__name__": "__main__"})

    assert captured['train_split_fraction'] == 0.99
    assert captured['valid_ratio'] == 0.01
    assert captured['baseline_parameters'] is constants.baseline_params_fast
    assert captured['NNTQ_parameters']['epochs'] == 2            # fast
    assert constants.NNTQ_PARAMETERS == ref                      # untouched


# ---------------------------------------------------------------------------
# append_csv_row: refuses a row whose columns differ from the file
# ---------------------------------------------------------------------------
def test_append_csv_row(tmp_path):
    import run
    path = str(tmp_path / "s.csv")
    run.append_csv_row(pd.DataFrame([{"a": 1., "b": 2.}]), path)   # creates
    run.append_csv_row(pd.DataFrame([{"a": 3., "b": 4.}]), path)   # appends
    assert pd.read_csv(path).to_dict("list") == {"a": [1., 3.], "b": [2., 4.]}
    for bad in ({"a": 5., "c": 6.},              # renamed column
                {"a": 5., "b": 6., "c": 7.},     # new column
                {"b": 5., "a": 6.}):             # same names, other order
        with pytest.raises(ValueError, match="Rename the file"):
            run.append_csv_row(pd.DataFrame([bad]), path)
    assert len(pd.read_csv(path)) == 2                              # untouched
    empty = tmp_path / "empty.csv"
    empty.write_text("")                                            # empty file:
    run.append_csv_row(pd.DataFrame([{"a": 1.}]), str(empty))      # new, header
    assert pd.read_csv(empty).to_dict("list") == {"a": [1.]}


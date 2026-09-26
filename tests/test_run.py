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
  passed on, and the constants are not modified; the settings of the other
  modes (Bayes: 40 trials, silent, RUN_FAST ignored);
* ``run_model``: mode -> search stage and csv path; the 'once' tail appends
  its row to parameter_search_one-off.csv; 'statistics', 'stats_only' and
  'load_input' pass the right settings to run_model_once.

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
        60, metrics, cov, weights, 2.5, 0,       # as run_model_once
        df_metrics_search=metrics, quantile_delta_coverage_test=cov)


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
    """The objective: search_* columns (first half of the test period),
    never the reported test_* ones."""
    import run
    _mk = lambda prefix, v: {f"{prefix}_{m}_{k}": v
                             for m in ['NNTQ', 'LR', 'RF', 'LGBM', 'meta_LR', 'meta_NN']
                             for k in ['bias', 'RMSE', 'MAE']}
    metrics = _mk("test", 1.)                              # objective, as keyed
    cov = {'q10': .01, 'q25': -.02, 'q50': 0., 'q75': .03, 'q90': -.01}
    rows = [dict(timestamp="2026-09-26 10:00:00", **_mk("search", 1.),
                 **_mk("test", 7.), **cov,               # reported: ignored
                 avg_abs_worst_days_search=1.5, num_runs=n,
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



# ---------------------------------------------------------------------------
# search objective on the first half of the test period, second half reported
# ---------------------------------------------------------------------------
def test_postprocess_objective_is_the_search_half():
    import run
    models = ['NNTQ', 'LR', 'RF', 'LGBM', 'meta LR', 'meta NN']
    search = pd.DataFrame(np.ones((6, 3)), index=models,
                          columns=['bias', 'RMSE', 'MAE'])
    test   = search * 5.
    cov    = {'q10': .01, 'q25': -.02, 'q50': 0., 'q75': .03, 'q90': -.01}
    cov_t  = {k: v + .1 for k, v in cov.items()}
    weights = pd.Series([.4, .2, .2, .2], index=['NNTQ_q50', 'LR', 'RF', 'LGBM'])
    row, (loss_NNTQ, loss_meta) = run.postprocess(
        copy.deepcopy(constants.BASELINES_PARAMETERS),
        copy.deepcopy(constants.NNTQ_PARAMETERS),
        copy.deepcopy(constants.METAMODEL_NN_PARAMETERS),
        60, test, cov, weights, 2.5, 0,
        df_metrics_search=search, quantile_delta_coverage_test=cov_t)

    # both halves in the row, the objective from the search half only
    assert row['search_meta_NN_MAE'] == 1. and row['test_meta_NN_MAE'] == 5.
    assert row['q10'] == cov['q10'] and row['test_coverage_q10'] == cov_t['q10']
    assert row['avg_abs_worst_days_search'] == 2.5
    flat = {f"test_{m}_{k}": 1. for m in ['NNTQ', 'LR', 'RF', 'LGBM',
                                          'meta_LR', 'meta_NN']
            for k in ['bias', 'RMSE', 'MAE']}
    assert loss_meta == pytest.approx(run.loss_meta(flat), abs=1e-5)
    assert loss_NNTQ == pytest.approx(run.loss_NNTQ(cov, 2.5), abs=1e-2)


def test_search_columns_are_not_parameters():
    import Bayes_search
    cols = Bayes_search.cols_not_paras()
    for c in ['search_meta_NN_MAE', 'test_meta_NN_MAE', 'test_coverage_q10',
              'avg_abs_worst_days_search']:
        assert c in cols, c



# ---------------------------------------------------------------------------
# predict_elec: the other modes
# ---------------------------------------------------------------------------
def _exec_predict_elec(monkeypatch, mode, run_fast):
    import os, run
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "predict_elec.py")
    src = open(path, encoding='utf-8').read()
    src, n1 = re.subn(r"MODE = '[A-Za-z_]+'", f"MODE = '{mode}'", src, count=1)
    src, n2 = re.subn(r"RUN_FAST\s*=\s*(True|False)", f"RUN_FAST = {run_fast}",
                      src, count=1)
    assert n1 == n2 == 1, "MODE / RUN_FAST not found in predict_elec.py"
    captured = {}
    monkeypatch.setattr(run, "run_model", lambda **kw: captured.update(kw))
    exec(compile(src, path, "exec"), {"__name__": "__main__"})
    return captured


@pytest.mark.parametrize("mode, run_fast, num_trials, verbose, split", [
    ('Bayes_NNTQ', True,  40, 0, None),     # RUN_FAST: one-off only
    ('Bayes_meta', False, 40, 0, None),
    ('once',       False,  1, 1, None),
    ('statistics', False,  1, 1, (0.99, 0.01)),
    ('stats_only', False,  0, 1, None),
])
def test_predict_elec_modes(monkeypatch, mode, run_fast, num_trials, verbose, split):
    """Mode, number of trials, verbosity, split and parameter bundles passed
    to run.run_model (the constants themselves, RUN_FAST ignored in the
    searches)."""
    kw = _exec_predict_elec(monkeypatch, mode, run_fast)
    assert kw['mode'] == mode and kw['num_trials'] == num_trials
    assert kw['verbose'] == verbose
    assert (kw['train_split_fraction'], kw['valid_ratio']) == \
        (split or (constants.TRAIN_SPLIT_FRACTION, constants.VALID_RATIO))
    assert kw['baseline_parameters'] is constants.BASELINES_PARAMETERS
    assert kw['NNTQ_parameters'] is constants.NNTQ_PARAMETERS
    assert kw['metamodel_NN_parameters'] is constants.METAMODEL_NN_PARAMETERS
    assert kw['dict_input_csv_fnames'] is constants.DICT_INPUT_CSV_FNAMES
    assert (kw['forecast_hour'], kw['seed'], kw['minutes_per_step']) == \
        (constants.FORECAST_HOUR, constants.SEED, constants.MINUTES_PER_STEP)


def test_predict_elec_rejects_an_unknown_mode(monkeypatch):
    """(a Bayes mode without a stage is rejected by run_model, below)"""
    with pytest.raises(ValueError, match="not a valid mode"):
        _exec_predict_elec(monkeypatch, 'whatever', False)


# ---------------------------------------------------------------------------
# run_model: mode -> stage, csv path
# ---------------------------------------------------------------------------
def _run_model(mode, **kw):
    import run
    args = dict(
        mode=mode, num_trials=3,
        baseline_parameters=constants.BASELINES_PARAMETERS,
        NNTQ_parameters=constants.NNTQ_PARAMETERS,
        metamodel_NN_parameters=constants.METAMODEL_NN_PARAMETERS,
        dict_input_csv_fnames={}, minutes_per_step=30,
        train_split_fraction=.8, valid_ratio=.25, forecast_hour=12, seed=0,
        force_calc_baselines=False, validate_every=1, display_every=5,
        plot_conv_every=5, cache_dir="some_cache")
    args.update(kw)
    return run.run_model(**args)


@pytest.mark.parametrize("mode, stage, csv", [
    ('Bayes_NNTQ',         'NNTQ', 'parameter_search_NNTQ.csv'),
    ('Bayes_meta',         'meta', 'parameter_search_meta.csv'),
    ('Bayesian_metamodel', 'meta', 'parameter_search_meta.csv'),
    ('Bayes_all',          'all',  'parameter_search_all.csv'),
])
def test_run_model_search_stage_and_csv(monkeypatch, mode, stage, csv):
    """The Bayes mode picks the stage and its csv; the bundles and settings
    are passed on unchanged."""
    import run
    from constants import Stage
    captured = []
    monkeypatch.setattr(run.Bayes_search, "run_Bayes_search",
                        lambda **kw: captured.append(kw))
    monkeypatch.setattr(run, "run_model_once",
                        lambda **kw: pytest.fail("no single run in a search"))
    _run_model(mode)
    assert len(captured) == 1
    kw = captured[0]
    assert kw['stage'] is Stage(stage) and kw['trials_csv_path'] == csv
    assert kw['num_trials'] == 3 and kw['cache_dir'] == "some_cache"
    assert kw['base_baseline_params'] is constants.BASELINES_PARAMETERS
    assert kw['base_NNTQ_params'] is constants.NNTQ_PARAMETERS
    assert kw['base_meta_NN_params'] is constants.METAMODEL_NN_PARAMETERS


@pytest.mark.parametrize("mode", ['Bayes', 'Bayes_foo', 'foo'])
def test_run_model_rejects_an_invalid_mode(monkeypatch, mode):
    import run
    monkeypatch.setattr(run.Bayes_search, "run_Bayes_search",
                        lambda **kw: pytest.fail("search started"))
    with pytest.raises(ValueError, match="not a valid mode"):
        _run_model(mode)


# ---------------------------------------------------------------------------
# run_model: single-run modes
# ---------------------------------------------------------------------------
def _fake_once(captured, row):
    from types import SimpleNamespace

    def once(**kw):
        captured.append(kw)
        if not kw['do_run_model']:
            return None
        return (SimpleNamespace(train="TRAIN"), dict(row), None, None, None,
                (20, 1.), (row['loss_NNTQ'], row['loss_meta']))
    return once


@pytest.mark.parametrize("mode", ['once', 'statistics'])
def test_run_model_once_tail_appends_the_one_off_csv(tmp_path, monkeypatch, mode):
    """'once' / 'statistics': model run (NNTQ cached), row appended to
    parameter_search_one-off.csv in the working directory, at each run;
    'statistics': whole-period diagnostics and plots on the training split."""
    import run
    from constants import Split
    monkeypatch.chdir(tmp_path)
    captured, plotted = [], []
    row = {'run': 0, 'a': 1.25, 'loss_NNTQ': 20.5, 'loss_meta': 2.25}
    monkeypatch.setattr(run, "run_model_once", _fake_once(captured, row))
    monkeypatch.setattr(run.plot_statistics, "thermosensitivity_per_time_of_day",
                        lambda **kw: plotted.append(kw['data_split']))
    _run_model(mode, num_trials=1)
    _run_model(mode, num_trials=1)

    df = pd.read_csv(tmp_path / "parameter_search_one-off.csv")
    assert df.to_dict("list") == {k: [v, v] for k, v in row.items()}
    kw = captured[0]
    assert kw['do_run_model'] is True and kw['save_cache_NNTQ'] is True
    assert kw['run_id'] == 0 and kw['cache_dir'] == "some_cache"
    stats = mode == 'statistics'
    assert kw['do_plot_statistics'] is stats
    assert kw['split_diagnostics'] is (Split.complete if stats else Split.test)
    assert plotted == (["TRAIN"] * 2 if stats else [])


@pytest.mark.parametrize("mode, stats", [('stats_only', True), ('load_input', False)])
def test_run_model_without_model_writes_nothing(tmp_path, monkeypatch, mode, stats):
    """'stats_only' / 'load_input': run_model_once without the model
    (do_run_model=False), nothing written."""
    import os, run
    monkeypatch.chdir(tmp_path)
    captured = []
    monkeypatch.setattr(run, "run_model_once", _fake_once(captured, {}))
    assert _run_model(mode, num_trials=0) is None
    assert captured[0]['do_run_model'] is False
    assert captured[0]['do_plot_statistics'] is stats
    assert os.listdir(tmp_path) == []

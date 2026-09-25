"""
Tests for ``run`` (outside the model itself) and ``predict_elec``:

* ``postprocess`` builds the csv row without modifying its input dicts;
* ``load_and_create_df``: ``dates_df``, part of the cache keys, depends
  neither on ``verbose`` nor on today's date;
* ``run_model``: warning when ``num_trials`` would not be used;
* ``enforce_ranges`` (csv maintenance) works with pandas >= 3;
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

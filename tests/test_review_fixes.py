"""
Regression tests for the small fixes of the 25/09 review
(claude/code_review_2026-09-25.md in the project; item numbers below).

Each test fails on the code before the fix.
"""
import copy
import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch = pytest.importorskip("torch")

import architecture, constants
from constants import Stage

pytestmark = [
    pytest.mark.filterwarnings("ignore:batch_size"),
    # plots call plt.show(), a no-op (with a warning) under the Agg backend
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# 23. postprocess must not modify the caller's parameter dicts
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
# 18. reloading the csv keeps small learning rates / weight decays
# ---------------------------------------------------------------------------
def test_reloaded_weight_decay_keeps_its_significant_digits(tmp_path):
    optuna = pytest.importorskip("optuna")
    import Bayes_search as bs
    nntq = copy.deepcopy(constants.NNTQ_PARAMETERS)
    nntq.update(weight_decay=1.312e-9, learning_rate=0.0032)
    meta = copy.deepcopy(constants.METAMODEL_NN_PARAMETERS)
    meta.update(weight_decay=2.3449e-05, learning_rate=0.0045)
    row, _ = _postprocess_row(nntq, meta)
    row.update(loss_NNTQ=20., loss_meta=2.3)
    csv = tmp_path / "search.csv"
    pd.DataFrame([row]).to_csv(csv, index=False, float_format="%.6f")  # as the search

    trials = bs.load_frozen_trials(
        str(csv), bs.DISTRIBUTIONS_BASELINES | bs.DISTRIBUTIONS_NNTQ |
        bs.DISTRIBUTIONS_METAMODEL_NN, Stage.NNTQ)
    p = trials[0].params
    assert p['weight_decay'] == pytest.approx(1.312e-9, rel=1e-6)   # was 1e-9
    assert p['learning_rate'] == 0.0032
    assert p['metaNN_weight_decay'] == pytest.approx(2.3449e-05, rel=1e-6)
    assert p['metaNN_learning_rate'] == 0.0045                    # on its grid


# ---------------------------------------------------------------------------
# 19. dates_df (part of the cache keys) independent of verbose and of today
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
# 4. public holidays after 2026
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
# 24. predict_elec: 'statistics' split and RUN_FAST baselines are passed on,
#     and the constants are not modified
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
# 39. num_trials warning in single-run modes
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
# 9. best-model saver: NaN never "best"; restore without save is explicit
# ---------------------------------------------------------------------------
def test_best_model_saver_ignores_nan():
    model = torch.nn.Linear(2, 1)
    saver = architecture.BestModelSaver(model)
    for epoch, loss in enumerate([5., float('nan'), 6.]):
        with torch.no_grad():
            model.weight.fill_(epoch)
        saver(loss, model, epoch)
    assert saver.best_epoch == 0 and saver.best_loss == 5.
    saver.restore(model)
    assert (model.weight == 0).all()


def test_best_model_saver_restore_without_save():
    model = torch.nn.Linear(2, 1)
    saver = architecture.BestModelSaver(model)
    saver(float('nan'), model, 0)
    with pytest.raises(RuntimeError, match="No best model"):
        saver.restore(model)


# ---------------------------------------------------------------------------
# 12. no autocast / GradScaler on CPU
# ---------------------------------------------------------------------------
def test_no_mixed_precision_on_cpu(tmp_path, monkeypatch):
    smoke = pytest.importorskip("test_run_smoke")
    import containers
    seen = []
    real = torch.amp.autocast

    def spy(*a, **k):
        seen.append(k.get('enabled', True))
        return real(*a, **k)
    monkeypatch.setattr(torch.amp, "autocast", spy)

    scalers = []
    real_scaler = torch.amp.GradScaler

    def spy_scaler(*a, **k):
        s = real_scaler(*a, **k)
        scalers.append(s)
        return s
    monkeypatch.setattr(torch.amp, "GradScaler", spy_scaler)

    smoke._once(tmp_path, monkeypatch, do_metamodel=False)
    assert seen and not any(seen)                       # autocast disabled
    assert scalers and not any(s.is_enabled() for s in scalers)


# ---------------------------------------------------------------------------
# 36. meta-NN: no UnboundLocalError when no epoch improves
# ---------------------------------------------------------------------------
def test_meta_NN_runs_when_validation_never_improves():
    import metamodel
    bugs = pytest.importorskip("test_open_bugs")
    df = bugs._toy_meta_frame(2)
    valid = df.copy()
    valid['y_true'] = np.nan                            # NaN loss: never "best"
    nets, weights = metamodel.train_meta_model(
        df_train=df, df_valid=valid, cols_features=["Tavg_degC"], valid_length=2,
        dropout=0., num_cells=[8, 8], epochs=2, learning_rate=1e-2,
        weight_decay=0., batch_size=8, patience=2, factor=.5, device="cpu")
    assert len(nets) == 2 and weights.shape[-1] == 4


# ---------------------------------------------------------------------------
# 16. n_valid must be positive
# ---------------------------------------------------------------------------
def test_make_X_and_y_rejects_empty_validation():
    n = 48 * 60
    dates = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    names_cols = {'y_nation': ['consumption_GW'], 'Y_regions': ['consumption_NE_GW'],
                  'features': ['f0'], 'ML_preds': ['consumption_LR']}
    with pytest.raises(AssertionError):
        architecture.make_X_and_y(
            np.zeros((n, 4), np.float32), dates, np.zeros(n, np.float32),
            int(n * .8), 0, names_cols, False, {'NE': 1.}, 30, 144, 72, True, 16)


# ---------------------------------------------------------------------------
# 42, 48, 49. plots
# ---------------------------------------------------------------------------
def test_time_of_day_grouping_in_local_time():
    import plots
    # a peak at 19:00 Paris time, in winter (18:00 UTC) and summer (17:00 UTC)
    idx = pd.date_range("2022-01-01", "2022-12-31 23:30", freq="30min", tz="UTC")
    local = idx.tz_convert("Europe/Paris")
    s = pd.Series(((local.hour == 19) & (local.minute == 0)).astype(float), index=idx)
    by_time = plots._apply_groupby(s, 'timeofday')
    assert by_time.idxmax() == 19.0 and by_time.max() == pytest.approx(1.0)
    by_day = plots._apply_groupby(s, 'dayofweek')
    assert np.allclose(np.sort(by_day[by_day > 0].index % 1), 19 / 24)


def test_loss_per_horizon_default_title():
    import plots
    plt.figure()
    plots.loss_per_horizon({'a': np.ones(4)}, 30)
    assert "class" not in plt.gca().get_title()


def test_data_accepts_series_and_single_curve_has_no_legend():
    import plots
    s = pd.Series(np.arange(5.), name="x")
    plots.data(s)                                        # used to IndexError
    assert plt.gca().get_legend() is None
    plots.data(pd.DataFrame({"a": np.arange(5.), "b": np.arange(5.)}))
    assert plt.gca().get_legend() is not None


def test_scatter_has_legend_with_two_clouds():
    import plots
    idx = pd.date_range("2022-01-01", periods=20, freq="D", tz="UTC")
    true = pd.Series(np.arange(20.), index=idx)
    x = pd.Series(np.arange(20.) + 1, index=idx)
    plots.scatter(true, {"q50": true * 1.1}, None, None, x_axis_series=x)
    assert plt.gca().get_legend() is not None


def test_curves_with_a_single_non_median_quantile():
    import plots
    idx = pd.date_range("2022-01-01", periods=48 * 10, freq="30min", tz="UTC")
    true = pd.Series(np.sin(np.arange(len(idx)) / 10), index=idx)
    plots.curves(true, {"q10": true - 1}, {}, {})       # used to KeyError 'q50'
    labels = [l.get_label() for l in plt.gca().get_lines()]
    assert "NNTQ (q10)" in labels


# ---------------------------------------------------------------------------
# 44, 45. plot_statistics
# ---------------------------------------------------------------------------
def test_drift_with_time_runs():
    import plot_statistics
    idx = pd.date_range("2016-01-01", "2020-12-31 23:30", freq="30min", tz="UTC")
    t = np.arange(len(idx))
    temp = pd.Series(12 + 8 * np.sin(2 * np.pi * t / (48 * 365)), index=idx,
                     name="Tavg_degC")
    cons = 55 - 1.5 * (temp - 12) + 0.001 * t / 48
    plot_statistics.drift_with_time(cons, temp, 48)     # used to UnboundLocalError


def test_threshold_zero_degC_is_a_threshold():
    import plot_statistics
    idx = pd.date_range("2022-01-01", periods=48 * 200, freq="30min", tz="UTC")
    T = pd.Series(np.linspace(-10, 10, len(idx)), index=idx)
    y = T ** 3                                           # slope 3T^2: ~0 at 0 degC
    out = plot_statistics.threshold_temp_sensitivity(
        y, {'q50': y}, {}, {}, T, idx, 'T_degC', [-1., 0., 1.], '==', 48, 0.5)
    slopes = out.iloc[:, 0]
    # at 0 degC the slope is local (small), as at +-1 degC, not the global fit
    assert abs(slopes.loc[0.]) < 5 and abs(slopes.loc[0.]) <= abs(slopes.loc[1.]) + 1


# ---------------------------------------------------------------------------
# 47. plot_optuna parameter importance
# ---------------------------------------------------------------------------
def test_plot_optuna_importance_table(capsys):
    optuna = pytest.importorskip("optuna")
    import Bayes_search as bs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        trial.suggest_int("LR_max_iter", 2000, 2000)
        if x > .5:                                       # conditional parameter
            trial.suggest_float("cond", 0, 1)
        return x
    study.optimize(objective, n_trials=40)
    bs.plot_optuna(study, Stage.meta, list_parameters_hist=["x"],
                   num_best_runs_params=5, num_best_runs_hist=10)
    out = capsys.readouterr().out
    table = out.split("Parameter Importance")[1].split("shape:")[0]
    names = [l.split()[0] for l in table.strip().splitlines()[2:]]
    assert "median" not in names and "LR_max_iter" not in names
    assert "cond" in names
    cond_line = [l for l in table.splitlines() if l.startswith("cond")][0]
    assert "NaN" not in cond_line.split()[1]


# ---------------------------------------------------------------------------
# 38. enforce_ranges with pandas >= 3 (infer_datetime_format removed)
# ---------------------------------------------------------------------------
def test_enforce_ranges(tmp_path):
    import run
    csv = tmp_path / "s.csv"
    pd.DataFrame({"timestamp": ["2026-09-25 10:00:00"] * 3,
                  "a": [1., 5., 9.]}).to_csv(csv, index=False)
    run.enforce_ranges(str(csv), {"a": (2., 10.)})
    assert pd.read_csv(csv)["a"].tolist() == [5., 9.]


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

"""
End-to-end smoke tests of ``run.run_model_once`` on a small synthetic dataset
(tiny NNTQ, fast baselines, 1-epoch meta-NN), with ``utils.df_features``
replaced by a generator: no input csv, no GPU needed.

They exercise the paths no unit test reaches:
* training -> metrics -> postprocess (csv row), metamodels on or off;
* the NNTQ cache: a second identical call loads the pickle instead of training,
  and the baseline predictions of the loaded bundle are replaced by the fresh
  ones (aligned on dates: the date-alignment bug of 25/09 would fail here);
* the NNTQ variants of the metamodel Bayesian search (N+2 trainings, middle N);
* the NNTQ cache key: each thing the NNTQ depends on gives another pickle;
* no pickle with save_cache_NNTQ=False; do_run_model=False returns early.
(Variants with use_ML_features, names_cols modified in place: test_open_bugs.py,
which reuses _once from here.)

Slow-ish (a few NNTQ trainings of a tiny model on CPU): ~1 min in total.
"""
import copy
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")            # headless: no window

torch = pytest.importorskip("torch", reason="run imports torch")
pytest.importorskip("lightgbm", reason="baselines need lightgbm")

import constants
import containers
import run

# the tiny synthetic training set has few batches: expected, not a problem here
pytestmark = pytest.mark.filterwarnings("ignore:batch_size")


N_DAYS  = 200
REGIONS = {'NE': 0.6, 'S': 0.4}


def _synthetic_features(seed: int = 0):
    """A frame shaped like utils.df_features' output (UTC half-hourly)."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range("2022-01-01", periods=48 * N_DAYS, freq="30min", tz="UTC")
    t    = np.arange(len(idx))
    day  = 2 * np.pi * t / 48
    Tavg = 10 + 8 * np.sin(2 * np.pi * t / (48 * 365)) + rng.normal(0, 2, len(t))
    cons = 55 + 6 * np.sin(day - 1) - 0.8 * (Tavg - 10) + rng.normal(0, 1, len(t))

    df = pd.DataFrame(index=idx)
    df["consumption_GW"]         = cons
    df["consumption_NE_GW"]      = 0.6 * cons + rng.normal(0, .3, len(t))
    df["consumption_S_GW"]       = 0.4 * cons + rng.normal(0, .3, len(t))
    df["consumption_SMA_1wk_GW"] = pd.Series(cons, index=idx).shift(48)\
                                     .rolling(48 * 7, min_periods=1).mean().bfill()
    df["Tavg_degC"]  = Tavg
    df["is_holiday"] = (idx.dayofweek >= 5).astype(float)
    df["sin_24h"]    = np.sin(day)
    df["cos_24h"]    = np.cos(day)
    df["year"]       = idx.year
    df["month"]      = idx.month
    df["timeofday"]  = t % 48
    df["price_euro_per_MWh"] = 80.
    dates_df = pd.DataFrame(columns=["start", "end"])
    return df, dates_df, dict(REGIONS)


def _parameters():
    base, nntq, meta = constants.fast_parameters(
        copy.deepcopy(constants.NNTQ_PARAMETERS),
        copy.deepcopy(constants.METAMODEL_NN_PARAMETERS))
    base = copy.deepcopy(base)
    nntq.update(device=torch.device('cpu'), epochs=1, input_length=48 * 3,
                batch_size=32, warmup_steps=5, patch_length=48, stride=24,
                model_dim=16, num_heads=2)
    meta.update(device=torch.device("cpu"), batch_size=8)
    return base, nntq, meta


def _once(tmp_path, monkeypatch, nntq_overrides=None, **kwargs):
    monkeypatch.setattr(run.utils, "df_features",
                        lambda *a, **k: _synthetic_features())
    return run.run_model_once(**_once_args(tmp_path, nntq_overrides, **kwargs))


def _once_args(tmp_path, nntq_overrides=None, **kwargs):
    base, nntq, meta = _parameters()
    nntq.update(nntq_overrides or {})
    args = dict(
        baseline_parameters=base, NNTQ_parameters=nntq,
        metamodel_NN_parameters=meta, dict_input_csv_fnames={},
        minutes_per_step=30, train_split_fraction=0.8, valid_ratio=0.25,
        forecast_hour=12, seed=0,
        force_calc_baselines=False, save_cache_baselines=False,
        save_cache_NNTQ=False, do_run_model=True,
        validate_every=1, display_every=999, plot_conv_every=999, run_id=0,
        cache_dir=str(tmp_path), do_plot_statistics=False, verbose=0)
    args.update(kwargs)
    return args


# ---------------------------------------------------------------------------
# one run, metamodels on / off
# ---------------------------------------------------------------------------
def test_full_run_produces_finite_losses_and_row(tmp_path, monkeypatch):
    data, row, metrics, w_meta, cov, _, (loss_NNTQ, loss_meta) = \
        _once(tmp_path, monkeypatch)                  # save_cache_NNTQ=False
    assert not list(tmp_path.glob("NNTQ_preds_*.pkl"))   # no pickle written
    assert np.isfinite(loss_NNTQ) and np.isfinite(loss_meta)
    assert {'NNTQ', 'LR', 'RF', 'LGBM', 'meta LR', 'meta NN'} <= set(metrics.index)
    assert np.isfinite(metrics.to_numpy()).all()
    assert row["loss_NNTQ"] == loss_NNTQ and row["loss_meta"] == loss_meta
    assert set(cov) == {'q10', 'q25', 'q50', 'q75', 'q90'}
    # objective on the first half of the test period, second half reported
    assert row['test_NNTQ_MAE'] == metrics.loc['NNTQ', 'MAE']
    assert np.isfinite(row['search_NNTQ_MAE']) and \
        row['search_NNTQ_MAE'] != row['test_NNTQ_MAE']
    assert np.isfinite(row['test_coverage_q50']) and \
        np.isfinite(row['avg_abs_worst_days_search'])
    cut = containers.search_period_end(data.test)
    assert data.test.true_nation_GW.index.min() < cut \
        < data.test.true_nation_GW.index.max()
    # the price is statistics only: never a model feature
    assert all('price' not in c for c in data.train.X_columns)


def test_run_without_metamodel_keeps_row_schema(tmp_path, monkeypatch):
    _, row_full, *_ = _once(tmp_path, monkeypatch)
    _, row_skip, _, w_meta, _, _, (loss_NNTQ, loss_meta) = \
        _once(tmp_path, monkeypatch, do_metamodel=False)
    assert w_meta is None and np.isnan(loss_meta) and np.isfinite(loss_NNTQ)
    assert list(row_full) == list(row_skip)


# ---------------------------------------------------------------------------
# NNTQ cache: second call loads, baselines refreshed on dates
# ---------------------------------------------------------------------------
def test_cache_is_loaded_and_baselines_refreshed(tmp_path, monkeypatch):
    data1, _, _, _, cov1, _, (loss1, _) = _once(tmp_path, monkeypatch,
                                                save_cache_NNTQ=True)
    assert len(list(tmp_path.glob("NNTQ_preds_*.pkl"))) == 1

    # other baseline parameters: same NNTQ key, different baseline predictions
    base, _, _ = _parameters()
    base['LR']['alpha'] *= 20
    base['RF']['max_depth'] = 3

    # NNTQ training must NOT happen again
    monkeypatch.setattr(run.containers.NeuralNet, "run",
                        lambda *a, **k: pytest.fail("NNTQ retrained"))
    data2, _, _, _, cov2, _, (loss2, _) = _once(
        tmp_path, monkeypatch, save_cache_NNTQ=True, baseline_parameters=base)

    assert cov2 == cov1 and loss2 == loss1              # same NNTQ
    for name in ('train', 'valid', 'test', 'complete'):
        s1, s2 = getattr(data1, name), getattr(data2, name)
        pd.testing.assert_series_equal(s1.dict_preds_NNTQ['q50'],
                                       s2.dict_preds_NNTQ['q50'])
        old = pd.DataFrame(s1.dict_preds_ML)
        new = pd.DataFrame(s2.dict_preds_ML)
        assert old.index.equals(new.index) and list(old) == list(new)
        assert not np.allclose(old['RF'], new['RF'])   # refreshed, not cached
        assert np.isfinite(new.to_numpy()).all()


def _other_regions():
    return _synthetic_features()[:2] + (dict(REGIONS, NE=0.7, S=0.3),)


def _extra_feature_column():
    df, dates_df, regions = _synthetic_features()
    df["sin_12h"] = np.sin(4 * np.pi * np.arange(len(df)) / 48)
    return df, dates_df, regions


def _other_baselines():
    base, _, _ = _parameters()
    base['LR']['alpha'] *= 20
    base['RF']['max_depth'] = 3
    return base


_ML = {'use_ML_features': 1}

# case: (settings of both runs, changes in the second run, features of the
#        second run, number of pickles expected)
CACHE_KEY_CASES = {
    # /!\ the key ignored the validation split, the region weights, the number
    #     of worst days (stale avg_abs_worst_days, hence loss_NNTQ, reloaded)
    "valid_ratio":    ({}, dict(valid_ratio=0.2), None, 2),
    "regions":        ({}, {}, _other_regions, 2),
    "num_worst_days": ({}, dict(num_worst_days=5), None, 2),
    "NNTQ parameter": ({}, dict(nntq_overrides={'dropout': 0.3}), None, 2),
    "feature column": ({}, {}, _extra_feature_column, 2),
    "forecast_hour":  ({}, dict(forecast_hour=13), None, 2),
        # (pred_length not adapted: only the key matters here)
    "baselines, ML features": (dict(nntq_overrides=_ML),
                               dict(baseline_parameters='other'), None, 2),
    # without ML features the NNTQ does not depend on the baselines: same key
    "baselines, no ML features": ({}, dict(baseline_parameters='other'), None, 1),
}


@pytest.mark.parametrize("case", list(CACHE_KEY_CASES))
def test_cache_key_follows_what_the_NNTQ_depends_on(tmp_path, monkeypatch, case):
    """A change of what the NNTQ depends on (validation split, region weights,
    number of worst days, NNTQ parameters, feature columns, forecast hour, the
    baselines when they are features) gives another NNTQ pickle; the baselines
    alone do not when use_ML_features=0."""
    both, changes, features, expected = CACHE_KEY_CASES[case]
    _once(tmp_path, monkeypatch, save_cache_NNTQ=True, do_metamodel=False, **both)
    assert len(list(tmp_path.glob("NNTQ_preds_*.pkl"))) == 1

    if features is not None:
        monkeypatch.setattr(run.utils, "df_features", lambda *a, **k: features())
    kwargs = dict(both, **changes)
    if kwargs.get('baseline_parameters') == 'other':
        kwargs['baseline_parameters'] = _other_baselines()
    run.run_model_once(**_once_args(tmp_path, save_cache_NNTQ=True,
                                    do_metamodel=False, **kwargs))
    assert len(list(tmp_path.glob("NNTQ_preds_*.pkl"))) == expected


def test_no_model_run_returns_early(tmp_path, monkeypatch):
    """do_run_model=False (statistics only): the data is loaded, then None is
    returned, before the baselines and the NNTQ."""
    monkeypatch.setattr(run.baselines, "create_baselines",
                        lambda *a, **k: pytest.fail("baselines computed"))
    assert _once(tmp_path, monkeypatch, do_run_model=False) is None
    assert not list(tmp_path.glob("NNTQ_preds_*.pkl"))


def test_regional_errors_in_national_units(tmp_path, monkeypatch):
    """The NNTQ gets regions_to_nation = region std / national std (training
    split), used by its regional loss."""
    built = []
    real = run.containers.NeuralNet

    def spy(*a, **k):
        net = real(*a, **k)
        built.append(net)
        return net
    monkeypatch.setattr(run.containers, "NeuralNet", spy)
    data, *_ = _once(tmp_path, monkeypatch, nntq_overrides={'lambda_regions': .05})
    ratio = np.asarray(built[0].regions_to_nation)
    df, _, _ = _synthetic_features()
    n_train = len(data.train.dates)
    expected = df[["consumption_NE_GW", "consumption_S_GW"]].iloc[:n_train].std(ddof=0) \
               / df["consumption_GW"].iloc[:n_train].std(ddof=0)
    np.testing.assert_allclose(ratio, expected.to_numpy(), rtol=1e-3)


def test_fresh_baselines_are_aligned_on_dates(tmp_path, monkeypatch):
    """After a cache load, dict_preds_ML must equal what a fresh (uncached) run
    computes, date by date."""
    _once(tmp_path, monkeypatch, save_cache_NNTQ=True)
    data_cached, *_ = _once(tmp_path, monkeypatch, save_cache_NNTQ=True)
    data_fresh,  *_ = _once(tmp_path / "other", monkeypatch)
    for name in ('train', 'valid', 'test'):
        pd.testing.assert_frame_equal(
            pd.DataFrame(getattr(data_cached, name).dict_preds_ML),
            pd.DataFrame(getattr(data_fresh,  name).dict_preds_ML))


def test_cache_pickle_is_slim(tmp_path, monkeypatch):
    """The NNTQ pickle holds neither the torch datasets / loaders nor the whole
    unscaled arrays; the bundle returned by the run keeps them (shared, not
    copied); a run from the pickle gives the same results (metamodels
    included, see test_metamodels_independent_of_the_cache)."""
    data, *_ = _once(tmp_path, monkeypatch, save_cache_NNTQ=True)
    path = next(tmp_path.glob("NNTQ_preds_*.pkl"))
    with open(path, "rb") as f:
        cached, _, _ = pickle.load(f)
    for name in ('train', 'valid', 'test', 'complete'):
        s_cached, s_run = getattr(cached, name), getattr(data, name)
        assert s_cached.loader is None and s_cached.dataset_scaled is None
        assert s_run.loader is not None and s_run.dataset_scaled is not None
        np.testing.assert_array_equal(s_cached.X, s_run.X)
        assert s_cached.origin_times == s_run.origin_times   # non-field attribute
    assert cached.X is None and data.X is not None

    # for_cache shares the arrays: no copy in memory
    slim = data.for_cache()
    assert slim.train.X is data.train.X
    assert slim.train.dict_preds_NNTQ is data.train.dict_preds_NNTQ

    # much smaller than the full bundle
    full = len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    assert path.stat().st_size < 0.8 * full, (path.stat().st_size, full)


@pytest.mark.filterwarnings("ignore")   # plots under Agg, legends, etc.
def test_verbose_run_from_the_cache(tmp_path, monkeypatch, capsys):
    """verbose=2 (comparisons, diagnostic and thermosensitivity plots) with the
    NNTQ trained, then loaded from the slim pickle: nothing needed downstream
    is missing from it (/!\\ the verbose >= 2 plots crashed on a TypeError);
    every figure is closed."""
    import matplotlib.pyplot as plt
    for _ in range(2):                           # trained, then from the cache
        _, _, metrics, *_ = _once(tmp_path, monkeypatch, save_cache_NNTQ=True,
                                  verbose=2)
        assert np.isfinite(metrics.to_numpy()).all()
        assert plt.get_fignums() == []
    assert "Loading NNTQ predictions" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NNTQ variants (metamodel Bayesian search)
# ---------------------------------------------------------------------------
def test_variants_built_once_then_reused(tmp_path, monkeypatch):
    num = 2
    _, _, _, _, cov0, _, _ = _once(tmp_path, monkeypatch,
                                   NNTQ_variant=0, num_NNTQ_variants=num)
    files = sorted(os.listdir(tmp_path))
    assert [f for f in files if f.endswith("_v0.pkl")] and \
           [f for f in files if f.endswith("_v1.pkl")]
    assert not [f for f in files if "_seed" in f]       # temporary files removed
    summary = json.load(open(next(tmp_path.glob("NNTQ_variants_*.json"))))
    assert len(summary["all"]) == num + 2 and len(summary["variants"]) == num

    # variant 1: loaded, not retrained, and a different NNTQ than variant 0
    monkeypatch.setattr(run.containers.NeuralNet, "run",
                        lambda *a, **k: pytest.fail("NNTQ retrained"))
    _, _, _, _, cov1, _, _ = _once(tmp_path, monkeypatch,
                                   NNTQ_variant=1, num_NNTQ_variants=num)
    assert cov1 != cov0

    with open(next(tmp_path.glob("NNTQ_preds_*_v1.pkl")), "rb") as f:
        _, cov_file, _ = pickle.load(f)
    assert cov_file == cov1


def test_variant_out_of_range_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        _once(tmp_path, monkeypatch, NNTQ_variant=2, num_NNTQ_variants=2)


# ---------------------------------------------------------------------------
# no autocast / GradScaler on CPU (bfloat16 there, ~3x slower)
# ---------------------------------------------------------------------------
def test_no_mixed_precision_on_cpu(tmp_path, monkeypatch):
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

    _once(tmp_path, monkeypatch, do_metamodel=False)
    assert seen and not any(seen)                       # autocast disabled
    assert scalers and not any(s.is_enabled() for s in scalers)


# ---------------------------------------------------------------------------
# metamodels: same result whether the NNTQ was trained or loaded
# ---------------------------------------------------------------------------
def test_metamodels_independent_of_the_cache(tmp_path, monkeypatch):
    """Training the NNTQ consumes random numbers, loading it does not: the
    metamodels are reseeded, so a cached run gives the same meta results."""
    _, _, m_fresh, *_ = _once(tmp_path, monkeypatch, save_cache_NNTQ=True)
    _, _, m_cached, *_ = _once(tmp_path, monkeypatch, save_cache_NNTQ=True)
    pd.testing.assert_frame_equal(m_fresh, m_cached)


# ---------------------------------------------------------------------------
# early stopping / best model: only on validated epochs
# ---------------------------------------------------------------------------
def test_early_stopping_only_on_validated_epochs(tmp_path, monkeypatch):
    import architecture
    seen = []
    real = architecture.EarlyStopping.__call__

    def spy(self, loss):
        seen.append(loss)
        return real(self, loss)
    monkeypatch.setattr(architecture.EarlyStopping, "__call__", spy)
    _once(tmp_path, monkeypatch, nntq_overrides={'epochs': 5, 'patience': 5},
          validate_every=3, do_metamodel=False)
    assert len(seen) == 2              # epochs 1 and 3 (index 0 and 2), not 5


# ---------------------------------------------------------------------------
# search objective on the first half of the test period, second half reported
# ---------------------------------------------------------------------------
def test_objective_on_the_first_half_of_the_test_period(tmp_path, monkeypatch):
    """Coverage and worst days (loss_NNTQ) on the first half of the test
    period, the reported coverage on the second half; recomputed here from
    the predictions."""
    data, row, _, _, cov, (top_n, worst), _ = _once(tmp_path, monkeypatch)
    cut   = containers.search_period_end(data.test)
    true  = data.test.true_nation_GW
    halves= {'search': true.index < cut, 'test': true.index >= cut}

    def coverage(key, keep):
        pred = data.test.dict_preds_NNTQ[key].reindex(true.index)
        return float(np.mean(true[keep] <= pred[keep])) - int(key[1:]) / 100

    whole = np.ones(len(true), bool)
    assert any(coverage(k, halves['search']) != coverage(k, whole) for k in cov)
    for key in cov:
        assert cov[key] == pytest.approx(coverage(key, halves['search']))
        assert row[f"test_coverage_{key}"] == \
            pytest.approx(coverage(key, halves['test']))

    def worst_days(keep):
        err = (data.test.dict_preds_NNTQ['q50'].reindex(true.index) - true)[keep]
        err = err.abs().round(2)
        days = err.groupby(err.index.tz_convert("Europe/Paris").normalize())
        daily = days.mean()[days.size() == 48]            # whole days only
        return float(daily.sort_values(ascending=False).head(top_n).mean())

    assert worst_days(halves['search']) != pytest.approx(worst_days(whole), abs=.01)
    assert worst == pytest.approx(worst_days(halves['search']), abs=.01)
    assert row['avg_abs_worst_days_search'] == worst


    # metrics (loss_meta): search_* on the first half, test_* on the second
    def mae(model, keep):
        preds = {'NNTQ': data.test.dict_preds_NNTQ['q50'], **data.test.dict_preds_ML}
        df = pd.DataFrame({'true': true, **{k: v.reindex(true.index)
                                            for k, v in preds.items()}})[keep].dropna()
        return float((df[model] - df['true']).abs().mean())

    for model in ['NNTQ', 'LR']:
        assert mae(model, halves['search']) != pytest.approx(mae(model, whole), abs=1e-3)
        assert row[f"search_{model}_MAE"] == pytest.approx(mae(model, halves['search']),
                                                          abs=1e-3)
        assert row[f"test_{model}_MAE"]   == pytest.approx(mae(model, halves['test']),
                                                          abs=1e-3)

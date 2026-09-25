"""
End-to-end smoke tests of ``run.run_model_once`` on a small synthetic dataset
(tiny NNTQ, fast baselines, 1-epoch meta-NN), with ``utils.df_features``
replaced by a generator: no input csv, no GPU needed.

They exercise the paths no unit test reaches:
* training -> metrics -> postprocess (csv row), metamodels on or off;
* the NNTQ cache: a second identical call loads the pickle instead of training,
  and the baseline predictions of the loaded bundle are replaced by the fresh
  ones (aligned on dates: the date-alignment bug of 25/09 would fail here);
* the NNTQ variants of the metamodel Bayesian search (N+2 trainings, middle N).
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
    return run.run_model_once(**args)


# ---------------------------------------------------------------------------
# one run, metamodels on / off
# ---------------------------------------------------------------------------
def test_full_run_produces_finite_losses_and_row(tmp_path, monkeypatch):
    data, row, metrics, w_meta, cov, _, (loss_NNTQ, loss_meta) = \
        _once(tmp_path, monkeypatch)
    assert np.isfinite(loss_NNTQ) and np.isfinite(loss_meta)
    assert {'NNTQ', 'LR', 'RF', 'LGBM', 'meta LR', 'meta NN'} <= set(metrics.index)
    assert np.isfinite(metrics.to_numpy()).all()
    assert row["loss_NNTQ"] == loss_NNTQ and row["loss_meta"] == loss_meta
    assert set(cov) == {'q10', 'q25', 'q50', 'q75', 'q90'}
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

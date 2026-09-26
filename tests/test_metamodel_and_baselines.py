"""
Tests for the baselines (``baselines`` imports lightgbm, not torch):

* Ridge predictions finite;
* RF and LGBM: finite, better than the mean, deterministic with
  ``random_state``, cached by configuration; Ridge never cached.

* the scaler of the linear baselines is fit on the training rows only.

Open bugs (item 4b, meta-NN selection on the train split and df_valid=None
crash): test_open_bugs.py.
"""
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Ridge: finite predictions (scaler: test_scaler_fit_on_train_only)
# ---------------------------------------------------------------------------
class TestRidgePredictions:
    def _fit(self, tmp_path):
        baselines = pytest.importorskip("baselines",
                                        reason="needs lightgbm/sklearn")
        rng = np.random.default_rng(0)
        F, n_train, n_rest = 4, 200, 100
        X = np.vstack([rng.normal(0, 1, size=(n_train, F)),
                       rng.normal(50, 5, size=(n_rest, F))]).astype(np.float32)
        y = (X[:, 0] * 2 + rng.normal(0, 0.1, size=n_train + n_rest)).astype(np.float32)
        dates = pd.date_range("2020-01-01", periods=len(X), freq="D")
        dates_df = pd.DataFrame({"start": [], "end": []})
        series, _ = baselines.regression_and_forest(
            X=X, y=y, cols_y_nation=["consumption_GW"],
            cols_features=[f"f{i}" for i in range(F)],
            dates=dates, dates_df=dates_df, train_end=160, val_end=200,
            models_cfg={"LR": {"type": "ridge", "alpha": 1.0}},
            cache_dir=str(tmp_path), save_cache_baselines=False,
            cache_id_dict={}, force_calculation=True, verbose=0)
        return series

    def test_predictions_are_finite(self, tmp_path):
        series = self._fit(tmp_path)
        assert np.isfinite(series["LR"].values).all()

# ---------------------------------------------------------------------------
# RF and LGBM baselines
# ---------------------------------------------------------------------------
RF_CFG   = {"type": "rf", "n_estimators": 20, "max_depth": 5,
            "min_samples_leaf": 3, "random_state": 0, "n_jobs": 1}
LGBM_CFG = {"type": "lgbm", "objective": "regression", "n_estimators": 30,
            "num_leaves": 7, "learning_rate": 0.1, "random_state": 0,
            "n_jobs": 1, "verbose": -1}


def _tree_baselines(tmp_path, cfg, save=True, force=False):
    baselines = pytest.importorskip("baselines", reason="needs lightgbm/sklearn")
    rng = np.random.default_rng(0)
    n, F = 400, 4
    X = rng.normal(size=(n, F)).astype(np.float32)
    y = (3 * X[:, 0] + np.sin(2 * X[:, 1]) + rng.normal(0, .1, n)).astype(np.float32)
    series, _ = baselines.regression_and_forest(
        X=X, y=y, cols_y_nation=["consumption_GW"],
        cols_features=[f"f{i}" for i in range(F)],
        dates=pd.date_range("2020-01-01", periods=n, freq="h"),
        dates_df=pd.DataFrame({"start": [], "end": []}),
        train_end=240, val_end=320, models_cfg=cfg, cache_dir=str(tmp_path),
        save_cache_baselines=save, cache_id_dict={}, force_calculation=force,
        verbose=0)
    return series, y


@pytest.mark.parametrize("name, cfg", [("RF", RF_CFG), ("LGBM", LGBM_CFG)])
def test_tree_baselines_finite_and_better_than_mean(tmp_path, name, cfg):
    series, y = _tree_baselines(tmp_path, {name: cfg})
    pred = series[name].to_numpy()
    assert len(pred) == len(y) and np.isfinite(pred).all()
    test = slice(320, None)
    assert np.mean((pred[test] - y[test])**2) < 0.5 * np.var(y[test])


@pytest.mark.parametrize("name, cfg", [("RF", RF_CFG), ("LGBM", LGBM_CFG)])
def test_tree_baselines_cached_by_configuration(tmp_path, name, cfg):
    s1, _ = _tree_baselines(tmp_path, {name: cfg})
    files = list(tmp_path.glob(f"{name}_preds_*.pkl"))
    assert len(files) == 1
    # same configuration: loaded from the cache, identical
    s2, _ = _tree_baselines(tmp_path, {name: cfg})
    pd.testing.assert_series_equal(s1[name], s2[name])
    assert list(tmp_path.glob(f"{name}_preds_*.pkl")) == files
    # other configuration: another cache file, other predictions
    other = dict(cfg, max_depth=2) if name == "RF" else dict(cfg, num_leaves=3)
    s3, _ = _tree_baselines(tmp_path, {name: other})
    assert len(list(tmp_path.glob(f"{name}_preds_*.pkl"))) == 2
    assert not np.allclose(s1[name], s3[name])
    # forced: recomputed, identical (random_state)
    s4, _ = _tree_baselines(tmp_path, {name: cfg}, force=True)
    pd.testing.assert_series_equal(s1[name], s4[name])


def test_ridge_is_never_cached(tmp_path):
    _tree_baselines(tmp_path, {"LR": {"type": "ridge", "alpha": 1.0}})
    assert not list(tmp_path.glob("LR_preds_*.pkl"))


# ---------------------------------------------------------------------------
# meta-NN: no crash when no epoch improves
# ---------------------------------------------------------------------------
def test_meta_NN_runs_when_validation_never_improves():
    pytest.importorskip("torch", reason="metamodel is a torch module")
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
# meta-NN: data per horizon built once, not at every epoch
# ---------------------------------------------------------------------------
def test_meta_NN_builds_tensors_once(monkeypatch):
    pytest.importorskip("torch", reason="metamodel is a torch module")
    import metamodel
    bugs = pytest.importorskip("test_open_bugs")
    df = bugs._toy_meta_frame(3)
    calls = []
    real = metamodel.to_tensors
    monkeypatch.setattr(metamodel, "to_tensors",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    nets, weights = metamodel.train_meta_model(
        df_train=df, df_valid=df.copy(), cols_features=["Tavg_degC"],
        valid_length=3, dropout=0., num_cells=[8, 8], epochs=4,
        learning_rate=1e-2, weight_decay=0., batch_size=8, patience=2,
        factor=.5, device="cpu")
    assert len(calls) == 2 * 3                  # train + valid, per horizon
    assert len(nets) == 3 and np.isfinite(weights).all()


def test_meta_NN_reproducible_with_a_seed():
    """Loaders built once still reshuffle at each epoch: same seed, same result."""
    torch = pytest.importorskip("torch", reason="metamodel is a torch module")
    import metamodel
    bugs = pytest.importorskip("test_open_bugs")
    df = bugs._toy_meta_frame(2)
    out = []
    for _ in range(2):
        torch.manual_seed(0)
        _, w = metamodel.train_meta_model(
            df_train=df, df_valid=df.copy(), cols_features=["Tavg_degC"],
            valid_length=2, dropout=0., num_cells=[8, 8], epochs=3,
            learning_rate=1e-2, weight_decay=0., batch_size=8, patience=2,
            factor=.5, device="cpu")
        out.append(w)
    np.testing.assert_array_equal(out[0], out[1])


# ---------------------------------------------------------------------------
# oracle baseline and metamodel context columns
# ---------------------------------------------------------------------------
def test_oracle_baseline_is_returned(tmp_path):
    with pytest.warns(UserWarning, match="Using the oracle"):   # deliberate
        series, y = _tree_baselines(tmp_path, {"oracle": {}, "LR": {"type": "ridge",
                                                                  "alpha": 1.0}})
    assert set(series) == {"oracle", "LR"}
    np.testing.assert_array_equal(series["oracle"].to_numpy(), y)


def test_metamodel_context_excludes_predictions():
    pytest.importorskip("torch", reason="metamodel is a torch module")
    import metamodel
    cols = ['Tavg_degC', 'consumption_LR', 'consumption_RF', 'consumption_LGBM',
            'consumption_NNTQ', 'NNTQ_inter']
    assert metamodel._context_cols(cols) == ['Tavg_degC', 'NNTQ_inter']


def test_scaler_fit_on_train_only(tmp_path):
    """
    Behavioural leak check. Fit the baselines on data whose valid and test
    blocks are drawn from two other distributions, then compare the returned
    TEST predictions to a leakage-free reference (Ridge on train-only-scaled
    features). They match only when the scaler is fit on the training rows
    alone, not on train + valid nor on all rows.
    (/!\\ it was fit on train+valid+test)
    """
    baselines = pytest.importorskip("baselines", reason="needs lightgbm/sklearn")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    F, n_train, n_rest = 4, 200, 100
    # train (160 rows), valid (40) and test (100) from three distributions: a
    #   scaler that saw the valid or test rows gives other predictions
    X = np.vstack([rng.normal(0, 1, size=(160, F)),
                   rng.normal(20, 3, size=(40, F)),
                   rng.normal(50, 5, size=(n_rest, F))]).astype(np.float32)
    y = (X[:, 0] * 2 + rng.normal(0, 0.1, size=n_train + n_rest)).astype(np.float32)
    dates = pd.date_range("2020-01-01", periods=len(X), freq="D")
    dates_df = pd.DataFrame({"start": [], "end": []})

    series, _ = baselines.regression_and_forest(
        X=X, y=y, cols_y_nation=["consumption_GW"],
        cols_features=[f"f{i}" for i in range(F)],
        dates=dates, dates_df=dates_df, train_end=160, val_end=200,
        models_cfg={"LR": {"type": "ridge", "alpha": 1.0}},
        cache_dir=str(tmp_path), save_cache_baselines=False,
        cache_id_dict={}, force_calculation=True, verbose=0)
    their_test = series["LR"].iloc[200:].values

    scaler = StandardScaler().fit(X[:160])          # train-only reference
    Xs = scaler.transform(X)
    ref = Ridge(alpha=1.0).fit(Xs[:160], y[:160]).predict(Xs[200:])

    np.testing.assert_allclose(their_test, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# metamodel horizon: half-hour of the Paris day
# ---------------------------------------------------------------------------
def test_metamodel_horizon_is_the_paris_half_hour():
    """One network per horizon: horizon h must be the same local time (and lead
    time after the noon Paris origin) in winter and in summer.
    (/!\\ was the UTC half-hour: 1 h apart between CET and CEST)"""
    pytest.importorskip("torch", reason="metamodel imports torch")
    import metamodel
    stamps = pd.DatetimeIndex(["2022-01-10 23:00", "2022-07-10 22:00",   # 00:00 Paris
                               "2022-01-11 22:30", "2022-07-11 21:30"],  # 23:30 Paris
                              tz="UTC")
    assert metamodel.horizon(stamps).tolist() == [0, 0, 47, 47]
    naive = pd.DatetimeIndex(["2022-01-10 00:00", "2022-01-10 23:30"])  # local already
    assert metamodel.horizon(naive).tolist() == [0, 47]


def test_meta_data_horizons_cover_each_paris_day():
    """prepare_meta_data on Paris-day stamps (as prediction_day_ahead makes
    them): each day has horizons 0..47 once, 0..3 and 6..47 on the spring-forward
    Sunday (02:00-03:00 does not exist)."""
    pytest.importorskip("torch", reason="metamodel imports torch")
    import metamodel
    local = pd.date_range("2022-03-25", "2022-03-30", freq="30min",
                          tz="Europe/Paris", inclusive="left")
    dates = local.tz_convert("UTC")
    n = len(dates)
    s = lambda v: pd.Series(v, index=dates)
    df = metamodel.prepare_meta_data(
        "test", {'q25': s(np.full(n, 49.)), 'q50': s(np.full(n, 50.)),
                 'q75': s(np.full(n, 51.))},
        {'LR': s(np.full(n, 50.)), 'RF': s(np.full(n, 50.)), 'LGBM': s(np.full(n, 50.))},
        np.zeros((n, 1)), np.full(n, 50.), dates, ['f0'])
    per_day = df.groupby(df.index.tz_convert("Europe/Paris").date)['horizon']
    for day, h in per_day:
        expected = [k for k in range(48) if not (str(day) == "2022-03-27" and k in (4, 5))]
        assert sorted(h.tolist()) == expected, day


# ---------------------------------------------------------------------------
# meta-LR: minimum weight
# ---------------------------------------------------------------------------
def _meta_LR_data(seed=0, n=300):
    """Four 'predictions' of y; y leans negatively on C: an unconstrained fit
    gives C a negative weight."""
    rng = np.random.default_rng(seed)
    truth = 50 + 10 * np.sin(np.arange(n) / 20)
    X = pd.DataFrame({name: truth + rng.normal(0, s, n)
                      for name, s in [("NNTQ", 1.), ("LR", 2.), ("RF", 3.), ("LGBM", 2.5)]},
                     index=pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC"))
    y = pd.Series(1.3 * X["NNTQ"] + 0.2 * X["LR"] - 0.4 * X["RF"] + 0. * X["LGBM"]
                  + rng.normal(0, .1, n), index=X.index)
    return X, y


@pytest.mark.parametrize("min_weight", [0., 0.05])
def test_meta_LR_respects_min_weight(min_weight):
    """coef_ >= min_weight for every predictor, on data where the
    unconstrained least squares put one weight below it (and that weight then
    sits on its bound); the returned weights and predictions match coef_."""
    pytest.importorskip("torch", reason="metamodel imports torch")
    import metamodel
    X, y = _meta_LR_data()
    unconstrained = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)[0]
    assert unconstrained.min() < min_weight - 0.1               # the case at stake

    model = metamodel.ConstrainedLinearRegression(fit_intercept=False,
                                                  min_weight=min_weight).fit(X, y)
    assert (model.coef_ >= min_weight - 1e-12).all()
    assert model.coef_[2] == pytest.approx(min_weight, abs=1e-6)   # RF on its bound

    weights, pred_in, pred1, pred2 = metamodel.weights_LR_metamodel(
        X.iloc[:200], y.iloc[:200], X.iloc[200:], None, min_weight=min_weight)
    assert list(weights) == list(X.columns)
    assert min(weights.values()) >= min_weight - 1e-3
    assert pred2 is None and pred1.index.equals(X.index[200:])
    assert np.isfinite(pred_in).all() and np.isfinite(pred1).all()


# ---------------------------------------------------------------------------
# meta-NN: the best epoch (on validation) is restored
# ---------------------------------------------------------------------------
def test_meta_NN_restores_its_best_epoch(monkeypatch):
    """Validation losses scripted to 5, 3, 1, 4, 2: the returned network is
    the one at the end of epoch 3 (not the last one), and the returned weights
    are those of that epoch's validation."""
    torch = pytest.importorskip("torch", reason="metamodel is a torch module")
    import metamodel
    bugs = pytest.importorskip("test_open_bugs")
    df = bugs._toy_meta_frame(1, per_h=60)
    script = [5., 3., 1., 4., 2.]
    states, valid_weights = [], []

    RealMSE = torch.nn.MSELoss

    class ScriptedMSE(RealMSE):
        """Real loss when training; scripted loss for the validation."""
        def forward(self, pred, target):
            if torch.is_grad_enabled():
                return super().forward(pred, target)
            return torch.tensor(script[len(valid_weights) - 1])

    class Recorded(metamodel.MetaNet):
        """Records, at each validation, its parameters and output weights."""
        def forward(self, context, preds):
            y, w = super().forward(context, preds)
            if not torch.is_grad_enabled():
                states.append({k: v.clone() for k, v in self.state_dict().items()})
                valid_weights.append(w.clone())
            return y, w

    monkeypatch.setattr(metamodel.nn, "MSELoss", ScriptedMSE)
    monkeypatch.setattr(metamodel, "MetaNet", Recorded)
    torch.manual_seed(0)
    nets, weights = metamodel.train_meta_model(
        df_train=df, df_valid=df.copy(), cols_features=["Tavg_degC"], valid_length=1,
        dropout=0., num_cells=[8, 8], epochs=len(script), learning_rate=5e-2,
        weight_decay=0., batch_size=32, patience=10, factor=.5, device="cpu")

    assert len(states) == len(script)                  # one validation batch per epoch
    best = int(np.argmin(script))
    final = nets[0].state_dict()
    for k, v in final.items():
        torch.testing.assert_close(v, states[best][k])
    assert any(not torch.equal(final[k], states[-1][k]) for k in final)  # not the last
    np.testing.assert_array_equal(weights, valid_weights[best].numpy())


# ---------------------------------------------------------------------------
# meta-NN data: complete inputs keep every row
# ---------------------------------------------------------------------------
def _meta_inputs(n=96, drop_baseline=None, quantiles=("q25", "q50", "q75")):
    dates = pd.date_range("2022-01-10 23:00", periods=n, freq="30min", tz="UTC")
    s = lambda v: pd.Series(np.full(n, v), index=dates)
    preds = {q: s(50. + k) for k, q in enumerate(quantiles)}
    baselines = {b: s(50.) for b in ("LR", "RF", "LGBM") if b != drop_baseline}
    return preds, baselines, np.zeros((n, 1)), np.full(n, 50.), dates


def test_meta_data_complete_inputs_keep_every_row():
    """Control for the test above: with every input, no row is lost."""
    import metamodel
    preds, baselines, X, y, dates = _meta_inputs()
    df = metamodel.prepare_meta_data("train", preds, baselines, X, y, dates, ["f0"])
    assert len(df) == len(dates)
    assert (df["NNTQ_inter"] == 2.).all()

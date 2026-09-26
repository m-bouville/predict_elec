"""
Tests for ``architecture``: the learning-rate schedule (fix 3), RMSNorm's eps
placement (fix 10), and the day-ahead dataset windows.

``architecture`` imports torch, so the whole file skips without it.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch",
                            reason="architecture is a torch module")

import architecture


# ---------------------------------------------------------------------------
# fix 3: warmup + cosine schedule
# ---------------------------------------------------------------------------
class TestLRWarmupCosine:
    # a day-ahead setup: ~25 batches/epoch, 22 epochs, warmup 40 steps (short
    #   enough not to be capped: 25% of 550 = 137)
    PER_EPOCH = 25
    EPOCHS = 22
    WARMUP = 40

    def total(self):
        return self.EPOCHS * self.PER_EPOCH

    def test_first_step_is_nonzero(self):
        """
        The very first optimizer step must train with a non-zero LR.  The old
        schedule returned step/warmup = 0/warmup at step 0, so epoch 0 ran at
        LR exactly 0.
        """
        m = architecture.lr_warmup_cosine(0, self.WARMUP, self.EPOCHS, self.PER_EPOCH)
        assert m > 0.0

    def test_warmup_is_monotonic_up_to_full_lr(self):
        ramp = [architecture.lr_warmup_cosine(s, self.WARMUP, self.EPOCHS,
                                              self.PER_EPOCH)
                for s in range(self.WARMUP)]
        assert all(b >= a for a, b in zip(ramp, ramp[1:]))          # non-decreasing
        peak = architecture.lr_warmup_cosine(self.WARMUP - 1, self.WARMUP,
                                             self.EPOCHS, self.PER_EPOCH)
        assert peak == pytest.approx(1.0, abs=1e-9)

    def test_decays_after_warmup_and_stays_in_unit_interval(self):
        mult = [architecture.lr_warmup_cosine(s, self.WARMUP, self.EPOCHS,
                                              self.PER_EPOCH)
                for s in range(self.total())]
        assert min(mult) >= 0.0
        assert max(mult) <= 1.0 + 1e-9
        # strictly past the warmup peak the multiplier comes back down
        mid = architecture.lr_warmup_cosine(self.total() // 2, self.WARMUP,
                                            self.EPOCHS, self.PER_EPOCH)
        assert mid < 1.0
        end = architecture.lr_warmup_cosine(self.total() - 1, self.WARMUP,
                                            self.EPOCHS, self.PER_EPOCH)
        assert end < mid                                            # cosine decay

    def test_oversized_warmup_is_clamped_not_dead(self):
        """
        The original bug: warmup expressed in the wrong unit (3000) exceeded the
        whole run, so the LR never left warmup and peaked near 0.7%.  With the
        clamp, even an absurd warmup must still reach full LR within the run.
        """
        huge = 100 * self.total()
        peak = max(architecture.lr_warmup_cosine(s, huge, self.EPOCHS,
                                                 self.PER_EPOCH)
                   for s in range(self.total()))
        assert peak == pytest.approx(1.0, abs=1e-6)

    def test_warmup_capped_at_25_percent(self):
        """A warmup longer than 25% of the run (e.g. the default 3100 steps) is
        capped: the LR peaks exactly at step total//4 - 1, then decays."""
        cap = self.total() // 4
        mult = [architecture.lr_warmup_cosine(s, 3100, self.EPOCHS, self.PER_EPOCH)
                for s in range(self.total())]
        assert mult[cap - 1] == pytest.approx(1.0, abs=1e-9)
        assert mult[cap - 2] < 1.0
        assert mult[cap + 10] < 1.0
        # same schedule as an explicit warmup of exactly 25%
        assert mult == [architecture.lr_warmup_cosine(s, cap, self.EPOCHS,
                                                      self.PER_EPOCH)
                        for s in range(self.total())]

    def test_no_warmup_branch_is_plain_cosine(self):
        start = architecture.lr_warmup_cosine(0, 0, self.EPOCHS, self.PER_EPOCH)
        end = architecture.lr_warmup_cosine(self.total(), 0, self.EPOCHS,
                                            self.PER_EPOCH)
        assert start == pytest.approx(1.0, abs=1e-9)
        assert end == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# fix 10: RMSNorm eps must guard the division by the norm
# ---------------------------------------------------------------------------
class TestRMSNorm:
    def _make(self, dim):
        # RMSNorm is a nested class inside the encoder layer: fail (not skip)
        #   if it moves, so that its tests cannot silently disappear
        RMSNorm = getattr(architecture, "RMSNorm", None) or \
            getattr(architecture.TransformerEncoderLayerWithAttn, "RMSNorm")
        return RMSNorm(dim)

    def test_zero_input_is_finite(self):
        """A zero vector has zero norm; eps must keep the output finite."""
        norm = self._make(8)
        out = norm(torch.zeros(4, 8))
        assert torch.isfinite(out).all()

    def test_unit_weight_normalises_rms_to_one(self):
        dim = 16
        norm = self._make(dim)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        x = torch.randn(32, dim) * 5.0 + 3.0
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # with default weights == 1, RMSNorm should map to unit RMS (eps tiny)
        np.testing.assert_allclose(rms.detach().numpy(),
                                   np.ones(32), atol=1e-3)


# ---------------------------------------------------------------------------
# DayAheadDataset window convention
# ---------------------------------------------------------------------------
# These tests pin the DATASET convention: origins at FORECAST_HOUR Paris time,
# the raw target spans the full horizon starting AT the origin, and the last
# valid_length steps line up with the Paris day ahead (the timestamps given to
# those steps by containers: test_predictions_and_metamodel_skip.py).  index_y_nation is a
# LIST (it is concatenated with indices_Y_regions inside __getitem__).
class TestForecastWindowConvention:
    def test_raw_target_spans_full_horizon_from_origin(self):
        F, n = 3, 48 * 6
        idx = pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")
        data = np.zeros((n, F), dtype=np.float32)
        data[:, 0] = np.arange(n)              # target column == row index
        temps = np.zeros(n, dtype=np.float32)

        pred_length = 72                       # noon -> +36 h
        ds = architecture.DayAheadDataset(
            data_subset=data, dates_subset=idx, temperatures_subset=temps,
            input_length=48, pred_length=pred_length, features_in_future=0,
            forecast_hour=12, index_y_nation=[0], indices_Y_regions=[])

        i = ds.start_indices_subset[0]
        y = np.asarray(ds[0][2]).ravel()       # raw target (length pred_length)
        assert len(y) == pred_length
        assert y[0] == i                       # first raw target IS the origin row
        assert y[-1] == i + pred_length - 1

    def test_last_valid_length_lines_up_with_day_ahead(self):
        """
        The last valid_length steps of the horizon (what containers scores) start
        pred_length - valid_length steps after a noon (Paris) origin, i.e. 00:00
        Paris of D+1, and end at 23:30 Paris of D+1 (a day without DST switch).
        """
        F, n = 3, 48 * 6
        idx = pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")
        data = np.zeros((n, F), dtype=np.float32)
        data[:, 0] = np.arange(n)
        temps = np.zeros(n, dtype=np.float32)

        pred_length, valid_length = 72, 48
        ds = architecture.DayAheadDataset(
            data_subset=data, dates_subset=idx, temperatures_subset=temps,
            input_length=48, pred_length=pred_length, features_in_future=0,
            forecast_hour=12, index_y_nation=[0], indices_Y_regions=[])

        i = ds.start_indices_subset[0]
        origin = ds.forecast_origins[0].tz_convert("Europe/Paris")
        y_valid = np.asarray(ds[0][2]).ravel()[-valid_length:]  # what containers scores

        offset = pred_length - valid_length                     # 24 (no +1)
        assert y_valid[0] == i + offset
        # first scored step is 00:00 (Paris) of the day AFTER the origin's day
        day_ahead_start = origin.normalize() + pd.DateOffset(days=1)
        assert idx[i + offset] == day_ahead_start
        assert idx[i + offset + valid_length - 1] == \
               day_ahead_start + pd.Timedelta(hours=23, minutes=30)

    @pytest.mark.parametrize("start, utc_hour", [("2022-01-10", 11),   # CET
                                                 ("2022-07-10", 10)])  # CEST
    def test_origins_are_noon_paris(self, start, utc_hour):
        """Origins at 12:00 Paris: 11:00 UTC in winter, 10:00 UTC in summer
        (/!\ were 12:00 UTC: 13:00 or 14:00 Paris, after the gate closure)."""
        n = 48 * 10
        idx = pd.date_range(start, periods=n, freq="30min", tz="UTC")
        ds = architecture.DayAheadDataset(
            data_subset=np.zeros((n, 2), np.float32), dates_subset=idx,
            temperatures_subset=np.zeros(n, np.float32), input_length=48,
            pred_length=72, features_in_future=0, forecast_hour=12,
            index_y_nation=[0], indices_Y_regions=[])
        assert len(ds) >= 7
        assert all(o.hour == utc_hour and o.minute == 0 for o in ds.forecast_origins)
        assert all(o.tz_convert("Europe/Paris").hour == 12 for o in ds.forecast_origins)

    @pytest.mark.parametrize("start", ["2022-03-20", "2022-10-23"])
    def test_one_origin_per_local_day_across_DST(self, start):
        """Around the DST switches: one origin per Paris day, and noon -> next
        Paris midnight is always 24 steps (the switch is at 2-3 am, on D+1)."""
        n = 48 * 14
        idx = pd.date_range(start, periods=n, freq="30min", tz="UTC")
        ds = architecture.DayAheadDataset(
            data_subset=np.zeros((n, 2), np.float32), dates_subset=idx,
            temperatures_subset=np.zeros(n, np.float32), input_length=48,
            pred_length=72, features_in_future=0, forecast_hour=12,
            index_y_nation=[0], indices_Y_regions=[])
        local = pd.DatetimeIndex(ds.forecast_origins).tz_convert("Europe/Paris")
        assert (local.hour == 12).all()
        assert local.normalize().is_unique
        assert (np.diff(local.tz_localize(None).normalize()) ==
                np.timedelta64(1, 'D')).all()
        for i, o in zip(ds.start_indices_subset, local):
            midnight = idx[i + 24].tz_convert("Europe/Paris")
            assert (midnight.hour, midnight.minute) == (0, 0)
            assert midnight.date() == (o + pd.DateOffset(days=1)).date()

    def test_features_in_future_window(self):
        """features_in_future (production setting): X covers input_length steps
        before the origin AND the pred_length steps of the horizon, without the
        target and region columns; y is still the horizon only."""
        n = 48 * 6
        idx = pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")
        data = np.zeros((n, 4), dtype=np.float32)
        data[:, 0] = np.arange(n)              # target
        data[:, 1] = -np.arange(n)             # a region
        data[:, 2] = 1000 + np.arange(n)       # feature a
        data[:, 3] = 2000 + np.arange(n)       # feature b
        temps = np.zeros(n, dtype=np.float32)
        input_length, pred_length = 48, 72

        for future in (0, 1):
            ds = architecture.DayAheadDataset(
                data_subset=data, dates_subset=idx, temperatures_subset=temps,
                input_length=input_length, pred_length=pred_length,
                features_in_future=future, forecast_hour=12,
                index_y_nation=[0], indices_Y_regions=[1])
            i = ds.start_indices_subset[0]
            X = np.asarray(ds[0][0])
            y = np.asarray(ds[0][2]).ravel()
            assert X.shape == (input_length + future * pred_length, 2)
            np.testing.assert_array_equal(
                X[:, 0], 1000 + np.arange(i - input_length, i + future * pred_length))
            np.testing.assert_array_equal(y, np.arange(i, i + pred_length))


# ---------------------------------------------------------------------------
# best-model saver: NaN never "best"; restore without save is explicit
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
# make_X_and_y: the validation split must not be empty
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
# make_X_and_y: baseline predictions as Series (not dicts of dicts)
# ---------------------------------------------------------------------------
def _small_bundle():
    import copy
    n = 48 * 60
    dates = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    names_cols = {'y_nation': ['consumption_GW'], 'Y_regions': ['consumption_NE_GW'],
                  'features': ['f0', 'f1'],
                  'ML_preds': ['consumption_LR', 'consumption_RF']}
    array = np.random.default_rng(0).normal(size=(n, 6)).astype(np.float32)
    data, _ = architecture.make_X_and_y(
        array, dates, np.zeros(n, np.float32), int(n * .8), int(n * .8 * .25),
        copy.deepcopy(names_cols), False, {'NE': 1.}, 30, 144, 72, True, 16)
    return data, array, dates


@pytest.mark.filterwarnings("ignore:batch_size")
def test_baseline_predictions_are_series_on_the_split_dates():
    data, array, dates = _small_bundle()
    for split in (data.train, data.valid, data.test, data.complete):
        assert list(split.dict_preds_ML) == ['LR', 'RF']
        for k, (name, series) in enumerate(split.dict_preds_ML.items()):
            assert isinstance(series, pd.Series) and series.dtype == np.float64
            assert series.index.equals(pd.DatetimeIndex(split.dates))
            # same values as before (float32 inputs, float64 as the former dicts)
            np.testing.assert_array_equal(
                series.to_numpy(),
                array[dates.get_indexer(series.index), 4 + k].astype(np.float64))
    # the DataFrame used by the metamodels is unchanged in content
    df = pd.DataFrame(data.test.dict_preds_ML)
    assert list(df.columns) == ['LR', 'RF'] and len(df) == len(data.test.dates)


# ---------------------------------------------------------------------------
# training loop: the epoch sums of the losses do not keep the autograd graph
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:batch_size")
def test_training_loss_sums_are_detached():
    import constants, containers
    data, _, _ = _small_bundle()
    params = dict(constants.NNTQ_PARAMETERS, device=torch.device('cpu'),
                  input_length=144, pred_length=72, valid_length=48,
                  model_dim=16, num_heads=2, num_layers=1, ffn_size=2,
                  num_geo_blocks=2, patch_length=48, stride=24)
    net = containers.NeuralNet(**params, len_train_data=len(data.train.loader),
                               num_features=data.num_features,
                               weights_regions={'NE': 1.})
    loss_h, dict_losses_h = architecture.subset_evolution_torch(net, data.train.loader)
    assert loss_h.shape == (48,) and torch.isfinite(loss_h).all()
    assert not loss_h.requires_grad
    assert not any(v.requires_grad for v in dict_losses_h.values())


# ---------------------------------------------------------------------------
# validation / test loss: the torch losses without gradients, on the device
# ---------------------------------------------------------------------------
def _small_net(data, **overrides):
    import constants, containers
    params = dict(constants.NNTQ_PARAMETERS, device=torch.device('cpu'),
                  input_length=144, pred_length=72, valid_length=48,
                  model_dim=16, num_heads=2, num_layers=1, ffn_size=2,
                  num_geo_blocks=2, patch_length=48, stride=24)
    params.update(overrides)
    torch.manual_seed(0)
    return containers.NeuralNet(**params, len_train_data=len(data.train.loader),
                                num_features=data.num_features,
                                weights_regions={'NE': 1.})


@pytest.mark.filterwarnings("ignore:batch_size")
@pytest.mark.parametrize("lambda_regions", [0., .05])
def test_subset_evaluation_is_the_training_loss_without_gradients(lambda_regions):
    """subset_evaluation = losses.quantile_torch (+ regions_torch) computed once
    over all the samples of the loader; float64 numpy out; no gradient.
    (/!\\ replaces subset_evolution_numpy and the numpy twins of the losses)"""
    import losses
    data, _, _ = _small_bundle()
    net = _small_net(data, lambda_regions=lambda_regions, lambda_deriv=.05,
                     lambda_median=.2, lambda_coverage=.01, lambda_cold=.2,
                     regions_to_nation=np.array([.3]))   # region std / national std
    loader = data.complete.loader     # ordered, several batches
    assert len(loader) > 1

    loss_h, dict_h = architecture.subset_evaluation(net, loader)

    V, names = 48, ['lambda_cross', 'lambda_coverage', 'lambda_deriv', 'lambda_median',
                    'smoothing_cross', 'saturation_cold_degC', 'threshold_cold_degC',
                    'lambda_cold']
    with torch.no_grad():
        batches = list(loader)
        pred, pred_regions = net.model(torch.cat([b[0] for b in batches]))
        Y_regions = torch.cat([b[1] for b in batches])
        y = torch.cat([b[2] for b in batches])
        T = torch.cat([b[3] for b in batches])
        expected, parts = losses.quantile_torch(
            pred[:, -V:], y[:, -V:, 0], net.quantiles,
            **{n: getattr(net, n) for n in names}, Tavg_current=T[:, -V:, 0])
        if lambda_regions > 0:
            expected = expected + losses.regions_torch(
                pred_regions[:, -V:], Y_regions[:, -V:],
                net.lambda_regions, net.lambda_regions_sum,
                regions_to_nation=torch.tensor([.3]))    # in national units

    assert loss_h.dtype == np.float64 and loss_h.shape == (V,)
    np.testing.assert_allclose(loss_h, expected.numpy(), rtol=1e-5)
    assert set(dict_h) == {'pinball', 'coverage', 'crossing', 'derivative', 'median'}
    for k in dict_h:
        np.testing.assert_allclose(dict_h[k], parts[k].numpy(),
                                   rtol=1e-5, atol=1e-7, err_msg=k)
    assert all(p.grad is None for p in net.model.parameters())
    assert not net.model.training


@pytest.mark.filterwarnings("ignore:batch_size")
def test_subset_evaluation_independent_of_the_batch_size():
    """(/!\\ was the average of per-batch losses: the coverage scale is a std
    over the batch, and the last batch is smaller)"""
    from torch.utils.data import DataLoader
    data, _, _ = _small_bundle()
    net = _small_net(data, lambda_coverage=.01)
    dataset = data.complete.loader.dataset
    results = [architecture.subset_evaluation(
                   net, DataLoader(dataset, batch_size=b, shuffle=False))[0]
               for b in (3, 7, len(dataset))]
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], rtol=1e-5)


@pytest.mark.filterwarnings("ignore:batch_size")
def test_training_losses_computed_in_float32(monkeypatch):
    """Under autocast (GPU) the model returns float16: the losses must be
    computed on float32 predictions (/!\\ they were computed inside autocast).
    The float16 output is simulated on CPU."""
    import losses
    data, _, _ = _small_bundle()
    net = _small_net(data)

    class Half(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, X):
            return tuple(o.half() for o in self.inner(X))
    net.model = Half(net.model)

    dtypes = []
    real = losses.quantile_torch
    monkeypatch.setattr(losses, "quantile_torch", lambda pred, *a, **k:
                        dtypes.append(pred.dtype) or real(pred, *a, **k))
    architecture.subset_evolution_torch(net, data.train.loader)
    assert dtypes and set(dtypes) == {torch.float32}


@pytest.mark.filterwarnings("ignore:batch_size")
def test_subset_evaluation_copies_to_cpu_only_at_the_end(monkeypatch):
    """One copy per result (total + 5 components), not one per batch."""
    data, _, _ = _small_bundle()
    net = _small_net(data)
    calls = []
    real_cpu = torch.Tensor.cpu
    monkeypatch.setattr(torch.Tensor, "cpu",
                        lambda self, *a, **k: calls.append(1) or real_cpu(self, *a, **k))
    architecture.subset_evaluation(net, data.complete.loader)
    assert len(data.complete.loader) > 1 and len(calls) == 6


@pytest.mark.filterwarnings("ignore:batch_size")
def test_subset_evaluation_keeps_every_tensor_on_the_model_device(monkeypatch):
    """Model on another device than the loader's (CPU) tensors: every input of
    the losses must be moved to it. The 'meta' device stands for a GPU (a
    tensor left on the CPU raises 'Expected all tensors to be on the same
    device'); copies out of it are replaced by zeros."""
    import containers
    data, _, _ = _small_bundle()
    for lambda_regions in (0., .05):
        net = _small_net(data, lambda_regions=lambda_regions)
        net.model.to('meta')
        net.device = torch.device('meta')
        real_cpu = torch.Tensor.cpu
        monkeypatch.setattr(torch.Tensor, "cpu", lambda self, *a, **k:
            torch.zeros(self.shape, dtype=self.dtype) if self.is_meta
            else real_cpu(self, *a, **k))
        loss_h, dict_h = architecture.subset_evaluation(net, data.complete.loader)
        assert loss_h.shape == (48,) and set(dict_h) >= {'pinball', 'coverage'}
        monkeypatch.undo()


@pytest.mark.filterwarnings("ignore:batch_size")
def test_training_loop_returns_the_profile_of_the_restored_model(monkeypatch):
    """The validation profile returned with the model is that of the model
    returned (the best one, restored), not of the last epoch.
    (restore is replaced by a visible change of the weights)"""
    data, _, _ = _small_bundle()
    net = _small_net(data, epochs=2)

    def restore(model, verbose=0):
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.5)
    monkeypatch.setattr(net.save_best_model, "restore", restore)

    *_, profile, parts = net.training_loop(
        data.train.loader, data.valid.loader, validate_every=1,
        display_every=999, plot_conv_every=999, verbose=0)
    expected, expected_parts = architecture.subset_evaluation(net, data.valid.loader)
    np.testing.assert_allclose(profile, expected, rtol=1e-6)
    for k in parts:
        np.testing.assert_allclose(parts[k], expected_parts[k], rtol=1e-6, atol=1e-9)


@pytest.mark.filterwarnings("ignore:batch_size")
def test_valid_and_test_forecast_from_their_first_day():
    """The first origin of valid and test is their first noon (Paris): the
    inputs before it come from the previous split (/!\\ each split was its own
    dataset: its first input_length steps, 14 days, were never forecast).
    Indices returned stay relative to the split."""
    data, array, dates = _small_bundle()
    for split in (data.valid, data.test):
        ds = split.loader.dataset
        start = pd.DatetimeIndex(split.dates)[0]
        first = ds.forecast_origins[0].tz_convert("Europe/Paris")
        assert first.hour == 12 and first - start.tz_convert("Europe/Paris") \
            < pd.Timedelta(days=1)
        X, _, y, _, idx, origin = ds[0]
        assert pd.Timestamp(int(origin), unit='s', tz='UTC') == ds.forecast_origins[0]
        assert pd.DatetimeIndex(split.dates)[int(idx)] == ds.forecast_origins[0]
        # inputs: the input_length rows before the origin, partly before the
        #   split, i.e. those of the same origin in the complete dataset
        g = dates.get_loc(ds.forecast_origins[0])
        assert g - 144 < dates.get_loc(start)
        complete = data.complete.loader.dataset
        k = complete.forecast_origins.index(ds.forecast_origins[0])
        np.testing.assert_array_equal(np.asarray(X), np.asarray(complete[k][0]))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(complete[k][2]))


@pytest.mark.filterwarnings("ignore:batch_size")
def test_patch_order_matters():
    """With the positional embedding, swapping two patches inside a pooling
    block changes the output (/!\\ without it the change was ~1e-8: attention
    and mean-pooling cannot tell the order)."""
    data, _, _ = _small_bundle()
    net = _small_net(data)
    model = net.model.eval()
    assert any(p is model.pos_embedding for p in model.parameters())
    assert model.pos_embedding.shape == (1, model.num_patches, model.dim_model)

    T, D = model.num_patches, model.dim_model
    tokens = torch.randn(2, T, D, generator=torch.Generator().manual_seed(0))
    start, end = model.block_ranges[0]
    assert end - start >= 2 and end < T          # two patches, not the last one
    swapped = tokens.clone()
    swapped[:, [start, start + 1]] = tokens[:, [start + 1, start]]

    X = torch.zeros(2, model.input_length + model.features_in_future * model.pred_length,
                    model.num_features)
    outputs = []
    for h in (tokens, swapped):
        model.patch_embed.forward = lambda X, h=h: h
        with torch.no_grad():
            outputs.append(model(X)[0])
    assert (outputs[0] - outputs[1]).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# the last whole day of a split is forecast
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("features_in_future", [0, 1])
def test_last_whole_day_of_the_data_is_forecast(features_in_future):
    """Data ending at 23:30 (Paris): the noon origin of the day before covers
    it, its pred_length rows being the last ones (/!\ `idx + pred_length <
    len`: that day was never forecast)."""
    end = pd.Timestamp("2024-11-30 23:30", tz="Europe/Paris").tz_convert("UTC")
    idx = pd.date_range(end=end, periods=48 * 10, freq="30min")
    n = len(idx)
    ds = architecture.DayAheadDataset(
        data_subset=np.arange(2 * n, dtype=np.float32).reshape(n, 2),
        dates_subset=idx, temperatures_subset=np.zeros(n, np.float32),
        input_length=48, pred_length=72, features_in_future=features_in_future,
        forecast_hour=12, index_y_nation=[0], indices_Y_regions=[])
    assert ds.forecast_origins[-1].tz_convert("Europe/Paris") == \
        pd.Timestamp("2024-11-29 12:00", tz="Europe/Paris")
    X, _, y, *_ = ds[len(ds) - 1]
    assert y.shape == (72, 1) and float(y[-1, 0]) == 2 * (n - 1)   # last row
    assert X.shape[0] == 48 + 72 * features_in_future


# ---------------------------------------------------------------------------
# NNTQ feature scaler fit on the training rows only
# ---------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:batch_size")
def test_features_scaled_on_the_training_rows_only():
    """The features trend upwards: scaled with the training rows, those have
    mean 0 / std 1 and the later rows do not (fit on all rows: both 0 / 1)."""
    import copy
    n = 48 * 60
    dates = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    names_cols = {'y_nation': ['consumption_GW'], 'Y_regions': ['consumption_NE_GW'],
                  'features': ['f0', 'f1'],
                  'ML_preds': ['consumption_LR', 'consumption_RF']}
    array = np.random.default_rng(0).normal(size=(n, 6)).astype(np.float32)
    array[:, 2:4] += np.linspace(0, 10, n, dtype=np.float32)[:, None]
    data, _ = architecture.make_X_and_y(
        array, dates, np.zeros(n, np.float32), int(n * .8), int(n * .8 * .25),
        copy.deepcopy(names_cols), False, {'NE': 1.}, 30, 144, 72, True, 16)

    X = data.complete.loader.dataset.data_subset[:, 2:4]     # after y, regions
    n_train = len(data.train.dates)
    np.testing.assert_allclose(X[:n_train].mean(0), 0., atol=1e-4)
    np.testing.assert_allclose(X[:n_train].std(0),  1., atol=1e-4)
    assert (X.mean(0) > .5).all()                              # later rows higher

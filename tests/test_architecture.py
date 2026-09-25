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
# NOTE: the 30-min off-by-one (item 2) is in containers' reconstruction
# (offset_steps): tested in test_open_bugs.py. These tests pin the DATASET
# convention that the fix relies on: the raw target spans the full horizon starting AT the origin, and
# the last valid_length steps line up with the day ahead.  index_y_nation is a
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
        pred_length - valid_length steps after a noon origin, i.e. 00:00 of D+1,
        and end at 23:30 of D+1.  (This is the convention; the +1 bug is in
        containers, tested in test_open_bugs.py.)
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
        origin = ds.forecast_origins[0]
        y_valid = np.asarray(ds[0][2]).ravel()[-valid_length:]  # what containers scores

        offset = pred_length - valid_length                     # 24 (no +1)
        assert y_valid[0] == i + offset
        # first scored step is 00:00 of the day AFTER the (noon) origin's day
        day_ahead_start = origin.normalize() + pd.Timedelta(days=1)
        assert origin + pd.Timedelta(minutes=30 * offset) == day_ahead_start
        # last scored step is 23:30 of that same day-ahead
        assert origin + pd.Timedelta(minutes=30 * (offset + valid_length - 1)) == \
               day_ahead_start + pd.Timedelta(hours=23, minutes=30)

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

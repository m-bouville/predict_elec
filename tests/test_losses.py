"""
Tests for ``losses``.

Fix 7 (cold penalty): the cold-weather weight was ``.mean()``-ed over the whole
(shuffled) batch, collapsing it to a near-constant scalar so cold days were not
actually up-weighted.  It is now one weight per sample.

The module keeps a NumPy twin of every torch loss and the file comments insist
the two "MUST remain equivalent"; we test that invariant directly.

``losses`` imports torch, so the file skips without it.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="losses is a torch module")

import losses


# ---------------------------------------------------------------------------
# cold penalty ramp + per-sample behaviour
# ---------------------------------------------------------------------------
def test_cold_penalty_ramp_numpy():
    sat, thr = -8.0, 2.0        # saturation colder than threshold
    T = np.array([[5.0], [2.0], [-3.0], [-8.0], [-20.0]])  # (B, 1)
    p = losses.penalty_nation_cold_numpy(sat, thr, T)
    # warm -> 0, at/below saturation -> 1, monotonically non-increasing in T
    assert p[0, 0] == pytest.approx(0.0)     # 5 degC, warmer than threshold
    assert p[1, 0] == pytest.approx(0.0)     # exactly at threshold
    assert p[3, 0] == pytest.approx(1.0)     # at saturation
    assert p[4, 0] == pytest.approx(1.0)     # colder than saturation, clipped
    assert 0.0 < p[2, 0] < 1.0               # on the ramp
    assert p.shape == T.shape                # same shape as input (not scalar)


def test_cold_penalty_is_per_sample_not_batch_constant():
    """
    The core of fix 7: in a batch mixing a very cold day with mild days, the
    penalty must differ across samples.  The old ``.mean()`` gave every sample
    the same value.
    """
    sat, thr = -8.0, 2.0
    T = np.full((10, 4), 12.0)      # mild
    T[0] = -10.0                    # one saturated-cold day
    p = losses.penalty_nation_cold_numpy(sat, thr, T)
    assert p[0].mean() == pytest.approx(1.0)
    assert p[1:].mean() == pytest.approx(0.0)
    assert p.std() > 0.1            # genuinely varies across the batch


def _random_case(seed, B=12, V=48, Q=5):
    rng = np.random.default_rng(seed)
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    base = rng.normal(60, 8, size=(B, V))
    # build monotone quantile predictions around the (noisy) truth
    spread = np.linspace(-6, 6, Q)
    y_pred = base[..., None] + spread[None, None, :] + rng.normal(0, 0.5,
                                                                 size=(B, V, Q))
    y_pred = np.sort(y_pred, axis=-1)          # ensure no crossing in the input
    y_true = base + rng.normal(0, 4, size=(B, V))
    Tavg = rng.normal(6, 8, size=(B, V))       # some cold, some mild
    return quantiles, y_pred, y_true, Tavg


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_torch_and_numpy_losses_agree(seed):
    """The torch and numpy quantile losses must return the same numbers."""
    quantiles, y_pred, y_true, Tavg = _random_case(seed)
    consts = dict(quantiles=quantiles, lambda_cross=0.5, lambda_coverage=0.3,
                  smoothing=0.1, saturation_cold_degC=-8.0,
                  threshold_cold_degC=2.0, lambda_cold=0.2)

    loss_np, parts_np = losses.quantile_with_crossing_numpy(
        y_pred, y_true, **consts, Tavg_current=Tavg)

    loss_t, parts_t = losses.quantile_with_crossing_torch(
        torch.tensor(y_pred), torch.tensor(y_true), **consts,
        Tavg_current=torch.tensor(Tavg))

    np.testing.assert_allclose(loss_t.detach().numpy(), loss_np,
                               rtol=1e-5, atol=1e-5)
    for key in ("pinball", "coverage", "crossing"):
        np.testing.assert_allclose(parts_t[key].detach().numpy(), parts_np[key],
                                   rtol=1e-5, atol=1e-5, err_msg=key)


def test_colder_batch_raises_pinball():
    """
    Sanity: with lambda_cold > 0, the same prediction error costs more when the
    batch is cold than when it is mild -- the whole point of the penalty.
    """
    quantiles = (0.1, 0.5, 0.9)
    B, V, Q = 8, 48, 3
    y_true = np.full((B, V), 60.0)
    y_pred = np.repeat(np.array([54.0, 60.0, 66.0])[None, None, :], B, 0)
    y_pred = np.repeat(y_pred, V, 1)
    y_pred = y_pred + 3.0            # constant bias -> non-zero pinball
    consts = dict(quantiles=quantiles, lambda_cross=0.0, lambda_coverage=0.0,
                  smoothing=0.1, saturation_cold_degC=-8.0,
                  threshold_cold_degC=2.0, lambda_cold=0.5)

    mild = np.full((B, V), 15.0)
    cold = np.full((B, V), -8.0)
    loss_mild, _ = losses.quantile_with_crossing_numpy(y_pred, y_true, **consts,
                                                       Tavg_current=mild)
    loss_cold, _ = losses.quantile_with_crossing_numpy(y_pred, y_true, **consts,
                                                       Tavg_current=cold)
    assert loss_cold.sum() > loss_mild.sum()


# ---------------------------------------------------------------------------
# derivative loss (lambda_deriv)
# ---------------------------------------------------------------------------
def test_derivative_zero_for_shifted_prediction():
    """Only the shape counts: prediction = truth + constant -> 0."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(60, 5, size=(6, 48))
    y_pred = np.repeat((y_true + 3.)[..., None], 5, axis=-1)
    np.testing.assert_allclose(losses.derivative_numpy(y_pred, y_true), 0., atol=1e-12)


def test_derivative_hand_computed_and_h0_zero():
    y_true = np.array([[0., 1., 2.]])               # slope 1
    y_pred = np.array([[[0.], [3.], [6.]]])         # slope 3 -> error 2 per step
    out = losses.derivative_numpy(y_pred, y_true)
    np.testing.assert_allclose(out, [0., 4., 4.])   # h=0 has no derivative


@pytest.mark.parametrize("seed", [0, 3])
def test_derivative_torch_numpy_agree(seed):
    _, y_pred, y_true, _ = _random_case(seed)
    np.testing.assert_allclose(
        losses.derivative_torch(torch.tensor(y_pred), torch.tensor(y_true)).numpy(),
        losses.derivative_numpy(y_pred, y_true), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 5])
def test_quantile_wrapper_torch_numpy_agree(seed):
    """quantile_* = quantile_with_crossing + lambda_deriv * derivative
    + lambda_median * |median bias|, identical in torch and numpy."""
    quantiles, y_pred, y_true, Tavg = _random_case(seed)
    consts = dict(quantiles=quantiles, lambda_cross=.07, lambda_coverage=.01,
                  lambda_deriv=.05, lambda_median=.2, smoothing_cross=.02,
                  saturation_cold_degC=-7.6, threshold_cold_degC=-.2,
                  lambda_cold=.17)
    loss_np, parts_np = losses.quantile_numpy(y_pred, y_true, **consts,
                                              Tavg_current=Tavg)
    loss_t, parts_t = losses.quantile_torch(torch.tensor(y_pred),
                                            torch.tensor(y_true), **consts,
                                            Tavg_current=torch.tensor(Tavg))
    np.testing.assert_allclose(loss_t.numpy(), loss_np, rtol=1e-5, atol=1e-5)
    for key in ("derivative", "median"):
        np.testing.assert_allclose(parts_t[key].numpy(), parts_np[key],
                                   rtol=1e-5, atol=1e-5, err_msg=key)
    # the derivative and median terms are really added
    base, _ = losses.quantile_numpy(y_pred, y_true, **dict(consts, lambda_deriv=0.,
                                    lambda_median=0.), Tavg_current=Tavg)
    np.testing.assert_allclose(loss_np, base + parts_np["derivative"] +
                               parts_np["median"], rtol=1e-9)


# ---------------------------------------------------------------------------
# regional loss (lambda_regions, lambda_regions_sum)
# ---------------------------------------------------------------------------
def test_regions_zero_when_perfect_or_disabled():
    Y = np.random.default_rng(0).normal(size=(4, 48, 3))
    np.testing.assert_allclose(losses.regions_numpy(Y, Y, .05, .5), 0.)
    np.testing.assert_allclose(losses.regions_numpy(Y + 1, Y, 0., .5), 0.)


def test_regions_hand_computed():
    """Errors +1 and -1 in two regions: MAE term 2 per step, national sum 0;
    errors +1 and +1: MAE term 2, national sum term 2 * lambda_regions_sum."""
    true = np.zeros((2, 3, 2))
    opposite = true + np.array([1., -1.])
    same     = true + np.array([1.,  1.])
    np.testing.assert_allclose(losses.regions_numpy(opposite, true, .1, .5), .1 * 2.)
    np.testing.assert_allclose(losses.regions_numpy(same,     true, .1, .5),
                               .1 * (2. + .5 * 2.))


@pytest.mark.parametrize("seed", [0, 2])
def test_regions_torch_numpy_agree(seed):
    rng = np.random.default_rng(seed)
    pred, true = rng.normal(size=(8, 48, 4)), rng.normal(size=(8, 48, 4))
    np.testing.assert_allclose(
        losses.regions_torch(torch.tensor(pred), torch.tensor(true), .05, .3).numpy(),
        losses.regions_numpy(pred, true, .05, .3), rtol=1e-6, atol=1e-8)

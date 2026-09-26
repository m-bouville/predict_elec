"""
Tests for ``losses`` (torch only: one implementation, used with gradients in
training and without in validation / test, see test_architecture.py for
``subset_evaluation``; /!\ the numpy twins of every loss are gone).

Fix 7 (cold penalty): the cold-weather weight was ``.mean()``-ed over the whole
(shuffled) batch, collapsing it to a near-constant scalar so cold days were not
actually up-weighted.  It is now one weight per sample.

``losses`` imports torch, so the file skips without it.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="losses is a torch module")

import losses


def _t(x):
    return torch.tensor(np.array(x, dtype=np.float64))   # a copy: writable


# ---------------------------------------------------------------------------
# cold penalty ramp + per-sample behaviour
# ---------------------------------------------------------------------------
def test_cold_penalty_ramp():
    sat, thr = -8.0, 2.0        # saturation colder than threshold
    T = _t([[5.0], [2.0], [-3.0], [-8.0], [-20.0]])  # (B, 1)
    p = losses.penalty_nation_cold_torch(sat, thr, T).numpy()
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
    p = losses.penalty_nation_cold_torch(sat, thr, _t(T)).numpy()
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
    return quantiles, _t(y_pred), _t(y_true), _t(Tavg)


CONSTS = dict(lambda_cross=0., lambda_coverage=0., smoothing=0.01,
              saturation_cold_degC=-8.0, threshold_cold_degC=2.0, lambda_cold=0.)


# ---------------------------------------------------------------------------
# quantile loss: pinball, coverage, crossing (hand-computed)
# ---------------------------------------------------------------------------
def test_pinball_hand_computed():
    """Prediction 3 GW below the truth: pinball = tau * 3 for each quantile."""
    quantiles = (0.1, 0.5, 0.9)
    y_true = _t(np.full((4, 2), 60.))
    y_pred = _t(np.full((4, 2, 3), 57.))
    loss, parts = losses.quantile_with_crossing_torch(
        y_pred, y_true, quantiles, **CONSTS, Tavg_current=_t(np.full((4, 2), 10.)))
    np.testing.assert_allclose(parts['pinball'].numpy(), 3 * sum(quantiles))
    np.testing.assert_allclose(loss.numpy(), parts['pinball'].numpy())


@pytest.mark.parametrize("offset", [+10., -10.])
def test_coverage_hand_computed(offset):
    """All quantiles far above (or below) every observation: coverage 1 (or 0),
    and each quantile costs lambda_coverage (alpha * w * |err| = 1)."""
    quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
    y_true = _t([[-1.], [1.]])                        # std 1 per horizon
    y_pred = _t(np.full((2, 1, 5), offset))
    _, parts = losses.quantile_with_crossing_torch(
        y_pred, y_true, quantiles, **dict(CONSTS, lambda_coverage=0.3),
        Tavg_current=_t([[10.], [10.]]))
    np.testing.assert_allclose(parts['coverage'].numpy(), 0.3 * 5, rtol=1e-6)


def test_crossing_hand_computed():
    """q10 above q50 by 2: crossing = lambda_cross * 2."""
    y_pred = _t([[[62., 60., 65.]]])
    _, parts = losses.quantile_with_crossing_torch(
        y_pred, _t([[60.]]), (0.1, 0.5, 0.9), **dict(CONSTS, lambda_cross=.5),
        Tavg_current=_t([[10.]]))
    np.testing.assert_allclose(parts['crossing'].numpy(), .5 * 2.)


def test_colder_batch_raises_pinball():
    """
    Sanity: with lambda_cold > 0, the same prediction error costs more when the
    batch is cold than when it is mild -- the whole point of the penalty.
    """
    quantiles = (0.1, 0.5, 0.9)
    B, V = 8, 48
    y_true = _t(np.full((B, V), 60.0))
    y_pred = _t(np.broadcast_to(np.array([54.0, 60.0, 66.0]) + 3., (B, V, 3)))
    consts = dict(CONSTS, lambda_cold=0.5)
    loss_mild, _ = losses.quantile_with_crossing_torch(
        y_pred, y_true, quantiles, **consts, Tavg_current=_t(np.full((B, V), 15.)))
    loss_cold, _ = losses.quantile_with_crossing_torch(
        y_pred, y_true, quantiles, **consts, Tavg_current=_t(np.full((B, V), -8.)))
    assert loss_cold.sum() > loss_mild.sum()


def test_coverage_scale_computed_once_same_result():
    """(the smoothing scale is now computed once, not per quantile): same
    numbers as recomputing it for each quantile."""
    quantiles, y_pred, y_true, Tavg = _random_case(0)
    consts = dict(CONSTS, lambda_coverage=.3, smoothing=.1, lambda_cold=.2)
    _, parts = losses.quantile_with_crossing_torch(y_pred, y_true, quantiles,
                                                   **consts, Tavg_current=Tavg)
    w_cold = 1. + .2 * losses.penalty_nation_cold_torch(-8., 2., Tavg).mean(
        dim=-1, keepdim=True)
    expected = torch.zeros(y_true.shape[1], dtype=torch.float64)
    for i, tau in enumerate(quantiles):
        scale = torch.std(y_true, dim=0, unbiased=False).clamp_min(1e-3)
        z = torch.clamp(-(y_true - y_pred[..., i]) / (.1 * scale), -20., 20.)
        cov = (w_cold * torch.sigmoid(z)).mean(dim=0) / w_cold.mean(dim=0)
        err = cov - tau
        expected += .3 * torch.where(err > 0, tau, 1 - tau) * err.abs() / (tau * (1 - tau))
    np.testing.assert_allclose(parts["coverage"].numpy(), expected.numpy(), rtol=1e-6)
    # (float32: the accumulators of quantile_with_crossing_torch)


# ---------------------------------------------------------------------------
# derivative loss (lambda_deriv)
# ---------------------------------------------------------------------------
def test_derivative_zero_for_shifted_prediction():
    """Only the shape counts: prediction = truth + constant -> 0."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(60, 5, size=(6, 48))
    y_pred = np.repeat((y_true + 3.)[..., None], 5, axis=-1)
    np.testing.assert_allclose(
        losses.derivative_torch(_t(y_pred), _t(y_true)).numpy(), 0., atol=1e-12)


def test_derivative_hand_computed_and_h0_zero():
    y_true = _t([[0., 1., 2.]])                     # slope 1
    y_pred = _t([[[0.], [3.], [6.]]])               # slope 3 -> error 2 per step
    out = losses.derivative_torch(y_pred, y_true).numpy()
    np.testing.assert_allclose(out, [0., 4., 4.])   # h=0 has no derivative


def test_quantile_wrapper_adds_derivative_and_median():
    """quantile_torch = quantile_with_crossing + lambda_deriv * derivative
    + lambda_median * |median bias|."""
    quantiles, y_pred, y_true, Tavg = _random_case(5)
    consts = dict(lambda_cross=.07, lambda_coverage=.01, lambda_deriv=.05,
                  lambda_median=.2, smoothing_cross=.02,
                  saturation_cold_degC=-7.6, threshold_cold_degC=-.2, lambda_cold=.17)
    loss, parts = losses.quantile_torch(y_pred, y_true, quantiles, **consts,
                                        Tavg_current=Tavg)
    base, _ = losses.quantile_torch(y_pred, y_true, quantiles,
                                    **dict(consts, lambda_deriv=0., lambda_median=0.),
                                    Tavg_current=Tavg)
    np.testing.assert_allclose(parts['derivative'].numpy(),
        .05 * losses.derivative_torch(y_pred, y_true).numpy(), rtol=1e-12)
    np.testing.assert_allclose(parts['median'].numpy(),
        .2 * (y_pred[..., 2] - y_true).mean(dim=0).abs().numpy(), rtol=1e-12)
    np.testing.assert_allclose(loss.numpy(), (base + parts['derivative'] +
                                              parts['median']).numpy(), rtol=1e-9)


# ---------------------------------------------------------------------------
# regional loss (lambda_regions, lambda_regions_sum)
# ---------------------------------------------------------------------------
def test_regions_zero_when_perfect_or_disabled():
    Y = _t(np.random.default_rng(0).normal(size=(4, 48, 3)))
    np.testing.assert_allclose(losses.regions_torch(Y, Y, .05, .5).numpy(), 0.)
    np.testing.assert_allclose(losses.regions_torch(Y + 1, Y, 0., .5).numpy(), 0.)


def test_regions_hand_computed():
    """Errors +1 and -1 in two regions: MAE term 2 per step, national sum 0;
    errors +1 and +1: MAE term 2, national sum term 2 * lambda_regions_sum."""
    true = np.zeros((2, 3, 2))
    opposite = true + np.array([1., -1.])
    same     = true + np.array([1.,  1.])
    np.testing.assert_allclose(
        losses.regions_torch(_t(opposite), _t(true), .1, .5).numpy(), .1 * 2.)
    np.testing.assert_allclose(
        losses.regions_torch(_t(same), _t(true), .1, .5).numpy(), .1 * (2. + .5 * 2.))


def test_no_numpy_twins_left():
    assert not [name for name in dir(losses) if name.endswith('_numpy')]

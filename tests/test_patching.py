"""
Patch embedding: the number of patches predicted in TimeSeriesTransformer.__init__
must equal what the Conv1d patch embedding actually produces, for every stride /
patch_length allowed by DISTRIBUTIONS_NNTQ (i.e. that a csv row or the search
can hold), with and without future features.

Regression: the right padding was computed from input_length only while
num_patches used input_length + pred_length, so e.g. stride 18 / patch 36 with
future features crashed with AssertionError (41, 40).

The grid is read from Bayes_search (not hard-coded), keeping only
patch_length >= stride: with shorter patches some time steps would be seen by
no patch. The search itself never samples such pairs (tested below).
"""
import itertools

import pytest

torch = pytest.importorskip("torch", reason="architecture imports torch")
optuna = pytest.importorskip("optuna", reason="Bayes_search imports optuna")

import architecture
import Bayes_search
import constants

INPUT_LENGTH, PRED_LENGTH, F = 14 * 48, 72, 4


def _values(dist):
    return range(dist.low, dist.high + 1, dist.step)


PAIRS = [(s, p) for s, p in itertools.product(
             _values(Bayes_search.DISTRIBUTIONS_NNTQ['stride']),
             _values(Bayes_search.DISTRIBUTIONS_NNTQ['patch_length']))
         if p >= s]


class _NotOffered(Exception):
    pass


class _EnumeratingTrial:
    """Stands in for optuna.Trial: `stride` fixed by the test, every other
    parameter at an allowed value; records the patch_length range offered."""
    def __init__(self, stride):
        self.stride, self.patch_range = stride, None

    def suggest_int(self, name, low, high, step=1, **k):
        if name == 'stride':
            if not (low <= self.stride <= high and (self.stride - low) % step == 0):
                raise _NotOffered
            return self.stride
        if name == 'patch_length':
            self.patch_range = range(low, high + 1, step)
        return low

    def suggest_float(self, name, low, high, **k):
        return low

    def suggest_categorical(self, name, choices):
        return choices[-1]


def _sampled_pairs():
    """Every (stride, patch_length) sample_NNTQ_parameters can return."""
    pairs = []
    for stride in _values(Bayes_search.DISTRIBUTIONS_NNTQ['stride']):
        trial = _EnumeratingTrial(stride)
        try:
            Bayes_search.sample_NNTQ_parameters(trial, dict(constants.NNTQ_PARAMETERS))
        except _NotOffered:
            continue                        # stride not offered by the search
        pairs += [(stride, p) for p in trial.patch_range]
    return pairs


def test_sampled_pairs_are_in_the_grid_and_cover_the_sequence():
    sampled = _sampled_pairs()
    assert sampled                                       # the enumeration works
    assert all(p > s for s, p in sampled)                # overlapping patches
    assert set(sampled) <= set(PAIRS)                    # all tested below


def _model(stride, patch_length, features_in_future):
    return architecture.TimeSeriesTransformer(
        num_features=F, dim_model=8, num_heads=2, num_layers=1,
        input_length=INPUT_LENGTH, patch_length=patch_length, stride=stride,
        pred_length=PRED_LENGTH, features_in_future=features_in_future,
        dropout=0., ffn_mult=1, num_quantiles=5, num_regions=2,
        num_geo_blocks=2, geo_block_ratio=1)


@pytest.mark.parametrize("features_in_future", [True, False])
@pytest.mark.parametrize("stride, patch_length", PAIRS)
def test_forward_runs_for_every_searchable_patching(stride, patch_length,
                                                   features_in_future):
    model = _model(stride, patch_length, features_in_future).eval()
    L = INPUT_LENGTH + int(features_in_future) * PRED_LENGTH
    with torch.no_grad():
        q, r = model(torch.randn(2, L, F))          # asserts inside forward()
    assert q.shape == (2, PRED_LENGTH, 5) and r.shape == (2, PRED_LENGTH, 2)
    # padding is minimal: less than one stride
    assert 0 <= model.pad_length < stride
    # the last patch ends exactly at the end of the padded sequence
    assert (model.num_patches - 1) * stride + patch_length == L + model.pad_length


@pytest.mark.parametrize("stride, patch_length", [(12, 36), (12, 48), (24, 48)])
def test_configurations_used_so_far_are_unchanged(stride, patch_length):
    """The strides/patches of the existing trials needed no padding and keep
    exactly the same number of patches as before the fix."""
    model = _model(stride, patch_length, True)
    assert model.pad_length == 0
    assert model.num_patches == \
        (INPUT_LENGTH + PRED_LENGTH - patch_length) // stride + 1


def test_block_size_assertion_message_is_formatted():
    """The consistency check on the geometric blocks reports its numbers
    (the message used to lack its f prefix: '{T}' printed literally)."""
    model = _model(24, 48, True).eval()
    model.block_sizes = [1] + list(model.block_sizes)       # sizes now wrong
    with pytest.raises(AssertionError, match=r"num_tokens \(\d+\)"):
        with torch.no_grad():
            model(torch.randn(2, INPUT_LENGTH + PRED_LENGTH, F))

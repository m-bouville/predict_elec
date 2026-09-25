"""
Tests for the configuration warnings:
* constants.py: per-head dimension (model_dim / num_heads) not a multiple of 8;
* architecture.make_X_and_y: batch_size too large for the training set
  (no training step at all because of drop_last, or fewer than 10 per epoch).

Both modules import torch, so the file skips without it.
"""
import copy
import inspect
import re
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="constants/architecture import torch")

import architecture, constants


def _per_head_warning(model_dim: int, num_heads: int) -> bool:
    """Execute constants.py with other model_dim/num_heads values (the check runs
    at import time); True if the per-head warning was emitted."""
    src = inspect.getsource(constants)
    # the substitutions must hit NNTQ_PARAMETERS, or the test would silently
    #   check the default values
    src, n1 = re.subn(r"('model_dim'\s*:\s*)\d+", rf"\g<1>{model_dim}", src, count=1)
    src, n2 = re.subn(r"('num_heads'\s*:\s*)\d+", rf"\g<1>{num_heads}", src, count=1)
    assert n1 == n2 == 1, "model_dim / num_heads not found in constants.py"
    ns = {"__name__": "constants_under_test"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile(src, "constants_under_test", "exec"), ns)
    assert ns["NNTQ_PARAMETERS"]["model_dim"] == model_dim
    assert ns["NNTQ_PARAMETERS"]["num_heads"] == num_heads
    return any("per-head dimension" in str(w.message) for w in caught)


@pytest.mark.parametrize("model_dim, num_heads, expected", [
    (500, 5, True),    # 100: not a multiple of 8
    (520, 5, False),   # 104
    (480, 5, False),   #  96
    (512, 8, False),   #  64
    (300, 6, True),    #  50
])
def test_per_head_warning(model_dim, num_heads, expected):
    assert _per_head_warning(model_dim, num_heads) == expected


def _make(batch_size):
    """make_X_and_y on ~128 training samples; returns the warning messages."""
    n = 48 * 220
    dates = pd.date_range("2022-01-01", periods=n, freq="30min", tz="UTC")
    names_cols = {'y_nation': ['consumption_GW'], 'Y_regions': ['consumption_NE_GW'],
                  'features': ['f0', 'f1'], 'ML_preds': ['consumption_LR']}
    array = np.random.default_rng(0).normal(size=(n, 5)).astype(np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data, _ = architecture.make_X_and_y(
            array, dates, np.zeros(n, np.float32), int(n * .8), int(n * .2),
            copy.deepcopy(names_cols), False, {'NE': 1.}, 30, 144, 72, True,
            batch_size)
    return data, [str(w.message) for w in caught if "batch_size" in str(w.message)]


def test_no_warning_for_reasonable_batch():
    data, msgs = _make(8)
    assert len(data.train.loader) >= 10
    assert msgs == []


def test_warning_for_few_steps_per_epoch():
    data, msgs = _make(64)
    assert 0 < len(data.train.loader) < 10
    assert len(msgs) == 1 and "optimizer steps per epoch" in msgs[0]


def test_warning_when_no_training_step():
    data, msgs = _make(4096)
    assert len(data.train.loader) == 0           # drop_last: nothing to train on
    assert len(msgs) == 1 and "NO training step" in msgs[0]

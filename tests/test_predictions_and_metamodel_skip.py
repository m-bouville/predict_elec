"""
Tests for:
* the vectorized ``DataSplit.prediction_day_ahead`` (containers.py): it must give
  exactly what the former per-(sample, horizon) Python loop gave;
  (the timestamps themselves, 30 min late, open item 2: test_open_bugs.py);
* ``do_metamodel=False`` (run.py): skipping the metamodels must keep the csv row
  schema (same columns, same order), with NaN metamodel columns and an
  unchanged ``loss_NNTQ``. That the NNTQ search passes do_metamodel=False is
  tested in test_search_objective.py.

All modules involved import torch, so the file skips without it.
"""
import copy
import inspect

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="containers/run import torch")

import architecture, containers, constants

# the small synthetic training sets have few batches: expected here (the
#   warning itself is tested in test_warnings.py)
pytestmark = pytest.mark.filterwarnings("ignore:batch_size")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
INPUT_LENGTH, PRED_LENGTH, VALID_LENGTH, MINUTES = 48 * 3, 72, 48, 30


@pytest.fixture(scope="module")
def data_and_model():
    """Small synthetic bundle + an (untrained) tiny NNTQ model, on CPU."""
    n = 48 * 200
    dates = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    names_cols = {'y_nation':  ['consumption_GW'],
                  'Y_regions': ['consumption_NE_GW', 'consumption_S_GW'],
                  'features':  [f'f{i}' for i in range(5)],
                  'ML_preds':  ['consumption_LR', 'consumption_RF',
                                'consumption_LGBM']}
    array = rng.normal(50, 8, size=(n, 11)).astype(np.float32)
    temps = rng.normal(8, 6, size=n).astype(np.float32)
    data, _ = architecture.make_X_and_y(
        array, dates, temps, int(n * .8), int(n * .8 * .25),
        copy.deepcopy(names_cols), False, {'NE': .6, 'S': .4}, MINUTES,
        INPUT_LENGTH, PRED_LENGTH, True, 16)

    params = dict(constants.NNTQ_PARAMETERS, device=torch.device('cpu'),
                  input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
                  valid_length=VALID_LENGTH, quantiles=QUANTILES,
                  model_dim=16, num_heads=2, num_layers=1, ffn_size=2,
                  num_geo_blocks=2, patch_length=48, stride=24)
    net = containers.NeuralNet(**params, len_train_data=len(data.train.loader),
                               num_features=data.num_features,
                               weights_regions={'NE': .6, 'S': .4})
    return data, net


def reference_prediction(split, model, scaler_y_nation, offset_steps,
                         valid_length, minutes_per_step, quantiles):
    """The former implementation (Python loop, one dict per record).
    `offset_steps` is taken from the implementation under test (its value, the
    open item 2, is tested separately by test_prediction_timestamps_...)."""
    model.eval()
    records = []
    min_o, max_o = np.inf, -np.inf
    with torch.no_grad():
        for (X, _, y, _, idx, origin) in split.loader:
            idx, origin = idx.numpy(), origin.numpy()
            pred = model(X)[0][:, -valid_length:].numpy()
            B, V, Q = pred.shape
            y_GW = scaler_y_nation.inverse_transform(
                y[:, -valid_length:, 0].numpy().reshape(-1, 1)).reshape(B, V)
            p_GW = scaler_y_nation.inverse_transform(
                pred.reshape(-1, Q)).reshape(B, V, Q)
            for s in range(B):
                t0 = pd.Timestamp(origin[s], unit='s', tz='UTC')
                min_o, max_o = min(min_o, origin[s]), max(max_o, origin[s])
                for h in range(V):
                    i = idx[s] + h + offset_steps
                    if i >= split.X.shape[0]:
                        continue
                    row = {"time_current": t0 + pd.Timedelta(
                               minutes=minutes_per_step * (h + offset_steps)),
                           "y_true": y_GW[s, h]}
                    for qi, tau in enumerate(quantiles):
                        row[f"q{int(100*tau)}"] = p_GW[s, h, qi]
                    records.append(row)
    df = pd.DataFrame.from_records(records).set_index("time_current").sort_index()
    return df, (pd.Timestamp(min_o, unit='s', tz='UTC'),
                pd.Timestamp(max_o, unit='s', tz='UTC'))


# ---------------------------------------------------------------------------
# vectorized prediction_day_ahead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("split_name", ["train", "valid", "test", "complete"])
def test_vectorized_prediction_matches_former_loop(data_and_model, split_name):
    data, net = data_and_model
    split = getattr(data, split_name)

    # train loader: shuffle + drop_last -> same seed so that both drop the same
    #   partial batch
    torch.manual_seed(0)
    split.prediction_day_ahead(net.model, data.scaler_y_nation,
                               torch.device('cpu'), INPUT_LENGTH, PRED_LENGTH,
                               VALID_LENGTH, MINUTES, QUANTILES)
    # offset used by the implementation: the earliest stamp is h=0 of the
    #   earliest origin
    offset_steps = (split.true_nation_GW.index.min() - split.origin_times[0]) \
                       // pd.Timedelta(minutes=MINUTES)
    assert offset_steps in (PRED_LENGTH - VALID_LENGTH, PRED_LENGTH - VALID_LENGTH + 1)

    torch.manual_seed(0)
    ref_df, ref_origins = reference_prediction(
        split, net.model, data.scaler_y_nation, offset_steps, VALID_LENGTH,
        MINUTES, QUANTILES)

    # same timestamps (including time resolution) and values, same names
    pd.testing.assert_series_equal(split.true_nation_GW, ref_df["y_true"])
    for tau in QUANTILES:
        key = f"q{int(100*tau)}"
        pd.testing.assert_series_equal(split.dict_preds_NNTQ[key], ref_df[key])
    assert split.origin_times == ref_origins


def test_prediction_index_is_sorted_and_unique(data_and_model):
    data, net = data_and_model
    split = data.complete
    split.prediction_day_ahead(net.model, data.scaler_y_nation,
                               torch.device('cpu'), INPUT_LENGTH, PRED_LENGTH,
                               VALID_LENGTH, MINUTES, QUANTILES)
    idx = split.true_nation_GW.index
    assert idx.is_monotonic_increasing and idx.is_unique
    assert str(idx.tz) == "UTC" and idx.name == "time_current"


# ---------------------------------------------------------------------------
# do_metamodel=False
# ---------------------------------------------------------------------------
def test_run_model_once_runs_metamodel_by_default():
    import run
    param = inspect.signature(run.run_model_once).parameters['do_metamodel']
    assert param.default is True      # modes other than the NNTQ search: unchanged


def _postprocess(metrics, weights):
    import run
    cov = {'q10': .01, 'q25': -.02, 'q50': .0, 'q75': .03, 'q90': -.01}
    return run.postprocess(copy.deepcopy(constants.BASELINES_PARAMETERS),
                           copy.deepcopy(constants.NNTQ_PARAMETERS),
                           copy.deepcopy(constants.METAMODEL_NN_PARAMETERS),
                           60, metrics, cov, weights, 2.5, 0)


def test_csv_row_schema_unchanged_without_metamodel():
    models = ['NNTQ', 'LR', 'RF', 'LGBM', 'meta LR', 'meta NN']
    metrics = pd.DataFrame(np.random.default_rng(0).uniform(1, 3, (6, 3)),
                           index=models, columns=['bias', 'RMSE', 'MAE'])
    weights = pd.Series([.4, .2, .2, .2], index=['NNTQ_q50', 'LR', 'RF', 'LGBM'])

    row_full, (loss_NNTQ_full, loss_meta_full) = _postprocess(metrics, weights)
    row_skip, (loss_NNTQ_skip, loss_meta_skip) = _postprocess(
        metrics.loc[models[:4]], None)          # metamodels not run

    # same columns, same order: rows can be appended to the existing csv
    assert list(row_full) == list(row_skip)
    # NNTQ loss unaffected, metamodel loss NaN
    assert loss_NNTQ_full == loss_NNTQ_skip
    assert not np.isnan(loss_meta_full) and np.isnan(loss_meta_skip)

    meta_cols = ([f"avg_weight_meta_NN_{m}" for m in ['NNTQ_q50', 'LR', 'RF', 'LGBM']]
                 + [f"test_meta_{m}_{k}" for m in ['LR', 'NN']
                    for k in ['bias', 'RMSE', 'MAE']] + ['loss_meta'])
    assert all(np.isnan(row_skip[c]) for c in meta_cols)
    # everything else identical (except the timestamp)
    for c in row_full:
        if c not in meta_cols + ['timestamp']:
            assert row_full[c] == row_skip[c] or \
                (pd.isna(row_full[c]) and pd.isna(row_skip[c])), c

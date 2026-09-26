"""
Tests for:
* ``DataSplit.prediction_day_ahead`` (containers.py): the vectorized version
  gives exactly what a per-(sample, horizon) Python loop gives; each prediction
  is stamped with the date of the row it predicts (/!\ was 30 min late), and
  only the Paris day after a noon (Paris) origin is kept, including DST days
  (46 or 50 steps) and missing rows;
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
PARIS = "Europe/Paris"


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
                         valid_length, quantiles):
    """Reference implementation: a Python loop, one dict per record. A step is
    stamped with the date of its row and kept only if that date is the Paris
    day after the origin."""
    model.eval()
    records = []
    min_o, max_o = np.inf, -np.inf
    with torch.no_grad():
        for (X, _, y, _, idx, origin) in containers.ordered_loader(split.loader):
            idx, origin = idx.numpy(), origin.numpy()
            pred = model(X)[0][:, -valid_length:].numpy()
            B, V, Q = pred.shape
            y_GW = scaler_y_nation.inverse_transform(
                y[:, -valid_length:, 0].numpy().reshape(-1, 1)).reshape(B, V)
            p_GW = scaler_y_nation.inverse_transform(
                pred.reshape(-1, Q)).reshape(B, V, Q)
            for s in range(B):
                t0 = pd.Timestamp(origin[s], unit='s', tz='UTC')
                day_ahead = (t0.tz_convert(PARIS) + pd.DateOffset(days=1)).date()
                min_o, max_o = min(min_o, origin[s]), max(max_o, origin[s])
                for h in range(V):
                    i = idx[s] + h + offset_steps
                    if i >= len(split.dates):
                        continue
                    t = split.dates[i]
                    if t.tz_convert(PARIS).date() != day_ahead:
                        continue
                    row = {"time_current": t, "y_true": y_GW[s, h]}
                    for qi, tau in enumerate(quantiles):
                        row[f"q{int(100*tau)}"] = p_GW[s, h, qi]
                    records.append(row)
    df = pd.DataFrame.from_records(records)
    df.index = pd.DatetimeIndex(df.pop("time_current"), name="time_current")
    return df.sort_index(), (pd.Timestamp(min_o, unit='s', tz='UTC'),
                             pd.Timestamp(max_o, unit='s', tz='UTC'))


# ---------------------------------------------------------------------------
# vectorized prediction_day_ahead
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("split_name", ["train", "valid", "test", "complete"])
def test_vectorized_prediction_matches_loop(data_and_model, split_name):
    data, net = data_and_model
    split = getattr(data, split_name)

    # predictions use every sample in order (train loader: shuffle + drop_last)
    torch.manual_seed(0)
    split.prediction_day_ahead(net.model, data.scaler_y_nation,
                               torch.device('cpu'), INPUT_LENGTH, PRED_LENGTH,
                               VALID_LENGTH, MINUTES, QUANTILES)
    torch.manual_seed(0)
    ref_df, ref_origins = reference_prediction(
        split, net.model, data.scaler_y_nation, PRED_LENGTH - VALID_LENGTH,
        VALID_LENGTH, QUANTILES)

    # same timestamps (including time resolution) and values, same names
    pd.testing.assert_series_equal(split.true_nation_GW, ref_df["y_true"])
    for tau in QUANTILES:
        key = f"q{int(100*tau)}"
        pd.testing.assert_series_equal(split.dict_preds_NNTQ[key], ref_df[key])
    assert split.origin_times == ref_origins


# ---------------------------------------------------------------------------
# timestamps and scored day (Paris time)
# ---------------------------------------------------------------------------
def _predict_row_numbers(dates, split_name="complete"):
    """prediction_day_ahead on a dataset whose target is the row number, with
    an untrained tiny model: returns (y_true series, split)."""
    n = len(dates)
    rng = np.random.default_rng(0)
    names_cols = {'y_nation': ['consumption_GW'], 'Y_regions': ['consumption_NE_GW'],
                  'features': ['f0', 'f1'], 'ML_preds': ['consumption_LR']}
    array = rng.normal(size=(n, 5)).astype(np.float32)
    array[:, 0] = np.arange(n)                       # target = row number
    data, _ = architecture.make_X_and_y(
        array, dates, np.zeros(n, np.float32), int(n * .8), int(n * .8 * .25),
        copy.deepcopy(names_cols), False, {'NE': 1.}, MINUTES,
        INPUT_LENGTH, PRED_LENGTH, True, 16)
    params = dict(constants.NNTQ_PARAMETERS, device=torch.device('cpu'),
                  input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
                  valid_length=VALID_LENGTH, quantiles=QUANTILES, model_dim=16,
                  num_heads=2, num_layers=1, ffn_size=2, num_geo_blocks=2,
                  patch_length=48, stride=24)
    net = containers.NeuralNet(**params, len_train_data=len(data.train.loader),
                               num_features=data.num_features,
                               weights_regions={'NE': 1.})
    split = getattr(data, split_name)
    split.prediction_day_ahead(net.model, data.scaler_y_nation,
                               torch.device('cpu'), INPUT_LENGTH, PRED_LENGTH,
                               VALID_LENGTH, MINUTES, QUANTILES)
    return split.true_nation_GW, split


def test_prediction_timestamps_match_observations():
    """y_true stamped at time t is the row whose date is t (/!\ was 30 min
    late), and the first scored step of a noon origin is 00:00 Paris of D+1."""
    dates = pd.date_range("2021-01-01", periods=48 * 60, freq="30min", tz="UTC")
    y, split = _predict_row_numbers(dates, "test")
    observed_at = dates[np.rint(y.to_numpy()).astype(int)]
    assert (y.index == observed_at).all(), \
        f"stamped {y.index[0]}, observed at {observed_at[0]}"
    first_origin = split.origin_times[0].tz_convert(PARIS)
    assert first_origin.hour == 12
    assert y.index[0] == first_origin.normalize() + pd.DateOffset(days=1)


def test_scored_days_are_paris_days_across_DST_and_missing_rows():
    """Each origin scores exactly its Paris D+1: 48 steps, 46 on the
    spring-forward Sunday; on the fall-back Sunday, RTE data lack one hour
    (2 rows), so its 48 rows are all scored. Stamps stay the observation dates
    even after the missing rows."""
    full  = pd.date_range("2021-02-01", "2021-12-01", freq="30min", tz="UTC",
                          inclusive="left")
    missing = pd.DatetimeIndex(["2021-10-31 00:00", "2021-10-31 00:30"], tz="UTC")
    dates = full.difference(missing)
    y, _ = _predict_row_numbers(dates)

    observed_at = dates[np.rint(y.to_numpy()).astype(int)]
    assert (y.index == observed_at).all()
    assert y.index.is_unique

    per_day = pd.Series(1, index=y.index.tz_convert(PARIS)).groupby(
        lambda t: t.date()).size()
    per_day.index = pd.to_datetime(per_day.index)
    inner = per_day.iloc[1:-1]                        # edges may be partial
    assert per_day[pd.Timestamp("2021-03-28")] == 46  # spring forward
    assert per_day[pd.Timestamp("2021-10-31")] == 48  # 50 minus the RTE gap
    others = inner.drop([pd.Timestamp("2021-03-28"), pd.Timestamp("2021-10-31")])
    assert (others == 48).all(), others[others != 48]
    # every Paris day scored starts at 00:00 Paris
    first = pd.Series(y.index.tz_convert(PARIS)).groupby(
        lambda i: y.index[i].tz_convert(PARIS).date()).min()
    assert all((t.hour, t.minute) == (0, 0) for t in first.iloc[1:])


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
                           60, metrics, cov, weights, 2.5, 0,
                           df_metrics_search=metrics + .5,
                           quantile_delta_coverage_test=cov)


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
                 + [f"{p}_meta_{m}_{k}" for p in ['search', 'test']
                    for m in ['LR', 'NN'] for k in ['bias', 'RMSE', 'MAE']]
                 + ['loss_meta'])
    assert all(np.isnan(row_skip[c]) for c in meta_cols)
    # everything else identical (except the timestamp)
    for c in row_full:
        if c not in meta_cols + ['timestamp']:
            assert row_full[c] == row_skip[c] or \
                (pd.isna(row_full[c]) and pd.isna(row_skip[c])), c


# ---------------------------------------------------------------------------
# predictions: every origin, including the training split
# ---------------------------------------------------------------------------
def test_train_predictions_cover_every_origin(data_and_model):
    """The training loader shuffles and drops its last partial batch:
    predictions must use every sample anyway."""
    data, net = data_and_model
    split = data.train
    assert split.loader.drop_last          # the case this protects against
    split.prediction_day_ahead(net.model, data.scaler_y_nation,
                               torch.device('cpu'), INPUT_LENGTH, PRED_LENGTH,
                               VALID_LENGTH, MINUTES, QUANTILES)
    ds     = split.loader.dataset
    offset = PRED_LENGTH - VALID_LENGTH
    local  = split.dates.tz_convert(PARIS)
    N = split.X.shape[0]
    # steps of each origin's Paris D+1 within its window (46 on the DST Sunday)
    expected = sum(
        sum(local[r].date() == (o.tz_convert(PARIS) + pd.DateOffset(days=1)).date()
            for r in range(s + offset, min(N, s + offset + VALID_LENGTH)))
        for s, o in zip(ds.start_indices_subset, ds.forecast_origins))
    assert len(split.true_nation_GW) == expected    # no origin dropped

"""
The objectives of the Bayesian searches and their multi-run logic.

* ``run.loss_NNTQ``: quantile-coverage gaps (weighted, sorted, largest first)
  + worst days;
* ``run.loss_meta``: weighted bias / RMSE / MAE of the models (NNTQ weight 0);
* ``Bayes_search.run_Bayes_search`` objective, with ``run.run_model_once``
  replaced by a script of losses: single runs during warm-up, extra runs only
  for promising trials, early stop, ``clean_avg`` (min and max dropped), csv row,
  one NNTQ variant per run in the meta search;
* the history: csv rows loaded into the study before the new trials (values
  loss_<stage>, numbering, warm-up and extra-run threshold from them);
* each stage samples its own bundles and passes what it sampled to
  run_model_once (NNTQ search: baselines and meta-NN at base values, 1 epoch).

(Bayes_all KeyError: test_open_bugs.py.)
"""
import copy

import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")            # the search ends with plots

torch  = pytest.importorskip("torch", reason="run imports torch")
optuna = pytest.importorskip("optuna")

import Bayes_search
import constants
import run
from constants import Stage

optuna.logging.set_verbosity(optuna.logging.WARNING)

QS = ['q10', 'q25', 'q50', 'q75', 'q90']


# ---------------------------------------------------------------------------
# loss_NNTQ
# ---------------------------------------------------------------------------
def test_loss_NNTQ_zero_when_perfect():
    assert run.loss_NNTQ({q: 0. for q in QS}, 0.) == 0.


def test_loss_NNTQ_hand_computed():
    # all quantiles 1% too high: spread 0, bias = mean of the 6 terms = 0.05/6
    # weighted |gaps| sorted: .02 .02 .015 .015 .01 .00833 0, weights 7..1 (/28)
    cov = {q: 0.01 for q in QS}
    expected = 100 * (7*.02 + 6*.02 + 5*.015 + 4*.015 + 3*.01 + 2*(.05/6) + 0) / 28
    assert run.loss_NNTQ(cov, 0.) == pytest.approx(expected, abs=1e-3)
    # worst days: + scale * weight_worst_days = 2 per GW
    assert run.loss_NNTQ(cov, 1.5) == pytest.approx(expected + 3., abs=1e-3)


def test_loss_NNTQ_spread_and_dominant_gap():
    """Same |gaps|: a too-narrow distribution (q10 too high, q90 too low) costs
    more than a shifted one, through the spread term (weight 2)."""
    narrow  = {'q10': .03, 'q25': .02, 'q50': 0., 'q75': -.02, 'q90': -.03}
    shifted = {q: abs(v) for q, v in narrow.items()}
    assert run.loss_NNTQ(narrow, 0.) > run.loss_NNTQ(shifted, 0.) > 0.
    # the worst quantile dominates: one large gap costs more than spread-out ones
    one_big  = {**{q: 0. for q in QS}, 'q10': .05}
    diffused = {q: .01 for q in QS}
    assert run.loss_NNTQ(one_big, 0.) > run.loss_NNTQ(diffused, 0.)


@pytest.mark.parametrize("cov, worst, expected", [
    # shifted down, wider than it should be: spread = ((q90-q10) + (q75-q25))/2
    #   = (.05 + .02)/2 = +.035; bias = mean(-.06 -.04 -.05 -.02 -.01 +.035)
    #   = -.145/6. Weighted |terms| (q10 .12, q25 .06, q50 .05, q75 .03,
    #   q90 .02, spread 2 x .035, bias .145/6), sorted, weights 7..1 (/28)
    ({'q10': -.06, 'q25': -.04, 'q50': -.05, 'q75': -.02, 'q90': -.01}, .5,
     100 * (7*.12 + 6*.07 + 5*.06 + 4*.05 + 3*.03 + 2*(.145/6) + 1*.02) / 28
     + 100 * .02 * .5),
    # shifted down, too narrow: spread = (-.09 - .06)/2 = -.075 -> |.075|;
    #   bias = mean(.01 -.01 -.06 -.07 -.08 +.075) = -.0225
    #   terms: q10 .02, q25 .015, q50 .06, q75 .105, q90 .16, spread .15, bias .0225
    ({'q10': .01, 'q25': -.01, 'q50': -.06, 'q75': -.07, 'q90': -.08}, 0.,
     100 * (7*.16 + 6*.15 + 5*.105 + 4*.06 + 3*.0225 + 2*.02 + 1*.015) / 28),
], ids=["wide", "narrow"])
def test_loss_NNTQ_hand_computed_spread_and_negative_bias(cov, worst, expected):
    """Pins each term: spread weight 2, spread = mean of (q90-q10) and
    (q75-q25) (the /2), |spread|, bias = mean of the 5 gaps and |spread| with
    weight 1 taken in absolute value (negative here), worst days x 2."""
    assert run.loss_NNTQ(cov, worst) == pytest.approx(expected, abs=1e-3)


def test_worst_days_average_the_top_n_days():
    """avg_abs_diff (the worst-days term of loss_NNTQ) is the mean over the
    top_n worst days (/!\\ was the mean over all days); days with a DST switch
    (46 or 50 steps) are left out."""
    import utils
    from constants import Split
    local = pd.date_range("2022-03-01", "2022-04-05", freq="30min",
                          tz="Europe/Paris", inclusive="left")
    idx = local.tz_convert("UTC")
    naive = local.tz_localize(None).normalize()           # calendar days, DST-proof
    day_number = (naive - naive[0]).days                  # 0..34
    y_true = pd.Series(50., index=idx)
    y_pred = pd.Series(50. + day_number + 1., index=idx)   # day k: error k+1 GW
    frame = pd.Series(10., index=idx)
    worst, avg = utils.worst_days_by_loss(
        Split.test, y_true, y_pred, frame, frame * 0, num_steps_per_day=48, top_n=5)
    # 35 days, 27 March (error 27) is a 46-step day: worst 5 are 35, 34, 33, 32, 31
    assert avg == pytest.approx((35 + 34 + 33 + 32 + 31) / 5)
    assert len(worst) == 5


# ---------------------------------------------------------------------------
# loss_meta
# ---------------------------------------------------------------------------
MODELS = ['NNTQ', 'LR', 'RF', 'LGBM', 'meta_LR', 'meta_NN']


def _flat(values):
    """{model: (bias, RMSE, MAE)} -> {'test_<model>_<metric>': value}"""
    return {f"test_{m}_{k}": v for m, vals in values.items()
            for k, v in zip(['bias', 'RMSE', 'MAE'], vals)}


def test_loss_meta_all_ones_is_one():
    assert run.loss_meta(_flat({m: (1., 1., 1.) for m in MODELS})) == \
        pytest.approx(1.)


def test_loss_meta_ignores_NNTQ_and_uses_abs_bias():
    base = {m: (1., 2., 1.5) for m in MODELS}
    ref  = run.loss_meta(_flat(base))
    assert run.loss_meta(_flat({**base, 'NNTQ': (9., 9., 9.)})) == ref  # weight 0
    assert run.loss_meta(_flat({**base, 'LR': (-1., 2., 1.5)})) == ref  # |bias|


def test_loss_meta_hand_computed():
    # model weights LR 1, RF 1, LGBM 1, meta_LR 1.5, meta_NN 2 (sum 6.5);
    # metric weights bias 2, RMSE 1, MAE 1 (sum 4)
    vals = {'NNTQ': (5., 5., 5.), 'LR': (1., 3., 2.), 'RF': (1., 3., 2.),
            'LGBM': (1., 3., 2.), 'meta_LR': (.5, 2., 1.), 'meta_NN': (0., 1., 1.)}
    w = {'LR': 1, 'RF': 1, 'LGBM': 1, 'meta_LR': 1.5, 'meta_NN': 2}
    avg = [sum(w[m] * abs(vals[m][k]) for m in w) / 6.5 for k in range(3)]
    expected = (2 * avg[0] + avg[1] + avg[2]) / 4
    assert run.loss_meta(_flat(vals)) == pytest.approx(expected, abs=1e-4)


def test_loss_meta_meta_NN_counts_most():
    base = {m: (1., 1., 1.) for m in MODELS}
    worse_LR     = run.loss_meta(_flat({**base, 'LR':      (1., 2., 1.)}))
    worse_metaNN = run.loss_meta(_flat({**base, 'meta_NN': (1., 2., 1.)}))
    assert worse_metaNN > worse_LR > 1.


# ---------------------------------------------------------------------------
# objective: multi-run logic (run_model_once replaced by a script)
# ---------------------------------------------------------------------------
SEED = 3
# loss of (trial, run): warm-up trials 0-2 single run; then
#   t3: 1.5 > best 0.9 + wiggle 0.1         -> 1 run
#   t4: 0.95 <= 1.0 -> all 4 runs, clean_avg(.95 .8 .7 .75) = mean(.8, .75)
#   t5: 0.85 <= 0.875, then 1.5, 1.6: avg without the worst (1.175) > 0.875
#       -> early stop after 3 runs, clean_avg = 1.5
SCRIPT = {0: [1.0], 1: [2.0], 2: [0.9], 3: [1.5],
          4: [0.95, 0.8, 0.7, 0.75], 5: [0.85, 1.5, 1.6, 0.1]}


@pytest.fixture
def search(tmp_path, monkeypatch):
    """_search(stage, num_trials, script=SCRIPT, history=None) runs the search
    with run_model_once replaced by `script` ({trial number: losses of its
    runs}, loss_meta; loss_NNTQ = 10 x). `history`: rows already in the csv
    (full rows, as run.postprocess writes them), then also the template of the
    new rows. Returns (calls, csv, values of all the study's trials);
    _search.trials: the study's trials."""
    calls = []
    state = {'script': SCRIPT, 'template': None}

    def fake_run_model_once(**kw):
        trial = kw['run_id']                            # = trial.number
        i     = sum(c['trial'] == trial for c in calls)  # runs already done
        calls.append(dict(trial=trial, run=i, seed=kw['seed'],
                          variant=kw.get('NNTQ_variant'),
                          num_variants=kw.get('num_NNTQ_variants'),
                          do_metamodel=kw.get('do_metamodel'),
                          save_cache_baselines=kw.get('save_cache_baselines'),
                          baseline=copy.deepcopy(kw['baseline_parameters']),
                          nntq=copy.deepcopy(kw['NNTQ_parameters']),
                          meta=copy.deepcopy(kw['metamodel_NN_parameters'])))
        loss = state['script'][trial][i]
        row = {'run': trial, 'num_runs': 1,         # as run.postprocess
               'loss_NNTQ': loss * 10, 'loss_meta': loss}
        if state['template'] is not None:           # full row (history)
            row = dict(state['template'], **row)
        return (None, row, None, None, None, (20, 0.),
                (loss * 10, loss))          # (loss_NNTQ, loss_meta)

    monkeypatch.setattr(run, "run_model_once", fake_run_model_once)
    monkeypatch.setattr(Bayes_search, "plot_optuna", lambda *a, **k: None)

    def _search(stage, num_trials=len(SCRIPT), script=SCRIPT, history=None):
        csv = tmp_path / f"search_{stage.value}.csv"
        state['script'] = script
        if history is not None:
            for row in history:                 # as the search writes them
                run.append_csv_row(pd.DataFrame([row]), str(csv))
            state['template'] = history[0]
        study_values = []
        orig_optimize = optuna.Study.optimize

        def optimize(self, func, n_trials, **k):
            orig_optimize(self, func, n_trials=n_trials, **k)
            study_values.extend(t.value for t in self.trials)
            _search.trials = list(self.trials)
        monkeypatch.setattr(optuna.Study, "optimize", optimize)

        Bayes_search.run_Bayes_search(
            stage=stage, num_trials=num_trials,
            base_baseline_params=constants.BASELINES_PARAMETERS,
            base_NNTQ_params=dict(constants.NNTQ_PARAMETERS),
            base_meta_NN_params=dict(constants.METAMODEL_NN_PARAMETERS),
            dict_input_csv_fnames={}, trials_csv_path=str(csv),
            minutes_per_step=30, train_split_fraction=.8, valid_ratio=.25,
            forecast_hour=12, seed=SEED, cache_dir=str(tmp_path),
            num_runs      ={Stage.NNTQ: 4, Stage.meta: 4, Stage.all: 4},
            min_num_trials={Stage.NNTQ: 2, Stage.meta: 2, Stage.all: 2},
            wiggle_value  ={Stage.NNTQ: 1., Stage.meta: .1, Stage.all: 1.1})
        return calls, pd.read_csv(csv), study_values
    return _search


def _runs_per_trial(calls):
    return pd.Series([c['trial'] for c in calls]).value_counts().sort_index().tolist()


def test_meta_warmup_single_runs_then_extra_runs_and_early_stop(search):
    calls, df, values = search(Stage.meta)
    assert _runs_per_trial(calls) == [1, 1, 1, 1, 4, 3]
    assert values == pytest.approx([1.0, 2.0, 0.9, 1.5, 0.775, 1.5])
    # csv: averaged losses and number of runs for the multi-run trials
    assert df['loss_meta'].tolist()[4:] == pytest.approx([0.775, 1.5])
    assert df['num_runs'].tolist()[4:] == [4, 3]
    assert df['loss_NNTQ'].tolist()[4:] == pytest.approx([7.75, 15.])


def test_meta_one_NNTQ_variant_per_run(search):
    calls, _, _ = search(Stage.meta)
    assert len({c['seed'] for c in calls}) == len(calls)   # a new seed per run
    assert all(c['variant'] == c['run'] and c['num_variants'] == 4 for c in calls)
    assert all(c['do_metamodel'] for c in calls)
    assert all(c['save_cache_baselines'] for c in calls)


def test_NNTQ_search_no_variant_no_metamodel(search, monkeypatch):
    # no sampling: the scripted losses stay independent of the parameters
    monkeypatch.setattr(Bayes_search, "sample_NNTQ_parameters",
                        lambda trial, p: dict(p))
    calls, df, values = search(Stage.NNTQ)
    assert all(c['variant'] is None and c['do_metamodel'] is False for c in calls)
    # same script, scaled by 10 (loss_NNTQ), wiggle 1.0: same decisions
    assert _runs_per_trial(calls) == [1, 1, 1, 1, 4, 3]
    assert values == pytest.approx([10., 20., 9., 15., 7.75, 15.])


# ---------------------------------------------------------------------------
# the search refuses to append to a csv with other columns
# ---------------------------------------------------------------------------
def test_search_appends_through_the_checked_writer(search, monkeypatch):
    import run
    written = []
    real = run.append_csv_row
    monkeypatch.setattr(run, "append_csv_row",
                        lambda df, path, **k: written.append(path) or real(df, path, **k))
    calls, df, _ = search(Stage.meta, num_trials=2)
    assert len(written) == 2 and len(df) == 2



# ---------------------------------------------------------------------------
# history: the rows already in the csv enter the study before the new trials
# ---------------------------------------------------------------------------
HISTORY_META = [1.0, 2.0, 0.9]          # loss_meta of the csv rows (NNTQ: x10)
# new trials, numbered after the 3 rows, i.e. past warm-up (min_num_trials 2):
#   t3: 1.05 > loaded best 0.9 + wiggle 0.1 -> 1 run
#       (with no history: warm-up, or threshold inf -> not a single run)
#   t4: 0.95 <= 1.0                        -> 4 runs, clean_avg = 0.775
HISTORY_SCRIPT = {3: [1.05, 9., 9., 9.], 4: [0.95, 0.8, 0.7, 0.75]}


def _history_rows():
    _postprocess_row = pytest.importorskip("test_run")._postprocess_row
    rows = []
    for k, loss in enumerate(HISTORY_META):
        row, _ = _postprocess_row()
        row.update(run=k, loss_NNTQ=loss * 10, loss_meta=loss)
        rows.append(row)
    return rows


@pytest.mark.parametrize("stage", [Stage.NNTQ, Stage.meta])
def test_history_is_loaded_before_the_new_trials(search, stage):
    """The csv rows are the study's first trials, valued loss_<stage>; the new
    trials are numbered from N (the history counts in the warm-up) and the
    extra-run threshold uses the best loaded value."""
    scale = 10 if stage == Stage.NNTQ else 1
    calls, df, values = search(stage, num_trials=2, script=HISTORY_SCRIPT,
                               history=_history_rows())
    trials = search.trials
    n = len(HISTORY_META)
    assert [t.number for t in trials] == list(range(n + 2))
    assert [t.state for t in trials[:n]] == \
        [optuna.trial.TrialState.COMPLETE] * n
    assert values[:n] == pytest.approx([scale * v for v in HISTORY_META])
    assert calls[0]['trial'] == n                         # first new trial: N
    assert _runs_per_trial(calls) == [1, 4]
    assert values[n:] == pytest.approx([scale * 1.05, scale * 0.775])
    assert len(df) == n + 2 and df['num_runs'].tolist()[n:] == [1, 4]


# ---------------------------------------------------------------------------
# each stage samples its own bundles only, and uses what it sampled
# ---------------------------------------------------------------------------
def _meta_key(name):
    return name[len('metaNN_'):]


def test_NNTQ_search_samples_NNTQ_only(search):
    """NNTQ search: the NNTQ bundle passed to run_model_once holds the trial's
    sampled values; baselines and meta-NN stay at their base values, except
    meta-NN epochs = 1."""
    calls, _, _ = search(Stage.NNTQ, num_trials=3)
    base_nntq = dict(constants.NNTQ_PARAMETERS)
    for c in calls:
        params = search.trials[c['trial']].params
        assert params and set(params) <= set(Bayes_search.DISTRIBUTIONS_NNTQ)
        assert {k: c['nntq'][k] for k in params} == params
        assert {k: v for k, v in c['nntq'].items() if k not in params} == \
               {k: v for k, v in base_nntq.items() if k not in params}
        assert c['baseline'] == constants.BASELINES_PARAMETERS
        assert c['meta'] == dict(constants.METAMODEL_NN_PARAMETERS, epochs=1)
    assert any(c['nntq'] != base_nntq for c in calls)


def test_meta_search_samples_baselines_and_meta_NN_only(search):
    """Meta search: the baselines and meta-NN bundles hold the trial's sampled
    values (<model>_<key>, metaNN_<key>, num_cells_0/_1, epochs sampled, not
    1); the NNTQ stays at its base values."""
    calls, _, _ = search(Stage.meta, num_trials=3)
    for c in calls:
        params = search.trials[c['trial']].params
        assert set(params) <= set(Bayes_search.DISTRIBUTIONS_BASELINES) | \
                              set(Bayes_search.DISTRIBUTIONS_METAMODEL_NN)
        assert c['nntq'] == dict(constants.NNTQ_PARAMETERS)
        meta = {k: v for k, v in params.items() if k.startswith('metaNN_')}
        assert c['meta']['num_cells'] == [meta.pop('metaNN_num_cells_0'),
                                          meta.pop('metaNN_num_cells_1')]
        assert meta and {_meta_key(k): c['meta'][_meta_key(k)] for k in meta} == \
               {_meta_key(k): v for k, v in meta.items()}
        assert c['meta']['epochs'] == params['metaNN_epochs'] >= 10
        base = {k: v for k, v in params.items() if not k.startswith('metaNN_')}
        assert base
        for name, value in base.items():
            model, key = name.split('_', 1)
            assert c['baseline'][model][key] == value, name

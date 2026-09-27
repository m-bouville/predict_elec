"""
Compatibility of the ranges in Bayes_search.py.

1. Every value the sampling functions can draw lies inside DISTRIBUTIONS_* (the
   distributions used to reload the csv): otherwise the next run crashes when
   `load_frozen_trials` / `study.add_trial` reads back the rows just written.
2. Every trial of the existing csv files fits DISTRIBUTIONS_* and, once loaded
   into a study, the sampling functions still work with the TPE sampler used by
   the search. With TPE, a categorical parameter must keep exactly the choices of
   DISTRIBUTIONS_* (removing or adding a choice raises "CategoricalDistribution
   does not support dynamic value space"), except when it has a single choice:
   the value is then fixed without consulting the sampler.
   Numeric ranges may differ from DISTRIBUTIONS_* (as long as sampled values fit).
3. Reloading the csv keeps the significant digits of small values (learning
   rates, weight decays are saved x1e6), the csv being written by
   ``run.append_csv_row`` (its float_format), as in the search.
4. The sampling functions put each sampled value into the parameter dicts
   actually used (key mapping: metaNN_ prefix, num_cells_0/_1, <model>_<key>),
   and leave the other entries (and their input) unchanged.

Bayes_search imports run -> torch, and optuna: skipped without them.
"""
import copy
import os

import pytest

pytest.importorskip("torch", reason="Bayes_search imports run -> torch")
optuna = pytest.importorskip("optuna")

import Bayes_search as bs
import constants
import run
from constants import Stage

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DISTRIBUTIONS = bs.DISTRIBUTIONS_BASELINES | bs.DISTRIBUTIONS_NNTQ | \
                    bs.DISTRIBUTIONS_METAMODEL_NN
NUM_SAMPLES = 300


def _sample(trial, stage) -> bool:
    """Sample as `objective` does for this stage (Stage.all: everything).
    Returns False if the sampling pruned the trial (e.g. a rejected batch_size)."""
    try:
        if stage in (Stage.meta, Stage.all):
            bs.sample_baseline_parameters(trial,
                                          copy.deepcopy(constants.BASELINES_PARAMETERS))
            bs.sample_metamodel_NN_parameters(trial,
                                              dict(constants.METAMODEL_NN_PARAMETERS))
        if stage in (Stage.NNTQ, Stage.all):
            bs.sample_NNTQ_parameters(trial, dict(constants.NNTQ_PARAMETERS))
    except optuna.TrialPruned:
        return False
    return True


def _tell(study, trial, completed: bool) -> None:
    if completed:
        study.tell(trial, 0.)
    else:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)


def _contains(dist, value) -> bool:
    return dist._contains(dist.to_internal_repr(value))


# ---------------------------------------------------------------------------
# 1. sampled values fit the distributions used to reload the csv
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", [Stage.NNTQ, Stage.meta])
def test_sampled_values_fit_reload_distributions(stage):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    problems = set()
    for _ in range(NUM_SAMPLES):
        trial = study.ask()
        completed = _sample(trial, stage)
        for name, value in trial.params.items():
            dist = ALL_DISTRIBUTIONS.get(name)
            if dist is None:
                problems.add(f"{name}: sampled but missing from DISTRIBUTIONS_*")
            elif not _contains(dist, value):
                problems.add(f"{name}: sampled {value!r} outside {dist}")
        _tell(study, trial, completed)
    assert not problems, "\n".join(sorted(problems))


@pytest.mark.parametrize("stage", [Stage.NNTQ, Stage.meta])
def test_categorical_choices_identical(stage):
    """A categorical suggest must use exactly the choices of DISTRIBUTIONS_*
    (single-choice categoricals excepted: TPE is not consulted for them)."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    problems = set()
    for _ in range(50):
        trial = study.ask()
        completed = _sample(trial, stage)
        for name, dist in trial.distributions.items():
            ref = ALL_DISTRIBUTIONS.get(name)
            if isinstance(dist, optuna.distributions.CategoricalDistribution) and \
                    ref is not None and len(dist.choices) > 1 and \
                    tuple(dist.choices) != tuple(ref.choices):
                problems.add(f"{name}: sampled from {dist.choices}, "
                             f"DISTRIBUTIONS_* has {ref.choices}")
        _tell(study, trial, completed)
    assert not problems, "\n".join(sorted(problems))


# ---------------------------------------------------------------------------
# 2. the existing csv files still load, and sampling still works after loading
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", [Stage.NNTQ, Stage.meta])
def test_csv_loads_and_sampling_works(stage):
    path = os.path.join(ROOT, f"parameter_search_{stage.value}.csv")
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not found")

    trials = bs.load_frozen_trials(path, ALL_DISTRIBUTIONS, stage)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        # TPE, as in the search: unlike RandomSampler, it reads the loaded trials
    for t in trials:
        study.add_trial(t)               # validates each value against its distribution

    for _ in range(50):                  # then sample as a new search would
        trial = study.ask()
        completed = _sample(trial, stage)  # raises on incompatible categorical choices
        _tell(study, trial, completed)


# ---------------------------------------------------------------------------
# 3. reloading keeps the significant digits of small values
#    (round(9) turned weight_decay 1.312e-9 into 1e-9)
# ---------------------------------------------------------------------------
def test_reloaded_weight_decay_keeps_its_significant_digits(tmp_path):
    import pandas as pd
    _postprocess_row = pytest.importorskip("test_run")._postprocess_row
    nntq = copy.deepcopy(constants.NNTQ_PARAMETERS)
    nntq.update(weight_decay=1.312e-9, learning_rate=0.0032)
    meta = copy.deepcopy(constants.METAMODEL_NN_PARAMETERS)
    meta.update(weight_decay=2.3449e-05, learning_rate=0.0045)
    row, _ = _postprocess_row(nntq, meta)
    row.update(loss_NNTQ=20., loss_meta=2.3)
    csv = tmp_path / "search.csv"
    run.append_csv_row(pd.DataFrame([row]), str(csv))   # the search's writer

    trials = bs.load_frozen_trials(
        str(csv), bs.DISTRIBUTIONS_BASELINES | bs.DISTRIBUTIONS_NNTQ |
        bs.DISTRIBUTIONS_METAMODEL_NN, Stage.NNTQ)
    p = trials[0].params
    assert p['weight_decay'] == pytest.approx(1.312e-9, rel=1e-6)   # was 1e-9
    assert p['learning_rate'] == 0.0032
    assert p['metaNN_weight_decay'] == pytest.approx(2.3449e-05, rel=1e-6)
    assert p['metaNN_learning_rate'] == 0.0045                    # on its grid


def test_numeric_rf_max_features_reloads(tmp_path):
    """A numeric RF max_features (a categorical choice such as '0.4') next to
    'sqrt' in the csv reloads as its choice (a guard: the column is then written
    as text, so float_format does not turn 0.4 into 0.400000)."""
    import pandas as pd
    _postprocess_row = pytest.importorskip("test_run")._postprocess_row
    rows = []
    for max_features in ('sqrt', 0.4):     # mixed column: read back as strings
        base = copy.deepcopy(constants.BASELINES_PARAMETERS)
        base['RF']['max_features'] = max_features
        row, _ = _postprocess_row(base=base)
        row.update(loss_NNTQ=20., loss_meta=2.3)
        rows.append(row)
    csv = tmp_path / "search.csv"
    run.append_csv_row(pd.DataFrame(rows), str(csv))   # the search's writer
        # (one row at a time, as the search does, 0.4 is written 0.400000 and
        #  not reloaded: test_open_bugs_B.py)
    trials = bs.load_frozen_trials(str(csv), ALL_DISTRIBUTIONS, Stage.meta)
    assert [t.params['RF_max_features'] for t in trials] == ['sqrt', '0.4']



def test_rows_outside_the_current_ranges_are_skipped(tmp_path):
    """A row whose value is no longer allowed (a batch size dropped from the
    choices, a learning rate above the new maximum) is left out; the others
    load (/!\ add_trial raised on the first such row: the search did not
    start)."""
    import pandas as pd
    _postprocess_row = pytest.importorskip("test_run")._postprocess_row
    nntq = copy.deepcopy(constants.NNTQ_PARAMETERS)
    rows = []
    for batch_size, learning_rate in [(96, .002), (10_000, .002), (96, 10.)]:
        nntq.update(batch_size=batch_size, learning_rate=learning_rate)
        row, _ = _postprocess_row(copy.deepcopy(nntq))
        row.update(loss_NNTQ=20., loss_meta=2.3)
        rows.append(row)
    csv = tmp_path / "search.csv"
    for row in rows:                       # through the search's writer
        run.append_csv_row(pd.DataFrame([row]), str(csv))

    trials = bs.load_frozen_trials(str(csv), ALL_DISTRIBUTIONS, Stage.NNTQ)

    assert [t.params['batch_size'] for t in trials] == [96]
    study = optuna.create_study()
    for t in trials:
        study.add_trial(t)                 # the ones loaded are valid


# ---------------------------------------------------------------------------
# 4. sampled values reach the parameter dicts actually used
# ---------------------------------------------------------------------------
class _ScriptedTrial:
    """Stands for an optuna trial: every suggest_* returns the value scripted
    for its name (all different from the defaults in constants), and records
    the names asked."""
    def __init__(self, values):
        self.values, self.asked = values, []

    def _get(self, name, *a, **k):
        self.asked.append(name)
        return self.values[name]
    suggest_int = suggest_float = suggest_categorical = _get


NNTQ_SCRIPT = dict(
    use_ML_features=1, stride=12, patch_length=36, input_length=576, epochs=12,
    batch_size=64, learning_rate=0.001, weight_decay=2e-8, dropout=0.06,
    lambda_cross=0.01, lambda_coverage=0.02, lambda_deriv=0.03,
    lambda_median=0.04, smoothing_cross=0.05, threshold_cold_degC=1.5,
    saturation_cold_degC=-3.5, lambda_cold=0.09, lambda_regions=0.016,
    lambda_regions_sum=0.2, ffn_size=3, num_heads=5, model_dim=340,
    num_layers=5, num_geo_blocks=8, warmup_steps=1200, patience=7,
    min_delta=0.02)

META_SCRIPT = dict(
    metaNN_epochs=15, metaNN_batch_size=32, metaNN_learning_rate=0.002,
    metaNN_weight_decay=3e-6, metaNN_dropout=0.05, metaNN_num_cells_0=44,
    metaNN_num_cells_1=8, metaNN_patience=5, metaNN_factor=0.8)

BASELINES_SCRIPT = dict(
    LR_type='lasso', LR_alpha=0.9,
    RF_n_estimators=300, RF_max_depth=20, RF_min_samples_leaf=9,
    RF_min_samples_split=15, RF_max_features='0.4',
    LGBM_boosting_type='dart', LGBM_num_leaves=31, LGBM_max_depth=4,
    LGBM_learning_rate=0.05, LGBM_n_estimators=500, LGBM_min_child_samples=8,
    LGBM_subsample=0.7, LGBM_colsample_bytree=0.8, LGBM_reg_alpha=0.12,
    LGBM_reg_lambda=0.05)


def _unchanged_except(out, base, changed_keys):
    return {k: v for k, v in out.items() if k not in changed_keys} == \
           {k: v for k, v in base.items() if k not in changed_keys}


def test_NNTQ_sampling_reaches_the_parameters():
    """Each NNTQ parameter is asked under its own name and its value lands in
    p[name]; the other entries (device, quantiles...) and the input dict are
    unchanged."""
    base = copy.deepcopy(constants.NNTQ_PARAMETERS)
    base.update({k: None for k in NNTQ_SCRIPT})   # every sampled value visible,
    ref  = copy.deepcopy(base)                    #   whatever the defaults
    trial = _ScriptedTrial(NNTQ_SCRIPT)
    p = bs.sample_NNTQ_parameters(trial, base)
    assert sorted(trial.asked) == sorted(NNTQ_SCRIPT)
    assert {k: p[k] for k in NNTQ_SCRIPT} == NNTQ_SCRIPT
    assert _unchanged_except(p, base, NNTQ_SCRIPT)
    assert base == ref


def test_metamodel_NN_sampling_reaches_the_parameters():
    """metaNN_<key> -> p[key]; metaNN_num_cells_0 / _1 -> p['num_cells'], in
    that order; metaNN_epochs -> p['epochs']."""
    base = copy.deepcopy(constants.METAMODEL_NN_PARAMETERS)
    ref  = copy.deepcopy(base)
    trial = _ScriptedTrial(META_SCRIPT)
    p = bs.sample_metamodel_NN_parameters(trial, base)
    assert sorted(trial.asked) == sorted(META_SCRIPT)
    expected = {k[len('metaNN_'):]: v for k, v in META_SCRIPT.items()
                if 'num_cells' not in k}
    expected['num_cells'] = [44, 8]
    assert {k: p[k] for k in expected} == expected
    assert p['epochs'] == 15 != base['epochs']
    assert _unchanged_except(p, base, expected)
    assert base == ref


def test_baseline_sampling_reaches_the_parameters():
    """<model>_<key> -> p[model][key] (numeric max_features given as text ->
    float); the other entries (random_state, n_jobs...) and the input dict
    are unchanged."""
    base = copy.deepcopy(constants.BASELINES_PARAMETERS)
    ref  = copy.deepcopy(base)
    trial = _ScriptedTrial(BASELINES_SCRIPT)
    p = bs.sample_baseline_parameters(trial, base)
    assert sorted(trial.asked) == sorted(BASELINES_SCRIPT)
    for name, value in BASELINES_SCRIPT.items():
        model, key = name.split('_', 1)
        expected = 0.4 if name == 'RF_max_features' else value
        assert p[model][key] == expected, name
        assert base[model][key] != expected, name        # visible change
    for model in base:
        keys = {n.split('_', 1)[1] for n in BASELINES_SCRIPT
                if n.startswith(model + '_')}
        assert _unchanged_except(p[model], base[model], keys), model
    assert base == ref

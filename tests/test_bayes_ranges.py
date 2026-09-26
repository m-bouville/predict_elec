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
   rates, weight decays are saved x1e6).

Bayes_search imports run -> torch, and optuna: skipped without them.
"""
import copy
import os
import pandas as pd

import pytest

pytest.importorskip("torch", reason="Bayes_search imports run -> torch")
optuna = pytest.importorskip("optuna")

import Bayes_search as bs
import constants
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
    if 'search_NNTQ_MAE' not in pd.read_csv(path, nrows=0).columns:
        pytest.skip(f"{os.path.basename(path)}: objective on the whole test "
                    "period (before the split in two): not reloadable")

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
    pd.DataFrame([row]).to_csv(csv, index=False, float_format="%.6f")  # as the search

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
    pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.6f")
    trials = bs.load_frozen_trials(str(csv), ALL_DISTRIBUTIONS, Stage.meta)
    assert [t.params['RF_max_features'] for t in trials] == ['sqrt', '0.4']


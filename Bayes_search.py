###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# Bayes_search.py
# Bayesian search for hyperparameters with Optuna
#
###############################################################################


import os
import copy
# import warnings

from   typing   import Dict, List, Any, Optional  # Tuple, Sequence,

from   datetime import timedelta  # datetime

import numpy    as np
import pandas   as pd

import optuna
from   optuna.distributions import \
                IntDistribution, FloatDistribution, \
                CategoricalDistribution #, BoolDistribution

# import matplotlib.pyplot as plt


import run
from   constants import Stage



# -------------------------------------------------------
# Distributions
# -------------------------------------------------------

DISTRIBUTIONS_BASELINES = {
    'LR_type':       CategoricalDistribution(choices=['lasso', 'ridge']),
    'LR_alpha':      FloatDistribution(low=0.00, high=2.0,step=0.05),
    # 'LR_alpha_lasso': FloatDistribution(low=0.005, high=0.04, log=True),
    # 'LR_alpha_ridge': FloatDistribution(low=0.5, high=2.0, log=True),
    'LR_max_iter':   IntDistribution(low=2000, high=2000),  # constant

    # random forest
    'RF_n_estimators': IntDistribution(low=200, high=700, step=10),
    'RF_max_depth':    IntDistribution(low=15, high=28, step=1),
    'RF_min_samples_leaf': IntDistribution(low= 8, high=20, step=1),
    'RF_min_samples_split':IntDistribution(low=12, high=25, step=1),
    'RF_max_features': CategoricalDistribution(choices=['sqrt', '0.4','0.5','0.6']),

    # gradient boosting (LGBM)
    'LGBM_boosting_type':CategoricalDistribution(choices=['gbdt']),
    'LGBM_num_leaves':   CategoricalDistribution(choices=[8-1, 16-1, 32-1, 64-1]),
    'LGBM_max_depth':    IntDistribution  (low=1,  high=10, step=1),
    'LGBM_learning_rate':FloatDistribution(low=0.001, high=0.17,step=0.001),
    'LGBM_n_estimators': IntDistribution  (low=300,  high=850, step=10),
    'LGBM_min_child_samples':IntDistribution(low= 6, high=25,  step=1),
    'LGBM_subsample':    FloatDistribution(low=0.6,  high=1.0, step=0.02),
    'LGBM_colsample_bytree':FloatDistribution(low=0.6,high=1.0,step=0.02),
    'LGBM_reg_alpha':    FloatDistribution(low=0.04, high=0.3, step=0.01),
    'LGBM_reg_lambda':   FloatDistribution(low=0.0, high=0.3, step=0.005)
}

DISTRIBUTIONS_NNTQ = {
    'use_ML_features':IntDistribution(low=0, high=1, step=1),  # Boolean

    # number of steps
    'patch_length':IntDistribution(low=12, high=48, step=6),
    'stride':      IntDistribution(low= 6, high=24, step=3),
    'input_length':IntDistribution(low=12*48,high=16*48,step=2*48),

    'epochs':      IntDistribution(low=10, high=30, step=1),
    'batch_size':  CategoricalDistribution(choices=[32, 64, 96, 128]),
    'learning_rate':FloatDistribution(low=0.0004,high=0.018, step=0.0004),
    'weight_decay':FloatDistribution(low=1e-9,  high=1e-5, log=True),
    'dropout':     FloatDistribution(low=0,     high=0.4, step=0.01),

    # quantile loss
    'lambda_cross':   FloatDistribution(low=0., high=0.1, step=0.002),
    'lambda_coverage':FloatDistribution(low=0., high=0.4, step=0.004),
    'lambda_deriv':   FloatDistribution(low=0., high=0.1, step=0.002),
    'lambda_median':  FloatDistribution(low=0., high=0.1, step=0.004),
    'smoothing_cross':FloatDistribution(low=0.004, high=0.072, step=0.001),
        # temperature-dependence (pinball loss, coverage penalty)
    'threshold_cold_degC': FloatDistribution(low=-1., high= 5., step=0.1),
    'saturation_cold_degC':FloatDistribution(low=-8., high=-2., step=0.1),
    'lambda_cold':    FloatDistribution(low=0.04,high=0.22, step=0.01),
        # régional consumption
    'lambda_regions': FloatDistribution(low=0.,   high=0.1, step=0.002),
    'lambda_regions_sum':FloatDistribution(low=0.,high=0.6, step=0.02),

    'model_dim':    IntDistribution(low=100, high=700, step=1),
    'ffn_size':     IntDistribution(low=2, high=7, step=1),
    'num_heads':    IntDistribution(low=3, high=9, step=1),
    'num_layers':   IntDistribution(low=1, high=7, step=1),
    'geo_block_ratio':FloatDistribution(low=1., high=1.),  # constant
    'num_geo_blocks': IntDistribution(low=2, high=12, step=1),

    'warmup_steps': IntDistribution(low=1000, high=4000, step=100),
    'patience':     IntDistribution(low=3, high=6, step=1),
    'min_delta':    FloatDistribution(low=0.020, high=0.048, step=0.001),
}

DISTRIBUTIONS_METAMODEL_NN = {
    'metaNN_epochs':      IntDistribution(low=1, high=20, step=1),  # 1: NNTQ search
    'metaNN_batch_size':  CategoricalDistribution(
        choices=[16, 32, 64, 96, 128, 192, 256, 384, 512, 640]),
    'metaNN_learning_rate':FloatDistribution(low=0.0005, high=0.0050, step=0.0001),
    'metaNN_weight_decay':FloatDistribution(low=5e-9, high=10e-5, log=True),
    'metaNN_dropout':     FloatDistribution(low=0., high=0.4,step=0.001),
    'metaNN_num_cells_0': IntDistribution  (low=24, high=72, step=4),
    'metaNN_num_cells_1': IntDistribution  (low= 4, high=32, step=4),
    'metaNN_patience':    CategoricalDistribution(choices=[2, 3, 4, 5, 6]),
    'metaNN_factor':      FloatDistribution(low=0.6, high=0.85, step=0.01),
}



# -------------------------------------------------------
# Sampling
# -------------------------------------------------------

def sample_baseline_parameters(
        trial: optuna.Trial,
        base_params: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
    """
    Sample baseline model parameters using Optuna.

    Args:
        trial: Optuna trial object for hyperparameter sampling.
        base_params: Dictionary of baseline model parameters to optimize.

    Returns:
        Updated dictionary of baseline model parameters.
    """
    p0 = copy.deepcopy(base_params)

    for _baseline in p0.keys():
        p = p0[_baseline]  # Get the parameters for the current baseline model

        if _baseline == 'LR':
            if 'type' in p:
                p['type'] = trial.suggest_categorical('LR_type', ['ridge'])  # 'lasso',
            if ('type' in p) and ('alpha' in p):
                if p['type'] == 'lasso':
                    p['alpha'] = trial.suggest_float('LR_alpha', 0.010, 0.030,step=0.004)
                else:  # ridge
                    p['alpha'] = trial.suggest_float('LR_alpha', 0.5, 1.5,step=0.05)

        elif _baseline == 'RF':
            if 'n_estimators' in p:  # number of trees in the Random Forest
                p['n_estimators'] = trial.suggest_int('RF_n_estimators', 200, 600, step=10)
            if 'max_depth' in p:     # maximum depth of the trees
                p['max_depth'] = trial.suggest_int('RF_max_depth', 15, 28, step=1)
            if 'min_samples_leaf' in p:   # minimum number of samples required a leaf node
                p['min_samples_leaf']= trial.suggest_int('RF_min_samples_leaf',  8, 18,step=1)
            if 'min_samples_split' in p:   # min number of samples to split an internal node
                p['min_samples_split']=trial.suggest_int('RF_min_samples_split',12, 22,step=1)
            if 'max_features' in p:    # number of features when looking for the best split
                p['max_features'] = trial.suggest_categorical(
                            'RF_max_features', ['sqrt'])  # , '0.4', '0.5', '0.6'])
                if p['max_features'] != 'sqrt':  # this is a float stored as str
                    p['max_features'] = float(p['max_features'])

        elif _baseline == 'LGBM':
            if 'boosting_type' in p:   # TODO add more?
                p['boosting_type'] = trial.suggest_categorical('LGBM_boosting_type', ['gbdt'])
            if 'num_leaves' in p:
                p['num_leaves']=trial.suggest_categorical('LGBM_num_leaves',[8-1,16-1,32-1,64-1])
            if 'max_depth' in p:
                p['max_depth'] = trial.suggest_int('LGBM_max_depth', 1, 10)
            if 'learning_rate' in p:
                p['learning_rate']=trial.suggest_float('LGBM_learning_rate',0.001,0.17,step=0.001)
            if 'n_estimators' in p:
                p['n_estimators'] = trial.suggest_int('LGBM_n_estimators', 300, 850, step=10)
            if 'min_child_samples' in p:
                p['min_child_samples']=trial.suggest_int('LGBM_min_child_samples',6,22,step=1)
            if 'subsample' in p:
                p['subsample'] = trial.suggest_float('LGBM_subsample', 0.6, .94, step=0.02)
            if 'colsample_bytree' in p:   # fraction of features used for each tree
                p['colsample_bytree']=trial.suggest_float('LGBM_colsample_bytree',.6,.94,step=0.02)
            if 'reg_alpha' in p:   # L1 regularization
                p['reg_alpha'] = trial.suggest_float('LGBM_reg_alpha', 0.04, 0.15, step=0.01)
            if 'reg_lambda' in p:   # L2 regularization
                p['reg_lambda']= trial.suggest_float('LGBM_reg_lambda',0., 0.15, step=0.005)

    return p0


def sample_NNTQ_parameters(
            trial: optuna.Trial,
            base_params: Dict[str, Any]
        ) -> Dict[str, Any]:

    p = base_params.copy()

    if 'use_ML_features' in p:
        p['use_ML_features']=trial.suggest_int('use_ML_features', 0, 0, step=1)

    # number of steps
    if 'stride' in p:
        p['stride'       ] = trial.suggest_int  ('stride',       12, 24, step=12)
    if 'patch_length' in p:  # safer if a multiple of stride
        _s = p['stride']
        p['patch_length' ] = trial.suggest_int  (
                                'patch_length', (24//_s+1)*_s, (48//_s)*_s, step=_s)
    if 'input_length' in p:
        p['input_length' ] = trial.suggest_int  ('input_length',14*48,14*48,step=2*48)

    if 'epochs' in p:
        p['epochs'        ] = trial.suggest_int  ('epochs', 10, 24, step=2)
    if 'batch_size' in p:
        p['batch_size'    ] = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    if 'learning_rate' in p:
        p['learning_rate' ] = trial.suggest_float('learning_rate',0.0004,0.018,step=0.0004)
    if 'weight_decay' in p:
        p['weight_decay'  ] = trial.suggest_float('weight_decay',1e-9,1e-5,log=True)
    if 'dropout' in p:
        p['dropout'       ] = trial.suggest_float('dropout', 0.0, 0.4, step=0.02)

    # quantile loss weights
    if 'lambda_cross' in p:
        p['lambda_cross'  ] = trial.suggest_float('lambda_cross',   0., 0.08, step=0.002)
    if 'lambda_coverage' in p:
        p['lambda_coverage']= trial.suggest_float('lambda_coverage',0.0,0.10, step=0.004)
    if 'lambda_deriv' in p:
        p['lambda_deriv'  ] = trial.suggest_float('lambda_deriv',   0., 0.096, step=0.004)
    if 'lambda_median' in p:
        p['lambda_median' ] = trial.suggest_float('lambda_median',  0., 0., step=0.008)
    if 'smoothing_cross' in p:
        p['smoothing_cross']=trial.suggest_float('smoothing_cross',0.02,0.072,step=0.002)

        # temperature-dependence (pinball loss, coverage penalty)
    if 'threshold_cold_degC' in p:
        p['threshold_cold_degC']=trial.suggest_float('threshold_cold_degC',-1.,5.,step=0.1)
    if 'saturation_cold_degC' in p:
        p['saturation_cold_degC']=trial.suggest_float('saturation_cold_degC',
                                                     -8., -2., step=0.1)
    if 'lambda_cold' in p:
        p['lambda_cold'        ]= trial.suggest_float('lambda_cold', 0.04, 0.22,step=0.01)

        # régional consumption
    if 'lambda_regions' in p:
        p['lambda_regions'   ]= trial.suggest_float('lambda_regions', 0.0, 0.072,step=0.008)
    if 'lambda_regions_sum' in p:
        p['lambda_regions_sum']=trial.suggest_float('lambda_regions_sum',0.32,0.6,step=0.04)

    # Architecture
    if 'ffn_size' in p:
        p['ffn_size'   ] = trial.suggest_int('ffn_size',  2, 7)
    if 'num_heads' in p:
        p['num_heads'  ] = trial.suggest_int('num_heads', 4, 7)
    if 'model_dim' in p:  # must be a multiiple of num_heads
        _step = 4 * p['num_heads']
        p['model_dim'  ] = trial.suggest_int(
                 'model_dim', (300//_step+1)*_step, (700//_step)*_step, step=_step)
    if 'num_layers' in p:
        p['num_layers' ] = trial.suggest_int('num_layers',3, 7)

    if 'num_geo_blocks' in p:
        p['num_geo_blocks'] = trial.suggest_int('num_geo_blocks', 5, 12)

    # Early stopping
    if 'warmup_steps' in p:
        p['warmup_steps'] = trial.suggest_int  ('warmup_steps', 1500,4000,step=100)
    if 'patience' in p:
        p['patience'    ] = trial.suggest_int  ('patience', 3, 6)
    if 'min_delta' in p:
        p['min_delta'   ] = trial.suggest_float('min_delta', 0.028, 0.048, step=0.002)


    # derived
    if 'model_dim' in p and 'num_heads' in p and p['model_dim'] % p['num_heads'] != 0:
        # _old = p['model_dim']
        p['model_dim'] = int(p['num_heads'] * round(p['model_dim'] / p['num_heads']))
        # if _old != p['model_dim']:
        #     print(f"model_dim: {_old} -> {p['model_dim']}")

    # if 'input_length' in p and 'features_in_future' in p and 'pred_length' in p and \
    #             'patch_length' in p and 'stride' in p:
    #     p['num_patches'] = (
    #         p['input_length']
    #         + int(p['features_in_future']) * p['pred_length']
    #         - p['patch_length']
    #     ) // p['stride'] + 1
    #     print(f"num_patches: {p['num_patches']} = ({p['input_length']} "
    #           f"+ {int(p['features_in_future']) * p['pred_length']} "
    #           f"- {p['patch_length']}) // {p['stride']} + 1")

    return p


def sample_metamodel_NN_parameters(
            trial: optuna.Trial,
            base_params: Dict[str, Any]
        ) -> Dict[str, Any]:

    p = base_params.copy()

    if 'epochs' in p:
        p['epochs']      = trial.suggest_int('metaNN_epochs', 10, 20)
    if 'batch_size' in p:
        p['batch_size']  = trial.suggest_categorical(
            'metaNN_batch_size', [16, 32, 64, 96, 128, 192, 256, 384, 512, 640])
    if 'learning_rate' in p:
        p['learning_rate']=trial.suggest_float('metaNN_learning_rate',
                                               low=0.0005, high=0.0050, step=0.0005)
    if 'weight_decay' in p:    # BUG: 0 in csv at start
        p['weight_decay']= trial.suggest_float('metaNN_weight_decay',5e-9,10e-5,log=True)
    if 'dropout' in p:
        p['dropout']     = trial.suggest_float('metaNN_dropout', 0.0, 0.2, step=0.01)

    if 'num_cells' in p:
        p['num_cells']= [trial.suggest_int('metaNN_num_cells_0', 40, 72, step=4),
                         trial.suggest_int('metaNN_num_cells_1',  4, 32, step=4)]

    # Early stopping
    if 'metaNN_patience' in p:
        p['patience'] = trial.suggest_categorical('patience', [2, 3, 4, 5, 6])
    if 'metaNN_factor' in p:
        p['factor'  ] = trial.suggest_float('factor', 0.6, 0.85, step=0.01)

    return p



# -------------------------------------------------------
# run Bayesian search
# -------------------------------------------------------

def run_Bayes_search(
            stage               : Stage,  # in [Stage.NNTQ, Stage.meta, Stage.all]
            num_trials          : int,

            # configuration bundles
            base_baseline_params: Dict[str, Dict[str, Any]],
            base_NNTQ_params    : Dict[str, Any],
            base_meta_NN_params : Dict[str, Any],
            dict_input_csv_fnames:Dict[str, str],
            trials_csv_path     : str,

            # statistics of the dataset
            minutes_per_step    : int,
            train_split_fraction: float,
            valid_ratio         : float,
            forecast_hour       : int,
            seed                : int,
            force_calc_baselines: bool   = False,
            cache_dir           : Optional[str] = None,

            # multi-run for best candidtes (robustness)
            num_runs            : Dict[Stage, int]  ={Stage.NNTQ: 7, Stage.meta:5},
            min_num_trials      : Dict[Stage, int]  ={Stage.NNTQ:40, Stage.meta:20},
            wiggle_value        : Dict[Stage, float]={Stage.NNTQ:2., Stage.meta:0.03},

            verbose             : int  = 0
        ):


    num_runs     = num_runs    [stage]
    wiggle_value = wiggle_value[stage]

    # average after excluding min and max
    def clean_avg(l: List[float]) -> float:
        if len(l) < 3:
            return np.mean(l)

        _list = l.copy()
        _list.remove(min(_list))
        _list.remove(max(_list))
        return np.mean(_list)


    def objective(trial: optuna.Trial) -> float:
        # print(f"** Starting run {trial.number} out of {num_runs}")

        baseline_parameters = copy.deepcopy(base_baseline_params)

        # three possible bahaviors:
        #   - all:  we sample everything
        #   - NNTQ: we sample NNTQ parameters only
        #       (metamodel is not used, baselines are used, but frozen)
        #   - meta: we sample baselines and metamodel parameters
        #       (NNTQ parameters are frozen to values found in `NNTQ` search)

        baseline_parameters= sample_baseline_parameters   (trial, baseline_parameters)\
            if stage in [      Stage.meta, Stage.all] else baseline_parameters
        NNTQ_parameters    = sample_NNTQ_parameters       (trial, base_NNTQ_params)\
            if stage in [Stage.NNTQ,       Stage.all] else base_NNTQ_params.copy()
        metamodel_parameters=sample_metamodel_NN_parameters(trial,base_meta_NN_params)\
            if stage in [      Stage.meta, Stage.all] else base_meta_NN_params.copy()

        if stage == Stage.NNTQ:
            metamodel_parameters['epochs'] = 1  # for speed


        _threshold = trial.study.best_trial.value + wiggle_value
        _stop_now  = False

        # print(stage, trial.number, num_runs)

        for i in range(num_runs):
            _, dict_row, df_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
                (num_worst_days, worst_days_test), (_loss_NNTQ, _loss_meta) = \
                    run.run_model_once(
                      # configuration bundles
                      baseline_parameters= baseline_parameters  .copy(),
                      NNTQ_parameters   = NNTQ_parameters       .copy(),
                      metamodel_NN_parameters=metamodel_parameters.copy(),
                      dict_input_csv_fnames= dict_input_csv_fnames,
                      # trials_csv_path   = trials_csv_path,

                      # statistics of the dataset
                      minutes_per_step  = minutes_per_step,
                      train_split_fraction=train_split_fraction,
                      valid_ratio       = valid_ratio,
                      forecast_hour     = forecast_hour,
                      seed              = seed*100 + trial.number*10 + i,

                      force_calc_baselines=force_calc_baselines,
                      save_cache_baselines= stage == Stage.NNTQ,  # baselines not sampled
                      save_cache_NNTQ     = stage == Stage.meta,  # NNTQ      not sampled

                      # XXX_EVERY (in epochs)
                      validate_every    =   1,
                      display_every     = 999,  # dummy
                      plot_conv_every   = 999,  # dummy

                      run_id            = trial.number,
                      cache_dir         = cache_dir,
                      verbose           = verbose
                )
            # print(trial.number, i, num_runs,
            #    (_loss_NNTQ_1run, _loss_meta), trial.study.best_trial.value)

            _loss_1run = _loss_NNTQ if stage == Stage.NNTQ else _loss_meta

            if num_runs > 1 and trial.number > min_num_trials[stage]:
                if i == 0: # we must decide whether to run more
                    if _loss_1run <= _threshold:
                        _list_losses_NNTQ = [_loss_NNTQ]
                        _list_losses_meta = [_loss_meta]
                        print(f"potential new record: {_loss_1run} "
                              f"(best so far: {trial.study.best_trial.value}) "
                              f"=> starting up to {num_runs-1} more runs...")
                    else:
                        break  # no need to run more than once

                else:
                    _list_losses_NNTQ.append(_loss_NNTQ)
                    _list_losses_meta.append(_loss_meta)
                    # print(i, _list_losses_NNTQ)

                    # is running for longer likely to be worth our while?
                    # we compare an optimistic metric --average after
                    #   removing the worst (largest) loss-- to the threshold
                    _losses = _list_losses_NNTQ.copy() if stage == Stage.NNTQ \
                        else  _list_losses_meta.copy()
                    # _losses_complete = _losses
                    _losses.remove(max(_losses))
                    if (verbose >= 2) and (i < num_runs-1):
                        print(f"{i}: comparing avg({_losses}) = "
                              f"{np.mean(_losses):.2f} to {_threshold}")
                    if np.mean(_losses) > _threshold:
                        _stop_now = True

                if (i == num_runs-1) or _stop_now:  # we just finished the last run
                    dict_row['num_runs'] = i+1

                    # average after excluding min and max
                    dict_row['loss_NNTQ'] = round(clean_avg(_list_losses_NNTQ), 2)
                    dict_row['loss_meta'] = round(clean_avg(_list_losses_meta), 5)

                    _loss_NNTQ = dict_row['loss_NNTQ']
                    _loss_meta = dict_row['loss_meta']

                    print(f"{i+1} runs: losses "
                          f"{_list_losses_NNTQ if stage==Stage.NNTQ else _list_losses_meta} "
                          f"-> avg {dict_row[f'loss_{stage.value}']}")
                    break

            if trial.number <= min_num_trials[stage]:
                break


        df_row = pd.DataFrame([dict_row])
        # /_\ with multiple runs, the metrics other than losses are just the last run

        df_row.to_csv(
            trials_csv_path,
            mode   = "a",
            header = not os.path.exists(trials_csv_path),
            index  = False,
            float_format="%.6f"
        )

        print()

        # return the relevant loss
        if stage == Stage.NNTQ:
            return _loss_NNTQ
        if stage == Stage.meta:
            return _loss_meta
        if stage == Stage.all:
            return _loss_NNTQ + _loss_meta



    # Create a study with the previous trials
    study   = optuna.create_study(direction= 'minimize',
                                  sampler  = optuna.samplers.TPESampler())

    _ALL_DISTRIBUTIONS = DISTRIBUTIONS_BASELINES | \
                         DISTRIBUTIONS_NNTQ | DISTRIBUTIONS_METAMODEL_NN

    # Load the CSV file containing MC runs
    if os.path.exists(trials_csv_path):
        trials = load_frozen_trials(trials_csv_path, _ALL_DISTRIBUTIONS, stage)

        # Add previous trials to the study
        for index, trial in enumerate(trials):
            # print(index)
            study.add_trial(trial)

    # Plotting
    _list_parameters_hist = ['model_dim', 'dropout', 'learning_rate', 'lambda_cross'] \
        if stage == Stage.NNTQ else \
            [ 'LGBM_min_child_samples', 'LGBM_reg_alpha', 'LGBM_colsample_bytree',
             'metaNN_learning_rate', 'metaNN_num_cells_1', 'RF_min_samples_split'
            # 'LR_alpha', 'RF_max_depth', 'LGBM_max_depth', 'metaNN_learning_rate'
             ]
    # plot_optuna(study, ist_parameters_hist=_list_parameters_hist)


    # Run optimization
    print(f"\nStarting {num_trials} Bayesian trials ({stage.value})...")
    study.optimize(objective, n_trials=num_trials)


    # Print the best parameters found
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # print(f"Best hyperparameters: {study.best_params}")


    # Plotting
    plot_optuna(study, stage, list_parameters_hist=_list_parameters_hist)


    return study.best_params



# -------------------------------------------------------
# run several plotting functions
# -------------------------------------------------------

def plot_optuna(study,
                stage                   : Stage,
                list_parameters_hist    : List[str],
                num_best_runs_params    : int =  5,
                num_best_runs_hist      : int = 15,
                num_important_parameters: int = 12) -> None:
    print("Plotting Optuna results so far...")


    # BUG (strangely) does not display anything
    # # Plot convergence
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()

    # # Plot parameter importances
    # optuna.visualization.plot_param_importances   (study).show()

    # optuna.visualization.plot_parallel_coordinate(
    #     study,
    #     params=list(study.trials[0].params.keys())[:3]
    # ).show()

    # # Parallel Coordinate Plot (structure discovery)
    # optuna.visualization.plot_parallel_coordinate(
    #     study,
    #     params=["learning_rate", "dropout", "num_layers"]
    # ).show()


    # Convergence
    ######################
    df = study.trials_dataframe()
    df = df.sort_values("number")
    df["moving_median"]= df["value"].rolling(20, min_periods=15, center=True).median()
    df["best_so_far"]  = df["value"].cummin()

    import matplotlib.pyplot as plt
    plt.figure()
    df.plot(x="number", y=["value", "moving_median", "best_so_far"])
    plt.xlabel("trial number")
    plt.ylabel(f"{stage.value} loss")
    plt.yscale('log')
    plt.show()


    # Parameter importance
    ######################
    cols_unusable = ['params_weight_decay', 'LR_max_iter', 'geo_block_ratio']
                     # 'patch_length', 'stride']
    dict_corr = dict()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols
                if col not in ['value', 'number', 'best_so_far'] + cols_unusable\
                and not np.issubdtype(df[col].dtype, np.timedelta64)]
        # `weight_decay` was so small it was initially rounded down to zero

    for p in numeric_cols:
        dict_corr[p[7:]] = -100 * np.corrcoef(df[p], df["value"])[0, 1]
            # `7:` removes "params_"
    df_corr = pd.Series(dict_corr, name="neg_corr_pc")

    df_corr_sorted = df_corr.abs().sort_values(ascending=False)

    df_corr = pd.concat([df_corr.round(2),
                         pd.Series(study.best_trial.params, name='parameter')],
                        axis = 1)
    df_corr_sorted_signed = df_corr.loc[df_corr_sorted.index]

    print("Parameter Importance (_negative_ correlation with `value`, in %)")
    print(df_corr_sorted_signed[:num_important_parameters].round(4))
    # plt.figure()
    # df_corr_sorted_signed.plot(kind="bar",
    #   title="Parameter Importance (Correlation with 'value')")
    # plt.show()


    # best parameters (more robust than just the very best)
    ######################
    print(f"shape: {df.shape} -> {df[numeric_cols + ['value']].shape}")

    # Get the best N trials based on the objective value
    best_trials_df = df[numeric_cols + ['value']].\
        sort_values(by='value', ascending=True).head(num_best_runs_params)

    avg_value   = float(best_trials_df[['value']].mean().iloc[0])
    list_values = best_trials_df['value'].round(4).tolist()

    # Extract parameters from these trials
    params_df = best_trials_df[numeric_cols]

    # Calculate the median for each parameter
    avg_params = params_df.mean()  # .drop(['number', 'best_so_far'])

    print(f"\nAverage parameters from the best {num_best_runs_params} runs "
          f"(mean of {list_values} = {avg_value:.2f}):")
    print(avg_params)


    # histogram
    ######################
    df  = study.trials_dataframe()
    top = df.nsmallest(num_best_runs_hist, "value")
    top[['params_' + e for e in list_parameters_hist]].hist(figsize=(10,8))



# -------------------------------------------------------
# subroutines for Bayesian search
# -------------------------------------------------------

def cols_not_paras() -> List[str]:
    # columns from the csv that do not contain optimisable parameters
    # parameters we do not optimise
    cols_not_paras = ['run'] + \
        ['quantiles_0','quantiles_1','quantiles_2','quantiles_3','quantiles_4'] +\
        ['RF_type', 'RF_random_state', 'RF_n_jobs'] + \
        ['LGBM_type', 'LGBM_objective'] + \
        ['LGBM_random_state','LGBM_n_jobs',	'LGBM_verbose'] + \
        ['pred_length', 'valid_length'] + \
        ['device', 'metaNN_device', 'features_in_future']
        # ,'GB_boosting_type', 'input_length', 'num_patches',

    # output
    cols_not_paras.extend(['q10', 'q25', 'q50', 'q75', 'q90'])  # coverage
    for _model in ['NNTQ_q50', 'LR', 'RF', 'LGBM']:
        cols_not_paras.append(f'avg_weight_meta_NN_{_model}')
    for _model in ['NNTQ', 'LR', 'RF', 'LGBM', 'meta_LR', 'meta_NN']:
        for _metric in ['bias', 'RMSE', 'MAE']:
            cols_not_paras.append(f'test_{_model}_{_metric}')
    cols_not_paras.extend(['num_features', 'avg_abs_worst_days_test'])
    # print(cols_not_paras)
    # print([e for e in results_df.columns if e not in cols_not_paras])

    return cols_not_paras


def load_frozen_trials(csv_path     : str,
                       distributions: dict,
                       stage        : Stage) -> List[optuna.Trial]:
    results_df = pd.read_csv(csv_path, index_col=False)

    _cols_not_paras = cols_not_paras()

    # print(list(results_df.columns))
    _superfluous = set(_cols_not_paras) - set(results_df.columns)
    assert len(_superfluous) == 0, _superfluous

    results_df.drop(columns=_cols_not_paras, inplace=True)
    print(f"{csv_path} loaded: {results_df.shape}")

    results_df[['learning_rate', 'weight_decay',
                'metaNN_learning_rate', 'metaNN_weight_decay']] =\
        (results_df[['learning_rate', 'weight_decay',
                'metaNN_learning_rate', 'metaNN_weight_decay']] * 1e-6).round(9)
            # learning_rate and weight_decay are small, prone to round-off errors:
            #    save them multiplied by a million (and round to avoid 0.999999)

    # can be 'sqrt' or a float => type issues
    results_df['RF_max_features'] = results_df['RF_max_features'].astype(str)

    # check consistency
    _special_keys = {'timestamp', 'num_runs', 'loss_NNTQ', 'loss_meta'}
    assert set(distributions.keys()) - set(results_df.columns) == set(), \
        set(distributions.keys()) - set(results_df.columns)
    assert set(results_df.columns) - set(distributions.keys()) == \
                _special_keys, \
                    set(results_df.columns) - set(distributions.keys())


    # Create a list of FrozenTrial objects
    trials = []
    distributions_keys = distributions.keys()
    for index, row in results_df.iterrows():
        # print(index)  #, row)
        _params = {k: row[k] for k in row.keys() if k not in _special_keys}
        assert set(_params.keys()) - set(distributions_keys) == set(), \
               set(_params.keys()) - set(distributions_keys)
        assert set(distributions_keys) - set(_params.keys()) == set(), \
               set(distributions_keys) - set(_params.keys())

        # relevant loss
        _value = row['loss_NNTQ'] + row['loss_meta'] if stage == Stage.all \
            else row[f'loss_{stage.value}']

        trial = optuna.trial.FrozenTrial(
            number        = index,  # Trial number
            state         = optuna.trial.TrialState.COMPLETE,  # state of trial
            datetime_start=   pd.to_datetime(row['timestamp'])- \
                timedelta(minutes=1.5),
            datetime_complete=pd.to_datetime(row['timestamp']),
            value         = _value,  # Objective value
            params        = _params,  # hyperparameters
            user_attrs    = {},  # Additional attributes (can be empty)
            distributions = distributions,
            system_attrs  = {},  # system attributes (can be empty)
            intermediate_values={},  # (can be empty)
            trial_id      = index  # ID of the trial
        )
        trials.append(trial)

    return trials




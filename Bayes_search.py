# import os
import copy
import warnings

from   typing   import Dict, Any, Optional  # List, Tuple, Sequence,

from   datetime import timedelta  # datetime

import numpy    as np
import pandas   as pd

import optuna
from   optuna.distributions import \
                IntDistribution, FloatDistribution, CategoricalDistribution

# import matplotlib.pyplot as plt


import run



# -------------------------------------------------------
# Distributions
# -------------------------------------------------------

distributions_baselines = {
    'lasso_alpha':    FloatDistribution(low=0.005, high=0.04, log=True),
    'lasso_max_iter': IntDistribution(low=2000, high=2000),  # constant
    'LR_type':        CategoricalDistribution(choices=['lasso', 'ridge']),
    'LR_alpha':      FloatDistribution(low=0.005, high=2.0, log=True),
    # 'LR_alpha_lasso': FloatDistribution(low=0.005, high=0.04, log=True),
    # 'LR_alpha_ridge': FloatDistribution(low=0.5, high=2.0, log=True),
    'LR_max_iter': IntDistribution(low=2000, high=2000),  # constant
    # random forest
    'RF_n_estimators': IntDistribution(low=300, high=600, step=10),
    'RF_max_depth': IntDistribution(low=15, high=25, step=1),
    'RF_min_samples_leaf': IntDistribution(low=10, high=20, step=1),
    'RF_min_samples_split': IntDistribution(low=15, high=25, step=1),
    'RF_max_features': CategoricalDistribution(choices=['sqrt', '0.4', '0.5', '0.6']),
    # gradient boosting
    'GB_boosting_type': CategoricalDistribution(choices=['gbdt']),
    'GB_num_leaves': CategoricalDistribution(choices=[15, 31]),
    'GB_max_depth':  CategoricalDistribution(choices=[3, 4, 5, 6]),
    'GB_learning_rate':FloatDistribution(low=0.002, high=0.15, log=True),
    'GB_n_estimators': IntDistribution(low=300, high=600, step=10),
    'GB_min_child_samples':IntDistribution(low=15, high=25, step=1),
    'GB_subsample':   FloatDistribution(low=0.6, high=1.0),
    'GB_colsample_bytree':FloatDistribution(low=0.6, high=1.0),
    'GB_reg_alpha':  FloatDistribution(low=0.03, high=0.3, log=True),
    'GB_reg_lambda': FloatDistribution(low=0.03, high=0.3, log=True)
}

distributions_NNTQ = {
    'patch_length': IntDistribution(low=24, high=24),  # constant
    'stride':     IntDistribution(low=12, high=12),  # constant
    'epochs':     IntDistribution(low=20, high=40, step=1),
    'batch_size': CategoricalDistribution(choices=[32, 64, 96, 128]),
    'learning_rate':FloatDistribution(low=0.002, high=0.3, log=True),
    'weight_decay': FloatDistribution(low=0., high=1e-5),  # BUG: 0 in csv at start
    'dropout':    FloatDistribution(low=0.02, high=0.15),
    # quantile loss
    'lambda_cross':   FloatDistribution(low=0., high=2.0),
    'lambda_coverage':FloatDistribution(low=0., high=1.0),
    'lambda_deriv':   FloatDistribution(low=0., high=0.3),
    'lambda_median':  FloatDistribution(low=0., high=1.0),
    'smoothing_cross':FloatDistribution(low=0.005,high=0.05),
        # temperature-dependence (pinball loss, coverage penalty)
    'threshold_cold_degC': FloatDistribution(low= 0., high= 5., step=0.1),
    'saturation_cold_degC':FloatDistribution(low=-8., high=-2., step=0.1),
    'lambda_cold':    FloatDistribution(low=0., high=1.),

    'model_dim':    IntDistribution(low=90, high=255, step=1),
    'ffn_size':     IntDistribution(low=2, high=7, step=1),
    'num_heads':    IntDistribution(low=2, high=8, step=1),
    'num_layers':   IntDistribution(low=1, high=6, step=1),
    'geo_block_ratio':FloatDistribution(low=1., high=1.),  # constant
    'num_geo_blocks': IntDistribution(low=2, high=8, step=1),
    'warmup_steps': IntDistribution(low=1500, high=4000, step=50),
    'patience':     IntDistribution(low=3, high=10, step=1),
    'min_delta':    FloatDistribution(low=15e-3, high=30e-3),
}

distributions_metamodel_NN = {
    'metaNN_epochs':      IntDistribution(low=1, high=15, step=1),  # 1: NNTQ search
    'metaNN_batch_size':  CategoricalDistribution(choices=[128, 192, 256, 384, 512, 640]),
    'metaNN_learning_rate':FloatDistribution(low=0.15e-3, high=1.5e-3, log=True),
    'metaNN_weight_decay':FloatDistribution(low=5e-9, high=10e-5, log=True),
    'metaNN_dropout':     FloatDistribution(low=0., high=0.4),
    'metaNN_num_cells_0': CategoricalDistribution(choices=[24, 32, 40, 48]),
    'metaNN_num_cells_1': CategoricalDistribution(choices=[12, 16, 20, 24]),
    'metaNN_patience':    CategoricalDistribution(choices=[2, 3, 4, 5, 6]),
    'metaNN_factor':      FloatDistribution(low=0.6, high=0.85),
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

        # if _baseline == 'lasso':
        #     if 'alpha' in p:
        #         p['alpha'] = trial.suggest_float('lasso_alpha', 0.005, 0.04, log=True)

        if _baseline == 'LR':
            if 'type' in p:
                p['type'] = trial.suggest_categorical('LR_type', ['lasso', 'ridge'])
            if ('type' in p) and ('alpha' in p):
                if p['type'] == 'lasso':
                    p['alpha'] = trial.suggest_float('LR_alpha', 0.005, 0.04,log=True)
                else:  # ridge
                    p['alpha'] = trial.suggest_float('LR_alpha', 0.2, 2.0, log=True)

        elif _baseline == 'RF':
            if 'n_estimators' in p:  # number of trees in the Random Forest
                p['n_estimators'] = trial.suggest_int('RF_n_estimators', 300, 600, step=10)
            if 'max_depth' in p:     # maximum depth of the trees
                p['max_depth'] = trial.suggest_int('RF_max_depth', 15, 25)
            if 'min_samples_leaf' in p:   # minimum number of samples required a leaf node
                p['min_samples_leaf'] = trial.suggest_int('RF_min_samples_leaf', 10, 20)
            if 'min_samples_split' in p:   # min number of samples to split an internal node
                p['min_samples_split'] = trial.suggest_int('RF_min_samples_split', 15, 25)
            if 'max_features' in p:    # number of features when looking for the best split
                p['max_features'] = trial.suggest_categorical(
                            'RF_max_features', ['sqrt', '0.4', '0.5', '0.6'])
                if p['max_features'] != 'sqrt':  # this is a float stored as str
                    p['max_features'] = float(p['max_features'])

        elif _baseline == 'GB':
            if 'boosting_type' in p:   # TODO add more?
                p['boosting_type'] = trial.suggest_categorical('GB_boosting_type', ['gbdt'])
            if 'num_leaves' in p:
                p['num_leaves'] = trial.suggest_categorical('GB_num_leaves', [16-1, 32-1])
            if 'max_depth' in p:
                p['max_depth'] = trial.suggest_categorical('GB_max_depth', [3, 4, 5, 6])
            if 'learning_rate' in p:
                p['learning_rate']=trial.suggest_float('GB_learning_rate', 0.015,0.15,log=True)
            if 'n_estimators' in p:
                p['n_estimators'] = trial.suggest_int('GB_n_estimators', 300, 600, step=10)
            if 'min_child_samples' in p:
                p['min_child_samples']=trial.suggest_int('GB_min_child_samples', 15, 25)
            if 'subsample' in p:
                p['subsample'] = trial.suggest_float('GB_subsample', 0.6, 1.0)
            if 'colsample_bytree' in p:   # fraction of features used for each tree
                p['colsample_bytree'] = trial.suggest_float('GB_colsample_bytree', 0.6, 1.0)
            if 'reg_alpha' in p:   # L1 regularization
                p['reg_alpha'] = trial.suggest_float('GB_reg_alpha', 0.03, 0.3, log=True)
            if 'reg_lambda' in p:   # L2 regularization
                p['reg_lambda'] = trial.suggest_float('GB_reg_lambda', 0.03, 0.3, log=True)

    return p0


def sample_NNTQ_parameters(
            trial: optuna.Trial,
            base_params: Dict[str, Any]
        ) -> Dict[str, Any]:

    p = base_params.copy()

    if 'epochs' in p:
        p['epochs'        ] = trial.suggest_int  ('epochs', 20, 40)
    if 'batch_size' in p:
        p['batch_size'    ] = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    if 'learning_rate' in p:
        p['learning_rate' ] = trial.suggest_float('learning_rate', 0.003, 0.3, log=True)
    if 'weight_decay' in p:     # BUG: 0 in csv at start
        p['weight_decay'  ] = trial.suggest_float('weight_decay', 0., 1e-5, log=False)
    if 'dropout' in p:
        p['dropout'       ] = trial.suggest_float('dropout', 0.02, 0.15)

    # quantile loss weights
    if 'lambda_cross' in p:
        p['lambda_cross'  ] = trial.suggest_float('lambda_cross',   0.,  0.8)
    if 'lambda_coverage' in p:
        p['lambda_coverage']= trial.suggest_float('lambda_coverage',0.,  0.8)
    if 'lambda_deriv' in p:
        p['lambda_deriv'  ] = trial.suggest_float('lambda_deriv',   0.0, 0.2)
    if 'lambda_median' in p:
        p['lambda_median' ] = trial.suggest_float('lambda_median',  0.,  0.8)
    if 'smoothing_cross' in p:
        p['smoothing_cross']= trial.suggest_float('smoothing_cross',0.005,0.05)

        # temperature-dependence (pinball loss, coverage penalty)
    if 'threshold_cold_degC' in p:
        p['threshold_cold_degC']=trial.suggest_float('threshold_cold_degC',0.,5.,step=0.1)
    if 'saturation_cold_degC' in p:
        p['saturation_cold_degC']=trial.suggest_float('saturation_cold_degC',
                                                      -8., -2., step=0.1)
    if 'lambda_cold' in p:
        p['lambda_cold'        ]= trial.suggest_float('lambda_cold', 0., 1.)

    # Architecture
    if 'model_dim' in p:
        p['model_dim'  ] = trial.suggest_int('model_dim', 95, 255, step=16)
    if 'ffn_size' in p:
        p['ffn_size'   ] = trial.suggest_int('ffn_size',  3, 7)
    if 'num_heads' in p:
        p['num_heads'  ] = trial.suggest_int('num_heads', 4, 8)
    if 'num_layers' in p:
        p['num_layers' ] = trial.suggest_int('num_layers',2, 6)

    if 'epochs' in p:
        p['num_geo_blocks'] = trial.suggest_int('num_geo_blocks', 2, 8)

    # Early stopping
    if 'warmup_steps' in p:
        p['warmup_steps'] = trial.suggest_int  ('warmup_steps', 1500, 4000, step=50)
    if 'patience' in p:
        p['patience'    ] = trial.suggest_int  ('patience', 4, 10)
    if 'min_delta' in p:
        p['min_delta'   ] = trial.suggest_float('min_delta', 15e-3, 30e-3)

    # derived
    if 'model_dim' in p and 'num_heads' in p:
        if p['model_dim'] % p['num_heads'] != 0:
            p['model_dim'] = p['num_heads'] * (p['model_dim'] // p['num_heads'])

    if 'input_length' in p and 'features_in_future' in p and 'pred_length' in p and \
                'patch_length' in p and 'stride' in p:
        p['num_patches'] = (
            p['input_length']
            + p['features_in_future'] * p['pred_length']
            - p['patch_length']
        ) // p['stride'] + 1

    return p


def sample_metamodel_NN_parameters(
            trial: optuna.Trial,
            base_params: Dict[str, Any]
        ) -> Dict[str, Any]:

    p = base_params.copy()

    if 'epochs' in p:
        p['epochs']      = trial.suggest_int('metaNN_epochs', 8, 15)
    if 'batch_size' in p:
        p['batch_size']  = trial.suggest_categorical('metaNN_batch_size',
                                                     [128, 192, 256, 384, 512, 640])
    if 'learning_rate' in p:
        p['learning_rate']=trial.suggest_float('metaNN_learning_rate',
                                               0.15e-3,1.5e-3,log=True)
    if 'weight_decay' in p:    # BUG: 0 in csv at start
        p['weight_decay']= trial.suggest_float('metaNN_weight_decay',0.,1e-5,log=False)
    if 'dropout' in p:
        p['dropout']     = trial.suggest_float('metaNN_dropout', 0.05, 0.2)

    if 'num_cells' in p:
        p['num_cells']= [trial.suggest_categorical('metaNN_num_cells_0', [24,32,40,48]),
                         trial.suggest_categorical('metaNN_num_cells_1', [12,16,20,24])]

    # Early stopping
    if 'metaNN_patience' in p:
        p['patience'] = trial.suggest_categorical('patience', [2, 3, 4, 5, 6])
    if 'metaNN_factor' in p:
        p['factor'  ] = trial.suggest_float('factor', 0.6, 0.85)

    return p



# -------------------------------------------------------
# run several plotting functions
# -------------------------------------------------------

def plot_optuna(study,
                num_best_runs: int = 15,
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
    ######################♠
    df = study.trials_dataframe()
    df = df.sort_values("number")

    df["best_so_far"] = df["value"].cummin()
    df.plot(x="number", y=["value", "best_so_far"])


    # Parameter importance
    ######################
    cols_unusable = ['params_weight_decay', 'LR_max_iter', 'geo_block_ratio', \
                     'lasso_max_iter', 'patch_length', 'stride']
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
    ######################♠
    print(f"shape: {df.shape} -> {df[numeric_cols + ['value']].shape}")
    # print(df[numeric_cols + ['value']])
    # Get the best 15 trials based on the objective value
    best_trials_df = df[numeric_cols + ['value']].\
        sort_values(by='value', ascending=False).head(num_best_runs)

    # Extract parameters from these trials
    params_df = best_trials_df[numeric_cols]

    # Calculate the median for each parameter
    median_params = params_df.median()  # .drop(['number', 'best_so_far'])

    print(f"\nMedian parameters from the best {num_best_runs} runs:")
    print(median_params)


    # histogram
    ######################♠
    df  = study.trials_dataframe()
    top = df.nsmallest(num_best_runs, "value")
    top[["params_learning_rate", "params_dropout"]].hist()



# -------------------------------------------------------
# run Bayesian search
# -------------------------------------------------------

def run_Bayes_search(
            stage               : str,    # in ['NNTQ', 'meta', 'all']
            num_runs            : int,

            # configuration bundles
            base_baseline_params: Dict[str, Dict[str, Any]],
            base_NNTQ_params    : Dict[str, Any],
            base_meta_NN_params : Dict[str, Any],
            dict_fnames         : Dict[str, str],

            # statistics of the dataset
            minutes_per_step    : int,
            train_split_fraction: float,
            val_ratio           : float,
            forecast_hour       : int,
            seed                : int,
            force_calc_baselines: bool   = False,
            cache_dir           : Optional[str] = None,
            csv_path            : str    = 'parameter_search.csv'
        ):

    # from   sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO fix for real


    def objective(trial: optuna.Trial) -> float:
        # print(f"Starting run {run_id} out of {num_runs}")

        # lasso controls features and is thus relevant to everyone
        baseline_parameters = copy.deepcopy(base_baseline_params)
        if 'alpha' in baseline_parameters['lasso'] :
            baseline_parameters['lasso']['alpha'] = \
                trial.suggest_float('lasso_alpha', 0.005, 0.04, log=True)

        # three possible bahaviors:
        #   - all:  we sample everything
        #   - NNTQ: we sample NNTQ parameters only
        #       (metamodel is not used, baselines are used, but frozen)
        #   - meta: we sample baselines and metamodel parameters
        #       (NNTQ parameters are frozen to values found in `NNTQ` search)

        baseline_parameters= sample_baseline_parameters   (trial, baseline_parameters)\
            if stage in [        'meta', 'all'] else baseline_parameters
        NNTQ_parameters    = sample_NNTQ_parameters       (trial, base_NNTQ_params)\
            if stage in ['NNTQ',         'all'] else base_NNTQ_params.copy()
        metamodel_parameters=sample_metamodel_NN_parameters(trial,base_meta_NN_params)\
            if stage in [        'meta', 'all'] else base_meta_NN_params.copy()

        if stage == 'NNTQ':
            metamodel_parameters['epochs'] = 1  # for speed

        dict_row, df_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
            (num_worst_days, avg_abs_worst_days_train), (loss_NNTQ, loss_meta) = \
                run.run_model_once(
                  # configuration bundles
                  baseline_parameters= baseline_parameters,
                  NNTQ_parameters   = NNTQ_parameters,
                  metamodel_NN_parameters=metamodel_parameters,
                  dict_fnames       = dict_fnames,

                  # statistics of the dataset
                  minutes_per_step  = minutes_per_step,
                  train_split_fraction=train_split_fraction,
                  val_ratio         = val_ratio,
                  forecast_hour     = forecast_hour,
                  seed              = seed + trial.number,

                  force_calc_baselines=force_calc_baselines,
                  save_cache_baselines= stage == 'NNTQ',  # baselines not sampled
                  save_cache_NNTQ     = stage == 'meta',  # NNTQ  not sampled

                  # XXX_EVERY (in epochs)
                  validate_every    = 999,
                  display_every     = 999,  # dummy
                  plot_conv_every   = 999,  # dummy

                  run_id            = trial.number,
                  cache_dir         = cache_dir,
                  verbose           = 0
        )

        # return the relevant loss
        if stage == 'NNTQ':
            return loss_NNTQ
        if stage == 'meta':
            return loss_meta
        if stage == 'all':
            return loss_NNTQ + loss_meta


    # Load the CSV file containing MC runs
    results_df = pd.read_csv(csv_path, index_col=False)

    # columns from the csv that do not contain optimisable parameters
    # parameters we do not optimise
    cols_not_paras = ['run'] + \
        ['quantiles_0', 'quantiles_1', 'quantiles_2', 'quantiles_3', 'quantiles_4'] + \
        ['RF_type', 'RF_random_state', 'RF_n_jobs'] + \
        ['GB_type', 'GB_objective'] + \
        ['GB_random_state','GB_n_jobs',	'GB_verbose'] + \
        ['input_length', 'pred_length', 'valid_length', 'num_patches'] + \
        ['device', 'metaNN_device', 'features_in_future']
        # ,'GB_boosting_type'

    # output
    cols_not_paras.extend(['q10', 'q25', 'q50', 'q75', 'q90'])  # coverage
    for _model in ['NN', 'LR', 'RF', 'GB']:
        cols_not_paras.append(f'avg_weight_meta_NN_{_model}')
    for _model in ['NN', 'LR', 'RF', 'GB', 'meta_LR', 'meta_NN']:
        for _metric in ['bias', 'RMSE', 'MAE']:
            cols_not_paras.append(f'test_{_model}_{_metric}')
    cols_not_paras.append('avg_abs_worst_days_train')
    # print(cols_not_paras)
    # print([e for e in results_df.columns if e not in cols_not_paras])
    _superfluous = [e for e in cols_not_paras if e not in results_df.columns]
    assert len(_superfluous) == 0, _superfluous

    results_df.drop(columns=cols_not_paras, inplace=True)
    print(f"{csv_path} loaded: {results_df.shape}")

    results_df[['learning_rate', 'weight_decay',
                'metaNN_learning_rate', 'metaNN_weight_decay']] =\
        results_df[['learning_rate', 'weight_decay',
                    'metaNN_learning_rate', 'metaNN_weight_decay']] * 1e-6

    _all_distributions = distributions_baselines | \
                         distributions_NNTQ | distributions_metamodel_NN
    assert set(_all_distributions.keys()) - set(results_df.columns) == set(), \
        set(_all_distributions.keys()) - set(results_df.columns)
    assert set(results_df.columns) - set(_all_distributions.keys()) == \
                {'timestamp', 'loss_NNTQ', 'loss_meta'}, \
                    set(results_df.columns) - set(_all_distributions.keys())

    # can be 'sqrt' or a float => type issues
    results_df['RF_max_features'] = results_df['RF_max_features'].astype(str)



    # Create a study with the previous trials
    study   = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler())

    distributions_keys = _all_distributions.keys()

    # Create a list of FrozenTrial objects
    trials = []
    for index, row in results_df.iterrows():
        # print(index, row)
        _params = {k: row[k] for k in row.keys()
            if k not in ['timestamp', 'loss_NNTQ', 'loss_meta']}
        assert set(_params.keys()) - set(distributions_keys) == set(), \
               set(_params.keys()) - set(distributions_keys)
        assert set(distributions_keys) - set(_params.keys()) == set(), \
               set(distributions_keys) - set(_params.keys())

        # relevant loss
        _value = row['loss_NNTQ'] + row['loss_meta'] if stage == 'all' \
            else row[f'loss_{stage}']

        trial = optuna.trial.FrozenTrial(
            number        = index,  # Trial number
            state         = optuna.trial.TrialState.COMPLETE,  # State of the trial
            datetime_start=   pd.to_datetime(row['timestamp'])-timedelta(minutes=1.5),
            datetime_complete=pd.to_datetime(row['timestamp']),
            value         = _value,  # Objective value
            params        = _params,  # hyperparameters
            user_attrs    = {},  # Additional attributes (can be empty)
            distributions = _all_distributions,
            system_attrs  = {},  # system attributes (can be empty)
            intermediate_values={},  # (can be empty)
            trial_id      = index  # ID of the trial
        )
        trials.append(trial)

    del results_df


    # Add previous trials to the study
    for index, trial in enumerate(trials):
        study.add_trial(trial)


    # Plotting
    # plot_optuna(study)


    # Run optimization
    print(f"\nStarting {num_runs} trials ({stage})...")
    study.optimize(objective, n_trials=num_runs)


    # Print the best parameters found
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # print(f"Best hyperparameters: {study.best_params}")


    # Plotting
    plot_optuna(study)


    return study.best_params



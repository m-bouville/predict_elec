###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# MC_search.py
# Monte Carlo (random) search for hyperparameters
#
###############################################################################


# import os
import copy
import warnings

from   tqdm     import trange

from   typing   import Dict, Any, Optional  # List, Tuple, Sequence,

# from   datetime import datetime

import numpy    as np
import pandas   as pd
import random


import run
from   constants import Stage



def sample_baseline_parameters(base_params: Dict[str, Dict[str, Any]])\
            -> Dict[str, Dict[str, Any]]:

    p0 = copy.deepcopy(base_params)  # Dict[str, Dict[str, Any]]

    for _baseline in p0.keys():
        p = p0[_baseline]   # Dict[str, Any]

        if _baseline == 'LR':
            if 'type' in p.keys():
                p['type']      = random.choice(['lasso'])  # 'ridge'
            if 'alpha' in p.keys():
                p['alpha']      = round(random.uniform(0.5 / 100., 4 / 100.), 5) \
                    if p['type'] == 'lasso'  else round(random.uniform(0.5, 2), 3)

        elif _baseline == 'RF':
            if 'n_estimators' in p.keys():
                p['n_estimators'   ] = int(round(random.uniform(400, 600), -1))
            if 'max_depth' in p.keys():
                p['max_depth'      ] = int(round(random.uniform( 15,  25)))
            if 'min_samples_leaf' in p.keys():
                p['min_samples_leaf']= int(round(random.uniform( 10,  20)))
            if 'min_samples_split' in p.keys():
                p['min_samples_split']=int(round(random.uniform( 15,  25)))
            if 'max_features' in p.keys():
                p['max_features'   ] = random.choice(['sqrt', 0.5])

        elif _baseline == 'LGBM':
            if 'boosting_type' in p.keys():
                p['boosting_type'  ] = random.choice(['gbdt'])  # TODO add more?
            if 'num_leaves' in p.keys():     # Default number of leaves
                p['num_leaves'     ] = random.choice([16, 32]) - 1
            if 'max_depth' in p.keys():
                p['max_depth'      ] = random.choice([3, 4, 5, 6])
            if 'learning_rate' in p.keys():   # Lower learning rate => more stable
                p['learning_rate'  ] = \
                    round(0.05 * 10**random.uniform(np.log10(0.3), np.log10( 3.)), 4)
            if 'n_estimators' in p.keys():   # More trees => more robust model
                p['n_estimators'   ] = int(round(random.uniform(400, 600), -1))
            if 'min_child_samples' in p.keys():
                p['min_child_samples']=int(round(random.uniform( 15,  25)))

            if 'learning_rate' in p.keys():
                p['learning_rate'  ] = \
                    round(0.05  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)

            if 'subsample' in p.keys():  # Fraction of samples used to train each tree
                p['subsample'      ] = round(random.uniform(0.6, 1. ), 3)

            if 'colsample_bytree' in p.keys(): # Fraction of features used for each tree
                p['colsample_bytree']= round(random.uniform(0.6, 1. ), 3)

            if 'reg_alpha' in p.keys():    # L1 regularization
                p['reg_alpha'      ] = \
                    round(0.1  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)

            if 'reg_lambda' in p.keys():   # L2 regularization
                p['reg_lambda'     ] = \
                    round(0.1  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)
    return p0



def sample_NNTQ_parameters(base_params: Dict[str, Any]) -> Dict[str, Any]:

    p = copy.deepcopy(base_params)

    if 'use_ML_features' in p.keys():
        p['use_ML_features'] = random.bool()

    if 'epochs' in p.keys():
        p['epochs']      = int(random.uniform(20, 40))
    if 'batch_size' in p.keys():
        p['batch_size']  = random.choice([32, 64, 96, 128])
    if 'learning_rate' in p.keys():
        p['learning_rate']= \
            round(0.02  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)
    if 'weight_decay' in p.keys():
        p['weight_decay']= \
            round(0.3e-6* 10 ** random.uniform(np.log10(0.1), np.log10(10.)), 9)
    if 'dropout' in p.keys():
        p['dropout']     = round(random.uniform(0.02, 0.15), 3)

    # quantile loss weights
    if 'lambda_cross' in p.keys():
        p['lambda_cross']   = round(random.uniform(0.2,  1.), 3)
    if 'lambda_coverage' in p.keys():
        p['lambda_coverage']= round(random.uniform(0.2,  1.), 2)
    if 'lambda_deriv' in p.keys():
        p['lambda_deriv']   = round(random.uniform(0.0,  0.3), 4)
    if 'lambda_median' in p.keys():
        p['lambda_median']  = round(random.uniform(0.2,  0.5), 3)
    if 'smoothing_cross' in p.keys():
        p['smoothing_cross']= round(random.uniform(0.005,0.05),4)

        # temperature-dependence (pinball loss, coverage penalty)
    if 'threshold_cold_degC' in p.keys():
        p['threshold_cold_degC']= round(random.uniform( 0.,  5.), 1)
    if 'saturation_cold_degC' in p.keys():
        p['saturation_cold_degC']=round(random.uniform(-8., -2.), 1)
    if 'lambda_cold' in p.keys():
        p['lambda_cold']= round(random.uniform(0., 0.2), 3)


    # # quantiles
    # p['quantiles'] = random.choice(modifiers['quantiles'])

    # # patch geometry
    # scale = random.uniform(*modifiers['patch_length_scale'])
    # patch_length = max(1, int(base_params['patch_length'] * scale))
    # p['patch_length'] = patch_length

    # stride_ratio = random.uniform(*modifiers['stride_ratio'])
    # p['stride'] = max(1, int(round(patch_length * stride_ratio)))


    # architecture
    if 'model_dim' in p.keys():
        p['model_dim']  = int(128 * random.choice([0.75, 1.0, 1.25]))
    if 'ffn_size' in p.keys():
        p['ffn_size']   = random.choice([2, 3, 4, 5, 6])
    if 'num_heads' in p.keys():
        p['num_heads']  = random.choice([2, 3, 4, 5, 6])
    if 'num_layers' in p.keys():
        p['num_layers'] = random.choice([1, 2, 3, 4, 5])

    # enforce divisibility (Transformer constraint)
    if 'model_dim' in p and 'num_heads' in p:
        if  p['model_dim'] % p['num_heads'] != 0:
            p['model_dim'] = p['num_heads'] * (p['model_dim'] // p['num_heads'])

    if 'num_geo_blocks' in p.keys():
        p['num_geo_blocks']= int(round(random.uniform(2,    8)))

    # early stopping
    if 'warmup_steps' in p.keys():
        p['warmup_steps']  = int(round(random.uniform(1500, 4000), -2))
    if 'patience' in p.keys():
        p['patience']      = int(      random.uniform(4,    10))
    if 'min_delta' in p.keys():
        p['min_delta']     =     round(random.uniform(15e-3,30e-3), 5)

    # derived quantities
    if 'input_length' in p and 'features_in_future' in p and \
                'pred_length' in p and 'patch_length' in p and 'stride' in p:
        p['num_patches'] = (
            p['input_length']
            + p['features_in_future'] * p['pred_length']
            - p['patch_length']
        ) // p['stride'] + 1

    return p



def sample_metamodel_NN_parameters(base_params : Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(base_params)

    if 'epochs' in p.keys():
        p['epochs']      = int(random.uniform(8, 15))

    if 'batch_size' in p.keys():
        p['batch_size']  = random.choice([192, 256, 384, 512])

    if 'learning_rate' in p.keys():
        p['learning_rate']= \
            round(0.5e-3* 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 5)

    if 'weight_decay' in p.keys():
        p['weight_decay']= \
            round(10e-6 * 10 ** random.uniform(np.log10(0.1), np.log10(10.)), 7)

    if 'dropout' in p.keys():
        p['dropout']     = round(random.uniform(0.02, 0.2), 2)

    # num_cells: scale but keep integers ≥ 4
    if 'num_cells' in p.keys():
        scales = [random.choice([0.75, 1.0, 1.25]),
              random.choice([0.75, 1.0, 1.25])]
        p['num_cells'] = [max(4, int(c * scales[i])) for i, c in enumerate([32, 16])]

    # early stopping
    if 'patience' in p.keys():
        p['patience'] =       random.choice ([2, 3, 4, 5, 6])
    if 'factor' in p.keys():
        p['factor']   = round(random.uniform(0.6, 0.85), 3)

    return p


# def sample_meta_params(base_cfg, rng):
#     cfg = deepcopy(base_cfg)
#     cfg["lr"]         = 10 ** rng.uniform(-4.5, -3.0)
#     cfg["dropout"]    = rng.uniform(0.05, 0.4)
#     # cfg["num_cells"]  = rng.choice([[32,16], [32,32], [64,32]])
#     return cfg


def run_Monte_Carlo_search(
            stage               : Stage,    # only Stage.all is implemented
            num_trials          : int,

            # configuration bundles
            base_baseline_params: Dict[str, Dict[str, Any]],
            base_NNTQ_params    : Dict[str, Any],
            base_meta_NN_params : Dict[str, Any],
            dict_input_csv_fnames:Dict[str, str],
            csv_path            : str,

            # statistics of the dataset
            minutes_per_step    : int,
            train_split_fraction: float,
            valid_ratio         : float,
            forecast_hour       : int,
            seed                : int,
            force_calc_baselines: bool,
            cache_dir           : Optional[str] = None,
            # csv_path            : str    = 'parameter_search.csv',
            verbose             : int  = 0
        ):

    if stage != Stage.all:
        warnings.warn(f"stage ({stage.value}) will not be used")
    # csv_path: str = 'parameter_search_all.csv'

    # from   sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO fix for real

    # rng = np.random.default_rng(seed)
    list_results = []

    for run_id in trange(num_trials, desc="MC runs"):
        # print(f"Starting run {run_id} out of {num_runs}")
        # meta_cfg = sample_meta_params(base_meta_NN_params, rng)


        baseline_parameters = sample_baseline_parameters    (base_baseline_params)
        NNTQ_parameters     = sample_NNTQ_parameters        (base_NNTQ_params)
        metamodel_parameters= sample_metamodel_NN_parameters(base_meta_NN_params)

        _, dict_row, df_metrics, avg_weights_meta_NN, quantile_delta_coverage, \
            (num_worst_days, worst_days_test), (_loss_NNTQ, _loss_meta) = \
                run.run_model_once(
                  # configuration bundles
                  baseline_parameters= baseline_parameters,
                  NNTQ_parameters   = NNTQ_parameters,
                  metamodel_NN_parameters=metamodel_parameters,
                  dict_input_csv_fnames= dict_input_csv_fnames,
                  csv_path          = csv_path,

                  # statistics of the dataset
                  minutes_per_step  = minutes_per_step,
                  train_split_fraction=train_split_fraction,
                  valid_ratio       = valid_ratio,
                  forecast_hour     = forecast_hour,
                  seed              = seed,

                  force_calc_baselines=False,
                  save_cache_baselines= False,  # these parameters
                  save_cache_NNTQ     = False,  #   will never be seen again

                  # XXX_EVERY (in epochs)
                  validate_every    = 1,
                  display_every     = 1,  # dummy
                  plot_conv_every   = 1,  # dummy
                  run_id            = run_id,

                  cache_dir         = cache_dir,
                  verbose           = verbose
        )

        list_results.append(dict_row)

    return pd.DataFrame(list_results)


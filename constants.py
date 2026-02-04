###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# constants.py
# Parameters for the two neural networks, regression, random forest, LGBM
#
###############################################################################


__all__ = ['SEED', 'TRAIN_SPLIT_FRACTION', 'VALID_RATIO',
           'VALIDATE_EVERY', 'DISPLAY_EVERY', 'PLOT_CONV_EVERY',
           'DICT_INPUT_CSV_FNAMES', 'CACHE_FNAME',
           'FORECAST_HOUR', 'MINUTES_PER_STEP', 'NUM_STEPS_PER_DAY',
           'BASELINES_PARAMETERS', 'NNTQ_PARAMETERS', 'METAMODEL_NN_PARAMETERS']


from   typing import Dict, Any  # Tuple, List, Sequence  #, Optional

import torch

from enum import Enum


# from   IO import normalize_name




class Split(Enum):
    train   = 'train'
    valid   = 'valid'
    test    = 'test'
    complete= 'complete'

class Stage(Enum):
    meta = 'meta'
    NNTQ = 'NNTQ'
    all  = 'all'


MINUTES_PER_STEP  = 30
NUM_STEPS_PER_DAY = int(round(24*60/MINUTES_PER_STEP))
def days_to_steps(num_days: float, num_steps_per_day=NUM_STEPS_PER_DAY) -> int:
    return int(round(num_days*num_steps_per_day))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. CONFIGURATION CONSTANTS
# ============================================================

SEED         =   0              # For reproducibility


TRAIN_SPLIT_FRACTION=0.8
VALID_RATIO  =   0.25           # validation from training set

FORECAST_HOUR:int = 12          # 12: noon



# NN model with Transformer and quantiles
_patch_length = days_to_steps(0.5)

NNTQ_PARAMETERS: dict = {
    'use_ML_features'  : 0,  # Boolean as int for compatibility with Optuna
    'device'           : DEVICE,

    'input_length'     : days_to_steps(14),  # How many half-hours the model sees
    'pred_length'      : days_to_steps( 1 + (24.-FORECAST_HOUR)/24),
        # start at noon, finish at midnight the next day
    'valid_length'     : days_to_steps( 1),       # 24h: full day ahead
    'features_in_future':True,                 # features do not stop at noon

    'epochs'           : 24,   # Number of training epochs  # Bayes: 20
    'batch_size'       : 96,   # Training batch size

    # architecture size
    'model_dim'        : 500,  # Transformer embedding dimension
    'num_layers'       : 5,    # Number of transformer encoder layers
    'num_heads'        : 5,    # Number of attention heads
    'ffn_size'         : 7,    # expansion factor
    'num_geo_blocks'   : 6,    # Number of geometric blocks

    # optimizer
    'learning_rate'    : 0.0036,  # Optimizer learning rate
    'weight_decay'     : 1.5e-7,
    'dropout'          : 0.38,
    'warmup_steps'     : 3000,

    # early stopping
    'patience'         : 5,
    'min_delta'        : 0.038,

    # PatchEmbedding
    'patch_length'     : 48,  # [half-hours]
    'stride'           : 24,  # [half-hours]  # max(int(round(_patch_length/2)), 1),

    # geometric blocks
    'geo_block_ratio'  : 1,
        # each block is a fraction of the size of the previous (geometric)

    # quantile loss
    'quantiles'        : (0.1, 0.25, 0.5, 0.75, 0.9),
    'lambda_cross'     : 0.064,   # enforcing correct order of quantiles
    'lambda_coverage'  : 0.084,
    'lambda_deriv'     : 0.062,   # derivative weight in loss function
    'lambda_median'    : 0.0,
    'smoothing_cross'  : 0.044,

        # temperature-dependence (pinball loss, coverage penalty):
        #   lambda * {1 + lambda_cold * [(threshold_cold_degC - Tavg_degC) / dT_K,
        #       clipped to interval [0, 1])]}
        #   where dT_K = (threshold_cold_degC - saturation_cold_degC)
    'saturation_cold_degC':-7.6,
    'threshold_cold_degC': -0.2,
    'lambda_cold'      :    0.22,

    'lambda_regions'   :    0.034,
    'lambda_regions_sum':   0.32,
}

# NNTQ_PARAMETERS['num_patches'] = \
#     (NNTQ_PARAMETERS['input_length'] + NNTQ_PARAMETERS['features_in_future'] * \
#                 NNTQ_PARAMETERS['pred_length'] - NNTQ_PARAMETERS['patch_length'])\
#         // NNTQ_PARAMETERS['stride'] + 1


# # Optional: plug in parameters averaged over the best 5 Bayesian trials
# NNTQ_PARAMETERS.update(

# )


VALIDATE_EVERY=  1
DISPLAY_EVERY=   2
PLOT_CONV_EVERY=10



# NN metamodel

METAMODEL_NN_PARAMETERS: dict = {
    'batch_size'       :   96,
    'num_cells'        : [60, 24],

    'epochs'           :   12,

    # optimizer
    'learning_rate'    :  3.6e-3,
    'weight_decay'     :  1.5e-7,
    'dropout'          :  0.11,

    # early stopping
    'patience'         :   4,
    'factor'           :   0.7,

    'device'           : DEVICE,
    }



DICT_INPUT_CSV_FNAMES = {
    "consumption":          'data/consommation-quotidienne-brute.csv',
    "consumption_by_region":'data/consommation-quotidienne-brute-regionale.csv',
    "temperature":          'data/temperature-quotidienne-regionale.csv',
    # "solar":     'data/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv',
    "price":                'data/wholesale_electricity_price_hourly.csv'
}
CACHE_FNAME = None  #  "cache/merged_aligned.csv"


BASELINES_PARAMETERS = {
    'LR': {"type": "ridge", "alpha": 1.3, 'max_iter': 2_000
    },
    'RF': {
        "type":            "rf",
        "n_estimators":    500,
        "max_depth":        25,
        "min_samples_leaf": 17,
        "min_samples_split":12,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'LGBM': {
        "type":          "lgbm",
        "objective":     "regression",
        "boosting_type": "gbdt",
        "num_leaves":       16-1,     # Default number of leaves
        "max_depth":         7,       # Moderate tree depth
        "learning_rate":     0.16,    # Lower learning rate for stability
        "n_estimators":    350,       # More trees for a robust model
        "min_child_samples":12,       # Minimum samples per leaf
        "subsample":         0.9,    # Fraction of samples used to train each tree
        "colsample_bytree":  0.62,    # Fraction of features used for each tree
        "reg_alpha":         0.09,   # L1 regularization
        "reg_lambda":        0.15,   # L2 regularization
        "random_state":      0,       # Seed for reproducibility
        "n_jobs":            4,       # Number of parallel jobs
        "verbose":          -1        # Suppress output
    }
}



# # Optional: plug in Bayesian best parameters
# for _model in ['LR', 'RF', 'LGBM']:
#     BASELINES_PARAMETERS[_model].update(_new_parameters[_model])
# METAMODEL_NN_PARAMETERS.update(_new_parameters['metaNN'])




# Checking
assert NNTQ_PARAMETERS['model_dim']  % NNTQ_PARAMETERS['num_heads'] == 0, \
    f"MODEL_DIM ({NNTQ_PARAMETERS['model_dim']}) must be divisible by " \
    f"NUM_HEADS ({NNTQ_PARAMETERS['num_heads']})."

assert 1 <= VALIDATE_EVERY <= min(NNTQ_PARAMETERS['epochs'],
                                  NNTQ_PARAMETERS['patience']), \
    (VALIDATE_EVERY, NNTQ_PARAMETERS['epochs'], NNTQ_PARAMETERS['patience'])

_quantiles = NNTQ_PARAMETERS['quantiles']
num_quantiles = len(_quantiles)
assert all([_quantiles[i] + _quantiles[num_quantiles - i - 1] == 1
            for i in range(num_quantiles // 2)]), \
    "quantiles should be symmetric"    # otherwise: hard to interpret
assert _quantiles[num_quantiles // 2] == 0.5, "middle quantile must be the median"
    # the code assumes it is





# ============================================================
# Parameters to run faster
# ============================================================

def fast_parameters(nntq_parameters        : Dict[str, Any],
                    metamodel_nn_parameters: Dict[str, Any]
                ) -> [Dict[str, Dict[str, Any]], Dict[str, Any],  Dict[str, Any]]:
    nntq_parameters     ['epochs'        ] =  2
    nntq_parameters     ['model_dim'     ] = 50
    nntq_parameters     ['num_layers'    ] =  1
    nntq_parameters     ['num_heads'     ] =  2
    nntq_parameters     ['ffn_size'      ] =  2
    nntq_parameters     ['num_geo_blocks'] =  2

    metamodel_nn_parameters['epochs'     ] =  1
    metamodel_nn_parameters['num_cells'  ] =  [20, 10]

    return (baseline_params_fast, nntq_parameters, metamodel_nn_parameters)


baseline_params_fast = {
    # "oracle": {1},  # (content is just a place-holder)
    'LR': {"type": "lasso", "alpha": 5 / 100., 'max_iter': 2_000},
    'RF': {
        "type":            "rf",
        "n_estimators":     50,     # was 300 -> fewer trees
        "max_depth":         6,     # shallower trees
        "min_samples_leaf": 10,     # more regularization
        "min_samples_split": 2,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'LGBM': {
        "type":     "lgbm",
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 16-1,           # Fewer leaves for simplicity
        "max_depth": 4,               # Shallower trees
        "learning_rate": 0.1,         # Learning rate
        "n_estimators": 50,           # Fewer trees for faster training
        "min_child_samples": 20,      # Minimum samples per leaf
        "subsample": 0.8,             # Fraction of samples used to train each tree
        "colsample_bytree": 0.8,      # Fraction of features training each tree
        "reg_alpha": 0.1,             # L1 regularization
        "reg_lambda": 0.1,            # L2 regularization
        "random_state": 0,            # Seed for reproducibility
        "n_jobs": 4,                  # Number of parallel jobs
        "verbose": -1                 # Suppress output
    }
}

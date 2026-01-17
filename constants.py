__all__ = ['SEED', 'TRAIN_SPLIT_FRACTION', 'VALID_RATIO',
           'VALIDATE_EVERY', 'DISPLAY_EVERY', 'PLOT_CONV_EVERY',
           'DICT_INPUT_CSV_FNAMES', 'CACHE_FNAME',
           'FORECAST_HOUR', 'MINUTES_PER_STEP', 'NUM_STEPS_PER_DAY',
           'BASELINES_PARAMETERS', 'NNTQ_PARAMETERS', 'METAMODEL_NN_PARAMETERS']


from   typing import Dict, Any  # Tuple, List, Sequence  #, Optional

import torch


# from   IO import normalize_name




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
    'device'           : DEVICE,
    'input_length'     : days_to_steps(14),  # How many half-hours the model sees
    'pred_length'      : days_to_steps( 1 + (24.-FORECAST_HOUR)/24),
        # start at noon, finish at midnight the next day
    'valid_length'     : days_to_steps( 1),       # 24h: full day ahead
    'features_in_future':True,                 # features do not stop at noon

    'epochs'           :  40,                  # Number of training epochs
    'batch_size'       :  32,                   # Training batch size

    # architecture size
    'model_dim'        :301,                # Transformer embedding dimension
    'num_layers'        : 2,               # Number of transformer encoder layers
    'num_heads'         : 7,                # Number of attention heads
    'ffn_size'          : 3,                # expansion factor
    'num_geo_blocks'   :  5,

    # optimizer
    'learning_rate'    :  0.011,            # Optimizer learning rate
    'weight_decay'     : 11.25e-9,
    'dropout'          :  0.24,
    'warmup_steps'     :2500,

    # early stopping
    'patience'         :  7,
    'min_delta'        :  0.022,

    # PatchEmbedding
    'patch_length'     : _patch_length,              # [half-hours]
    'stride'           : max(int(round(_patch_length/2)), 1), # [half-hours]

    # geometric blocks
    'geo_block_ratio'  : 1,
         # each block is a fraction of the size of the previous (geometric)

    # quantile loss
    'quantiles'        : (0.1, 0.25, 0.5, 0.75, 0.9),
    'lambda_cross'     : 0.028,          # enforcing correct order of quantiles
    'lambda_coverage'  : 0.34,
    'lambda_deriv'     : 0.064,         # derivative weight in loss function
    'lambda_median'    : 0.08,
    'smoothing_cross'  : 0.025,

        # temperature-dependence (pinball loss, coverage penalty):
        #   lambda * {1 + lambda_cold * [(threshold_cold_degC - Tavg_degC) / dT_K,
        #       clipped to interval [0, 1])]}
        #   where dT_K = (threshold_cold_degC - saturation_cold_degC)
    'saturation_cold_degC':-6.5,
    'threshold_cold_degC':  5.,
    'lambda_cold'      :    0.13,

    'lambda_regions'   :    0.05,
    'lambda_regions_sum':   0.1,
}

NNTQ_PARAMETERS['num_patches'] = \
    (NNTQ_PARAMETERS['input_length'] + NNTQ_PARAMETERS['features_in_future'] * \
                NNTQ_PARAMETERS['pred_length'] - NNTQ_PARAMETERS['patch_length'])\
        // NNTQ_PARAMETERS['stride'] + 1


# # Optional: plug in Bayesian best parameters
NNTQ_PARAMETERS.update(
    # {'patch_length': 36, 'stride': 12, 'input_length': 768, 'epochs': 15,
    #  'batch_size': 64, 'learning_rate': 0.0122, 'weight_decay': 1.5650e-09,
    #  'dropout': 0.08, 'lambda_cross': 0.016, 'lambda_coverage': 0.2,
    #  'lambda_deriv': 0.028, 'lambda_median': 0.04, 'smoothing_cross': 0.049,
    #  'threshold_cold_degC': 0.30, 'saturation_cold_degC': -3.6, 'lambda_cold': 0.09,
    #  'lambda_regions': 0.044, 'lambda_regions_sum': 0.42, 'model_dim': 350,
    #  'ffn_size': 4, 'num_heads': 5, 'num_layers': 1, 'num_geo_blocks': 8,
    #  'warmup_steps': 1500, 'patience': 6, 'min_delta': 0.034
    #  }  # trial 358: avg loss 18.90 over 7 runs [data -> 11/25]
    #     #   losses [11.11, 19.03, 18.63, 17.77, 20.1, 18.98, 20.16]

    # {'patch_length': 36, 'stride': 12, 'input_length': 576, 'epochs': 15,
    #  'batch_size': 64, 'learning_rate': 0.0138, 'weight_decay': 1.421e-09,
    #  'dropout': 0.1, 'lambda_cross': 0.008, 'lambda_coverage': 0.16,
    #  'lambda_deriv': 0.028, 'lambda_median': 0.04, 'smoothing_cross': 0.055,
    #  'threshold_cold_degC': -0.3, 'saturation_cold_degC': -3.9, 'lambda_cold': 0.09,
    #  'lambda_regions': 0.054, 'lambda_regions_sum': 0.42, 'model_dim': 375,
    #  'ffn_size': 4, 'num_heads': 5, 'num_layers': 1, 'num_geo_blocks': 8,
    #  'warmup_steps': 1500, 'patience': 6, 'min_delta': 0.034
    #  }  # trial 373: avg loss 15.86 over 7 runs [data -> 11/25]
    #     #   losses [17.84, 15.14, 21.44, 19.22, 15.58, 21.17, 12.12]

    # {'patch_length': 36, 'stride': 15, 'input_length': 576, 'epochs': 15,
    #  'batch_size': 64, 'learning_rate': 0.0158, 'weight_decay': 1.56054e-09,
    #  'dropout': 0.11, 'lambda_cross': 0.0, 'lambda_coverage': 0.18,
    #  'lambda_deriv': 0.048, 'lambda_median': 0.04, 'smoothing_cross': 0.057,
    #  'threshold_cold_degC': -0.8, 'saturation_cold_degC': -4.2, 'lambda_cold': 0.13,
    #  'lambda_regions': 0.052, 'lambda_regions_sum': 0.42, 'model_dim': 525,
    #  'ffn_size': 4, 'num_heads': 5, 'num_layers': 4, 'num_geo_blocks': 8,
    #  'warmup_steps': 2000, 'patience': 6, 'min_delta': 0.034
    #  }  # trial 420: avg loss 12.79 over 7 runs [data -> 11/25]
    #     #   7 runs: losses [8.73, 11.42, 18.16, 8.84, 18.88, 18.36, 21.07] -> avg 15.13

    {'patch_length': 36, 'stride': 15, 'input_length': 576, 'epochs': 15,
     'batch_size': 64, 'learning_rate': 0.0162, 'weight_decay': 1.4469e-09,
     'dropout': 0.12, 'lambda_cross': 0.0, 'lambda_coverage': 0.18,
     'lambda_deriv': 0.048, 'lambda_median': 0.04, 'smoothing_cross': 0.057,
     'threshold_cold_degC': -1.0, 'saturation_cold_degC': -4.3, 'lambda_cold': 0.13,
     'lambda_regions': 0.052, 'lambda_regions_sum': 0.42, 'model_dim': 525,
     'ffn_size': 4, 'num_heads': 5, 'num_layers': 4, 'num_geo_blocks': 8,
     'warmup_steps': 2000, 'patience': 6, 'min_delta': 0.034
     }  # trial 426: avg loss 8.81 over 7 runs [data -> 11/25]
        #   losses [8.62, 7.06, 7.37, 9.67, 9.45, 8.94, 17.88] -> avg 8.81
   )



VALIDATE_EVERY=  1
DISPLAY_EVERY=   2
PLOT_CONV_EVERY=10



# NN metamodel

METAMODEL_NN_PARAMETERS: dict = {
    'batch_size'       :  256,
    'num_cells'        : [40, 20],

    'epochs'           :   12,

    # optimizer
    'learning_rate'    :  5e-4,
    'weight_decay'     :  6e-6,
    'dropout'          :  0.1,

    # early stopping
    'patience'         :   4,
    'factor'           :   0.7,

    'device'           : DEVICE,
    }



DICT_INPUT_CSV_FNAMES = {
    "consumption":          "data/consommation-quotidienne-brute.csv",
    "consumption_by_region":'data/consommation-quotidienne-brute-regionale.csv',
    "temperature":          'data/temperature-quotidienne-regionale.csv',
    # "solar":                'data/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv'
}
CACHE_FNAME = None  #  "cache/merged_aligned.csv"


BASELINES_PARAMETERS = {
    # 'lasso': {"alpha": 0.00, 'max_iter': 2_000},
    # "oracle": {1},  # (content is just a place-holder)
    'LR': {"type": "ridge", "alpha": 0.25, 'max_iter': 2_000},
    'RF': {
        "type":            "rf",
        "n_estimators":    350,
        "max_depth":        15,
        "min_samples_leaf": 15,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'LGBM': {
        "type":          "lgbm",
        "objective":     "regression",
        "boosting_type": "gbdt",
        "num_leaves":       32-1,     # Default number of leaves
        "max_depth":         5,       # Moderate tree depth
        "learning_rate":     0.02,    # Lower learning rate for stability
        "n_estimators":    500,       # More trees for a robust model
        "min_child_samples":25,       # Minimum samples per leaf
        "subsample":         0.6,    # Fraction of samples used to train each tree
        "colsample_bytree":  0.7,    # Fraction of features used for each tree
        "reg_alpha":         0.1,     # L1 regularization
        "reg_lambda":        0.1,    # L2 regularization
        "random_state":      0,       # Seed for reproducibility
        "n_jobs":            4,       # Number of parallel jobs
        "verbose":          -1        # Suppress output
    }
}



# # Optional: plug in Bayesian best parameters
# _new_parameters = { #'lasso_alpha': 0.0,
#     'LR_type': 'ridge', 'LR_alpha': 1.45, 'RF_n_estimators': 490, 'RF_max_depth': 23,
#     'RF_min_samples_leaf': 11, 'RF_min_samples_split': 12, 'RF_max_features': 'sqrt',
#     'LGBM_boosting_type': 'gbdt', 'LGBM_num_leaves': 15, 'LGBM_max_depth': 2,
#     'LGBM_learning_rate': 0.07, 'LGBM_n_estimators': 630, 'LGBM_min_child_samples': 13,
#     'LGBM_subsample': 0.72, 'LGBM_colsample_bytree': 0.98, 'LGBM_reg_alpha': 0.06,
#     'LGBM_reg_lambda': 0.08, 'metaNN_epochs': 20, 'metaNN_batch_size': 96,
#     'metaNN_learning_rate': 0.0025, 'metaNN_weight_decay': 3.054e-08,
#     'metaNN_dropout': 0.11, 'metaNN_num_cells_0': 64, 'metaNN_num_cells_1': 32
# } # trial 51: 1.207 (Bayes) -> 1.23 (one-off)

# _new_parameters = {'LR_type': 'ridge', 'LR_alpha': 1.2, 'RF_n_estimators': 500,
#     'RF_max_depth': 20, 'RF_min_samples_leaf': 10, 'RF_min_samples_split': 15,
#     'RF_max_features': 'sqrt', 'LGBM_boosting_type': 'gbdt',
#     'LGBM_num_leaves': 15, 'LGBM_max_depth': 3, 'LGBM_learning_rate': 0.01,
#     'LGBM_n_estimators': 420, 'LGBM_min_child_samples': 12, 'LGBM_subsample': 1.0,
#     'LGBM_colsample_bytree': 0.9, 'LGBM_reg_alpha': 0.08, 'LGBM_reg_lambda': 0.09,
#     'metaNN_epochs': 18, 'metaNN_batch_size': 96, 'metaNN_learning_rate': 0.0035,
#     'metaNN_weight_decay': 2.2787e-07, 'metaNN_dropout': 0.0,
#     'metaNN_num_cells_0': 40, 'metaNN_num_cells_1': 16
# }  # trial 15: avg loss 1.3474 over 5 runs [data -> 11/25]
#    #   losses [1.3332, 1.3484, 1.3454, 1.3503, 1.3485]

# for _model in ['LR', 'RF', 'LGBM']:
#     BASELINES_PARAMETERS[_model].update(
#         {k.removeprefix(_model+'_'): v
#                         for (k, v) in _new_parameters.items() if _model in k})

# METAMODEL_NN_PARAMETERS.update({k.removeprefix('metaNN_'): v
#                         for (k, v) in _new_parameters.items() if 'metaNN' in k})




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
    nntq_parameters     ['model_dim'     ] = 48
    nntq_parameters     ['num_layers'    ] =  1
    nntq_parameters     ['num_heads'     ] =  2
    nntq_parameters     ['ffn_size'      ] =  2
    nntq_parameters     ['num_geo_blocks'] =  2

    metamodel_nn_parameters['epochs'     ] =  1
    metamodel_nn_parameters['num_cells'  ] =  [20, 10]

    return (baseline_params_fast, nntq_parameters, metamodel_nn_parameters)


baseline_params_fast = {
    # 'lasso': {"alpha": 0.04, 'max_iter': 2_000},
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
        "subsample": 0.8,             # Fraction of samples used for training each tree
        "colsample_bytree": 0.8,      # Fraction of features used for training each tree
        "reg_alpha": 0.1,             # L1 regularization
        "reg_lambda": 0.1,            # L2 regularization
        "random_state": 0,            # Seed for reproducibility
        "n_jobs": 4,                  # Number of parallel jobs
        "verbose": -1                 # Suppress output
    }
}

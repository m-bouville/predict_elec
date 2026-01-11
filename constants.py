__all__ = ['RUN_FAST', 'SEED', 'TRAIN_SPLIT_FRACTION', 'VALID_RATIO',
           'VALIDATE_EVERY', 'DISPLAY_EVERY', 'PLOT_CONV_EVERY',
           'DICT_INPUT_CSV_FNAMES', 'CACHE_FNAME',
           'FORECAST_HOUR', 'MINUTES_PER_STEP', 'NUM_STEPS_PER_DAY',
           'BASELINES_PARAMETERS', 'NNTQ_PARAMETERS', 'METAMODEL_NN_PARAMETERS']


import  torch




MINUTES_PER_STEP  = 30
NUM_STEPS_PER_DAY = int(round(24*60/MINUTES_PER_STEP))
def days_to_steps(num_days: float, num_steps_per_day=NUM_STEPS_PER_DAY) -> int:
    return int(round(num_days*num_steps_per_day))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. CONFIGURATION CONSTANTS
# ============================================================

RUN_FAST     = False         # True: smaller system: runs faster, for debugging

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

    'batch_size'       :  32,                   # Training batch size

    # optimizer
    'learning_rate'    :  0.011,      # Optimizer learning rate
    'weight_decay'     : 11.25e-9,
    'dropout'          :  0.24,

    # early stopping
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
}

EPOCHS       = [  2,  40] # Number of training epochs
MODEL_DIM    = [ 48, 301] # Transformer embedding dimension
NUM_HEADS    = [  2,   7] # Number of attention heads
FFN_SIZE     = [  4,   3] # expansion factor
NUM_LAYERS   = [  1,   2] # Number of transformer encoder layers
NUM_GEO_BLOCKS=[  2,   5]
WARMUP_STEPS =[4000,2500]
PATIENCE     = [  5,   7]  # DEBUG: patience > nb epochs

# Pick correct value from list of possibilites
IDX_RUN_FAST = {True: 0, False: 1}[RUN_FAST]

NNTQ_PARAMETERS['epochs']       = EPOCHS        [IDX_RUN_FAST]
NNTQ_PARAMETERS['model_dim']    = MODEL_DIM     [IDX_RUN_FAST]
NNTQ_PARAMETERS['warmup_steps'] = WARMUP_STEPS  [IDX_RUN_FAST]
NNTQ_PARAMETERS['patience']     = PATIENCE      [IDX_RUN_FAST]
NNTQ_PARAMETERS['num_layers']   = NUM_LAYERS    [IDX_RUN_FAST]
NNTQ_PARAMETERS['num_heads']    = NUM_HEADS     [IDX_RUN_FAST]
NNTQ_PARAMETERS['ffn_size']     = FFN_SIZE      [IDX_RUN_FAST]
NNTQ_PARAMETERS['num_geo_blocks']=NUM_GEO_BLOCKS[IDX_RUN_FAST]

NNTQ_PARAMETERS['num_patches'] = \
    (NNTQ_PARAMETERS['input_length'] + NNTQ_PARAMETERS['features_in_future'] * \
                NNTQ_PARAMETERS['pred_length'] - NNTQ_PARAMETERS['patch_length'])\
        // NNTQ_PARAMETERS['stride'] + 1


# # Optional: plug in Bayesian best parameters
# NNTQ_PARAMETERS.update({ #'lasso_alpha': 0.0,

#     # 'epochs': 41, 'batch_size': 32, 'learning_rate': 0.0098, 'weight_decay': 1.125e-08,
#     # 'dropout': 0.24, 'lambda_cross': 0.028, 'lambda_coverage': 0.34,
#     # 'lambda_deriv': 0.064, 'lambda_median': 0.08, 'smoothing_cross': 0.025,
#     # 'threshold_cold_degC': 4.9, 'saturation_cold_degC': -6.6, 'lambda_cold': 0.13,
#     # 'model_dim': 301, 'ffn_size': 3, 'num_heads': 7, 'num_layers': 2,
#     # 'num_geo_blocks': 5, 'warmup_steps': 2500, 'patience': 7, 'min_delta': 0.022
#     # }  # loss 13 -> 20  => these are the values used above

#     # 'epochs': 41, 'batch_size': 64, 'learning_rate': 0.0106, 'weight_decay': 1.26e-08,
#     # 'dropout': 0.24, 'lambda_cross': 0.032, 'lambda_coverage': 0.28,
#     # 'lambda_deriv': 0.044, 'lambda_median': 0.056, 'smoothing_cross': 0.041,
#     # 'threshold_cold_degC': 2.1, 'saturation_cold_degC': -2.9, 'lambda_cold': 0.07,
#     # 'model_dim': 348, 'ffn_size': 4, 'num_heads': 6, 'num_layers': 2,
#     # 'num_geo_blocks': 9, 'warmup_steps': 2000, 'patience': 6, 'min_delta': 0.03
#     # }  # trial 165, loss 21.5 -> 22

#     # 'epochs': 40, 'batch_size': 64, 'learning_rate': 0.0106, 'weight_decay': 1.68e-09,
#     # 'dropout': 0.24, 'lambda_cross': 0.032, 'lambda_coverage': 0.28,
#     # 'lambda_deriv': 0.012, 'lambda_median': 0.04, 'smoothing_cross': 0.027,
#     # 'threshold_cold_degC': 3.0, 'saturation_cold_degC': -3.0, 'lambda_cold': 0.07,
#     # 'model_dim': 348, 'ffn_size': 2, 'num_heads': 6, 'num_layers': 2,
#     # 'num_geo_blocks': 9, 'warmup_steps': 1500, 'patience': 5, 'min_delta': 0.03
#     # }  # loss 25 -> 30
# )



VALIDATE_EVERY=  1
DISPLAY_EVERY=   2
PLOT_CONV_EVERY=10



# NN metamodel

METAMODEL_NN_PARAMETERS: dict = {
    'batch_size'       :  256,
    'num_cells'        : [40, 20],

    # optimizer
    'learning_rate'    :  4e-4,
    'weight_decay'     :  6e-6,
    'dropout'          :  0.1,

    # early stopping
    'patience'         :   4,
    'factor'           :   0.7,

    'device'           : DEVICE,
    }

META_EPOCHS = [ 1, 12]
METAMODEL_NN_PARAMETERS['epochs'] = META_EPOCHS[IDX_RUN_FAST]



DICT_INPUT_CSV_FNAMES = {
    "consumption": "data/consommation-quotidienne-brute.csv",
    "temperature": 'data/temperature-quotidienne-regionale.csv',
    "solar":       'data/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv'
}
CACHE_FNAME = None  #  "cache/merged_aligned.csv"



baseline_params_fast = {
    'lasso': {"alpha": 0.04, 'max_iter': 2_000},
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

baseline_params_normal = {
    'lasso': {"alpha": 0.00, 'max_iter': 2_000},
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
        "colsample_bytree":  0.75,    # Fraction of features used for each tree
        "reg_alpha":         0.12,     # L1 regularization
        "reg_lambda":        0.08,    # L2 regularization
        "random_state":      0,       # Seed for reproducibility
        "n_jobs":            4,       # Number of parallel jobs
        "verbose":          -1        # Suppress output
    }
}

BASELINES_PARAMETERS = baseline_params_fast if RUN_FAST else baseline_params_normal




# Optional: plug in Bayesian best parameters
_new_parameters = { #'lasso_alpha': 0.0,
    'LR_type': 'ridge', 'LR_alpha': 1.45, 'RF_n_estimators': 490, 'RF_max_depth': 23,
    'RF_min_samples_leaf': 11, 'RF_min_samples_split': 12, 'RF_max_features': 'sqrt',
    'GB_boosting_type': 'gbdt', 'GB_num_leaves': 15, 'GB_max_depth': 2,
    'GB_learning_rate': 0.07, 'GB_n_estimators': 630, 'GB_min_child_samples': 13,
    'GB_subsample': 0.72, 'GB_colsample_bytree': 0.98, 'GB_reg_alpha': 0.06,
    'GB_reg_lambda': 0.08, 'metaNN_epochs': 20, 'metaNN_batch_size': 96,
    'metaNN_learning_rate': 0.0025, 'metaNN_weight_decay': 3.054e-08,
    'metaNN_dropout': 0.11, 'metaNN_num_cells_0': 64, 'metaNN_num_cells_1': 32
} # trial 51: 1.207 (Bayes) -> 1.23 (one-off)

for _model in ['LR', 'RF', 'LGBM']:
    BASELINES_PARAMETERS[_model].update(
        {k.strip(_model+'_'): v for (k, v) in _new_parameters.items() if _model in k})
METAMODEL_NN_PARAMETERS.update({k.strip('metaNN_'): v for (k, v) in _new_parameters.items() if 'metaNN' in k})




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



__all__ = ['SYSTEM_SIZE', 'SEED', 'TRAIN_SPLIT_FRACTION', 'VAL_RATIO',
           'VALIDATE_EVERY', 'DISPLAY_EVERY', 'PLOT_CONV_EVERY',
           'VERBOSE', 'DICT_FNAMES', 'CACHE_FNAME', 'BASELINE_CFG',
           'FORECAST_HOUR', 'MINUTES_PER_STEP', 'NUM_STEPS_PER_DAY',
           # NN model parameters,
           'NNTQ_PARAMETERS', 'METAMODEL_NN_PARAMETERS']


import  torch




MINUTES_PER_STEP  = 30
NUM_STEPS_PER_DAY = int(round(24*60/MINUTES_PER_STEP))
def days_to_steps(num_days: float, num_steps_per_day=NUM_STEPS_PER_DAY) -> int:
    return int(round(num_days*num_steps_per_day))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. CONFIGURATION CONSTANTS
# ============================================================

SYSTEM_SIZE  = 'SMALL'          # in ['DEBUG', 'SMALL', 'LARGE']

VERBOSE: int = 2 if SYSTEM_SIZE == 'DEBUG' else 1

SEED         =   0              # For reproducibility


TRAIN_SPLIT_FRACTION=0.8
VAL_RATIO    =   0.25           # validation from training set

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

    'batch_size'       :  64,                   # Training batch size

    # optimizer
    'learning_rate'    :  7.5e-3,      # Optimizer learning rate
    'weight_decay'     :  1.e-7,
    'dropout'          :  0.05,

    # early stopping
    'min_delta'        :   25 / 1000,

    # PatchEmbedding
    'patch_length'     : _patch_length,              # [half-hours]
    'stride'           : max(int(round(_patch_length/2)), 1), # [half-hours]

    # geometric blocks
    'geo_block_ratio'  : 1,
         # each block is a fraction of the size of the previous (geometric)

    # quantile loss
    'quantiles'        : (0.1, 0.25, 0.5, 0.75, 0.9),
    'lambda_cross'     : 1.,          # enforcing correct order of quantiles
    'lambda_coverage'  : 0.5,
    'lambda_deriv'     : 0.1,         # derivative weight in loss function
    'lambda_median'    : 0.5,
    'smoothing_cross'  : 0.02,
}


EPOCHS       = [  2,  30,  60] # Number of training epochs
MODEL_DIM    = [ 48, 128, 192] # Transformer embedding dimension
NUM_HEADS    = [  2,   4,   6] # Number of attention heads
FFN_SIZE     = [  4,   4,   6] # expansion factor
NUM_LAYERS   = [  1,   2,   3] # Number of transformer encoder layers
NUM_GEO_BLOCKS=[  2,   5,  10]
WARMUP_STEPS =[4000,2500,2250]
PATIENCE     = [  5,   5,  10]  # DEBUG: patience > nb epochas

# Pick correct value from list of possibilites
IDX_SYSTEM_SIZE = {'DEBUG': 0, 'SMALL': 1, 'LARGE': 2}[SYSTEM_SIZE]

NNTQ_PARAMETERS['epochs']       = EPOCHS        [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['model_dim']    = MODEL_DIM     [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['warmup_steps'] = WARMUP_STEPS  [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['patience']     = PATIENCE      [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['num_layers']   = NUM_LAYERS    [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['num_heads']    = NUM_HEADS     [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['ffn_size']     = FFN_SIZE      [IDX_SYSTEM_SIZE]
NNTQ_PARAMETERS['num_geo_blocks']=NUM_GEO_BLOCKS[IDX_SYSTEM_SIZE]

NNTQ_PARAMETERS['num_patches'] = \
    (NNTQ_PARAMETERS['input_length'] + NNTQ_PARAMETERS['features_in_future'] * \
                NNTQ_PARAMETERS['pred_length'] - NNTQ_PARAMETERS['patch_length'])\
        // NNTQ_PARAMETERS['stride'] + 1


VALIDATE_EVERY=  1
DISPLAY_EVERY=   2
PLOT_CONV_EVERY=10



# NN metamodel

METAMODEL_NN_PARAMETERS: dict = {
    'batch_size'       :  256,
    'num_cells'        : [32, 16],

    # optimizer
    'learning_rate'    :  5e-4,
    'weight_decay'     :  1e-5,
    'dropout'          :  0.05,

    # early stopping
    'patience'         :   4,
    'factor'           :   0.7,

    'device'           : DEVICE,
    }

META_EPOCHS     = [  1, 12, 15]
METAMODEL_NN_PARAMETERS['epochs']  = META_EPOCHS  [IDX_SYSTEM_SIZE]



DICT_FNAMES = {
    "consumption": "data/consommation-quotidienne-brute.csv",
    "temperature": 'data/temperature-quotidienne-regionale.csv',
    "solar":       'data/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv'
}
CACHE_FNAME = "cache/merged_aligned.csv"





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




baseline_cfg = [
    {  # 'DEBUG'
    'lasso': {"alpha": 5 / 100., 'max_iter': 1_000},
    # "oracle": {1},  # (content is just a place-holder)
    'LR': {"type": "lasso", "alpha": 5 / 100.},
    'RF': {
        "type":            "rf",
        "n_estimators":     50,     # was 300 -> fewer trees
        "max_depth":         6,     # shallower trees
        "min_samples_leaf": 10,     # more regularization
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'GB': {
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
},

{  # 'SMALL'
    'lasso': {"alpha": 2 / 100., 'max_iter': 2_000},
    # "oracle": {1},  # (content is just a place-holder)
    'LR': {"type": "lasso", "alpha": 2 / 100., 'max_iter': 2_000},
    'RF': {
        "type":            "rf",
        "n_estimators":    500,
        "max_depth":        20,
        "min_samples_leaf": 15,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'GB': {
        "type":          "lgbm",
        "objective":     "regression",
        "boosting_type": "gbdt",
        "num_leaves":       32-1,     # Default number of leaves
        "max_depth":         5,       # Moderate tree depth
        "learning_rate":     0.05,    # Lower learning rate for stability
        "n_estimators":    500,       # More trees for a robust model
        "min_child_samples":20,       # Minimum samples per leaf
        "subsample":         0.8,     # Fraction of samples used to train each tree
        "colsample_bytree":  0.8,     # Fraction of features used for each tree
        "reg_alpha":         0.1,     # L1 regularization
        "reg_lambda":        0.1,     # L2 regularization
        "random_state":      0,       # Seed for reproducibility
        "n_jobs":            4,       # Number of parallel jobs
        "verbose":          -1        # Suppress output
    }
},

{  # 'LARGE'
    'lasso': {"alpha": 0.5 / 100., 'max_iter': 5_000},
    # "oracle": {1},  # (content is just a place-holder)
    'LR': {"type": "lasso", "alpha": 0.5 / 100., 'max_iter': 5_000},
    'RF': {
        "type":            "rf",
        "n_estimators":    400,
        "max_depth":        15,
        "min_samples_leaf": 20,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
    'GB': {
        "type":     "lgbm",
        "objective": "regression",
        "boosting_type": "gbdt",
        "num_leaves": 64-1,           # More leaves for complex patterns
        "max_depth": 8,               # Deeper trees
        "learning_rate": 0.02,        # Lower learning rate for precision
        "n_estimators": 500,          # More trees for a robust model
        "min_child_samples": 30,      # Minimum samples per leaf
        "subsample": 0.7,             # Fraction of samples used to train each tree
        "colsample_bytree": 0.7,      # Fraction of features used for each tree
        "reg_alpha": 0.2,             # Stronger L1 regularization
        "reg_lambda": 0.2,            # Stronger L2 regularization
        "random_state": 0,            # Seed for reproducibility
        "n_jobs": 4,                  # Number of parallel jobs
        "verbose": -1                 # Suppress output
    }
}
]


BASELINE_CFG = baseline_cfg[IDX_SYSTEM_SIZE]

# BASELINE_CFG = {}
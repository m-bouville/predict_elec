__all__ = ['SYSTEM_SIZE', 'SEED', 'TRAIN_SPLIT_FRACTION', 'VAL_RATIO', 'INPUT_LENGTH',
           'PRED_LENGTH', 'BATCH_SIZE', 'EPOCHS', 'MODEL_DIM', 'NUM_HEADS', 'FFN_SIZE',
           'NUM_LAYERS', 'PATCH_LEN', 'STRIDE', 'FEATURES_IN_FUTURE',
           'LAMBDA_CROSS', 'LAMBDA_COVERAGE', 'LAMBDA_DERIV', 'LAMBDA_MEDIAN',
           'SMOOTHING_CROSS', 'QUANTILES', 'NUM_GEO_BLOCKS', 'GEO_BLOCK_RATIO',
           'LEARNING_RATE', 'WEIGHT_DECAY', 'DROPOUT', 'WARMUP_STEPS', 'PATIENCE',
           'MIN_DELTA', 'VALIDATE_EVERY', 'DISPLAY_EVERY', 'PLOT_CONV_EVERY',
           'VERBOSE', 'DICT_FNAMES', 'CACHE_FNAME', 'BASELINE_CFG',
           'FORECAST_HOUR', 'MINUTES_PER_STEP', 'NUM_STEPS_PER_DAY',
           'META_EPOCHS', 'META_LR', 'META_WEIGHT_DECAY', 'META_BATCH_SIZE',
           'META_DROPOUT', 'META_NUM_CELLS', 'META_PATIENCE', 'META_FACTOR']



# from   typing import Dict  # List


import LR_RF  # utils

MINUTES_PER_STEP  = 30
NUM_STEPS_PER_DAY = int(round(24*60/MINUTES_PER_STEP))
def days_to_steps(num_days: float, num_steps_per_day=NUM_STEPS_PER_DAY) -> int:
    return int(round(num_days*num_steps_per_day))

# ============================================================
# 1. CONFIGURATION CONSTANTS
# ============================================================

SYSTEM_SIZE  = 'SMALL'          # in ['DEBUG', 'SMALL', 'LARGE','HUGE']

SEED         =   0              # For reproducibility


TRAIN_SPLIT_FRACTION=0.8
VAL_RATIO    =   0.25           # validation from training set

FORECAST_HOUR:int = 12          # 12: noon

INPUT_LENGTH = days_to_steps(14)       # How many half-hours the model sees
PRED_LENGTH  = days_to_steps( 1)       # BUG should be 36h  How many future half-hours to predict

BATCH_SIZE   =  64              # Training batch size
EPOCHS       = [  2, 30, 60, 80] # Number of training epochs

MODEL_DIM    = [ 48,128,192,256] # Transformer embedding dimension
NUM_HEADS    = [  2,  4,  6,  8] # Number of attention heads
FFN_SIZE     = [  4,  4,  6,  8] # expansion factor
NUM_LAYERS   = [  1,  2,  3,  6] # Number of transformer encoder layers

# PatchEmbedding
PATCH_LEN    = days_to_steps(0.5)                # [half-hours]
STRIDE       = max(int(round(PATCH_LEN/2)), 1)   # [half-hours]
FEATURES_IN_FUTURE=True  # features do not stop at noon

# losses
LAMBDA_CROSS   = 1.              # enforcing correct order of quantiles
LAMBDA_COVERAGE= 0.5
LAMBDA_DERIV   = 0.1            # derivative weight in loss function
LAMBDA_MEDIAN  = 0.5
SMOOTHING_CROSS= 0.02
QUANTILES      = (0.1, 0.25, 0.5, 0.75, 0.9)

NUM_GEO_BLOCKS=[  1,  3,  4,  6]
GEO_BLOCK_RATIO= 0.5            # each block is half the size of the previous (geometric)

LEARNING_RATE=   7.5e-3          # Optimizer learning rate
WEIGHT_DECAY =   1.e-7
DROPOUT      =   0.05
WARMUP_STEPS =[4000,2500,2250,2500]
# SCHED_FACTOR =   0.5
# SCHED_PATIENCE=  5


PATIENCE     = [  5,  5, 10, 10]  # DEBUG: patience > nb epochas
MIN_DELTA    =   10 / 1000

VALIDATE_EVERY=  1
DISPLAY_EVERY=   2
PLOT_CONV_EVERY=10
# INCR_STEPS_TEST=24                # only test every n half-hours

# metamodel
META_EPOCHS     =  50
META_LR         =   2e-4  # learning rate
META_WEIGHT_DECAY=  1e-5
META_BATCH_SIZE = 256
META_DROPOUT    =   0.2
META_NUM_CELLS  = [32, 16]
META_PATIENCE   =  10
META_FACTOR     =   0.5

VERBOSE: int   = 2 if SYSTEM_SIZE == 'DEBUG' else 1


DICT_FNAMES = {
    "consumption": "data/consommation-quotidienne-brute.csv",
    "temperature": 'data/temperature-quotidienne-regionale.csv',
    "solar":       'data/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv'
}
CACHE_FNAME = "cache/merged_aligned.csv"



# Pick correct value from list of possibilites
IDX_SYSTEM_SIZE = {'DEBUG': 0, 'SMALL': 1, 'LARGE': 2, 'HUGE': 3}[SYSTEM_SIZE]
EPOCHS       = EPOCHS       [IDX_SYSTEM_SIZE]
MODEL_DIM    = MODEL_DIM    [IDX_SYSTEM_SIZE]
NUM_HEADS    = NUM_HEADS    [IDX_SYSTEM_SIZE]
FFN_SIZE     = FFN_SIZE     [IDX_SYSTEM_SIZE]
NUM_LAYERS   = NUM_LAYERS   [IDX_SYSTEM_SIZE]
NUM_GEO_BLOCKS=NUM_GEO_BLOCKS[IDX_SYSTEM_SIZE]
PATIENCE     = PATIENCE     [IDX_SYSTEM_SIZE]
WARMUP_STEPS = WARMUP_STEPS [IDX_SYSTEM_SIZE]
BASELINE_CFG = (LR_RF.baseline_cfg)[IDX_SYSTEM_SIZE]



# BASELINE_CFG = {}



# Checking
assert MODEL_DIM % NUM_HEADS == 0, \
    f"MODEL_DIM ({MODEL_DIM}) must be divisible by NUM_HEADS ({NUM_HEADS})."

assert 1 <= VALIDATE_EVERY <= min(EPOCHS, PATIENCE), \
    (VALIDATE_EVERY, EPOCHS, PATIENCE)

num_quantiles = len(QUANTILES)
assert all([QUANTILES[i] + QUANTILES[num_quantiles - i - 1] == 1
            for i in range(num_quantiles // 2)]), \
    "quantiles should be symmetric"    # otherwise: hard to interpret
assert QUANTILES[num_quantiles // 2] == 0.5, "middle quantile must be the median"
    # the code assumes it is


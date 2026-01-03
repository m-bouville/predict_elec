# import sys


import MC_search, run # containers, architecture, utils, LR_RF, IO, plots  # losses, metamodel,


# system dimensions
# B = BATCH_SIZE
# L = INPUT_LENGTH
# H = prediction horizon  =  PRED_LENGTH
# V = validation horizon  = VALID_LENGTH
# Q = number of quantiles = len(quantiles)
# F = number of features


# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.
# TODO make future T° noisy to mimic the uncertainty of forecasts
# BUG RF and Boosting generalize poorly
# TODO make the metamodel reduce the bias
# [done] save RF and GB pickles separately
# [done] use lasso with LR to select features, and use only these with other models
# TODO have separate public holidays, as with the school holidays
# BUG GB complains about pd vs. np
# [in progress] MC hyperparameter search
# [done] preparation: in `predict_elec.py`
#   - [done] create functions
#   - [done] make main call these functions
#   - [done] no plotting if verbose == 0



if __name__ == "__main__":
    num_runs: int  = 40

    from   constants import (SYSTEM_SIZE, SEED, TRAIN_SPLIT_FRACTION, VAL_RATIO,
               VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
               VERBOSE, DICT_FNAMES, CACHE_FNAME, BASELINE_CFG,
               FORECAST_HOUR, MINUTES_PER_STEP, NUM_STEPS_PER_DAY,
               NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS
               )

    if num_runs > 1:   # search for hyperparameters
        MC_search.run_Monte_Carlo_search(
                num_runs        = num_runs,
                csv_path        = "MC_results.csv",

                # configuration bundles
                baseline_cfg    = BASELINE_CFG,

                base_NNTQ_params= NNTQ_PARAMETERS,
                NNTQ_modifiers  = MC_search.NNTQ_SEARCH,
                base_meta_NN_params= METAMODEL_NN_PARAMETERS,
                meta_NN_modifiers= MC_search.METAMODEL_SEARCH,
                dict_fnames     = DICT_FNAMES,

                # statistics of the dataset
                minutes_per_step= MINUTES_PER_STEP,
                train_split_fraction=TRAIN_SPLIT_FRACTION,
                val_ratio       = VAL_RATIO,
                forecast_hour   = FORECAST_HOUR,
                seed            = SEED,
                force_calc_baselines=False, #VERBOSE >= 2, #SYSTEM_SIZE == 'DEBUG'
                cache_fname     = CACHE_FNAME,
            )


    else:  # single run
        run.run_model(
                  # configuration bundles
                  baseline_cfg      = BASELINE_CFG,
                  NNTQ_parameters   = NNTQ_PARAMETERS,
                  meta_NN_parameters= METAMODEL_NN_PARAMETERS,

                  dict_fnames       = DICT_FNAMES,

                  # statistics of the dataset
                  minutes_per_step  = MINUTES_PER_STEP,
                  train_split_fraction=TRAIN_SPLIT_FRACTION,
                  val_ratio         = VAL_RATIO,
                  forecast_hour     = FORECAST_HOUR,
                  seed              = SEED,
                  force_calc_baselines=False, #VERBOSE >= 2, #SYSTEM_SIZE == 'DEBUG',

                  # XXX_EVERY (in epochs)
                  validate_every    = VALIDATE_EVERY,
                  display_every     = DISPLAY_EVERY,
                  plot_conv_every   = PLOT_CONV_EVERY,

                  cache_fname       = CACHE_FNAME,
                  verbose          = VERBOSE
                  )

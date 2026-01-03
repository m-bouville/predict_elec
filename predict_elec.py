# import sys


import MC_search, run # containers, architecture, utils, LR_RF, IO, plots  # losses, metamodel,



# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.
# TODO make future T° noisy to mimic the uncertainty of forecasts
# BUG RF and Boosting generalize poorly
# TODO make the metamodel reduce the bias
# TODO have separate public holidays, as with the school holidays
# BUG GB complains about pd vs. np
# [in progress] MC hyperparameter search
#   - TODO add parameters for lasso, LR, RF, GB



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
                base_meta_NN_params= METAMODEL_NN_PARAMETERS,
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

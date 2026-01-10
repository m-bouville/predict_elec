# import sys

# import numpy  as np


import run
    # containers, architecture, baselines, IO, plots, losses, metamodel,
    # MC_search, Bayes_search,  utils

from   constants import (RUN_FAST, SEED, TRAIN_SPLIT_FRACTION, VALID_RATIO,
           VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           DICT_INPUT_CSV_FNAMES, FORECAST_HOUR, MINUTES_PER_STEP,
           BASELINES_PARAMETERS, NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS
           )



# BUG NNTQ misses whole days for no apparent reason
# BUG bias => bad coverage of quantiles.
# TODO make future T° noisy to mimic the uncertainty of forecasts
# BUG RF and Boosting generalize poorly
# TODO make the metamodel reduce the bias (how?)
# TODO have separate public holidays, as with the school holidays
# [done] GB complains about pd vs. np
# [done] split hyperparameter search: one for NNTQ, one for metamodel
# TODO add `q75` minus `q25` (uncertainty proxy) to NN metamodel
# BUG parameters found in search do not work when used in a one-off run







if __name__ == "__main__":


    mode = 'once'
        # in ['once', 'random', 'Bayes_NNTQ', 'Bayes_meta, 'Bayes_all']



    if mode in ['once']:
        num_runs =  1
        VERBOSE: int = 1  # 2 if RUN_FAST else 1
        force_calc_baselines = VERBOSE >= 3
    else:
        num_runs = 50
        force_calc_baselines = False
        VERBOSE: int = 0

    if 'Bayes' in mode:
        assert not RUN_FAST, "fast parameters are outside the Bayesian distributions"

    run.run_model(
        mode                = mode,
        num_runs            = num_runs,

        # configuration bundles
        baseline_parameters = BASELINES_PARAMETERS,
        NNTQ_parameters     = NNTQ_PARAMETERS,
        metamodel_NN_parameters= METAMODEL_NN_PARAMETERS,

        dict_input_csv_fnames= DICT_INPUT_CSV_FNAMES,

        # statistics of the dataset
        minutes_per_step    = MINUTES_PER_STEP,
        train_split_fraction= TRAIN_SPLIT_FRACTION,
        valid_ratio         = VALID_RATIO,
        forecast_hour       = FORECAST_HOUR,
        seed                = SEED,
        force_calc_baselines= force_calc_baselines,

        # XXX_EVERY (in epochs)
        validate_every      = VALIDATE_EVERY,
        display_every       = DISPLAY_EVERY,
        plot_conv_every     = PLOT_CONV_EVERY,

        cache_dir           = 'cache',
        verbose             = VERBOSE
    )



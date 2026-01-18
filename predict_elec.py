# import sys

# import numpy  as np


import constants, run
    # containers, architecture, baselines, IO, plots, losses, metamodel,
    # MC_search, Bayes_search,  utils

from   constants import (SEED, TRAIN_SPLIT_FRACTION, VALID_RATIO,
           VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           DICT_INPUT_CSV_FNAMES, FORECAST_HOUR, MINUTES_PER_STEP,
           BASELINES_PARAMETERS, NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS
           )




if __name__ == "__main__":


    RUN_FAST     = False         # True: smaller system => runs faster, for debugging

    MODE = 'once'
        # in ['once', 'random', 'Bayes_NNTQ', 'Bayes_meta, 'Bayes_all']

    if RUN_FAST:
        (BASELINE_PARAMS_FAST, NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS) = \
            constants.fast_parameters(NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS)


    if MODE in ['once']:
        num_trials =  1
        VERBOSE: int = 1  # 2 if RUN_FAST else 1
        force_calc_baselines = False  # VERBOSE >= 3
    else:
        num_trials = 10
        force_calc_baselines = False
        VERBOSE: int = 0


    if 'Bayes' in MODE:
        assert not RUN_FAST, "fast parameters are outside the Bayesian distributions"



    run.run_model(
        mode                = MODE,
        num_trials          = num_trials,

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



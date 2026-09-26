###############################################################################
#
# Neural Network based on Transformers, with Quantiles (NNTQ)
# by: Mathieu Bouville
#
# conso_elec.py
# Main file
#
###############################################################################


# import sys

import copy

import constants, run
    # containers, architecture, baselines, IO, plots, losses, metamodel,
    # MC_search, Bayes_search,  utils

from   constants import (SEED, TRAIN_SPLIT_FRACTION, VALID_RATIO,
           VALIDATE_EVERY, DISPLAY_EVERY, PLOT_CONV_EVERY,
           DICT_INPUT_CSV_FNAMES, FORECAST_HOUR, MINUTES_PER_STEP,
           BASELINES_PARAMETERS, NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS
           )




if __name__ == "__main__":


    MODE = 'Bayes_NNTQ'
        # in ['once', 'Bayes_NNTQ', 'Bayes_meta, 'Bayes_all', # no stats, model
        #     'statistics', #    stats,    model
        #     'stats_only', #    stats, no model
        #     'load_input'  # no stats, no model]

    VERBOSE:    int =  1
        # 0: no display, 1: normal, 2: more info, 3+: debugging

    FORCE_CALC_BASELINES = False

    RUN_FAST   = False    # True: smaller system => runs faster, for debugging
        # one-off only



    # defaults, possibly overridden below
    baseline_parameters  = BASELINES_PARAMETERS
    train_split_fraction = TRAIN_SPLIT_FRACTION
    valid_ratio          = VALID_RATIO

    if MODE in ['once', 'statistics']:
        NUM_TRIALS = 1

        # METAMODEL_NN_PARAMETERS['epochs'] =  2

        if RUN_FAST:
            # on copies: fast_parameters modifies its arguments in place
            (baseline_parameters, NNTQ_PARAMETERS, METAMODEL_NN_PARAMETERS) = \
                constants.fast_parameters(copy.deepcopy(NNTQ_PARAMETERS),
                                          copy.deepcopy(METAMODEL_NN_PARAMETERS))

        if 'stat' in MODE:    # works for `stats` and `statistics`
        # we will plot on the training (nearly complete) data set
            train_split_fraction = 0.99
            valid_ratio          = 0.01


    elif 'Bayes' in MODE:       # 'Bayes_NNTQ', 'Bayes_meta, 'Bayes_all'
        NUM_TRIALS = 80
        VERBOSE    = 0

    elif MODE in ['stats_only', 'load_input']:
        NUM_TRIALS = 0  # no NNTQ model, perhaps statistics

    else:
        raise ValueError(f"`{MODE}` is not a valid mode")


    # if 'Bayes' in MODE:
    #     assert not RUN_FAST, "fast parameters are outside Bayesian distributions"



    run.run_model(
        mode                = MODE,
        num_trials          = NUM_TRIALS,

        # configuration bundles
        baseline_parameters = baseline_parameters,
            # /!\ was BASELINES_PARAMETERS: RUN_FAST did not apply to baselines
        NNTQ_parameters     = NNTQ_PARAMETERS,
        metamodel_NN_parameters= METAMODEL_NN_PARAMETERS,

        dict_input_csv_fnames= DICT_INPUT_CSV_FNAMES,

        # statistics of the dataset
        minutes_per_step    = MINUTES_PER_STEP,
        train_split_fraction= train_split_fraction,
        valid_ratio         = valid_ratio,
            # /!\ were the constants: 'statistics' did not use its 0.99 / 0.01
        forecast_hour       = FORECAST_HOUR,
        seed                = SEED,
        force_calc_baselines= FORCE_CALC_BASELINES,

        # XXX_EVERY (in epochs)
        validate_every      = VALIDATE_EVERY,
        display_every       = DISPLAY_EVERY,
        plot_conv_every     = PLOT_CONV_EVERY,

        cache_dir           = 'cache',
        verbose             = VERBOSE
    )



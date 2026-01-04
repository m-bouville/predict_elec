import os
import copy
from   tqdm     import trange

from   typing   import Dict, Sequence, Any, Optional  # List, Tuple,

from   datetime import datetime

import numpy    as np
import pandas   as pd
import random


import run




def sample_baseline_parameters(base_params: Dict[str, Dict[str, Any]])\
            -> Dict[str, Dict[str, Any]]:

    p0 = copy.deepcopy(base_params)  # Dict[str, Dict[str, Any]]

    for _baseline in p0.keys():
        p = p0[_baseline]   # Dict[str, Any]
        if _baseline == 'lasso':
            if 'alpha' in p.keys():
                p['alpha']      = round(random.uniform(0.5 / 100., 4 / 100.), 5)

        elif _baseline == 'LR':
            if 'type' in p.keys():
                p['type']      = random.choice(['lasso'])  # 'ridge'
            if 'alpha' in p.keys():
                p['alpha']      = round(random.uniform(0.5 / 100., 4 / 100.), 5) \
                    if p['type'] == 'lasso'  else round(random.uniform(0.5, 2), 3)

        elif _baseline == 'RF':
            if 'n_estimators' in p.keys():
                p['n_estimators'   ] = int(round(random.uniform(300, 600), -1))
            if 'max_depth' in p.keys():
                p['max_depth'      ] = int(round(random.uniform( 15,  25)))
            if 'min_samples_leaf' in p.keys():
                p['min_samples_leaf']= int(round(random.uniform( 10,  20)))
            if 'min_samples_split' in p.keys():
                p['min_samples_split']= int(round(random.uniform( 15,  25)))
            if 'max_features' in p.keys():
                p['max_features'   ] = random.choice(['sqrt', 0.5])

        elif _baseline == 'GB':
            if 'boosting_type' in p.keys():
                p['boosting_type'  ] = random.choice(['gbdt'])  # TODO add more?
            if 'num_leaves' in p.keys():     # Default number of leaves
                p['num_leaves'     ] = random.choice([16, 32]) - 1
            if 'max_depth' in p.keys():
                p['max_depth'      ] = random.choice([3, 4, 5, 6])
            if 'learning_rate' in p.keys():   # Lower learning rate => more stable
                p['learning_rate'  ] = \
                    round(0.05 * 10**random.uniform(np.log10(0.3), np.log10( 3.)), 4)
            if 'n_estimators' in p.keys():   # More trees => more robust model
                p['n_estimators'   ] = int(round(random.uniform(300, 600), -1))
            if 'min_child_samples' in p.keys():
                p['min_child_samples']=int(round(random.uniform( 15,  25)))

            if 'learning_rate' in p.keys():
                p['learning_rate'  ] = \
                    round(0.01  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)

            if 'subsample' in p.keys():  # Fraction of samples used to train each tree
                p['subsample'      ] = round(random.uniform(0.6, 1. ), 3)

            if 'colsample_bytree' in p.keys(): # Fraction of features used for each tree
                p['colsample_bytree']= round(random.uniform(0.6, 1. ), 3)

            if 'reg_alpha' in p.keys():    # L1 regularization
                p['reg_alpha'      ] = \
                    round(0.1  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)

            if 'reg_lambda' in p.keys():   # L2 regularization
                p['reg_lambda'     ] = \
                    round(0.1  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)
    return p0



def sample_NNTQ_parameters(base_params: Dict[str, Any]) -> Dict[str, Any]:

    p = copy.deepcopy(base_params)

    if 'epochs' in p.keys():
        p['epochs']      = int(random.uniform(20, 40))
    if 'batch_size' in p.keys():
        p['batch_size']  = random.choice([32, 64, 96, 128])
    if 'learning_rate' in p.keys():
        p['learning_rate']= \
            round(0.01  * 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 4)
    if 'weight_decay' in p.keys():
        p['weight_decay']= \
            round(0.1e-6* 10 ** random.uniform(np.log10(0.1), np.log10(10.)), 9)
    if 'dropout' in p.keys():
        p['dropout']     = round(random.uniform(0.02, 0.15), 3)

    # quantile loss weights
    if 'lambda_cross' in p.keys():
        p['lambda_cross']   = round(random.uniform(0.5,  2.0), 3)
    if 'lambda_coverage' in p.keys():
        p['lambda_coverage']= round(random.uniform(0.2,  1.0), 2)
    if 'lambda_deriv' in p.keys():
        p['lambda_deriv']   = round(random.uniform(0.0,  0.3), 4)
    if 'lambda_median' in p.keys():
        p['lambda_median']  = round(random.uniform(0.2,  1.0), 3)
    if 'smoothing_cross' in p.keys():
        p['smoothing_cross']= round(random.uniform(0.005,0.05),4)


    # # quantiles
    # p['quantiles'] = random.choice(modifiers['quantiles'])

    # # patch geometry
    # scale = random.uniform(*modifiers['patch_length_scale'])
    # patch_length = max(1, int(base_params['patch_length'] * scale))
    # p['patch_length'] = patch_length

    # stride_ratio = random.uniform(*modifiers['stride_ratio'])
    # p['stride'] = max(1, int(round(patch_length * stride_ratio)))


    # architecture
    if 'model_dim' in p.keys():
        p['model_dim']  = int(128 * random.choice([0.75, 1.0, 1.25]))
    if 'ffn_size' in p.keys():
        p['ffn_size']   = random.choice([2, 3, 4, 5])
    if 'num_heads' in p.keys():
        p['num_heads']  = random.choice([2, 3, 4, 5])
    if 'num_layers' in p.keys():
        p['num_layers'] = random.choice([1, 2, 3, 4])

    # enforce divisibility (Transformer constraint)
    if  p['model_dim'] % p['num_heads'] != 0:
        p['model_dim'] = p['num_heads'] * (p['model_dim'] // p['num_heads'])

    if 'epochs' in p.keys():
        p['num_geo_blocks']= int(round(random.uniform(2,    8)))

    # early stopping
    if 'warmup_steps' in p.keys():
        p['warmup_steps']  = int(round(random.uniform(1500, 4000), -2))
    if 'patience' in p.keys():
        p['patience']      = int(      random.uniform(4,    10))
    if 'min_delta' in p.keys():
        p['min_delta']     =     round(random.uniform(15e-3,30e-3), 5)

    # derived quantities
    p['num_patches'] = (
        p['input_length']
        + p['features_in_future'] * p['pred_length']
        - p['patch_length']
    ) // p['stride'] + 1

    return p



def sample_metamodel_NN_parameters(base_params : Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(base_params)

    if 'epochs' in p.keys():
        p['epochs']      = int(random.uniform(8, 15))

    if 'batch_size' in p.keys():
        p['batch_size']  = random.choice([128, 192, 256, 384])

    if 'learning_rate' in p.keys():
        p['learning_rate']= \
            round(0.5e-3* 10 ** random.uniform(np.log10(0.3), np.log10( 3.)), 5)

    if 'weight_decay' in p.keys():
        p['weight_decay']= \
            round(10e-6 * 10 ** random.uniform(np.log10(0.1), np.log10(10.)), 7)

    if 'dropout' in p.keys():
        p['dropout']     = round(random.uniform(0.05, 0.2), 2)

    # num_cells: scale but keep integers ≥ 4
    if 'num_cells' in p.keys():
        scales = [random.choice([0.75, 1.0, 1.25]),
              random.choice([0.75, 1.0, 1.25])]
        p['num_cells'] = [max(4, int(c * scales[i])) for i, c in enumerate([32, 16])]

    # early stopping
    if 'patience' in p.keys():
        p['patience'] =       random.choice ([3, 4, 5, 6])
    if 'factor' in p.keys():
        p['factor']   = round(random.uniform(0.6, 0.85), 3)

    return p


# def sample_meta_params(base_cfg, rng):
#     cfg = deepcopy(base_cfg)
#     cfg["lr"]         = 10 ** rng.uniform(-4.5, -3.0)
#     cfg["dropout"]    = rng.uniform(0.05, 0.4)
#     # cfg["num_cells"]  = rng.choice([[32,16], [32,32], [64,32]])
#     return cfg


def expand_sequence(name: str, values: Sequence, length: int,
                    prefix: Optional[str]="", fill_value=np.nan) -> Dict[str, Any]:
    """
    Expand a list/tuple into fixed-length columns.
    Shorter lists are padded with fill_value.
    Longer lists raise an error (by default).
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be list or tuple, got {type(values)}")

    if len(values) > length:
        raise ValueError(f"{name} length {len(values)} > fixed length {length}")

    output = {}
    for i in range(length):
        output[f"{prefix}{name}_{i}"] = values[i] if i < len(values) else fill_value

    return output

def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dict using namespaced keys.
    Example:
      {"rf": {"n_estimators": 500}} →
      {"baseline__rf__n_estimators": 500}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def run_Monte_Carlo_search(
            num_runs            : int,
            # configuration bundles
            base_baseline_params: Dict[str, Dict[str, Any]],
            base_NNTQ_params    : Dict[str, Any],
            base_meta_NN_params : Dict[str, Any],
            dict_fnames         : Dict[str, str],
            # statistics of the dataset
            minutes_per_step    : int,
            train_split_fraction: float,
            val_ratio           : float,
            forecast_hour       : int,
            seed                : int,
            force_calc_baselines: bool,
            cache_fname         : Optional[str] = None,
            csv_path            : str    = "mc_results.csv"
        ):

    import warnings
    # from   sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=UserWarning)  # TODO fix for real

    # rng = np.random.default_rng(seed)
    results = []

    for run_id in trange(num_runs, desc="MC runs"):
        # print(f"Starting run {run_id} out of {num_runs}")
        # meta_cfg = sample_meta_params(base_meta_NN_params, rng)


        baseline_parameters = sample_baseline_parameters    (base_baseline_params)
        NNTQ_parameters     = sample_NNTQ_parameters        (base_NNTQ_params)
        metamodel_parameters= sample_metamodel_NN_parameters(base_meta_NN_params)

        df_metrics, avg_weights_meta_NN, quantile_delta_coverage = run.run_model(
                  # configuration bundles
                  baseline_cfg    = baseline_parameters,
                  NNTQ_parameters = NNTQ_parameters,
                  metamodel_NN_parameters=metamodel_parameters,
                  dict_fnames     = dict_fnames,

                  # statistics of the dataset
                  minutes_per_step= minutes_per_step,
                  train_split_fraction=train_split_fraction,
                  val_ratio       = val_ratio,
                  forecast_hour   = forecast_hour,
                  seed            = seed,
                  force_calc_baselines=False, #VERBOSE >= 2, #SYSTEM_SIZE == 'DEBUG',

                  # XXX_EVERY (in epochs)
                  validate_every  = 1,
                  display_every   = 1,  # dummy
                  plot_conv_every = 1,  # dummy

                  cache_fname     = cache_fname,
                  verbose         = 0
        )

        flat_metrics = {}
        for model in df_metrics.index:
            for metric in df_metrics.columns:
                key = f"test_{model}_{metric}".replace(" ", "_")
                flat_metrics[key] = df_metrics.loc[model, metric].astype(np.float32)


        # learning_rate and weight_decay are so small
        NNTQ_parameters    ['learning_rate' ]= NNTQ_parameters  ['learning_rate'] * 1e6
        metamodel_parameters['learning_rate']=metamodel_parameters['learning_rate']*1e6

        NNTQ_parameters    ['weight_decay' ]= NNTQ_parameters  ['weight_decay'] * 1e6
        metamodel_parameters['weight_decay']=metamodel_parameters['weight_decay']*1e6

        # flatten sequences
        _dict_quantiles = expand_sequence(name="quantiles",
               values=NNTQ_parameters["quantiles"],  length=5, prefix="")
        del NNTQ_parameters["quantiles"]
        NNTQ_parameters.update(_dict_quantiles)


        _dict_num_cells = expand_sequence(name="num_cells",
               values=metamodel_parameters["num_cells"], length=2, prefix="")
        del metamodel_parameters["num_cells"]
        metamodel_parameters.update(_dict_num_cells)

        _score =np.mean([abs(e) for e in(list(quantile_delta_coverage.values()) +\
                                         list(flat_metrics           .values()))])
        row = {
            "run"      : run_id,
            "timestamp": datetime.now(),   # Excel-compatible
            **(flatten_dict(baseline_parameters, parent_key="")),
            **NNTQ_parameters,
            **{"metaNN_"+key: value for (key, value) in metamodel_parameters.items()},
            **quantile_delta_coverage,
            **{"avg_weight_meta_NN_"+key: value
               for (key, value) in avg_weights_meta_NN.items()},
            **flat_metrics,
            "score": _score
        }
        # print(row)

        df_row = pd.DataFrame([row])
        df_row.to_csv(
            csv_path,
            mode   = "a",
            header = not os.path.exists(csv_path),
            index  = False,
            float_format="%.6f"
        )
    return pd.DataFrame(results)

import os
import copy
from   tqdm     import trange

from   typing   import Dict, Sequence, Any, Optional  # List, Tuple,

from   datetime import datetime

import numpy    as np
import pandas   as pd
import random


import run



NNTQ_SEARCH = {
    # batch size: powers of two near baseline
    "batch_size": [32, 64, 96, 128],

    # optimizer (log-scale)
    "learning_rate": (0.3, 3.0),
    "weight_decay":  (0.3, 3.0),

    # regularization
    "dropout":       (0.02, 0.12),

    # quantile-loss weights (absolute ranges)
    "lambda_cross":    (0.5, 2.0),
    "lambda_coverage": (0.2, 1.0),
    "lambda_deriv":    (0.0, 0.3),
    "lambda_median":   (0.2, 1.0),
    "smoothing_cross": (0.005, 0.05),

    # multiplicative jitter around SYSTEM_SIZE baseline
    "model_dim_scale":   [0.75, 1.0, 1.25],
    "ffn_size_scale":    [0.75, 1.0, 1.25],

    # discrete alternatives near baseline
    "num_layers_delta":  [-1, 0, +1],
    "num_heads_delta":   [-1, 0, +1],
    "num_geo_blocks_delta": [-2, 0, +2],

    # training dynamics
    "epochs_scale":      [0.75, 1.0, 1.25],
    "warmup_scale":      [0.7, 1.0, 1.3],
    "patience_delta":    [-2, 0, +2],

    # patch geometry (relative jitter)
    # "patch_length_scale": (0.8, 1.25),
    # "stride_ratio":       (0.4, 0.7),

    # # quantiles: optional mild perturbation
    # "quantiles": [
    #     (0.1, 0.25, 0.5, 0.75, 0.9),
    #     (0.05, 0.25, 0.5, 0.75, 0.95),
    # ],
}


METAMODEL_SEARCH = {
    # batch size: powers of two near baseline
    "batch_size": [128, 256, 512],

    # hidden cells: small structural jitter
    "num_cells_scale": [0.75, 1.0, 1.25],

    # optimizer (log-scale where appropriate)
    "learning_rate":  (0.3, 3.0),    # multiplicative
    "weight_decay":   (0.3, 3.0),
    "dropout":        (0.02, 0.12),

    # early stopping
    "patience":       [3, 4, 5, 6],
    "factor":         (0.6, 0.85),
}



def sample_NNTQ_parameters(
        base_params: dict,
        modifiers  : dict,
    ) -> dict:

    p = copy.deepcopy(base_params)


    p["epochs"] = max(
        1,
        int(base_params["epochs"] * random.choice(modifiers["epochs_scale"]))
    )

    # batch size
    p["batch_size"] = random.choice(modifiers["batch_size"])

    # optimizer (log-uniform)
    p["learning_rate"] = base_params["learning_rate"] * (
        10 ** random.uniform(
            np.log10(modifiers["learning_rate"][0]),
            np.log10(modifiers["learning_rate"][1]),
        )
    )
    p["weight_decay"] = base_params["weight_decay"] * (
        10 ** random.uniform(
            np.log10(modifiers["weight_decay"][0]),
            np.log10(modifiers["weight_decay"][1]),
        )
    )

    # dropout
    p["dropout"] = random.uniform(*modifiers["dropout"])

    # quantile loss weights
    for k in ["lambda_cross", "lambda_coverage",
              "lambda_deriv", "lambda_median", "smoothing_cross"]:
        p[k] = random.uniform(*modifiers[k])

    # # quantiles
    # p["quantiles"] = random.choice(modifiers["quantiles"])

    # # patch geometry
    # scale = random.uniform(*modifiers["patch_length_scale"])
    # patch_length = max(1, int(base_params["patch_length"] * scale))
    # p["patch_length"] = patch_length

    # stride_ratio = random.uniform(*modifiers["stride_ratio"])
    # p["stride"] = max(1, int(round(patch_length * stride_ratio)))


    # architecture
    p["model_dim"] = max(
        16,
        int(base_params["model_dim"] * random.choice(modifiers["model_dim_scale"]))
    )

    p["ffn_size"] = max(
        2,
        int(base_params["ffn_size"] * random.choice(modifiers["ffn_size_scale"]))
    )

    p["num_layers"] = max(
        1,
        base_params["num_layers"] + random.choice(modifiers["num_layers_delta"])
    )

    p["num_heads"] = max(
        1,
        base_params["num_heads"]  + random.choice(modifiers["num_heads_delta"])
    )

    # enforce divisibility (Transformer constraint)
    if p["model_dim"] % p["num_heads"] != 0:
        p["model_dim"] = p["num_heads"] * (p["model_dim"] // p["num_heads"])

    p["num_geo_blocks"] = max(
        1,
        base_params["num_geo_blocks"] + random.choice(modifiers["num_geo_blocks_delta"])
    )


    # early stopping
    p["warmup_steps"] = max(
        100,
        int(base_params["warmup_steps"] * random.choice(modifiers["warmup_scale"]))
    )

    p["patience"] = max(
        2,
        base_params["patience"]  + random.choice(modifiers["patience_delta"])
    )


    # derived quantities
    p["num_patches"] = (
        p["input_length"]
        + p["features_in_future"] * p["pred_length"]
        - p["patch_length"]
    ) // p["stride"] + 1

    return p



def sample_metamodel_NN_parameters(
            base_params : dict,
            modifiers   : dict
        ) -> dict:
    p = copy.deepcopy(base_params)

    # batch size
    p["batch_size"] = random.choice(modifiers["batch_size"])

    # num_cells: scale but keep integers ≥ 4
    scale = random.choice(modifiers["num_cells_scale"])
    p["num_cells"] = [
        max(4, int(c * scale))
        for c in base_params["num_cells"]
    ]

    # learning rate (log-uniform around base)
    p["learning_rate"] = base_params["learning_rate"] * (
        10 ** random.uniform(
            np.log10(modifiers["learning_rate"][0]),
            np.log10(modifiers["learning_rate"][1]),
        )
    )

    # weight decay (log-uniform around base)
    p["weight_decay"] = base_params["weight_decay"] * (
        10 ** random.uniform(
            np.log10(modifiers["weight_decay"][0]),
            np.log10(modifiers["weight_decay"][1]),
        )
    )

    # dropout (absolute range)
    p["dropout"]  = random.uniform(*modifiers["dropout"])

    # early stopping
    p["patience"] = random.choice (modifiers["patience"])
    p["factor"]   = random.uniform(*modifiers["factor"])

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
            baseline_cfg        : Dict[str, Dict[str, Any]],
            base_NNTQ_params    : Dict[str, Any],
            NNTQ_modifiers      : Dict[str, Any],
            base_meta_NN_params : Dict[str, Any],
            meta_NN_modifiers   : Dict[str, Any],
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

        NNTQ_parameters = sample_NNTQ_parameters(
            base_params= base_NNTQ_params,
            modifiers  = NNTQ_modifiers
            )

        metamodel_parameters = sample_metamodel_NN_parameters(
            base_params= base_meta_NN_params,
            modifiers  = meta_NN_modifiers
            )

        df_metrics, avg_weights_meta_NN, quantile_delta_coverage = run.run_model(
                  # configuration bundles
                  baseline_cfg    = baseline_cfg,
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



        _baseline_cfg     = baseline_cfg   .copy()
        _NNTQ_parameters  = NNTQ_parameters.copy()

        # flatten sequences
        _dict_quantiles = expand_sequence(name="quantiles",
               values=_NNTQ_parameters["quantiles"],  length=5, prefix="")
        del _NNTQ_parameters["quantiles"]
        _NNTQ_parameters.update(_dict_quantiles)


        _dict_num_cells = expand_sequence(name="num_cells",
               values=metamodel_parameters["num_cells"], length=2, prefix="")
        del metamodel_parameters["num_cells"]
        metamodel_parameters.update(_dict_num_cells)

        _score =np.mean([abs(e) for e in(list(quantile_delta_coverage.values()) +\
                                         list(flat_metrics           .values()))])
        row = {
            "run"      : run_id,
            "timestamp": datetime.now(),   # Excel-compatible
            **(flatten_dict(_baseline_cfg, parent_key="")),
            **_NNTQ_parameters,
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

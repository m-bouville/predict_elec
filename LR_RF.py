# ----------------------------------------------------------------------
#
# LR_RF.py
#
# linear regression & random forest
#
# ----------------------------------------------------------------------


import os, warnings

import json
import hashlib
import pickle

from   typing import List, Tuple, Dict # Sequence, Optional
# from   collections import defaultdict


import numpy  as np
import pandas as pd

from   sklearn.linear_model    import Ridge
from   sklearn.ensemble        import RandomForestRegressor

# import matplotlib.pyplot as plt


# import losses  # architecture




def temperature_correlation_matrix(df, verbose: int = 1) -> None:
    temp_cols = ["Tmin_degC", "Tmax_degC", "Tavg_degC", "Tavg_sat15_degC"]

    # Ensure they exist in the DataFrame
    temp_cols = [c for c in temp_cols if c in df.columns]

    print("Temperature Features Correlation Matrix (%):")
    corr = df[temp_cols].corr() * 100
    print(corr.round(1))

    if verbose > 1:
        # Also show pairs sorted by absolute correlation (most correlated first)
        print("Highly Correlated Temperature Feature Pairs (%):")
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
        )
        print(pairs.round(1))

def _build_model_from_cfg(cfg: Dict[str, dict]):
    """Factory that builds a model strictly from cfg."""
    cfg = cfg.copy()
    model_type = cfg.pop("type")

    if model_type == "ridge":
        return Ridge(**cfg)

    elif model_type == "rf":
        return RandomForestRegressor(**cfg)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

baseline_cfg = [
    {  # 'DEBUG'
    # "oracle": {1},  # (content is just a place-holder)
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":     50,     # was 300 -> fewer trees
        "max_depth":         6,     # shallower trees
        "min_samples_leaf": 10,     # more regularization
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'SMALL'
    # "oracle": {1},  # (content is just a place-holder)
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    250,
        "max_depth":        10,
        "min_samples_leaf": 15,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'LARGE'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    400,
        "max_depth":        15,
        "min_samples_leaf": 20,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
},

{  # 'HUGE'
    "lr": {"type": "ridge", "alpha": 1.0},
    "rf": {
        "type":            "rf",
        "n_estimators":    500,
        "max_depth":        20,
        "min_samples_leaf": 20,
        "min_samples_split":20,
        "max_features":   "sqrt",
        "random_state":      0,
        "n_jobs":            4
    },
}
]



def load_or_compute_regression_and_forest(
    compute_kwargs,
    cache_dir,
    cache_id_dict,
    force_calculation: bool = False,
    verbose: int = 0,
):
    """
    Generic cache wrapper for RandomForest predictions.

    Parameters
    ----------
    compute_kwargs : dict
        Keyword arguments passed to compute_fn
    cache_dir : str
        Directory to store cached predictions
    cache_id_dict : dict
        Dict describing data + RF config (used to build cache key)
    verbose : int
        Print cache hit/miss messages
    """
    os.makedirs(cache_dir, exist_ok=True)

    key_str    = json.dumps(cache_id_dict, sort_keys=True)
    cache_key  = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"rf_preds_{cache_key}.pkl")

    # either load...
    if os.path.exists(cache_path) and \
            'rf' in compute_kwargs['models_cfg'] and \
            not force_calculation:
        if verbose > 0:
            print("Loaded RandomForest predictions from cache")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ... or compute
    if verbose > 0:
        print("Training RandomForest (no cache found)")

    out = regression_and_forest(**compute_kwargs)

    # Save
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)

    if verbose > 0:
        print("Saved RandomForest predictions to cache")

    return out



## Version 3 -- older versions in: archives/utils-old-LR_RF-test_predictions.py
def regression_and_forest(
    df:          pd.DataFrame,
    # dates:       pd.DatetimeIndex,
    target_col:  str,
    feature_cols:List[str],
    train_end:   int,   # end of training set (exclusive)
    val_end:     int,     # end of validation set (exclusive)
    models_cfg:  Dict[str, dict],
    verbose:     int = 0
) -> Tuple[Dict[str, pd.Series], Ridge, Dict[str, Dict[str, float]]]:
    """
    Leakage-safe contemporaneous tabular baselines:
        y_t ~ features_t   (NO LAG)

    Parameters
    ----------
    models_cfg : dict
        Example:
        {
            "lr": {"type": "ridge", "alpha": 1.0},
            "rf": {"type": "rf", "n_estimators": 300, "max_depth": 12}
        }

    Returns
    -------
    features : Dict[str, pd.Series]
        {name -> OOF feature aligned with df.index}
    final_models : Dict[str, fitted model]
    losses : Dict[str, Dict[str, float]]
        {name -> {"train": mse, "valid": mse}}
    """

    # sigma_y_GW = 11.7  # TODO do not do by hand

    # -------------------------
    # 1. Extract matrices
    # -------------------------
    X_GW: np.ndarray = df[feature_cols].values.astype(np.float32)
    y_GW: np.ndarray = df[target_col  ].values.astype(np.float32)

    N: int = len(df)
    # print(f"indices: 0 < {train_end} < {val_end} < {N}")
    assert 0 < train_end < val_end <= N, "Invalid split indices"

    # -------------------------
    # 2. Define ranges
    # -------------------------
    train_idx: np.ndarray = np.arange(0,        train_end)
    valid_idx: np.ndarray = np.arange(train_end,val_end)
    test_idx : np.ndarray = np.arange(val_end,  N)

    X_train_GW: np.ndarray = X_GW[train_idx];  y_train_GW: np.ndarray = y_GW[train_idx]
    X_valid_GW: np.ndarray = X_GW[valid_idx];  y_valid_GW: np.ndarray = y_GW[valid_idx]
    X_test_GW : np.ndarray = X_GW[ test_idx];  y_test_GW : np.ndarray = y_GW[ test_idx]

    # -------------------------
    # 3. predictions on TRAIN only
    # -------------------------

    models            = dict()
    preds_GW          = dict()
    series_pred_GW    = pd.Series()

    for name, cfg in models_cfg.items():  # name = e.g. 'lr', 'rf'
        preds_GW          [name] = pd.Series()
        # losses_quantile_GW[name] = dict()

        if name != 'oracle':
            models[name] = _build_model_from_cfg(cfg)
            models[name].fit(X_train_GW, y_train_GW)

            pred_train_GW = models[name].predict(X_train_GW)
            pred_valid_GW = models[name].predict(X_valid_GW)
            pred_test_GW  = models[name].predict(X_test_GW )

        else:
            warnings.warn("Using the oracle!")
            models[name] = None # meaningless
            pred_train_GW = y_train_GW
            pred_valid_GW = y_valid_GW
            pred_test_GW  = y_test_GW

        series_pred_GW[name] = pd.Series(
            np.concatenate([pred_train_GW, pred_valid_GW, pred_test_GW]),
                            index = df.index)
        # print(series_pred_GW[name])


    # most relevant features
    if verbose >= 3 and {"lr", "rf"} <= models.keys():

        ridge = pd.Series(
            models["lr"].coef_ * 100.,
            index=feature_cols,
            name="ridge_coef_pc"
        ).astype(np.float32).round(2)

        rf = pd.Series(
            models["rf"].feature_importances_ * 100.,
            index=feature_cols,
            name="rf_importance_pc"
        )

        df_imp = pd.concat([ridge, rf], axis=1).astype(np.float32).round(2)

        # Normalize for comparability
        _source = df_imp["ridge_coef_pc"].abs()
        df_imp["ridge_norm"] = (_source / _source.max())

        _source = df_imp["rf_importance_pc"]
        df_imp["rf_norm"]    = _source / _source.max()

        # Overall relevance score
        df_imp["score_pc"] = (100. * df_imp[["ridge_norm", "rf_norm"]])\
            .mean(axis=1).astype(np.float32).round(2)

        # Final ordering
        df_imp = (
            df_imp
            .sort_values("score_pc", ascending=False)
            .drop(columns=["ridge_norm", "rf_norm"])
        )

        print("\n[Model diagnostics] Top features (Ridge + RF):")
        print(df_imp.astype(np.float32).round(2))  # .head(20)


    return series_pred_GW, models #, losses_quantile_GW

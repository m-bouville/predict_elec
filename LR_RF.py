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

from   sklearn.preprocessing   import StandardScaler
from   sklearn.linear_model    import Ridge, Lasso
from   sklearn.ensemble        import RandomForestRegressor
from   lightgbm                import LGBMRegressor

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

    if model_type == "lasso":
        return Lasso(**cfg)

    if model_type == "rf":
        return RandomForestRegressor(**cfg)

    if model_type == 'lgbm':
        return LGBMRegressor(**cfg)

    raise ValueError(f"Unknown model type: {model_type}")


baseline_cfg = [
    {  # 'DEBUG'
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
},

{  # 'HUGE'
    'LR': {"type": "lasso", "alpha": 0.2 / 100., 'max_iter': 10_000},
    'RF': {
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

def most_relevant_features(model_LR, model_RF, feature_cols: List[str]):

    lr = pd.Series(
        model_LR.coef_,
        index=feature_cols,
        name="LR_coef"
    ).astype(np.float32)

    rf = pd.Series(
        model_RF.feature_importances_,
        index=feature_cols,
        name="RF_importance"
    ).astype(np.float32)

    df_imp = pd.concat([lr, rf], axis=1).astype(np.float32).round(3)

    # Normalize for comparability
    _source = df_imp["LR_coef"].abs()
    df_imp["LR_norm_pc"] = (100.*_source/_source.quantile(0.95))

    _source = df_imp["RF_importance"]
    df_imp["RF_norm_pc"] = (100.*_source/_source.quantile(0.95))

    # Overall relevance score
    df_imp["score_pc"] = (df_imp[["LR_norm_pc", "RF_norm_pc"]]).mean(axis=1)


    # reference for RF: the second-lowest value
    rf_imp = df_imp["RF_norm_pc"]
    min_rf = rf_imp.replace(0, np.nan).min() * 1.001 + .0001
    rf_filtered = rf_imp.mask(rf_imp < min_rf, np.nan)
    rf_second_lowest = rf_filtered.min()
    print(f"rf_smallest_non_zero: {min_rf:.5f}, {rf_second_lowest:.5f}")

    lr_survivor = lr.abs() > 0
    rf_survivor = rf_imp > rf_second_lowest * 1.001
    both_survivor  = lr_survivor & rf_survivor
    either_survivor= lr_survivor | rf_survivor
    score_survivor = (df_imp["score_pc"] > 5)
    print(f"survivors out of {len(lr)} features: "
          f"LR {      lr_survivor.sum()}, RF {        rf_survivor.sum()}, "
          f"both {  both_survivor.sum()}, either {either_survivor.sum()}, "
          f"score {score_survivor.sum()}")


    # Final ordering
    df_imp = (
        df_imp
        .sort_values("score_pc", ascending=False)
        .drop(columns=["LR_coef", "RF_importance"])  # keep only normalized
    )

    print("\n[Model diagnostics] Top features (LR + RF):")
    print(df_imp.round(2).to_string(float_format="%6.2f"))  # .head(20)




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



## older versions in: archives/utils-old-LR_RF-test_predictions.py
def regression_and_forest(
    df:          pd.DataFrame,
    # dates:       pd.DatetimeIndex,
    target_col:  str,
    feature_cols:List[str],
    train_end:   int,   # end of training set (exclusive)
    val_end:     int,     # end of validation set (exclusive)
    models_cfg:  Dict[str, dict],
    verbose:     int = 0
) -> Tuple[Dict[str, pd.Series], object, pd.DataFrame, List[str]]:
    """
    Leakage-safe contemporaneous tabular baselines:
        y_t ~ features_t   (NO LAG)

    Parameters
    ----------
    models_cfg : dict
        Example:
        {
            'LR': {"type": "ridge", "alpha": 1.0},
            'RF': {"type": "rf", "n_estimators": 300, "max_depth": 12}
        }

    Returns
    -------
    features : Dict[str, pd.Series]
        {name -> OOF feature aligned with df.index}
    final_models : Dict[str, fitted model]
    df: pd.DataFrame
        with only selected features
    feature_cols: List[str]
        said features (df.columns[1:])
    """

    # sigma_y_GW = 11.7  # TODO do not do by hand

    # Extract matrices
    # -------------------------
    X_GW: np.ndarray = df[feature_cols].values.astype(np.float32)
    y_GW: np.ndarray = df[target_col  ].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_GW)

    N: int = len(df)
    # print(f"indices: 0 < {train_end} < {val_end} < {N}")
    assert 0 < train_end < val_end <= N, "Invalid split indices"

    # Define ranges
    # -------------------------
    train_idx: np.ndarray = np.arange(0,        train_end)
    valid_idx: np.ndarray = np.arange(train_end,val_end)
    test_idx : np.ndarray = np.arange(val_end,  N)

    X_train_scaled = X_scaled[train_idx];  y_train_GW = y_GW[train_idx]


    # lasso pass to select features
    # -------------------------
    cfg = models_cfg['LR'].copy()
    cfg.pop('type')

    model_lasso = Lasso(**cfg)
    model_lasso.fit(X_train_scaled, y_train_GW)

    coeffs_lasso= pd.Series(model_lasso.coef_, index=feature_cols).astype(np.float32)
    # print("coeffs_lasso:", coeffs_lasso)

    # Features with non-zero coefficients
    idx_coeffs   = np.where(coeffs_lasso != 0)[0]
    # print("idx_coeffs:", idx_coeffs)
    feature_cols = list(np.array(feature_cols)[idx_coeffs])
    feature_cols = [str(feature) for feature in feature_cols] # get rid of np.str_
    # print("feature_cols:", feature_cols)

    # print(df.shape, X_GW.shape)
    X_GW = X_GW[:, idx_coeffs]
    df   = df[[target_col] + feature_cols]
    # print(df.shape, X_GW.shape)


    # start over, with fewer features
    # -------------------------
    X_scaled = scaler.fit_transform(X_GW)

    X_train_scaled = X_scaled[train_idx];  y_train_GW = y_GW[train_idx]
    X_valid_scaled = X_scaled[valid_idx];  y_valid_GW = y_GW[valid_idx]
    X_test_scaled  = X_scaled[ test_idx];  y_test_GW  = y_GW[ test_idx]

    models            = dict()
    preds_GW          = dict()
    series_pred_GW    = pd.Series()

    for name, cfg in models_cfg.items():  # name = e.g. 'LR', 'rf'
        preds_GW          [name] = pd.Series()
        # losses_quantile_GW[name] = dict()

        if name == 'oracle':
            warnings.warn("Using the oracle!")
            models[name] = None # meaningless
            pred_train_GW = y_train_GW
            pred_valid_GW = y_valid_GW
            pred_test_GW  = y_test_GW

        else:
            models[name] = _build_model_from_cfg(cfg)
            models[name].fit(X_train_scaled, y_train_GW)

            pred_train_GW = models[name].predict(X_train_scaled)
            pred_valid_GW = models[name].predict(X_valid_scaled)
            pred_test_GW  = models[name].predict(X_test_scaled )


        series_pred_GW[name] = pd.Series(
            np.concatenate([pred_train_GW, pred_valid_GW, pred_test_GW]),
                            index = df.index)
        # print(series_pred_GW[name])


    # most relevant features
    if verbose >= -3 and {'LR', 'RF'} <= models.keys():
        most_relevant_features(models['LR'], models['RF'], feature_cols)


    return series_pred_GW, models, df, feature_cols

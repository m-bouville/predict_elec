# ----------------------------------------------------------------------
#
# LR_RF.py
#
# linear regression & random forest
#
# ----------------------------------------------------------------------


import os, warnings
import time

import json
import hashlib
import pickle

from   typing import List, Tuple, Dict, Any # Sequence, Optional
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



# ============================================================
# CREATE BASELINES (LINERA REGREASSION, RANDOM FOREST, GRADIENT BOOSTING)
# ============================================================

def create_baselines(df            : pd.DataFrame,
                     target_col    : str,
                     feature_cols  : List[str],
                     baseline_cfg  : Dict[str, Dict[str, Any]],
                     train_split   : float,
                     n_valid       : int,
                     cache_dir     : str = "cache",
                     force_calculation:bool = False,
                     verbose       : int = 0) \
        -> Tuple[pd.DataFrame, str, List[str]]:

    t_start = time.perf_counter()

    # Dict describing data + RF config (used to build cache key)
    cache_id = {
        "target":       target_col,
        "feature_cols": feature_cols,  # TODO add AFTER lasso
        'train_end':    train_split-n_valid,
        'val_end':      train_split,
        # "split": "v1",   # optional: data split identifier
    }



    # Extract matrices
    # -------------------------
    X_GW: np.ndarray = df[feature_cols].values.astype(np.float32)
    y_GW: np.ndarray = df[target_col  ].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_GW)


    # Define ranges
    # -------------------------
    train_idx: np.ndarray = np.arange(0,        train_split-n_valid)
    X_train_scaled = X_scaled[train_idx];  y_train_GW = y_GW[train_idx]


    # lasso pass to select features
    # -------------------------
    cfg = baseline_cfg['lasso'] # .copy()

    model_lasso = Lasso(**cfg)
    model_lasso.fit(X_train_scaled, y_train_GW)

    coeffs_lasso= pd.Series(model_lasso.coef_, index=feature_cols).astype(np.float32)
    # print("coeffs_lasso:", coeffs_lasso)

    # Features with non-zero coefficients
    idx_coeffs   = np.where(coeffs_lasso != 0)[0]
    feature_cols = list(np.array(feature_cols)[idx_coeffs])
    feature_cols = [str(feature) for feature in feature_cols] # get rid of np.str_
    # print("feature_cols:", feature_cols)
    if verbose >= 1:
        print(f"lasso: {X_GW.shape[1]} -> {len(feature_cols)} features")

    # print(df.shape, X_GW.shape)
    X_GW = X_GW[:, idx_coeffs]
    df   = df[[target_col] + feature_cols]
    # print(df.shape, X_GW.shape)


    # start over, with fewer features
    # -------------------------

    baseline_features_GW, baseline_models = \
        regression_and_forest(
            X               = X_GW,
            y               = y_GW,
            target_col      = target_col,
            feature_cols    = feature_cols,
            dates           = df.index,
            train_end       = train_split-n_valid,
            val_end         = train_split,
            models_cfg      = baseline_cfg,
            cache_dir       = cache_dir,
            cache_id_dict   = cache_id,
            force_calculation=force_calculation,
            verbose         = verbose
        )


    if verbose >= 1:
        print(f"LR + RF took: {time.perf_counter() - t_start:.2f} s")

    # Add features
    baseline_idx = dict()
    for name, series in baseline_features_GW.items():
        col_name     = f"consumption_{name}"
        df[col_name] = series
        feature_cols.append(col_name)
        baseline_idx[name] = feature_cols.index(col_name)
    # print(df['consumption_regression'].head(20))
    if verbose >= 3:
        print(f"baseline_idx: {baseline_idx}")

    if verbose >= 2:
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        print("Using target:", target_col)

    return df, feature_cols



def regression_and_forest(
    X:             np.ndarray,
    y:             np.ndarray,
    target_col:    str,
    feature_cols:  List[str],
    dates:         pd.DatetimeIndex,
    train_end:     int,     # end of training set (exclusive)
    val_end:       int,     # end of validation set (exclusive)
    models_cfg:    Dict[str, dict],
    cache_dir:     str,
    cache_id_dict: dict,
    force_calculation:bool = False,
    verbose:       int = 0
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

    train_idx: np.ndarray = np.arange(0,        train_end)
    valid_idx: np.ndarray = np.arange(train_end,val_end)
    test_idx : np.ndarray = np.arange(val_end,  len(X))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_scaled = X_scaled[train_idx];  y_train_GW = y[train_idx]
    X_valid_scaled = X_scaled[valid_idx]
    X_test_scaled  = X_scaled[ test_idx]

    models         = dict()
    preds_GW       = dict()
    series_pred_GW = dict()

    for name, cfg in models_cfg.items():  # name = e.g. 'LR', 'RF'
        preds_GW          [name] = pd.Series()
        # losses_quantile_GW[name] = dict()

        if name == 'oracle':
            warnings.warn("Using the oracle!")
            models[name] = None # meaningless
            pred_train_GW = y_train_GW
            pred_valid_GW = y[valid_idx]
            pred_test_GW  = y[ test_idx]

            continue

        if name == 'lasso':
            continue


        # normal models
        os.makedirs(cache_dir, exist_ok=True)

        key_str    = json.dumps(cache_id_dict | cfg, sort_keys=True)
        cache_key  = hashlib.md5(key_str.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{name}_preds_{cache_key}.pkl")

        # either load...
        if os.path.exists(cache_path) and name != 'LR' and not force_calculation:
            if verbose > 0:
                print(f"Loading {name} predictions from: {cache_path}...")
            with open(cache_path, "rb") as f:
                ((pred_train_GW, pred_valid_GW,
                  pred_test_GW, models[name])) = pickle.load(f)

        else:  # ... or compute
            if verbose > 0:
                if name == 'LR':
                    print(f"Training {name}...")
                elif force_calculation:
                    print(f"Training {name} (calculation forced)...")
                else:
                    print(f"Training {name} (no cache found)...")

            models[name] = _build_model_from_cfg(cfg)
            models[name].fit(X_train_scaled, y_train_GW)

            pred_train_GW = models[name].predict(X_train_scaled)
            pred_valid_GW = models[name].predict(X_valid_scaled)
            pred_test_GW  = models[name].predict(X_test_scaled )

            # Save
            if name != 'LR':
                with open(cache_path, "wb") as f:
                    pickle.dump((pred_train_GW, pred_valid_GW,
                                 pred_test_GW,  models[name]), f)
                if verbose > 0:
                    print(f"Saved {name} predictions to: {cache_path}")

        series_pred_GW[name] = pd.Series(
            np.concatenate([pred_train_GW, pred_valid_GW, pred_test_GW]),
                            index = dates)
        # print(series_pred_GW[name])


    # most relevant features
    if verbose >= 3 and {'LR', 'RF'} <= models.keys():
        most_relevant_features(models['LR'], models['RF'], feature_cols)


    return series_pred_GW, models

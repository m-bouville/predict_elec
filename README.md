# Electricity Consumption Forecasting — Architecture Overview

My purpose is to forecast electricity consumption. On top of historical consumption data, other information (temperature, school holidays) is used to provide context.

Input data (included) are from open sources, for the past ten years. The consumption I currently use is the half-hourly national French consumption (average: 53 GW). 

The model predicts not only the median consumption but also other quantiles (e.g. quartiles, deciles). The purpose is to estimate the precision of the forecast median.



## Current State

### Model overview
The current forecasting system predicts consumption using a **quantile-based neural network** (Transformer-style, direct multi-horizon prediction).

The neural network (NN) model outputs multiple quantiles (e.g. `q10`, `q50`, `q90`) for each forecast horizon.


### Features
The NN is trained using:
- Exogenous features (temperature, solar, school holidays)
- Lagged consumption features and engineered statistics
- Predictions from baseline models used as features:
  - **Linear Regression (LR)**
  - **Random Forest (RF)**

LR and RF are *not* ensembled directly; they are treated as informative input features.


### Training and validation
- Training objective: pinball (quantile) loss
	- Quantile crossing constraints applied sequentially
- Validation focuses on:
  - quantile coverage
  - calibration
  - sharpness
- Forecasts are produced using **direct multi-horizon prediction**
- No autoregression
- No aggregation across forecast origins prior to validation


### Limitations
- The `q50` (median) forecast is not guaranteed to be optimal as a **mean / point forecast**
- LR and RF predictors are intrinsically mean-oriented and may introduce bias when optimized under a quantile loss
- There is currently no learned mechanism to optimally combine predictors for point accuracy (e.g. RMSE or MAE)


### Problems in the current code and setup

TODO: Add:
- diagnostics
- how the new architecture will improve this.

- **Systematic bias in predictions**
  - Bias is visible across horizons and regimes, even when overall RMSE / MAE are reasonable.
  - The bias differs between LR, RF, and NN, and is not explicitly corrected anywhere in the pipeline.

- **Median vs mean ambiguity**
  - The neural network is trained with a quantile loss, so `q50` is a conditional median.
  - Operational evaluation and comparison (RMSE, MAE, bias) implicitly target the conditional mean.
  - Using `q50` as a point forecast mixes these two objectives and can lead to persistent bias.

- **LR and RF used outside their natural objective**
  - LR and RF are trained (or designed) to minimize mean-based losses.
  - When used as features in a quantile-trained NN, their signal is partially misaligned with the optimization objective.
  - The NN can compensate, but this creates unnecessary tension in training.

- **No learned combination of predictors**
  - LR, RF, and NN outputs are implicitly combined inside the quantile NN without an explicit objective for point accuracy.
  - There is no model whose sole purpose is to optimally weight or correct these predictors for mean performance.

- **Validation focuses on probabilistic quality only**
  - Quantile calibration and coverage are well defined and evaluated.
  - Point-forecast diagnostics (bias, RMSE) are secondary and not directly optimized.
  - This makes it hard to reason about trade-offs between uncertainty quality and operational accuracy.

- **Limited interpretability of bias sources**
  - Bias may come from:
    - miscalibrated baselines (LR / RF),
    - regime changes,
    - quantile loss asymmetry,
    - horizon-dependent effects.
  - The current architecture does not isolate these mechanisms clearly.

- **Architecture tightly couples uncertainty and point prediction**
  - Improving point accuracy (e.g. via tuning or regularization) risks degrading quantile calibration.
  - There is no clean way to improve one without affecting the other.



---

## Planned Architecture

### High-level design
The planned system introduces a **two-stage architecture** with a strict separation of responsibilities:

1. A **probabilistic model** estimates uncertainty: the Neural Network using Transformers for Quantiles (NNTQ)
2. A **deterministic mean meta-model** forecasts the operational point 

There is **no feedback loop** between the two stages.

---

### Stage 1 — Quantile Neural Network (probabilistic layer)

**Input**
- Raw exogenous and engineered features
- LR and RF predictions (as features)

**Output**
- Multiple conditional quantiles (e.g. `q10`, `q50`, `q90`)

**Training**
- Pinball loss with sequential crossing constraints
- Direct multi-horizon prediction

**Role**
- Learn the conditional distribution of electricity demand
- Provide calibrated uncertainty estimates (e.g. deciles)
- Produce a statistically meaningful median (`q50`)

This stage is **self-contained** and remains unchanged by downstream models.

---

### Stage 2 — Mean-Based Meta-Model (deterministic layer)

**Input**
- LR and RF predictions
- Median (`q50`) output from the Neural Network using Transformers for Quantiles (NNTQ)
- Potentially: same raw features used by the quantile NN

**Output**
- A single point forecast optimized for mean accuracy

**Training**
- Mean-based loss (e.g. MAE or MSE)
- Trained independently from the quantile NN
- No gradient flow or feedback to Stage 1

**Role**
- Learn optimal weights and nonlinear combinations of:
  - baseline predictors (LR, RF)
  - neural network median
- Correct systematic bias
- Optimize operational point accuracy

---

### Key design principles

- **Strict separation of concerns**
  - Quantile NN defines uncertainty
  - Meta-model defines the point forecast
- **One-way information flow**
  - Quantile outputs feed the meta-model
  - Meta-model does not influence quantile training
- **Probabilistic integrity**
  - Quantile calibration is preserved
  - Point forecast improvements do not distort uncertainty estimates

---

### Expected benefits
- Improved point forecast accuracy over using `q50` alone
- Better bias correction across regimes
- Retention of well-calibrated probabilistic forecasts
- Clean validation and interpretability at each stage


---
 
### Plan

The implementation will proceed incrementally, keeping the current system as a stable reference.

#### Stage 2: Mean-based meta-model
- version 0.1:
  - meta-model based on 3 inputs: Median (`q50`) from the Neural Network using Transformers for Quantiles (NNTQ), LR prediction, RF prediction;
  - constant weights set by hand;
  - this stage provides a simple, interpretable reference and validates the usefulness of a trained meta-model.
- version 0.2: 
  - linear regression with the same 3 inputs;
  - mean-based loss (initially MAE, possibly MSE later);
  - weights are still constant, but trained;
  - no feedback to the quantile model.
- version 1: 
  - small dense neural network;
  - extra features beyond the three predictions (similar to those to train the NNTQ);
  - **Motivation**
    - Capture regime-dependent weighting of predictors,
    - Allow nonlinear bias correction,
    - Improve robustness across seasons and extreme events;
  - still no influence on quantile training

---


#### Output strategy

- **Initial**
  - Direct multi-horizon prediction for all horizons
  - Strict horizon alignment between all predictors

- **Possible later extension**
  - Light (partial) decoder for the meta-model only
    - Parallel decoding
    - No autoregression
    - Explicit horizon conditioning if needed

Any decoder-based extension must remain compatible with:
- direct validation,
- stable feature semantics,
- and the absence of feedback to the quantile model.

---

#### Validation protocol

- **Quantile model**
  - Validated independently using:
    - pinball loss
    - coverage
    - calibration
    - sharpness

- **Meta-model**
  - Validated using:
    - RMSE
    - MAE
    - bias
  - Evaluated per horizon, without aggregation across origins

- **Reporting**
  - Probabilistic and point-forecast metrics reported separately
  - No metric mixing between stages

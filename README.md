# Electricity Consumption Forecasting — Architecture Overview


## Introduction
The purpose of this project is to forecast electricity consumption. On top of historical consumption data, other information (temperature, school holidays) is used to provide context.

The neural network (NN) model predicts not only the median consumption but also other quantiles (e.g. quartiles, deciles) in order to estimate the precision of the forecast median.

The focus is currently on the half-hourly national French consumption (average: 53 GW). Input data (included) are from open sources, for the past ten years. 


### Possible applications
- Grid operations, unit commitment
  - requires: avoiding underestimating peaks;
- Short-term electricity price forecasting and trading
  - requires: good short-term predictions (esp. ramps and extreme events);
- Long-term planning, energy transition analysis
  - requires: interpretability of exogenous variables as causes.


### Choice of application
In Europe, prices are set daily at noon for the next day (day-ahead price). Producers and consumers must let the market know what their 48 half-hourly consumptions will be, from _h_ + 12 to _h_ + 36. 

Gearing the model toward this specific case has two advantages:
- the goal is univocal: one compares each of the 48 half-hourly predictions to 48 actual consumptions (no issue of aggregation);
- the consumption at _t_ is predicted only once: the day before at noon, not _H_ times.


### A note on the system size
- The number of samples drops by nearly a factor of 50, but this number is artificially high:
  - noon-only: 3 650 _non-overlapping_ training samples in ten years;
  - all origins: 175 000 _overlapping_ training samples.
- Training only from _h_ + 12 to _h_ + 36 lets the model rely on the fact that time step number 24 is always midnight. No need for a decoder or extra degrees of freedom to get this pattern right.


---

## Current State

### Model overview
The current consumption forecasting system revolves around a Neural Network which uses Transformers to predict Quantiles (hereafter, NNTQ), with direct multi-horizon prediction.

The loss includes:
- pinball (quantile) loss,
- quantile crossing constraints applied sequentially (so that `q10` <= `q50` <= `q90`),
- sharpness (derivatives).

The NN is trained using:
- exogenous features (temperature, school holidays),
- week-ends plus Fourier-like sine waves at different scales (day, year),
- moving averages of recent consumption (but not so recent as to cause leaks),
- predictions from baseline models used as features: **linear regression (LR)** and **random forest (RF)**
  - LR and RF are *not* ensembled directly: they are treated as informative input features.


### Issues in the current model
- **Systematic bias in predictions**
  - Bias is visible in LR, RF and NN in validation and testing (but not training).
  - Adding moving averages as features (see above) halved the bias.

- **Median vs mean ambiguity**
  - The neural network is trained with a quantile loss, so `q50` is a conditional median.
  - Operational evaluation and comparison (RMSE, MAE, bias) implicitly target the conditional mean.
  - Using `q50` as a point forecast mixes these two objectives and can lead to persistent bias.
  - The Transformer is simultaneously responsible for learning uncertainty structure (quantiles) and producing a usable point forecast; any change improving point RMSE can degrade quantile calibration (and vice versa).

- **No learned combination of predictors**
  - The meta-model is less powerful than it could be: LR, RF and NN outputs are statically combined, weights are set by hand.
  - A learned meta-learner would allow nonlinear bias correction, make weights regime-dependent and improve robustness across seasons and extreme events.

- **Remote prdiction**
  - The code does not currently predict _h_ + 12 to _h_ + 36, but rather _h_ to _h_ + 24
  - The model must be allowed to skip the first 12 hours in validation: run from _h_ to _h_ + 36 but use only _h_ + 12 to _h_ + 36.


---

## Planned Architecture

### High-level design
The planned system introduces a **two-stage architecture** with a strict separation of responsibilities:
1. A **probabilistic model** estimates uncertainty: the Neural Network using Transformers for Quantiles (NNTQ)
2. A **deterministic mean meta-model** forecasts the operational point 

There is **no feedback loop** between the two stages.


### Stage 1 — Quantile Neural Network (probabilistic layer)

**Input**
- Raw exogenous and engineered features
- LR and RF predictions (as features)

**Output**
- Multiple conditional quantiles (e.g. `q10`, `q50`, `q90`)

**Training**
- Pinball loss with sequential crossing constraints
- Direct multi-horizon prediction
  - Possible later extension: a light (partial) decoder

**Role**
- Learn the conditional distribution of electricity demand to produce
  - calibrated uncertainty estimates (e.g. deciles),
  - a statistically meaningful median (`q50`).

This stage is **self-contained** and remains unchanged by downstream models.


### Stage 2 — Mean-Based Meta-Model (deterministic layer)

**Input**
- LR and RF predictions
- Median (`q50`) output from stage 1
- Potentially: raw features similar to those used in the quantile NN

**Output**
- A single point forecast optimized for mean accuracy

**Training**
- no quantiles (e.g. MAE or MSE)
- Trained independently from the quantile NN
- No gradient flow or feedback to Stage 1

**Role**
- Correct systematic bias
- Optimize operational point accuracy


#### Implementation plan
- currently:
  - meta-model based on 3 predictions: LR, RF and the median (`q50`) from the NNTQ;
  - constant weights set by hand.
- next: 
  - linear regression with the same 3 predictions as input (initially MAE, possibly MSE later);
  - weights are still constant, but trained.
- finally: 
  - small dense neural network;
  - extra features beyond the three predictions (similar to those to train the NNTQ);
  - still no influence on quantile training.


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


### Expected benefits
- Improved point forecast accuracy over using `q50` alone
- Retention of well-calibrated probabilistic forecasts
- Better bias correction across regimes
- Clean validation and interpretability at each stage

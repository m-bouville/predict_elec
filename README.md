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

## Architecture

The strategy is based on a **two-stage architecture** with a strict separation of responsibilities:
1. First, a Neural Network uses Transformers to predict Quantiles (hereafter, NNTQ), thus estimating uncertainty.
2. Then, a meta-model forecasts the operational point as the mean.

There is **no feedback loop** between the two stages.

Expected benefits
- Improved point forecast accuracy over using `q50` alone
- Retention of well-calibrated probabilistic forecasts
- Better bias correction across regimes
- Clean validation and interpretability at each stage

The main file is `predict_elec.py`.


### Stage 1 — Quantile Neural Network

**Input**
- exogenous features (temperature, school holidays),
- week-ends plus Fourier-like sine waves at different scales (day, year),
- moving averages of recent consumption (but not so recent as to cause leaks),
- predictions from baseline models (**linear regression, LR** and **random forest, RF**) used as features, *not* ensembled directly.

**Output**
- Multiple conditional quantiles (e.g. `q10`, `q50`, `q90`)

**Training**
- Pinball (quantile) loss with sequential crossing constraints (so that `q10` <= `q50` <= `q90`) and sharpness (derivatives),
- Direct multi-horizon prediction
  - Possible later extension: a light (partial) decoder

**Role**
- Learn the conditional distribution of electricity demand to produce
  - calibrated uncertainty estimates (e.g. deciles),
  - a statistically meaningful median (`q50`).

This stage is **self-contained** and remains unchanged by the downstream metamodel.


### Stage 2 — Mean-Based Meta-Model

**Input**
- LR and RF predictions
- Median (`q50`) output from stage 1
  - also, `q75` minus `q25` (uncertainty proxy)?

**Output**
- A single point forecast optimized for mean accuracy

**Training**
- A small dense neural network minimizing MSE, not quartiles
- Trained independently from the quantile NN: no gradient flow or feedback to Stage 1

**Role**
- Correct systematic bias
- Optimize operational point accuracy


  
---

## Issues in the current model
- **Systematic bias in predictions**
  - Bias is visible in LR, RF and NN in validation and testing (but not training).
  - Adding moving averages as features (see above) halved the bias.

- **Median vs mean ambiguity**
  - The neural network is trained with a quantile loss, so `q50` is a conditional median.
  - Operational evaluation and comparison (RMSE, MAE, bias) implicitly target the conditional mean.
  - Using `q50` as a point forecast mixes these two objectives and can lead to persistent bias.
  - The Transformer is simultaneously responsible for learning uncertainty structure (quantiles) and producing a usable point forecast; any change improving point RMSE can degrade quantile calibration (and vice versa).

- **Remote prdiction**
  - The model currently predicts _h_ to _h_ + 24: it must be allowed to skip the first 12 hours in validation -- run from _h_ to _h_ + 36 but vaidate on _h_ + 12 to _h_ + 36 only.

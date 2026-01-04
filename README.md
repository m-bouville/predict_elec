# Electricity Consumption Forecasting — Architecture Overview


## Introduction
The purpose of this project is to use neural network (NN) models to forecast electricity consumption. Information such as temperature and public and school holidays is used along historical consumption data to provide context.

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

Gearing the model toward this specific case has two (advantageous) consequences:
- the goal is univocal: one compares each of the 48 half-hourly predictions to 48 actual consumptions (no issue of aggregation);
- the consumption at _t_ is predicted only once: the day before at noon, not _H_ times.


### A note on the system size
- The number of samples drops by nearly a factor of 50, but this number is artificially high:
  - 3 650 _non-overlapping_ daily samples in ten years;
  - 175 000 _overlapping_ half-hourly samples.
- Training only from _h_ + 12 to _h_ + 36 lets the model rely on the fact that time step number 24 is always midnight. No need for a decoder or extra degrees of freedom to get this pattern right.


### Leakage Control
- All features used to predict _h_ + 36 are available at _h_: no consumption data from D+1 are used.
- Baseline model predictions are generated out-of-sample only.
- Meta-models are trained on validation predictions exclusively.


---

## Architecture

The strategy is based on a **two-stage architecture** with a strict separation of responsibilities:
1. First, a Neural Network uses Transformers to predict Quantiles (hereafter, NNTQ), in order to estimate the uncertainty of the forecast.
2. Then, a meta-model forecasts the operational point as the mean.

There is **no feedback loop** between the two stages.

The main file is `predict_elec.py`.


### Stage 1 — Quantile Neural Network

**Input**
- exogenous features (temperature, school holidays);
- week-ends plus Fourier-like sine waves at different scales (day, year);
- moving averages of recent consumption (but not so recent as to cause leaks);
- predictions from baseline models --**linear regression (LR)**, **random forest, (RF)** and **gradient boosting (GB)**-- used as features, *not* ensembled directly (yet).

**Output**
- Multiple conditional quantiles (e.g. `q10`, `q50`, `q90`)

**Training**
- Pinball (quantile) loss with sequential crossing constraints (so that `q10` <= `q50` <= `q90`) and sharpness (derivatives);
- Direct multi-horizon prediction

**Role**
- Learn the conditional distribution of electricity demand to produce
  - calibrated uncertainty estimates (e.g. deciles),
  - a statistically meaningful median (`q50`).

This stage is **self-contained** and remains unchanged by the downstream metamodel.


### Stage 2 — Mean-Based Meta-Model

**Input**
- LR, RF an GB predictions
- Median (`q50`) output from stage 1
- features similar to those used in stage 1.

**Output**
- A single point forecast optimized for mean accuracy

**Training**
- A small dense neural network minimizing MSE, not quartiles
- Trained independently from the quantile NN: no gradient flow or feedback to Stage 1

**Role**
- Correct systematic bias
- Optimize operational point accuracy


  
---

## Issues and plans
- **Systematic bias in predictions**
  - Bias is visible in LR, RF and NN in validation and testing (but not training).

- **Empirical coverage of predicted quantiles**
  - While `q10` and `q25` aim at approximating the first decile and quartile, this is not explicitly enforced; consequently there is a difference.
  
- **Plans**
  - `q75` minus `q25` (uncertainty proxy) could be used as a feature in the NN metamodel.

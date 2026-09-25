# Tests

Regression tests for the fixes applied to the forecasting pipeline, unit tests
of the objectives, and end-to-end smoke tests.

[Written automatically by Claude]


## Running

```bash
pip install pytest            # plus the project deps (torch, lightgbm, optuna, holidays, ...)
pytest                        # from the project root, ~1-2 min
```

On a full training environment every test runs. In a minimal environment the
files needing torch, lightgbm, optuna or holidays skip themselves via
`pytest.importorskip`. No GPU is needed: the smoke tests train a tiny model on
CPU. The Bayesian-range tests read `parameter_search_{NNTQ,meta}.csv` and skip
when they are absent.

## What each file covers

| File | What | Needs |
|------|------|-------|
| `test_run_smoke.py` | **end-to-end** `run.run_model_once` on synthetic data (tiny NNTQ, fast baselines): finite losses and metrics, csv row, price never a feature; metamodels off keeps the row schema; NNTQ cache loaded instead of retraining, baseline predictions refreshed and aligned on dates; NNTQ variants built once (N+2 trainings, N kept), then loaded | torch, lightgbm |
| `test_search_objective.py` | `loss_NNTQ` (hand-computed, spread, dominant gap, worst days) and `loss_meta` (hand-computed, NNTQ weight 0, abs bias, model weights); Bayesian objective with a scripted `run_model_once`: single runs during warm-up, extra runs only within `wiggle`, early stop, `clean_avg`, csv `num_runs`/losses, one NNTQ variant per meta run | torch, optuna |
| `test_bayes_ranges.py` | sampled values fit `DISTRIBUTIONS_*`; categorical choices identical (TPE); the existing csv files load and sampling works after loading | torch, optuna |
| `test_NNTQ_variants.py` | `build_NNTQ_variants` with faked training: seeds, middle N kept, median first, temporary files removed, summary json, cache valid only for the same N | torch |
| `test_predictions_and_metamodel_skip.py` | vectorized `prediction_day_ahead` identical to the former loop; `do_metamodel=False` keeps the csv schema (NaN meta columns) | torch |
| `test_patching.py` | patch count = what the Conv1d produces, for every stride / patch length in `DISTRIBUTIONS_NNTQ` with patch >= stride (grid read from `Bayes_search`); every pair the search can sample is in that grid and has overlapping patches; existing configurations unchanged | torch, optuna |
| `test_warnings.py` | per-head dimension not a multiple of 8 (constants.py re-executed with other values); `batch_size` too large (no step, or < 10 steps per epoch) | torch |
| `test_architecture.py` | LR warmup + cosine (first step non-zero, 25% cap); RMSNorm eps; `DayAheadDataset` windows with and without `features_in_future` | torch |
| `test_losses.py` | cold penalty per sample; torch/numpy twins agree (quantile + crossing, wrapper with derivative and median terms, derivative, regions); derivative and regional terms hand-computed | torch |
| `test_features.py` | daily sin/cos pairs, `sin_12mo`; past-consumption features only use data `lag` steps old, and `lag = pred_length` (no leak through `features_in_future`) | torch (via `IO`), holidays |
| `test_containers_and_columns.py` | `DataSplit.__post_init__` checks run; price excluded from features and not used to clip the range | torch |
| `test_io_timezone.py` | regional consumption localised to `Europe/Paris`, DST-nonexistent hour dropped; a pickle under the old (pre-tz) key is not reloaded; the cache is reloaded without parsing the csv | torch (via `IO`) |
| `test_metamodel_and_baselines.py` | baseline predictions finite; RF and LGBM finite, better than the mean, cached by configuration, deterministic; Ridge never cached | lightgbm |
| `test_plots_prepare_series.py` | **B8**: every curve goes through one MA -> range -> groupby pipeline | matplotlib |

## Limits

* `test_warnings.py` re-executes `constants.py` with `model_dim` / `num_heads`
  substituted by regex (the check runs at import); it fails loudly if the
  substitution no longer finds them.
* `test_bayes_ranges.py` also checks your csv files: a failure there can mean a
  csv row outside the current ranges (e.g. a `batch_size` no longer among the
  choices), not a code bug.
* Not tested: `plot_statistics`, most plots, the early stopping inside
  `NeuralNet.run` (only run end-to-end), geometric pooling in isolation,
  meta-LR weights.
* The caches have no code version: after a change to the NNTQ, data or baseline
  code, delete `cache/NNTQ_*` (and `cache/RF_*`, `cache/LGBM_*` for baselines).

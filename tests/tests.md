# Tests

Regression tests for the fixes applied to the forecasting pipeline, unit tests
of the objectives, and end-to-end smoke tests.

[Written automatically by Claude]


## Running

```bash
pip install pytest            # plus the project deps (torch, lightgbm, optuna, holidays, ...)
pytest                        # from the project root, about 1-1.5 min
```

On a full training environment every test runs. In a minimal environment the
files needing torch, lightgbm, optuna or holidays skip themselves via
`pytest.importorskip`. No GPU is needed: the smoke tests train a tiny model on
CPU. The Bayesian-range tests read `parameter_search_{NNTQ,meta}.csv` and skip
when they are absent or older than the test period cut in two.

## What each file covers

| File | What | Needs |
|------|------|-------|
| `test_run_smoke.py` | **end-to-end** `run.run_model_once` on synthetic data (tiny NNTQ, fast baselines): finite losses and metrics, csv row (objective on the first half of the test period, second half reported), price never a feature; metamodels off keeps the row schema; NNTQ cache loaded instead of retraining, another cache for another validation split or other region weights, the NNTQ gets the region / national std ratios for its regional loss, baseline predictions refreshed and aligned on dates; NNTQ variants built once (N+2 trainings, N kept), then loaded; no autocast / GradScaler on CPU; metamodels give the same result whether the NNTQ was trained or loaded; early stopping only on validated epochs; the NNTQ pickle holds no torch datasets / loaders nor the whole unscaled arrays (`DatasetBundle.for_cache`, shared arrays, run bundle unchanged); a verbose=2 run (comparisons, diagnostic and thermosensitivity plots), trained then from the cache, leaves no figure open | torch, lightgbm |
| `test_search_objective.py` | `loss_NNTQ` (hand-computed, spread, dominant gap, worst days: the mean over the top-n worst Paris days, DST days left out) and `loss_meta` (hand-computed, NNTQ weight 0, abs bias, model weights); Bayesian objective with a scripted `run_model_once`: single runs during warm-up, extra runs only within `wiggle`, early stop, `clean_avg`, csv `num_runs`/losses, one NNTQ variant per meta run; rows written through the checked `append_csv_row` | torch, optuna |
| `test_bayes_ranges.py` | sampled values fit `DISTRIBUTIONS_*`; categorical choices identical (TPE); the existing csv files load and sampling works after loading; small learning rates / weight decays keep their significant digits on reload; numeric `RF_max_features` next to `sqrt` reloads | torch, optuna |
| `test_NNTQ_variants.py` | `build_NNTQ_variants` with faked training: seeds, middle N kept, median first, temporary files removed, summary json, cache valid only for the same N; a rebuild with fewer variants removes the obsolete ones; variants pickled through `for_cache()` | torch |
| `test_predictions_and_metamodel_skip.py` | `prediction_day_ahead`: vectorized = loop; each prediction stamped with the date of its row; only the Paris day after the noon (Paris) origin is scored (48 steps, 46 on the spring-forward Sunday, missing rows such as the RTE fall-back hour handled); `do_metamodel=False` keeps the csv schema (NaN meta columns, both halves); train predictions cover every origin | torch |
| `test_patching.py` | patch count = what the Conv1d produces, for every stride / patch length in `DISTRIBUTIONS_NNTQ` with patch >= stride (grid read from `Bayes_search`); every pair the search can sample is in that grid and has overlapping patches; existing configurations unchanged; the block-size assertion message is formatted | torch, optuna |
| `test_warnings.py` | per-head dimension not a multiple of 8 (constants.py re-executed with other values); `batch_size` too large (no step, or < 10 steps per epoch) | torch |
| `test_architecture.py` | learned positional embedding: swapping two patches changes the output; LR warmup + cosine (first step non-zero, 25% cap); RMSNorm eps; `DayAheadDataset`: origins at noon Paris (11:00 / 10:00 UTC), one per Paris day across DST, scored window from 00:00 Paris D+1, valid and test forecast from their first day (inputs from the previous split), windows with and without `features_in_future`; best-model saver ignores NaN, explicit error if nothing saved; non-empty validation split; baseline predictions stored as Series on the split dates; training loss sums detached from the autograd graph; training losses computed in float32 even when the model returns float16 (autocast); validation / test loss (`subset_evaluation`) = the training losses computed once over all the samples (independent of the batch size), without gradients, copied to the CPU once at the end, every tensor on the model's device (checked on the 'meta' device); the validation profile returned is that of the restored (best) model | torch |
| `test_losses.py` | cold penalty per sample; pinball, coverage, crossing, derivative and regional terms hand-computed; regional errors in national units (the sum term is the national error); the wrapper adds the derivative and median terms; coverage scale computed once gives the same numbers; no numpy twin left | torch |
| `test_features.py` | calendar features in Paris local time (same values for the same local time in winter and summer, holidays over the Paris day); daily sin/cos pairs, `sin_12mo`; past-consumption features only use data `lag` steps old, and `lag = pred_length` (no leak through `features_in_future`); SMA windows of their named size (1, 2, 4 weeks); public holidays after 2026; school holidays NaN (unknown) after the last date of the calendar, not 0; vectorized `is_holiday` identical to a per-date lookup on the Paris dates; `IO.add_calendar_columns` | torch (via `IO`), holidays |
| `test_containers_and_columns.py` | `DataSplit.__post_init__` checks run; price excluded from features and not used to clip the range; `DatasetBundle` items() and []; the test period cut in two at a Paris midnight (whole days, DST day included), metrics of each half | torch |
| `test_io.py` | real-time consumption: the half-hourly value is the :00 / :30 reading; the data end at the end of the model inputs (not price / eco2mix), last temperature day whole; `load_data` in statistics mode parses (and plots) eco2mix once, from the pickle or not, and at verbose 3; `_read_or_download`: a missing csv is downloaded as served (no extra index column, no partial file on failure) and read with the same options as a local one; eco2mix's first download reads 'ND' (samples of the real files in `tests/data/`); no loader reads a URL directly; world temperatures: first run (parsed) and cached csv agree | torch (via `IO`) |
| `test_io_timezone.py` | regional consumption localised to `Europe/Paris`, DST-nonexistent hour dropped; a pickle under the old (pre-tz) key is not reloaded; the cache is reloaded without parsing the csv | torch (via `IO`) |
| `test_run.py` | `postprocess` leaves its inputs unchanged; `dates_df` (cache keys) independent of `verbose`/date; `num_trials` warning; `enforce_ranges` with pandas >= 3; `predict_elec` passes its statistics / fast settings and leaves the constants unchanged; `append_csv_row` refuses rows with other columns; a search warns when `validate_every` is not 1; `recalculate_loss` recomputes single-run rows from the search half, keeps multi-run rows; `postprocess`: objective from the search half (`search_*`, q10..q90, `avg_abs_worst_days_search`), reported half in `test_*`, `test_coverage_*`; these columns are not parameters for `Bayes_search`; the input pickle is keyed on the data files, obsolete ones removed | torch |
| `test_plots.py` | **B8**: every curve goes through one MA -> range -> groupby pipeline; time-of-day / day-of-week groupings in local time; single-quantile curves, Series input, legends, default title; `drift_with_time` runs, with a one-year average; eco2mix variation fits keep midnight; one figure per plot and no empty figure (`thermosensitivity_regions`, eco2mix, `plot_optuna`); 0 degC threshold; `plot_optuna` importance table; vectorized `date_of_year` identical to the former per-row map, and fast; `to_local_time` keeps values with their timestamps; `thermosensitivity_regions` leaves its input unchanged; `prices_per_season` in local time (hours and months, March included), seasons without data, or with a few hours only, skipped; `thermosensitivity_per_temperature_model` runs; one shared thermosensitivity helper for the season and hysteresis figures; `diagnostics` without baselines / metamodels; `plots.finish`: figures closed once shown (Agg, inline, under either name) but kept in windows (Qt, Tk, widgets), no figure left open, no bare `plt.show()` left | matplotlib; torch, optuna for some |
| `test_metamodel_and_baselines.py` | baseline predictions finite; RF and LGBM finite, better than the mean, cached by configuration, deterministic; Ridge never cached; meta-NN runs when no epoch improves; meta-NN tensors built once per horizon (not per epoch), results reproducible with a seed; oracle baseline returned; the scaler of the linear baselines fit on the training rows only; metamodel context excludes the predictions; metamodel horizon = half-hour of the Paris day (same in winter and summer, 0..47 each day, 46 values on the spring-forward Sunday) | lightgbm |

## Limits

* `test_warnings.py` re-executes `constants.py` with `model_dim` / `num_heads`
  substituted by regex (the check runs at import); it fails loudly if the
  substitution no longer finds them.
* `test_bayes_ranges.py` also checks your csv files: a failure there can mean a
  csv row outside the current ranges (e.g. a `batch_size` no longer among the
  choices), not a code bug.
* Not tested: most of `plot_statistics` and of the plots (a few run in
  `test_plots.py` and in the verbose smoke run), the early stopping inside
  `NeuralNet.run` (only run end-to-end), geometric pooling in isolation,
  meta-LR weights.
* The caches have no code version: after a change to the NNTQ, data or baseline
  code, delete `cache/NNTQ_*` (and `cache/RF_*`, `cache/LGBM_*` for baselines).

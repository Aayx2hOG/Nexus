# Training configurations for the final models

Recorded 7 October 2026 (IST), from the exported model configs and source experiment artifacts. Both final packages share **the same trained LightGBM and baseline AE**. Only their frozen decision rules and AE thresholds differ. The imported seed42 plain/latent model, optimized AEs, standalone RF/Isolation Forest and multiclass models are comparison experiments, not components of these packages.

## Authoritative artifacts

Artifact-folder model reference: [artifacts/config.md](../artifacts/config.md).

- [Uncertainty-band AE10 config](../models/uncertainty_band_ae10/config.json)
- [Mode-confidence AE5 config](../models/mode_confidence_ae05/config.json)
- Source checkpoints: `experiments/four_mode_ae_20261007/binary.joblib` and `autoencoder.joblib`.
- Original hyperparameter winner: `experiments/lightgbm_validated_v2/binary/selected_validation.json`.
- Final AE thresholds: `experiments/ae_fpr_5_10_20261007/frozen_thresholds.json`.

## Dataset and partitions

UNSW-NB15 official training CSV: 175,341 rows. Inputs are 42 predictors, excluding `id`, `label`, `attack_cat`. Categorical columns: `proto`, `service`, `state`; 39 numeric predictors. Labels: 0 normal, 1 attack. Identical predictor rows remain in the same partition. The saved split is `experiments/lightgbm_validated_v2/split_indices.npz`.

| Partition | Rows | Use |
|---|---:|---|
| fit | 99,778 | LightGBM initial fit; normal-only AE/preprocessing fit |
| early | 25,083 | LightGBM round selection; normal-only AE early stopping |
| calibration | 25,449 | Binary recall calibration and normal-only AE cutoff calibration |
| selection | 25,031 | Compare previously frozen candidate configurations |

LightGBM final refit uses fit + early = **124,861 rows**. AE fitting uses **31,861 normal fit rows**. Final AE FPR calibration uses **8,002 normal calibration rows**. Official testing has 82,332 rows: 37,000 normal and 45,332 attack.

## LightGBM training

Original search: 16 candidates, seed 42, maximum 2,500 rounds, early-stop patience 100; selected trial 4 and 1,318 refit rounds. Binary threshold calibration targets 95% recall using the existing one-sided 95% confidence procedure. Selection minimizes FPR subject to recall ≥95%. A 300-tree balanced Random Forest was the paired control; it is not included in the final packages.

The final base-model experiment retrained the selected configuration on fit + early with fixed rounds. LightGBM preprocessing learns categorical vocabularies on those refit rows, creates one-hot features, casts numeric values to float32 and uses LightGBM’s native handling of numeric NaNs. Unknown categories follow the saved encoder.

| Parameter | Frozen value |
|---|---|
| `boosting_type` | `"gbdt"` |
| `class_weight` | `{"0": 1.5684084913955534, "1": 0.7339928987960873}` |
| `colsample_bytree` | `0.85` |
| `importance_type` | `"split"` |
| `learning_rate` | `0.03` |
| `max_depth` | `-1` |
| `min_child_samples` | `100` |
| `min_child_weight` | `0.001` |
| `min_split_gain` | `0.0` |
| `n_estimators` | `1318` |
| `n_jobs` | `4` |
| `num_leaves` | `63` |
| `objective` | `"binary"` |
| `random_state` | `42` |
| `reg_alpha` | `0.0` |
| `reg_lambda` | `1.0` |
| `subsample` | `0.85` |
| `subsample_for_bin` | `200000` |
| `subsample_freq` | `1` |
| `verbosity` | `-1` |
| `deterministic` | `true` |
| `force_col_wise` | `true` |
| `metric` | `"None"` |

Binary decision: attack when raw LightGBM score ≥ **0.5776925765603604**. This is an uncalibrated model score, not a guaranteed attack probability.

## Baseline autoencoder training

| Setting | Value |
|---|---|
| Implementation | scikit-learn MLPRegressor reconstructing its input |
| Hidden layers | 128 → 32 → 128 |
| Activation / optimizer | ReLU / Adam |
| Learning rate | 0.001, constant |
| L2 alpha | 0.0001 |
| Training epochs / early-stop patience | Up to 20 / 4 stale epochs |
| Best restored epoch | 20 |
| Seed | 42 |
| Training batch size | 512; final short batch 117 |
| Epoch shuffling | Manual NumPy seeded permutation; estimator shuffle=False |
| Early stopping criterion | Mean reconstruction MSE on normal early partition |
| Loss / inference batch size | Mean squared reconstruction error / 2,048 |

Preprocessing is fitted only on normal fit rows: numeric SimpleImputer with its default mean strategy, OneHotEncoder(handle_unknown="ignore", float32), and numeric StandardScaler. Numeric imputation is cast to float32 before scaling. Categorical missing values use `__MISSING__`; one-hot columns remain unscaled. Reconstruction error is averaged over all transformed numeric and one-hot dimensions.

**Implementation detail:** fitting uses manual `partial_fit` epochs and custom early stopping. Thus serialized `max_iter=1` and `early_stopping=False` do not mean only one epoch was trained. Serialized `batch_size=117` is the last short mini-batch, not the configured 512-row training batch size.

Complete fitted estimator settings:

```json
{
  "activation": "relu",
  "alpha": 0.0001,
  "batch_size": 117,
  "beta_1": 0.9,
  "beta_2": 0.999,
  "early_stopping": false,
  "epsilon": 1e-08,
  "hidden_layer_sizes": [
    128,
    32,
    128
  ],
  "learning_rate": "constant",
  "learning_rate_init": 0.001,
  "loss": "squared_error",
  "max_fun": 15000,
  "max_iter": 1,
  "momentum": 0.9,
  "n_iter_no_change": 10,
  "nesterovs_momentum": true,
  "power_t": 0.5,
  "random_state": 42,
  "shuffle": false,
  "solver": "adam",
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": false,
  "warm_start": false
}
```

## Final AE FPR calibration and fusion rules

For target b, use normal calibration reconstruction errors and a conservative `>=` threshold allowing at most floor(b × N) normal alerts. Score comparisons are float64 so an excluded tied float32 score cannot round the threshold back onto the boundary. Four mode boundaries use tail rates b, b/2, b/10. Equality enters the higher mode.

| Setting | Uncertainty-band AE10 | Mode-confidence AE5 |
|---|---|---|
| AE target | 10% | 5% |
| Actual normal-calibration FPR | 9.997501% | 4.998750% |
| No/low boundary | 0.004530309699475766 | 0.008725384250283243 |
| Low/mid boundary | 0.008725384250283243 | 0.015471044927835466 |
| Mid/high boundary | 0.027579598128795627 | 0.04816404730081559 |
| LightGBM uncertainty interval | [0.10, 0.65], inclusive | Not used |
| Decision | Inside interval: none → normal, low → binary, mid/high → attack; outside: binary | None: p ≥0.65; all other modes: p ≥0.5776925765603604 |

5%/10% describe the AE’s calibration FPR, not combined-system test FPR. Mode-confidence AE5 uses the q95-like AE boundary shown here, not the earlier median confidence gate.

## Experiments that supplied the final settings

- `experiment_confidence_ae.py`: used LightGBM scores and calibration AE-error quantiles to search mode cutoffs; retained `[0.65, 0.5776925765603604, 0.5776925765603604, 0.5776925765603604]`.
- `experiment_uncertainty_residuals.py`: searched uncertainty bands and learned fusion controls. The retained interval is `[0.1, 0.65]`. Its logistic models are not components of either final package.
- `experiment_ae_fpr_sweep.py`: reused the frozen detectors and settings above, recalibrated AE tail cutoffs to 5% and 10%, and evaluated the selected rules. No base model was retrained for that sweep.
- `export_selected_fusion_models.py`: exports fitted preprocessing and weights; zero prediction mismatches on all 82,332 official-test rows for each package.

## Recorded environment

| Library | Version |
|---|---|
| python | 3.12.3 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| scikit_learn | 1.9.1 |
| lightgbm | 4.7.0 |
| joblib | 1.6.0 |

## Reproduction and safe output paths

To start a **new** validated binary search using the same split:

```sh
.venv/bin/python training/train_validated_lightgbm.py \
  --task binary --split-indices experiments/lightgbm_validated_v2/split_indices.npz \
  --trials 16 --max-rounds 2500 --patience 100 --rf-trees 300 \
  --n-jobs 4 --seed 42 --minimum-recall 0.95 --recall-confidence 0.95 \
  --experiment-dir experiments/lightgbm_validated_new
```

To rerun the base-model experiment from the existing selected binary configuration:

```sh
.venv/bin/python training/experiment_four_mode_ae.py \
  --output-dir experiments/four_mode_ae_new
```

These commands require the local raw CSVs and saved split/config artifacts. A newly named base experiment does not automatically update downstream input paths: the later scripts intentionally point to the original frozen source directories. Route new inputs/outputs explicitly before attempting a full replacement pipeline. `train_autoencoder.py` is retained for its actual fitting implementation; running its standalone CLI uses its separate default split and is not the exact final grouped experiment.

Existing final packages are already exported. The exporter refuses to overwrite their directories. Load them using `nexus.fusion_model.load_fusion_model`; see [working_v2.md](../working_v2.md). The cleanup did not retrain, recalibrate or change any existing package.

## Cleanup and limitations

See [README.md](README.md) for each retained script’s dependency role and [archive manifest](../experiments/archived_training_20261007/manifest.json) for removed active scripts. Historical manifests can reference archived/already-missing source files; source-fingerprint verification of those old experiments still requires their original source snapshot. Their hashes were not rewritten.

The official test has known predictor overlap with training and repeated development use. Selection of these two configurations used that development history; it is not a fresh-holdout performance claim.

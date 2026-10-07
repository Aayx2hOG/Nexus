# Final-model training pipeline

The active directory contains the scripts and dependencies used to create and verify the final **LightGBM + baseline AE** packages. Exact settings are in [configs.md](configs.md); model usage and results are in [working_v2.md](../working_v2.md).

## Pipeline

1. `train_validated_lightgbm.py` selects the binary LightGBM configuration.
2. `experiment_four_mode_ae.py` retrains that binary configuration and a normal-only baseline AE using `train_autoencoder.py`.
3. `experiment_confidence_ae.py` and `experiment_uncertainty_residuals.py` produce confidence cutoffs, the uncertainty interval and cached features.
4. `experiment_ae_fpr_sweep.py` calibrates AE 5%/10% FPR cutoffs and evaluates the rules.
5. `export_selected_fusion_models.py` packages the selected configurations and verifies every official-test prediction.

The scripts preserve their historical artifact paths. Existing output directories are not overwritten. Consult each script’s `--help` where supported and the provenance notes in configs.md before starting a new run; do not run the exporter again over the existing selected packages.

## Retained Python files

| File | Why it remains |
|---|---|
| [anomaly_detection_models.py](anomaly_detection_models.py) | Conservative FPR cutoff helper and imported seed42 classes used by the final sweep. |
| [compare_lightgbm.py](compare_lightgbm.py) | Shared LightGBM training/refit helpers imported by validated training. |
| [compare_working_v2_fusion.py](compare_working_v2_fusion.py) | Provides metrics used throughout the retained final-fusion scripts. |
| [evaluate_frozen_lightgbm.py](evaluate_frozen_lightgbm.py) | Validates frozen LightGBM artifacts; covered by the active validated-training tests. |
| [evaluate_imported_seed42_test.py](evaluate_imported_seed42_test.py) | Provides imported_scores used by the sweep to reproduce the seed42 comparison. |
| [experiment_ae_fpr_sweep.py](experiment_ae_fpr_sweep.py) | Calibrated the final 5%/10% AE boundaries and evaluated both selected configurations. |
| [experiment_confidence_ae.py](experiment_confidence_ae.py) | Produced the frozen confidence cutoffs used by the final mode-confidence rule. |
| [experiment_four_mode_ae.py](experiment_four_mode_ae.py) | Actually retrained the LightGBM + baseline AE checkpoints used in both final packages. |
| [experiment_uncertainty_residuals.py](experiment_uncertainty_residuals.py) | Produced the selected confidence band and cached features used by the final sweep. |
| [export_release_bundle.py](export_release_bundle.py) | Existing LightGBM serving-bundle exporter, also used by active API/inference tests. |
| [export_selected_fusion_models.py](export_selected_fusion_models.py) | Exports and verifies the two final self-contained model packages. |
| [fast_lightgbm_models.py](fast_lightgbm_models.py) | Fitted preprocessing/model wrappers required by training and source-model deserialization. |
| [multiclass_training_utils.py](multiclass_training_utils.py) | Imported by shared tuning helpers; no standalone multiclass trainer is retained. |
| [train_autoencoder.py](train_autoencoder.py) | Baseline normal-only AE fitting, scaling and reconstruction-error implementation. |
| [train_isolation_forest.py](train_isolation_forest.py) | AE imports its operating-point calibration helper; not a selected final detector. |
| [train_validated_lightgbm.py](train_validated_lightgbm.py) | Original binary LightGBM hyperparameter selection, grouped splits and refit. |
| [training_utils.py](training_utils.py) | Shared data loading and binary evaluation utilities. |
| [tune_lightgbm.py](tune_lightgbm.py) | Shared weights, thresholds, metrics and JSON helpers. |
| [tune_lightgbm_fast.py](tune_lightgbm_fast.py) | Grouped splitting, recall calibration and candidate evaluation helpers. |

## Removed from active training

26 unrelated research scripts and eight research-only test modules were moved to [the historical archive](../experiments/archived_training_20261007/README.md). The archive includes original paths and byte hashes, preserving existing local/staged edits. Baseline-AE and anomaly helper tests remain active; research-only cases from mixed test modules are preserved in their original archived copies.

Model weights, configs, datasets, saved experiment results and application serving were not changed. `FUSION_EXPERIMENTS.md` is historical documentation. The retained dependency modules sometimes contain older optional workflows; their imported helpers are still needed, so their source is preserved.

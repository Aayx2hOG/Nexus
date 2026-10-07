# Final model artifacts and training configuration

The final trained models are stored under `models/`. This document is the configuration reference in `artifacts/`; it does not refer to the older serving bundles or SQLite database in this directory.

## Trained models

| Selected model | Trained weights | Exact configuration |
|---|---|---|
| Uncertainty-band four-mode, AE 10% | [weights.joblib](../models/uncertainty_band_ae10/weights.joblib) | [config.json](../models/uncertainty_band_ae10/config.json) |
| Mode confidence, AE 5% | [weights.joblib](../models/mode_confidence_ae05/weights.joblib) | [config.json](../models/mode_confidence_ae05/config.json) |

Each weights file contains the fitted LightGBM, baseline autoencoder and all fitted preprocessing. Both packages use identical base detectors, with different final decision rules. They do not require the experiment folders for inference.

## Training pipeline

```text
train_validated_lightgbm.py  → selected LightGBM parameters
experiment_four_mode_ae.py  → trained LightGBM + baseline AE
  └─ train_autoencoder.py   → normal-only AE training
experiment_confidence_ae.py → confidence cutoffs
experiment_uncertainty_residuals.py → uncertainty interval
experiment_ae_fpr_sweep.py  → final 5% / 10% AE thresholds
export_selected_fusion_models.py → models/<selected_model>/
```

Full training protocol, preprocessing, dataset partitions and reproduction commands: [training/configs.md](../training/configs.md). Retained scripts: [training/README.md](../training/README.md).

## LightGBM training settings

Seed 42; selected trial 4 of 16. Initial search: maximum 2,500 rounds, early-stop patience 100. Final refit: 124,861 training rows, 1,318 boosting rounds. The binary score cutoff is **0.5776925765603604**.

```json
{
  "boosting_type": "gbdt",
  "class_weight": {
    "0": 1.5684084913955534,
    "1": 0.7339928987960873
  },
  "colsample_bytree": 0.85,
  "importance_type": "split",
  "learning_rate": 0.03,
  "max_depth": -1,
  "min_child_samples": 100,
  "min_child_weight": 0.001,
  "min_split_gain": 0.0,
  "n_estimators": 1318,
  "n_jobs": 4,
  "num_leaves": 63,
  "objective": "binary",
  "random_state": 42,
  "reg_alpha": 0.0,
  "reg_lambda": 1.0,
  "subsample": 0.85,
  "subsample_for_bin": 200000,
  "subsample_freq": 1,
  "verbosity": -1,
  "deterministic": true,
  "force_col_wise": true,
  "metric": "None"
}
```

## Baseline AE training settings

| Setting | Value |
|---|---|
| Training data | 31,861 normal fit rows |
| Hidden layers | 128 → 32 → 128 |
| Activation / optimizer | ReLU / Adam |
| Learning rate / L2 alpha | 0.001 / 0.0001 |
| Epoch limit / patience | 20 / 4 |
| Best restored epoch | 20 |
| Seed | 42 |
| Training batch size | 512; final short batch 117 |
| Inference batch size | 2,048 |
| Preprocessing | Normal-fit numeric mean imputation + standardization; one-hot categories |
| Error | Mean squared reconstruction error over transformed inputs |

Custom `partial_fit` training performs the epochs and early stopping. The saved estimator’s `max_iter=1` and last `batch_size=117` do not mean it trained for one epoch or used 117 as its configured batch size.

## Frozen decision configurations

### Uncertainty-band four-mode — AE target 10%

```json
{
  "binary_cutoff": 0.5776925765603604,
  "ae_calibration_fpr_target": 0.1,
  "ae_boundaries": [
    0.004530309699475766,
    0.008725384250283243,
    0.027579598128795627
  ],
  "mode_order": [
    "no_anomaly",
    "low",
    "mid",
    "high"
  ],
  "decision_rule": {
    "kind": "uncertainty_band_four_mode",
    "confidence_band": [
      0.1,
      0.65
    ]
  }
}
```

### Mode confidence — AE target 5%

```json
{
  "binary_cutoff": 0.5776925765603604,
  "ae_calibration_fpr_target": 0.05,
  "ae_boundaries": [
    0.008725384250283243,
    0.015471044927835466,
    0.04816404730081559
  ],
  "mode_order": [
    "no_anomaly",
    "low",
    "mid",
    "high"
  ],
  "decision_rule": {
    "kind": "mode_confidence",
    "mode_attack_cutoffs": [
      0.65,
      0.5776925765603604,
      0.5776925765603604,
      0.5776925765603604
    ]
  }
}
```

Uncertainty-band: outside the inclusive 10%–65% LightGBM score interval, follow binary. Inside it: no anomaly → normal; low → binary; mid/high → attack.

Mode-confidence: no anomaly requires LightGBM score ≥65%; all other modes require ≥57.76925765603604%. Equality enters the higher anomaly mode.

AE 5%/10% refers to normal calibration traffic, not a guarantee on test or combined-system FPR.

## Verified test results

Both packages reproduced all 82,332 official-test predictions exactly.

| Model | Recall | F1 | System FPR | Missed attacks | False alarms |
|---|---:|---:|---:|---:|---:|
| Uncertainty-band four-mode | 97.2029% | 92.3435% | 16.3216% | 1,268 | 6,039 |
| Mode confidence | 95.9543% | 92.3789% | 14.4405% | 1,834 | 5,343 |

## Load a trained model

```python
from nexus.fusion_model import load_fusion_model

model = load_fusion_model("models/uncertainty_band_ae10")
predictions = model.predict(flows)  # DataFrame containing the 42 raw flow features
```

See [working_v2.md](../working_v2.md) for both decision tables, confusion matrices and complete usage. The packages are selected experimental models; API serving was not automatically switched. The official test has known predictor overlap and repeated development use.

The JSON config beside each weights file is the authoritative machine-readable configuration and includes dependency versions, source hashes and AE training history.

# Working v2 — selected LightGBM + AE configurations

Updated 7 October 2026 (IST). This replaces the previous historical OR-fusion working record. The selected models are **Uncertainty-band four-mode at AE 10%** and **Mode confidence at AE 5%**. Both are exported with all fitted preprocessing under `models/`. They share the same frozen base detectors but use different decision rules. Exporting them does not switch the API’s serving configuration.

## Selected packages

| Model | Package | Exact config | Role |
|---|---|---|---|
| Uncertainty-band four-mode, AE 10% | [models/uncertainty_band_ae10](models/uncertainty_band_ae10/) | [config.json](models/uncertainty_band_ae10/config.json) | Higher-recall option |
| Mode confidence, AE 5% | [models/mode_confidence_ae05](models/mode_confidence_ae05/) | [config.json](models/mode_confidence_ae05/config.json) | Fewer-false-alarm option |

Each directory contains `weights.joblib`, `config.json`, `metrics.json`, `manifest.json`, `verification.json` and `README.md`. Weights contain fitted LightGBM, AE, and each model’s preprocessors. Neither package needs the experiment directories at inference time. JSON configs/manifests and README files are trackable; binary `.joblib` weights remain local artifacts under the repository’s existing ignore policy.

## Exact decision rules

Let **p** be the raw LightGBM attack score and **e** the AE mean squared reconstruction error. Base binary attack cutoff: **0.5776925765603604**. Output labels: **0 = normal, 1 = attack**. Boundary equality enters the higher AE mode.

| Mode | Uncertainty-band AE 10%: error range | Mode-confidence AE 5%: error range |
|---|---|---|
| No anomaly | e < 0.004530309699475766 | e < 0.008725384250283243 |
| Low | 0.004530309699475766 ≤ e < 0.008725384250283243 | 0.008725384250283243 ≤ e < 0.015471044927835466 |
| Mid | 0.008725384250283243 ≤ e < 0.027579598128795627 | 0.015471044927835466 ≤ e < 0.04816404730081559 |
| High | e ≥ 0.027579598128795627 | e ≥ 0.04816404730081559 |

### Uncertainty-band four-mode — AE 10%

- If **p < 0.10 or p > 0.65**, preserve the base binary prediction.
- Within **0.10 ≤ p ≤ 0.65**: no anomaly → normal; low → base binary; mid/high → attack.
- The uncertainty interval includes both endpoints. The high boundary affects the severity label; mid and high both predict attack within the interval.

### Mode confidence — AE 5%

- No anomaly: attack iff **p ≥ 0.65**.
- Low, mid or high: attack iff **p ≥ 0.5776925765603604**.
- Otherwise normal. This AE-5% no-anomaly boundary is **0.008725384250283243**, not the median boundary used in the earlier confidence experiment.

The 5% and 10% values describe **AE calibration FPR on normal calibration traffic**, not overall system FPR. Actual recent-AE test FPR is 10.31% at the 5% setting and 16.45% at the 10% setting.

## Verified official-test results

Both exports were independently loaded and scored from the same 82,332 raw test rows (37,000 normal; 45,332 attacks). **Every prediction matches the selected experiment**, with zero mismatches per package. AE reconstruction scores also match.

| Model | Accuracy | Precision | Attack recall | F1 | System FPR | Missed attacks | False alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uncertainty-band four-mode, AE 10% | 91.1250% | 87.9468% | 97.2029% | 92.3435% | 16.3216% | 1,268 | 6,039 |
| Mode confidence, AE 5% | 91.2829% | 89.0604% | 95.9543% | 92.3789% | 14.4405% | 1,834 | 5,343 |

Uncertainty-band detects **566 more attacks** than mode-confidence on this test, with **696 more false alarms**. Both are retained as requested; the API has not automatically selected one.

Rows below are actual; columns predicted.

### Uncertainty-band four-mode — AE 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,961 | 6,039 |
| Attack | 1,268 | 44,064 |

### Mode confidence — AE 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 31,657 | 5,343 |
| Attack | 1,834 | 43,498 |

## Loading and prediction

Run with this project installed, or set `PYTHONPATH=src`. Use the recorded dependency versions in each config. The loader is [src/nexus/fusion_model.py](src/nexus/fusion_model.py).

```python
import pandas as pd
from nexus.fusion_model import load_fusion_model

flows = pd.read_csv("data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv")
recall_model = load_fusion_model("models/uncertainty_band_ae10")
fewer_alerts_model = load_fusion_model("models/mode_confidence_ae05")

labels = recall_model.predict(flows)  # 0 normal, 1 attack
details = fewer_alerts_model.predict_details(flows)
print(details.head())
```

Input: a pandas DataFrame containing all 42 raw features listed in `config.json`. Extra `id`, `label`, and `attack_cat` columns are ignored; missing required features are rejected. Do not pre-scale or one-hot encode the input. Fitted preprocessing handles category vocabularies and unknown categories.

`predict_details()` returns `prediction`, `lightgbm_attack_score`, `binary_prediction`, `ae_error`, and `ae_mode`, retaining the input index. The LightGBM score is not a calibrated final fusion probability, so the wrapper does not expose a misleading `predict_proba()` for the rule-based decision.

## Base model configuration and provenance

Both packages use seed 42. LightGBM is the freshly retrained v2 trial-4 configuration: 1,318 trees, learning rate 0.03, 63 leaves, min_child_samples 100, balanced class weights and 0.85 row/feature subsampling. The normal-only AE uses hidden layers 128–32–128, ReLU/Adam, learning rate 0.001, up to 20 epochs, early-stop patience 4, training batch size 512, and best epoch 20. Full fitted estimator parameters, training history, dependency versions, source hashes, feature columns and exact thresholds are recorded in each package’s config.

Source models: `experiments/four_mode_ae_20261007/{binary,autoencoder}.joblib`. Selected rules and results: [AE FPR sweep report](experiments/ae_fpr_5_10_20261007/REPORT.md). Export command: `.venv/bin/python training/export_selected_fusion_models.py`. The exporter refuses to overwrite existing package directories.

Manifest SHA-256 checks cover weights, config and metrics before loading. The models are exported without training-module pickle dependencies; use the `nexus` loader and the matching scientific Python libraries.

## Evaluation limits and previous record

These configurations were selected after inspecting repeated development experiments. The official test has known training-feature overlap and is not a fresh holdout; the reported numbers are historical benchmark results, not a production guarantee.

Previous working record: [archived working_v2](reports/working_v2_before_selected_fusion_20261007.md). The historical OR evaluator is now in `experiments/archived_training_20261007/training/`; its preserved code writes a separate legacy report. Active training scripts and configurations are documented in [training/configs.md](training/configs.md).

# Working v2: retrained baseline AE and fusion comparison

Run date: 5 October 2026 (IST).

LightGBM alone wins the prespecified objective: minimum selection-set normal FPR with attack recall >=95%. The best eligible fusion is selective recovery. It gives a small recall gain with additional false alerts; it is not an across-the-board improvement.

## Historical confusion matrices checked before training

Source: `working_v2.md` and `reports/official_testing_lightgbm_v2_ae_evaluation.json`. Rows are actual normal/attack; columns predicted normal/attack.

- Frozen LightGBM: `[[30665, 6335], [1570, 43762]]`.
- Historical baseline AE: `[[31336, 5664], [7898, 37434]]`.
- Historical OR: `[[26965, 10035], [935, 44397]]`.

The current run exactly reproduces the historical LightGBM confusion matrix. The historical AE checkpoint/config is missing, so its matrix is checked from the saved record, not reproduced by inference.

## Protocol

Frozen LightGBM: `experiments/lightgbm_validated_v2/binary/model.joblib`, threshold 0.5776925765603604. All manifest artifact hashes and the original training CSV hash passed verification. Existing source files have changed; this is artifact verification, not a claim that the entire historical source tree is unchanged.

AE: baseline MLP architecture 128–32–128, ReLU/Adam, learning rate 0.001, batch 512, maximum 20 epochs, patience 4, seed 42. Trained on normal rows from the frozen grouped fit partition, with normal early-partition rows for early stopping. Imputation, encoding, and scaling are fitted only on normal fit rows. These split/preprocessing choices prevent calibration and selection rows from entering AE fitting; this is a controlled baseline retraining, not an exact reconstruction of the unavailable historical AE.

Best epoch: 20; normal fit rows: 31861; input features: 63; calibrated baseline AE threshold: 0.00453154556453228. AE threshold maximizes calibration recall subject to normal FPR <=10%.

Compared 34 candidates: LightGBM, AE, OR, AND, five weighted blends of LightGBM score and AE normal-fit empirical percentile, and 25 selective recovery rules. Weighted thresholds are calibrated for recall >=95%; selective AE thresholds use calibration-normal quantiles. All candidates are ranked on the separate selection partition. Selected rules were saved before loading the official test.

Best fusion rule:

```text
attack = (LightGBM score >= 0.5776925765603604)
         OR (LightGBM score >= 0.4 AND AE reconstruction MSE >= 0.027578630857169606)
```

Selection comparison: LightGBM FP=326, FN=799; selected fusion FP=327, FN=795. LightGBM therefore retains the lower FPR. Fusion recovers four selection attacks for one extra false positive.

## Official test results

All 82,332 official testing rows. Thresholds and selected fusion were frozen before scoring.

| Method | TN | FP | FN | TP | Recall | FPR | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 30665 | 6335 | 1570 | 43762 | 96.5367% | 17.1216% | 91.7164% |
| Baseline AE | 30916 | 6084 | 7259 | 38073 | 83.9870% | 16.4432% | 85.0898% |
| OR | 26631 | 10369 | 763 | 44569 | 98.3169% | 28.0243% | 88.8980% |
| AND | 34950 | 2050 | 8066 | 37266 | 82.2068% | 5.5405% | 88.0493% |
| Selective gate=0.4 q=0.99 | 30597 | 6403 | 1510 | 43822 | 96.6690% | 17.3054% | 91.7191% |

Selected fusion recovers 60 LightGBM misses with 68 additional false positives and no lost LightGBM detections. F1 rises by only about 0.0028 percentage points, while accuracy decreases slightly. Naive OR recovers 807 misses but adds 4,034 false positives. AND reduces false positives but fails the 95% recall requirement.

Recommendation: retain LightGBM for the recorded minimum-FPR objective. If the priority is combining both models while preserving LightGBM detections, selective recovery is the best eligible fusion among the tested candidates. Its small advantage needs confirmation on fresh data before adoption.

## Limitations and artifacts

This is one seed and a limited fusion grid, not a claim of a globally optimal combination. The selection partition previously helped choose LightGBM. The official test has repeatedly informed development and contains 8,541 predictor rows also found in training; these are historical benchmark results, not independent validation. The empirical 95% recall constraint is not a confidence bound. Working v2, frozen checkpoints, and application serving were not changed.

- `selection.csv`: every candidate on the selection split.
- `frozen_selection.json`: exact rules and winners, written before test loading.
- `official_test.csv`: standalone models, logical baselines, and selected fusion on test.
- `autoencoder.joblib` and `autoencoder_config.json`: retrained AE and matching preprocessing.
- `scores.npz`: paired selection/test scores and labels.
- `audit.json` and `artifact_hashes.json`: checks, limitations, and fingerprints.

Reproduce with `.venv/bin/python training/compare_working_v2_fusion.py`. The script refuses to overwrite its output directory; use a new OUT directory for another run.

Validation: four existing autoencoder tests passed. Runtime assertions verify frozen artifact hashes, training data identity, grouped split separation, finite test scores, and exact reproduction of the LightGBM test matrix.

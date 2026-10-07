# Four-mode AE fusion experiment — 7 October 2026

Both LightGBM and the autoencoder were freshly trained. The binary configuration comes from the previously selected v2 model; AE uses normal-only fitting, architecture 128–32–128, seed 42, up to 20 epochs and patience 4.

## Decision rule

| AE mode | Binary normal | Binary attack |
|---|---|---|
| No anomaly | Normal | Normal |
| Low | Normal | Attack |
| Mid | Attack | Attack |
| High | Attack | Attack |

Low + binary normal is interpreted as normal. Mid and high have identical binary decisions; their boundary only changes the severity label. These are score bands, not ground-truth anomaly severity classes.

## Training and threshold selection

Grouped train partitions: {'fit': 99778, 'early': 25083, 'calibration': 25449, 'selection': 25031}. AE fit uses 31,861 normal rows; best epoch 20. AE preprocessing is fitted only on those normal rows. LightGBM refits on fit + early using the previously selected 1,318 rounds. Binary cutoff 0.57769257656 is calibrated for 95% recall with the existing 95% confidence procedure.

AE boundaries are quantiles of normal calibration errors. Fixed bands use q90/q95/q99. The validation search varies the no/low and low/mid boundaries over the grids in protocol.json, with mid/high fixed at q99.99. It minimizes selection FPR subject to recall ≥95%, falling back to maximum recall if no candidate qualifies. All thresholds and models are saved before loading test data.

Selection recall constraint met: True. Candidates: 65.

| Variant | Quantiles | Error boundaries | Selection recall | Selection FPR |
|---|---|---|---:|---:|
| Fixed four-mode | [0.9, 0.95, 0.99] | [0.004529757192358377, 0.008725384250283241, 0.027578630857169606] | 80.5856% | 8.1167% |
| Selected four-mode | [0.01, 0.999, 0.9999] | [0.00016033523716032504, 0.16687584674358308, 0.274032324552536] | 95.2548% | 3.9790% |

For error e and boundaries t0 < t1 < t2: none e<t0; low t0≤e<t1; mid t1≤e<t2; high e≥t2.

## Official test results

Test rows: 82,332; normal 37,000; attack 45,332. Attack is the positive class.

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Binary only | 90.3986% | 87.3545% | 96.5367% | 91.7164% | 17.1216% |
| Fixed four-mode | 85.5196% | 89.5478% | 83.4400% | 86.3861% | 11.9324% |
| Selected four-mode | 90.2553% | 87.1493% | 96.5367% | 91.6031% | 17.4405% |

### Binary only: confusion matrix

Rows = actual; columns = predicted.

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

### Fixed four-mode: confusion matrix

Rows = actual; columns = predicted.

| | Normal | Attack |
|---|---:|---:|
| Normal | 32,585 | 4,415 |
| Attack | 7,507 | 37,825 |

Compared with binary only: recovered 558 attacks; lost 6,495 previously detected attacks; removed 4,284 false alarms; added 2,364 false alarms.

### Selected four-mode: confusion matrix

Rows = actual; columns = predicted.

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,547 | 6,453 |
| Attack | 1,570 | 43,762 |

Compared with binary only: recovered 0 attacks; lost 0 previously detected attacks; removed 0 false alarms; added 118 false alarms.

## Test breakdown by mode

| Variant | Mode | Rows | Actual normal | Actual attack | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed four-mode | No anomaly | 38173 | 30915 | 7258 | 30915 | 0 | 7258 | 0 |
| Fixed four-mode | Low | 3967 | 2270 | 1697 | 1670 | 600 | 249 | 1448 |
| Fixed four-mode | Mid | 13345 | 2629 | 10716 | 0 | 2629 | 0 | 10716 |
| Fixed four-mode | High | 26847 | 1186 | 25661 | 0 | 1186 | 0 | 25661 |
| Selected four-mode | No anomaly | 184 | 184 | 0 | 184 | 0 | 0 | 0 |
| Selected four-mode | Low | 76644 | 36655 | 39989 | 30363 | 6292 | 1570 | 38419 |
| Selected four-mode | Mid | 3041 | 103 | 2938 | 0 | 103 | 0 | 2938 |
| Selected four-mode | High | 2463 | 58 | 2405 | 0 | 58 | 0 | 2405 |

## Limits and reproduction

This is a single-seed experiment on an official test set already used in prior project development, not a fresh holdout. 8,541 test rows have feature-identical matches in the training CSV. Internal partitions are group-disjoint. Previously selected binary hyperparameters also reflect earlier development. Test data did not select this experiment’s AE boundaries. Low reconstruction error does not guarantee normal traffic; the lost-attack counts measure the cost of that override.

Run from repository root:

```sh
.venv/bin/python training/experiment_four_mode_ae.py --output-dir experiments/four_mode_ae_new
```

Artifacts: protocol.json, frozen_selection.json, ae_training.json, audit.json, test_results.json, mode_breakdown.csv, test_scores.npz, binary.joblib, autoencoder.joblib.

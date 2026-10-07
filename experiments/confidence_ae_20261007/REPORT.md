# AE boundaries and LightGBM confidence — 7 October 2026

## Findings

The recall-constrained confidence rule achieved test F1 91.8935% versus 91.7164% for the current binary. It removed 231 false alarms and lost 40 previously detected attacks. This is a small observed improvement, not evidence of statistical significance or a deployment recommendation.

In this run the selected rule requires 65% below the normal calibration-error median and 57.7693% elsewhere. Low, mid and high therefore collapse to the same decision condition. The maximum-F1 mode search likewise selected the same 40% cutoff in all modes: AE bands did not help that objective. The binary-only F1 search had a finer cutoff grid and chose 39%.

The isotonic percentage calibration improved selection Brier score but worsened test Brier score. Do not treat either score as a verified live attack probability. attack_percentages.csv shows every test row’s raw percentage, calibrated estimate, AE mode, required cutoff and prediction for the recall-constrained confidence rule.

## Experiment

Reused the LightGBM and normal-only AE freshly trained in the preceding experiment so changes measure decision rules, not model randomness. The binary model supplies `predict_proba(X)[:, 1]`: a score from 0 to 1, displayed as 0–100%.

Three families were searched: binary-only cutoffs, original hard AE overrides, and a separate LightGBM cutoff for each AE mode. In the last family, predict attack when `score >= cutoff[mode]`, otherwise normal. Cutoffs cannot increase as AE anomaly strength increases. No anomaly can therefore preserve a confident attack; mid/high can require binary evidence instead of forcing every row to attack.

Candidate counts: {'Binary threshold': 104, 'Mode confidence': 250920, 'Original override': 82}. Full grids are in protocol.json. AE boundaries use normal calibration-error quantiles. Equality enters the higher AE mode. A cutoff above 100% means always normal; 0% means always attack.

Two operating objectives were declared before test scoring: maximum selection F1, and minimum selection false-positive rate (FPR) subject to attack recall ≥95%. Each family gets a separate winner for each objective. Ties prefer lower FPR then higher recall for F1; higher recall then F1 for recall-constrained selection. A missing recall95 row means no candidate met that constraint. Binary-only cutoff search provides a control for gains achievable without the AE.

## Frozen rules

| Rule | AE quantiles | No-anomaly cutoff | Low | Mid | High | Selection recall | Selection FPR | Selection F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Binary threshold/f1 | None | 39.0000% | 39.0000% | 39.0000% | 39.0000% | 97.4819% | 7.0060% | 97.0496% |
| Binary threshold/recall95 | None | 59.0000% | 59.0000% | 59.0000% | 59.0000% | 95.0172% | 3.8081% | 96.5278% |
| Mode confidence/f1 | [0.01, 0.8, 0.99] | 40.0000% | 40.0000% | 40.0000% | 40.0000% | 97.3869% | 6.8351% | 97.0411% |
| Mode confidence/recall95 | [0.5, 0.8, 0.99] | 65.0000% | 57.7693% | 57.7693% | 57.7693% | 95.0172% | 3.7349% | 96.5453% |
| Original override/f1 | [0.01, 0.99, 0.995] | Always normal | 57.7693% | 0.0000% | 0.0000% | 95.3260% | 6.6520% | 96.0160% |
| Original override/recall95 | [0.3, 0.99, 0.995] | Always normal | 57.7693% | 0.0000% | 0.0000% | 95.0707% | 6.6276% | 95.8879% |
| Prior selected four-mode | [0.01, 0.999, 0.9999] | Always normal | 57.7693% | 0.0000% | 0.0000% | 95.2548% | 3.9790% | 96.6118% |
| Current binary | None | 57.7693% | 57.7693% | 57.7693% | 57.7693% | 95.2548% | 3.9790% | 96.6118% |

Exact reconstruction-error boundaries and cutoffs are in frozen_selection.json.

## Official test results

| Rule | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Binary threshold/f1 | 87.5905% | 82.6051% | 98.1249% | 89.6986% | 25.3162% |
| Binary threshold/recall95 | 90.5662% | 87.6954% | 96.3911% | 91.8379% | 16.5703% |
| Mode confidence/f1 | 87.7593% | 82.8641% | 98.0433% | 89.8169% | 24.8405% |
| Mode confidence/recall95 | 90.6306% | 87.7494% | 96.4484% | 91.8935% | 16.4973% |
| Original override/f1 | 89.6116% | 86.0021% | 96.9051% | 91.1286% | 19.3243% |
| Original override/recall95 | 89.6298% | 86.1704% | 96.6823% | 91.1242% | 19.0108% |
| Prior selected four-mode | 90.2553% | 87.1493% | 96.5367% | 91.6031% | 17.4405% |
| Current binary | 90.3986% | 87.3545% | 96.5367% | 91.7164% | 17.1216% |

## Changes relative to current binary

| Rule | Recovered attacks | Lost attacks | Removed false alarms | Added false alarms |
|---|---:|---:|---:|---:|
| Binary threshold/f1 | 720 | 0 | 0 | 3032 |
| Binary threshold/recall95 | 0 | 66 | 204 | 0 |
| Mode confidence/f1 | 683 | 0 | 0 | 2856 |
| Mode confidence/recall95 | 0 | 40 | 231 | 0 |
| Original override/f1 | 167 | 0 | 0 | 815 |
| Original override/recall95 | 167 | 101 | 116 | 815 |
| Prior selected four-mode | 0 | 0 | 0 | 118 |
| Current binary | 0 | 0 | 0 | 0 |

## Confusion matrices

Rows are actual; columns predicted. Attack is positive.

### Binary threshold/f1

| | Normal | Attack |
|---|---:|---:|
| Normal | 27,633 | 9,367 |
| Attack | 850 | 44,482 |

### Binary threshold/recall95

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,869 | 6,131 |
| Attack | 1,636 | 43,696 |

### Mode confidence/f1

| | Normal | Attack |
|---|---:|---:|
| Normal | 27,809 | 9,191 |
| Attack | 887 | 44,445 |

### Mode confidence/recall95

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

### Original override/f1

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,850 | 7,150 |
| Attack | 1,403 | 43,929 |

### Original override/recall95

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,966 | 7,034 |
| Attack | 1,504 | 43,828 |

### Prior selected four-mode

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,547 | 6,453 |
| Attack | 1,570 | 43,762 |

### Current binary

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

## Does the percentage reflect attack likelihood?

An 80% raw LightGBM score is not automatically an 80% real-world attack probability. The model uses class weighting, and traffic distributions differ. An isotonic calibrator was fitted only on the calibration partition; it did not change the raw-score rules above. Lower Brier, log loss and ECE are better. ECE uses ten equal-width score bins and depends on binning.

| Partition / score | Brier | Log loss | ECE |
|---|---:|---:|---:|
| selection_raw | 0.028849 | 0.089939 | 1.1572% |
| selection_calibrated | 0.028052 | 0.088877 | 0.3172% |
| test_raw | 0.072472 | 0.219103 | 7.3970% |
| test_calibrated | 0.082133 | 0.247676 | 9.1044% |

### test_raw: predicted percentage versus observed attacks

| Score bin | Rows | Average score | Observed attack rate |
|---|---:|---:|---:|
| 0–10% | 22,184 | 0.29% | 0.38% |
| 10–20% | 2,070 | 15.27% | 8.45% |
| 20–30% | 2,302 | 25.04% | 11.60% |
| 30–40% | 2,140 | 34.95% | 16.87% |
| 40–50% | 1,951 | 44.91% | 18.04% |
| 50–60% | 2,033 | 54.94% | 21.45% |
| 60–70% | 2,086 | 64.90% | 27.66% |
| 70–80% | 2,278 | 75.07% | 35.03% |
| 80–90% | 2,796 | 85.25% | 46.60% |
| 90–100% | 42,492 | 99.40% | 96.44% |

### test_calibrated: predicted percentage versus observed attacks

| Score bin | Rows | Average score | Observed attack rate |
|---|---:|---:|---:|
| 0–10% | 21,530 | 0.03% | 0.12% |
| 10–20% | 1,142 | 15.84% | 8.23% |
| 20–30% | 2,237 | 27.16% | 9.25% |
| 30–40% | 933 | 34.37% | 11.79% |
| 40–50% | 1,802 | 40.82% | 14.48% |
| 50–60% | 3,525 | 54.84% | 18.64% |
| 60–70% | 2,352 | 68.40% | 23.00% |
| 70–80% | 2,741 | 72.55% | 30.90% |
| 80–90% | 3,010 | 87.35% | 43.92% |
| 90–100% | 43,060 | 99.35% | 95.84% |

## Test rows without feature-identical training matches

8,541 official test rows match training features. The following is a secondary diagnostic excluding them, with the same frozen rules; it was not used to choose rules.

| Rule | Accuracy | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Binary threshold/f1 | 86.8168% | 97.8125% | 88.4985% | 25.0253% |
| Binary threshold/recall95 | 89.8673% | 95.7714% | 90.7425% | 16.4912% |
| Mode confidence/f1 | 86.9998% | 97.7184% | 88.6302% | 24.5440% |
| Mode confidence/recall95 | 89.9392% | 95.8419% | 90.8082% | 16.4180% |
| Original override/f1 | 88.8604% | 96.3803% | 89.9727% | 19.2383% |
| Original override/recall95 | 88.8808% | 96.1164% | 89.9644% | 18.9118% |
| Prior selected four-mode | 89.5462% | 95.9439% | 90.4925% | 17.3441% |
| Current binary | 89.7061% | 95.9439% | 90.6243% | 17.0119% |

## Limitations and reproduction

The official test and selection partitions have been used in previous development. This is an exploratory follow-up, not a fresh independent holdout. The large rule search can overfit selection. All current choices were frozen before current test scoring; no test metric selected a rule. The base models use one seed. Calibration on this dataset does not guarantee accurate percentages on live traffic. No serving configuration was changed.

```sh
.venv/bin/python training/experiment_confidence_ae.py --output-dir experiments/confidence_ae_new
```

Saved: source-model hashes, protocol, search summaries, frozen rules, validation and test scores, calibrated probability model, mode counts, audits and metrics.

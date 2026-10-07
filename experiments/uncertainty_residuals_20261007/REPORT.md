# Uncertainty-band and per-feature AE fusion — 7 October 2026

## Findings

Previous confidence-rule test F1: 91.8935%; uncertainty-band F1: 91.8877%; per-feature fusion F1: 91.1819%. These are descriptive comparisons of frozen validation-selected rules, not test-driven threshold choices.

Per-feature fusion recovered 392 attacks and lost 42 compared with the current binary, while removing 63 false alarms and adding 1040. Inspect family recall below to locate the change; the overall recall gain does not by itself justify the false alarms.

All methods target at least 95% selection recall, but their achieved test recalls differ. These are not comparisons at identical test recall. The learned-model threshold grid is deliberately limited to three OOF recall targets. No extra threshold tuning was performed after viewing test results.

## Design

This experiment tests the two proposed next steps: restrict AE overrides to an uncertain LightGBM score band, and learn fusion from individual reconstruction errors. The previous binary and normal-only AE models are reused, isolating fusion changes. New logistic fusion models are trained in this run.

Uncertainty band: outside [lower, upper], preserve the existing binary prediction. Inside the band, low AE error below veto forces normal; error at or above promote forces attack; otherwise follow binary. Both endpoints are included. AE thresholds are normal calibration-error quantiles, not probabilities.

Learned models: score-only logistic is a control; total-error logistic adds log(1+MSE) and its interaction with the LightGBM logit; per-feature logistic adds log(1+squared residual) for each original numeric feature and log(1+summed squared residuals) for each categorical one-hot block. Categories remain grouped by original input feature. All inputs are standardized within each meta-training fold.

The calibration partition is excluded from both base detectors’ fitting. Five stratified, group-disjoint folds generate out-of-fold meta-model scores. Those scores calibrate thresholds at 95%, 96% or 97% recall; final meta-models refit on calibration. A separate selection partition chooses regularization and threshold target. The OOF fold table is diagnostic: its thresholds use pooled OOF labels, so it is not nested-CV performance of the whole selection pipeline.

Every family selects minimum selection FPR subject to recall ≥95%, ties higher recall then F1. If infeasible, highest-recall fallback is explicitly flagged. Models, thresholds and choices are saved before test scoring. This is a fixed-base grouped-CV fusion experiment, not five independent retrainings of both base models.

Total candidates: 679. Full grids: protocol.json. All candidates: all_candidates.json.

## Selection results

| Method | Recall ≥95%? | Selection recall | Selection FPR | Selection F1 |
|---|---|---:|---:|---:|
| Binary threshold | True | 95.0172% | 3.8081% | 96.5278% |
| Uncertainty band | True | 95.0172% | 3.7349% | 96.5453% |
| Score-only logistic | True | 95.8843% | 4.6137% | 96.7897% |
| Total-error logistic | True | 95.0113% | 3.7593% | 96.5363% |
| Per-feature logistic | True | 95.8606% | 4.9554% | 96.6961% |
| Previous confidence rule | True | 95.0172% | 3.7349% | 96.5453% |
| Current binary | True | 95.2548% | 3.9790% | 96.6118% |

Selected uncertainty band: **10%–65%**. Within it, veto below normal-error q50 (0.000895940058); promote at/above q99.9 (0.166875847); otherwise use binary cutoff 57.7693%.

| Learned method | C | OOF recall target | Final score cutoff |
|---|---:|---:|---:|
| Score-only logistic | 0.01 | 96% | 0.645747551 |
| Total-error logistic | 10.0 | 95% | 0.671211021 |
| Per-feature logistic | 10.0 | 96% | 0.613602214 |

## Official test results

| Method | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Binary threshold | 90.5662% | 87.6954% | 96.3911% | 91.8379% | 16.5703% |
| Uncertainty band | 90.6233% | 87.7388% | 96.4484% | 91.8877% | 16.5135% |
| Score-only logistic | 89.7452% | 86.1820% | 96.9139% | 91.2334% | 19.0378% |
| Total-error logistic | 90.6318% | 87.7921% | 96.3889% | 91.8898% | 16.4216% |
| Per-feature logistic | 89.6371% | 85.7810% | 97.3087% | 91.1819% | 19.7622% |
| Previous confidence rule | 90.6306% | 87.7494% | 96.4484% | 91.8935% | 16.4973% |
| Current binary | 90.3986% | 87.3545% | 96.5367% | 91.7164% | 17.1216% |

## Errors changed versus current binary

| Method | Recovered attacks | Lost attacks | Removed false alarms | Added false alarms |
|---|---:|---:|---:|---:|
| Binary threshold | 0 | 66 | 204 | 0 |
| Uncertainty band | 0 | 40 | 231 | 6 |
| Score-only logistic | 171 | 0 | 0 | 709 |
| Total-error logistic | 9 | 76 | 262 | 3 |
| Per-feature logistic | 392 | 42 | 63 | 1040 |
| Previous confidence rule | 0 | 40 | 231 | 0 |
| Current binary | 0 | 0 | 0 | 0 |

## Confusion matrices

Actual classes are rows; predicted classes are columns.

### Binary threshold

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,869 | 6,131 |
| Attack | 1,636 | 43,696 |

### Uncertainty band

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,890 | 6,110 |
| Attack | 1,610 | 43,722 |

### Score-only logistic

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,956 | 7,044 |
| Attack | 1,399 | 43,933 |

### Total-error logistic

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,924 | 6,076 |
| Attack | 1,637 | 43,695 |

### Per-feature logistic

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,688 | 7,312 |
| Attack | 1,220 | 44,112 |

### Previous confidence rule

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

### Current binary

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

## Recall by attack family

| Family | Rows | Binary threshold | Uncertainty band | Score-only logistic | Total-error logistic | Per-feature logistic | Previous confidence rule | Current binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Analysis | 677 | 95.86% | 96.01% | 96.31% | 96.01% | 96.01% | 96.01% | 96.01% |
| Backdoor | 583 | 99.66% | 99.66% | 99.83% | 99.66% | 99.66% | 99.66% | 99.66% |
| DoS | 4089 | 99.68% | 99.66% | 99.78% | 99.68% | 99.66% | 99.66% | 99.68% |
| Exploits | 11132 | 99.04% | 98.99% | 99.19% | 99.05% | 99.09% | 98.99% | 99.07% |
| Fuzzers | 6062 | 75.87% | 76.36% | 79.28% | 75.82% | 82.61% | 76.36% | 76.87% |
| Generic | 18871 | 99.97% | 99.97% | 99.98% | 99.97% | 99.97% | 99.97% | 99.97% |
| Reconnaissance | 3496 | 99.71% | 99.71% | 99.77% | 99.71% | 99.74% | 99.71% | 99.71% |
| Shellcode | 378 | 97.88% | 98.15% | 98.15% | 97.88% | 98.15% | 98.15% | 98.15% |
| Worms | 44 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Excluding feature-identical training matches

8,541 test rows match training features. This secondary diagnostic uses the same frozen predictions and does not select models.

| Method | Accuracy | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| Binary threshold | 89.8673% | 95.7714% | 90.7425% | 16.4912% |
| Uncertainty band | 89.9310% | 95.8419% | 90.8015% | 16.4349% |
| Score-only logistic | 89.0461% | 96.3882% | 90.1240% | 18.8612% |
| Total-error logistic | 89.9392% | 95.7688% | 90.8019% | 16.3392% |
| Per-feature logistic | 88.9580% | 96.8586% | 90.0960% | 19.5508% |
| Previous confidence rule | 89.9392% | 95.8419% | 90.8082% | 16.4180% |
| Current binary | 89.7061% | 95.9439% | 90.6243% | 17.0119% |

## Limits and artifacts

Selection and official test have been used in prior development; this remains exploratory, not an untouched holdout. Prior binary hyperparameters were selected on this selection set. Grouped CV protects meta-training from duplicate leakage but cannot undo that development history. No test score selected a current candidate, and no serving model was changed. Small attack families have unstable recall estimates. Scores are not guaranteed live attack probabilities.

Saved: exact rules, meta-models, all validation candidates, fold assignments, OOF scores, validation residuals, test predictions, attack-family recall, source hashes and standardized residual coefficients. Coefficients describe fitted associations, not causal effects or reliable feature importance under correlated inputs.

```sh
.venv/bin/python training/experiment_uncertainty_residuals.py --output-dir experiments/uncertainty_residuals_new
```

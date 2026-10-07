# Fusion experiments with AE FPR targets of 5% and 10%

## Observed tradeoffs

At the 5% AE calibration target, uncertainty-band fusion has **96.6006% attack recall**, **15.1919% system FPR**, and **92.4407% F1**. Relative to binary only it detects +29 net attacks and changes false alarms by -714. OR fusion reaches 97.7676% recall at 23.5108% system FPR.

At the 10% AE calibration target, uncertainty-band fusion has **97.2029% attack recall**, **16.3216% system FPR**, and **92.3435% F1**. Relative to binary only it detects +302 net attacks and changes false alarms by -296. OR fusion reaches 98.3169% recall at 28.0243% system FPR.

These are observations of all predeclared test results, not a new test-selected deployment decision. The uncertainty-band configurations improve recall and false-alarm counts over binary-only on this test, while OR prioritizes higher recall at a much larger false-alarm cost. Confirmation on new data is still needed.

## Scope and interpretation

The 5% and 10% settings apply to the **AE alone on normal calibration traffic**. They are not constraints on the full system’s FPR, and do not guarantee AE test FPR. Every policy below uses the same official test: 82,332 rows, 37,000 normal and 45,332 attack. This isolates AE operating thresholds; model weights, LightGBM cutoffs and learned-fusion cutoffs remain frozen. No winner is selected on test.

The sweep covers the previously compared policy families: AE alone, OR, AND, four-mode hard overrides, mode-specific confidence, uncertainty-band overrides, selective recovery, score-only/total-error/per-feature logistic fusion, and the imported seed-42 plain-AE + latent fusion. Previous individual threshold-search trials are not all retrained or reselected.

## How the budgets apply

- Threshold T(b) is calibrated with a conservative empirical cutoff: no more than floor(b × normal calibration count) normal rows can meet error ≥ T(b).
- Four AE modes use tail budgets b, b/2 and b/10. At b=5%, these are 5%, 2.5%, 0.5%; at b=10%, they are 10%, 5%, 1%. No anomaly is below T(b).
- Hard four-mode rule: no anomaly → normal; low → binary; mid/high → attack.
- Confidence rule: no anomaly needs 65% LightGBM score; all other modes need 57.7693%. Unlike the prior median boundary, no-anomaly now ends at T(b).
- Uncertainty-band rule applies the hard four-mode rule only within the previously selected 10%–65% LightGBM score interval, preserving binary outside.
- Selective recovery preserves binary positives and adds attacks when score ≥40% and AE error ≥T(b). OR/AND use the same AE anomaly decision.
- Continuous learned fusion has no intrinsic AE FPR parameter. Its original output is retained as a budget-independent reference. A separately labeled AE-gated variant follows learned fusion only when AE error ≥T(b), otherwise binary.
- Imported seed42 uses its own normal calibration partition and plain-AE error threshold. Its binary and fusion cutoffs remain at the saved original 5% system calibration-budget setting for both AE sweep settings. Its latent input still comes from the denoising AE; that component is not recalibrated.

## AE calibration achieved

| AE | Target | Threshold | Normal rows | False alarms | Calibration FPR |
|---|---:|---:|---:|---:|---:|
| Recent AE | 5% | 0.00872538425 | 8002 | 400 | 4.9988% |
| Imported plain AE | 5% | 0.02851921879 | 5618 | 280 | 4.9840% |
| Recent AE | 10% | 0.004530309699 | 8002 | 800 | 9.9975% |
| Imported plain AE | 10% | 0.01599344797 | 5618 | 561 | 9.9858% |

## AE alone on official test

| AE | Calibration target | Actual test FPR | Attack recall |
|---|---:|---:|---:|
| Recent AE | 5% | 10.3108% | 80.2457% |
| Imported plain AE | 5% | 9.1919% | 78.1391% |
| Recent AE | 10% | 16.4459% | 83.9892% |
| Imported plain AE | 10% | 16.3189% | 82.7826% |

## All system results — AE target 5%

Attack recall and missed attacks are shown first because missing attacks is the priority. Actual system FPR remains necessary to assess false-alarm workload.

| Method | Recall | Missed attacks | System FPR | False alarms | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE alone | 80.2457% | 8,955 | 10.3108% | 3,815 | 90.5081% | 85.0685% | 84.4896% |
| OR | 97.7676% | 1,012 | 23.5108% | 8,699 | 83.5927% | 90.1262% | 88.2051% |
| AND | 79.0148% | 9,513 | 3.9216% | 1,451 | 96.1068% | 86.7267% | 86.6832% |
| Four-mode hard override | 79.7075% | 9,199 | 7.4514% | 2,757 | 92.9108% | 85.8042% | 85.4783% |
| Mode confidence | 95.9543% | 1,834 | 14.4405% | 5,343 | 89.0604% | 92.3789% | 91.2829% |
| Uncertainty-band four-mode | 96.6006% | 1,541 | 15.1919% | 5,621 | 88.6242% | 92.4407% | 91.3011% |
| Selective recovery (p>=40%) | 97.0087% | 1,356 | 17.9459% | 6,640 | 86.8816% | 91.6663% | 90.2881% |
| Binary only | 96.5367% | 1,570 | 17.1216% | 6,335 | 87.3545% | 91.7164% | 90.3986% |
| Tuned binary (59%) | 96.3911% | 1,636 | 16.5703% | 6,131 | 87.6954% | 91.8379% | 90.5662% |
| Score-only logistic (continuous reference) | 96.9139% | 1,399 | 19.0378% | 7,044 | 86.1820% | 91.2334% | 89.7452% |
| Total-error logistic (continuous reference) | 96.3889% | 1,637 | 16.4216% | 6,076 | 87.7921% | 91.8898% | 90.6318% |
| Total-error logistic (AE-gated) | 96.5124% | 1,581 | 17.1027% | 6,328 | 87.3640% | 91.7106% | 90.3938% |
| Per-feature logistic (continuous reference) | 97.3087% | 1,220 | 19.7622% | 7,312 | 85.7810% | 91.1819% | 89.6371% |
| Per-feature logistic (AE-gated) | 96.9712% | 1,373 | 17.8459% | 6,603 | 86.9408% | 91.6825% | 90.3124% |
| Prior confidence (reference) | 96.4484% | 1,610 | 16.4973% | 6,104 | 87.7494% | 91.8935% | 90.6306% |
| Imported binary (reference) | 94.6484% | 2,426 | 16.6459% | 6,159 | 87.4473% | 90.9054% | 89.5727% |
| Imported plain AE alone | 78.1391% | 9,910 | 9.1919% | 3,401 | 91.2397% | 84.1828% | 83.8325% |
| Imported OR | 96.6315% | 1,527 | 22.4351% | 8,301 | 84.0690% | 89.9136% | 88.0630% |
| Imported AND | 76.1559% | 10,809 | 3.4027% | 1,259 | 96.4815% | 85.1222% | 85.3423% |
| Imported plain+latent (continuous reference) | 94.8646% | 2,328 | 16.4162% | 6,074 | 87.6238% | 91.1005% | 89.7950% |
| Imported plain+latent (AE-gated) | 95.0212% | 2,257 | 16.8622% | 6,239 | 87.3484% | 91.0234% | 89.6808% |

## All system results — AE target 10%

Attack recall and missed attacks are shown first because missing attacks is the priority. Actual system FPR remains necessary to assess false-alarm workload.

| Method | Recall | Missed attacks | System FPR | False alarms | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| AE alone | 83.9892% | 7,258 | 16.4459% | 6,085 | 86.2202% | 85.0901% | 83.7937% |
| OR | 98.3169% | 763 | 28.0243% | 10,369 | 81.1260% | 88.8980% | 86.4791% |
| AND | 82.2090% | 8,065 | 5.5432% | 2,051 | 94.7836% | 88.0496% | 87.7132% |
| Four-mode hard override | 83.4400% | 7,507 | 11.9324% | 4,415 | 89.5478% | 86.3861% | 85.5196% |
| Mode confidence | 96.0558% | 1,788 | 14.7000% | 5,439 | 88.8961% | 92.3374% | 91.2221% |
| Uncertainty-band four-mode | 97.2029% | 1,268 | 16.3216% | 6,039 | 87.9468% | 92.3435% | 91.1250% |
| Selective recovery (p>=40%) | 97.2690% | 1,238 | 18.4595% | 6,830 | 86.5879% | 91.6182% | 90.2007% |
| Binary only | 96.5367% | 1,570 | 17.1216% | 6,335 | 87.3545% | 91.7164% | 90.3986% |
| Tuned binary (59%) | 96.3911% | 1,636 | 16.5703% | 6,131 | 87.6954% | 91.8379% | 90.5662% |
| Score-only logistic (continuous reference) | 96.9139% | 1,399 | 19.0378% | 7,044 | 86.1820% | 91.2334% | 89.7452% |
| Total-error logistic (continuous reference) | 96.3889% | 1,637 | 16.4216% | 6,076 | 87.7921% | 91.8898% | 90.6318% |
| Total-error logistic (AE-gated) | 96.4925% | 1,590 | 17.0622% | 6,313 | 87.3879% | 91.7148% | 90.4011% |
| Per-feature logistic (continuous reference) | 97.3087% | 1,220 | 19.7622% | 7,312 | 85.7810% | 91.1819% | 89.6371% |
| Per-feature logistic (AE-gated) | 97.0793% | 1,324 | 18.0216% | 6,668 | 86.8419% | 91.6757% | 90.2930% |
| Prior confidence (reference) | 96.4484% | 1,610 | 16.4973% | 6,104 | 87.7494% | 91.8935% | 90.6306% |
| Imported binary (reference) | 94.6484% | 2,426 | 16.6459% | 6,159 | 87.4473% | 90.9054% | 89.5727% |
| Imported plain AE alone | 82.7826% | 7,805 | 16.3189% | 6,038 | 86.1403% | 84.4280% | 83.1864% |
| Imported OR | 97.3904% | 1,183 | 28.2514% | 10,453 | 80.8560% | 88.3563% | 85.8670% |
| Imported AND | 80.0406% | 9,048 | 4.7135% | 1,744 | 95.4139% | 87.0537% | 86.8921% |
| Imported plain+latent (continuous reference) | 94.8646% | 2,328 | 16.4162% | 6,074 | 87.6238% | 91.1005% | 89.7950% |
| Imported plain+latent (AE-gated) | 95.0499% | 2,244 | 16.8946% | 6,251 | 87.3305% | 91.0268% | 89.6820% |

## Confusion matrices

Rows = actual; columns = predicted. Reference methods repeat unchanged across budgets by design.

### AE alone — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 33,185 | 3,815 |
| Attack | 8,955 | 36,377 |

### OR — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 28,301 | 8,699 |
| Attack | 1,012 | 44,320 |

### AND — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 35,549 | 1,451 |
| Attack | 9,513 | 35,819 |

### Four-mode hard override — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 34,243 | 2,757 |
| Attack | 9,199 | 36,133 |

### Mode confidence — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 31,657 | 5,343 |
| Attack | 1,834 | 43,498 |

### Uncertainty-band four-mode — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 31,379 | 5,621 |
| Attack | 1,541 | 43,791 |

### Selective recovery (p>=40%) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,360 | 6,640 |
| Attack | 1,356 | 43,976 |

### Binary only — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

### Tuned binary (59%) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,869 | 6,131 |
| Attack | 1,636 | 43,696 |

### Score-only logistic (continuous reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,956 | 7,044 |
| Attack | 1,399 | 43,933 |

### Total-error logistic (continuous reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,924 | 6,076 |
| Attack | 1,637 | 43,695 |

### Total-error logistic (AE-gated) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,672 | 6,328 |
| Attack | 1,581 | 43,751 |

### Per-feature logistic (continuous reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,688 | 7,312 |
| Attack | 1,220 | 44,112 |

### Per-feature logistic (AE-gated) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,397 | 6,603 |
| Attack | 1,373 | 43,959 |

### Prior confidence (reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

### Imported binary (reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,841 | 6,159 |
| Attack | 2,426 | 42,906 |

### Imported plain AE alone — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 33,599 | 3,401 |
| Attack | 9,910 | 35,422 |

### Imported OR — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 28,699 | 8,301 |
| Attack | 1,527 | 43,805 |

### Imported AND — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 35,741 | 1,259 |
| Attack | 10,809 | 34,523 |

### Imported plain+latent (continuous reference) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,926 | 6,074 |
| Attack | 2,328 | 43,004 |

### Imported plain+latent (AE-gated) — AE target 5%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,761 | 6,239 |
| Attack | 2,257 | 43,075 |

### AE alone — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,915 | 6,085 |
| Attack | 7,258 | 38,074 |

### OR — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 26,631 | 10,369 |
| Attack | 763 | 44,569 |

### AND — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 34,949 | 2,051 |
| Attack | 8,065 | 37,267 |

### Four-mode hard override — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 32,585 | 4,415 |
| Attack | 7,507 | 37,825 |

### Mode confidence — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 31,561 | 5,439 |
| Attack | 1,788 | 43,544 |

### Uncertainty-band four-mode — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,961 | 6,039 |
| Attack | 1,268 | 44,064 |

### Selective recovery (p>=40%) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,170 | 6,830 |
| Attack | 1,238 | 44,094 |

### Binary only — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

### Tuned binary (59%) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,869 | 6,131 |
| Attack | 1,636 | 43,696 |

### Score-only logistic (continuous reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,956 | 7,044 |
| Attack | 1,399 | 43,933 |

### Total-error logistic (continuous reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,924 | 6,076 |
| Attack | 1,637 | 43,695 |

### Total-error logistic (AE-gated) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,687 | 6,313 |
| Attack | 1,590 | 43,742 |

### Per-feature logistic (continuous reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 29,688 | 7,312 |
| Attack | 1,220 | 44,112 |

### Per-feature logistic (AE-gated) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,332 | 6,668 |
| Attack | 1,324 | 44,008 |

### Prior confidence (reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

### Imported binary (reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,841 | 6,159 |
| Attack | 2,426 | 42,906 |

### Imported plain AE alone — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,962 | 6,038 |
| Attack | 7,805 | 37,527 |

### Imported OR — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 26,547 | 10,453 |
| Attack | 1,183 | 44,149 |

### Imported AND — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 35,256 | 1,744 |
| Attack | 9,048 | 36,284 |

### Imported plain+latent (continuous reference) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,926 | 6,074 |
| Attack | 2,328 | 43,004 |

### Imported plain+latent (AE-gated) — AE target 10%

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,749 | 6,251 |
| Attack | 2,244 | 43,088 |

## Limits and reproduction

All AE cutoffs use calibration normals only and were saved before current test scoring. Selection results are saved separately for recent models; imported models have a different split protocol and are not scored on the recent selection set, which may overlap their fitting data. This is an operating-rule sweep, not a new model-training search. AE-gated learned variants are explicitly new conditions, not claims that the continuous model itself has a 5%/10% AE FPR.

The official test has known training-feature overlap and repeated development use. Imported seed42 withheld Exploits during training; recent models did not. Calibration FPR is empirical and need not transfer to test/live traffic. No serving settings changed.

Exact boundaries: frozen_thresholds.json. All metrics and per-baseline error changes: test_results.csv/json. Per-family recall: attack_family_recall.csv. All frozen decisions: test_predictions.npz. Sources: audit.json.

```sh
.venv/bin/python training/experiment_ae_fpr_sweep.py --output-dir experiments/ae_fpr_5_10_new
```

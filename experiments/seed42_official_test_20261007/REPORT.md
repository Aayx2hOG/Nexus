# Seed 42 versus recent confidence fusion — same official testing set

**Both models below are now evaluated on the exact same 82,332 test rows: 37,000 normal and 45,332 attacks.** No thresholds or model weights were changed.

Imported model: `experiments/seed_42/seed_42/models.joblib`, `fusion_ae_plain_latent`, seed 42, Exploits withheld during training. Its saved 5%-budget fusion threshold is **0.675778666446794**; its binary baseline uses **0.646919282734273**. The 5% is the original calibration budget, not a promised FPR on this test.

The imported fusion combines LightGBM score, log(1+plain-AE reconstruction error), and log(1+denoising-AE latent Mahalanobis distance). The latent component is from the denoising AE; it is not a plain-AE-only model.

Recent rule: 65% LightGBM cutoff below the normal-calibration AE-error median; 57.7693% elsewhere. Its saved predictions were independently reproduced on these test rows.

## Test metrics

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Imported seed42 binary | 89.5727% | 87.4473% | 94.6484% | 90.9054% | 16.6459% |
| Imported seed42 plain-AE + latent | 89.7950% | 87.6238% | 94.8646% | 91.1005% | 16.4162% |
| Recent binary | 90.3986% | 87.3545% | 96.5367% | 91.7164% | 17.1216% |
| Recent confidence rule | 90.6306% | 87.7494% | 96.4484% | 91.8935% | 16.4973% |

## Confusion matrices

Rows = actual; columns = predicted.

### Imported seed42 binary

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,841 | 6,159 |
| Attack | 2,426 | 42,906 |

### Imported seed42 plain-AE + latent

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,926 | 6,074 |
| Attack | 2,328 | 43,004 |

### Recent binary

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,665 | 6,335 |
| Attack | 1,570 | 43,762 |

### Recent confidence rule

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

## Direct paired comparison

Imported fusion versus recent confidence rule: F1 difference **-0.7930 percentage points**, recall difference **-1.5839 points**, and FPR difference **-0.0811 points**.

The imported fusion detects 314 attacks missed by the recent rule but misses 1,032 attacks the recent rule detects. It introduces 1,161 false alarms and removes 1,191 recent false alarms.

## Verification and limits

All six imported artifact hashes verified. Before test scoring, the full archived 68,643-row holdout was rescored: saved component scores matched within numerical tolerance, and binary/fusion predictions matched exactly at all three archived budgets. See archive_parity.json for maximum score differences.

This fixes the previous different-evaluation-set comparison. Training still differs: the imported run withheld Exploits, whereas the recent run included all attack families. Therefore this compares the actual frozen systems, not an isolated architecture effect. The official test has known training-feature overlap and has already been used in project development; it is not a fresh holdout.

Per-attack-family recall is in attack_family_recall.csv. Frozen predictions and scores are in test_predictions.npz. No serving configuration was changed.

```sh
.venv/bin/python training/evaluate_imported_seed42_test.py --output-dir experiments/seed42_official_test_new
```

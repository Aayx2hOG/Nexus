# Imported seed 42 versus the best recent experiment

## Finding

The imported run’s best recorded budget-compliant result is **fusion_ae_plain_latent** at a 5% FPR budget: **97.6830% F1**. The best recorded official-test F1 across the five recent experiment folders is **Mode confidence/recall95**, **91.8935% F1**. These evaluate different rows under different training and threshold protocols, so the larger archived F1 does not establish a better model.

“Best” here is a retrospective descriptive ranking, not a newly validated model selection. The archived winner was identified by its recorded holdout performance; the recent rule was originally chosen on selection data before test scoring.

## What seed 42 contains

Source: `experiments/seed_42/seed_42`. Its manifest identifies **Exploits withheld**: that family is absent from fitting, early stopping, fusion training and calibration. The official testing CSV was not used in that run. Both old and recent models use seed 42, so this is a comparison of protocols and models, not a seed-effect study.

All six manifest artifact hashes verified. All 60 saved confusion matrices and associated precision/recall/F1/FPR values reproduce from the stored decisions. The recent best matrix also reproduces.

| Property | Imported seed 42 | Recent best |
|---|---|---|
| Evaluation rows | 68,643 internal holdout | 82,332 official test |
| Normal / attack | 8,259 / 60,384 | 37,000 / 45,332 |
| Attack prevalence | 87.97% | 55.06% |
| Training | Exploits withheld; separate fusion partition | All attack families available |
| Threshold objective | Benign calibration FPR budget | Selection recall ≥95%, minimize FPR |
| Fusion | Plain AE reconstruction + latent representation | Mode-specific LightGBM cutoffs |

The difference in attack prevalence particularly affects precision and F1. FPR and recall also depend on which normal and attack traffic the holdout contains.

## Recorded results — not a matched head-to-head comparison

| Dataset | Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|
| Internal Exploits-withheld holdout | lightgbm | 95.7082% | 99.3437% | 95.7538% | 97.5157% | 4.6253% |
| Internal Exploits-withheld holdout | fusion_ae_plain_latent | 95.9909% | 99.3492% | 96.0718% | 97.6830% | 4.6010% |
| Official test | Current binary | 90.3986% | 87.3545% | 96.5367% | 91.7164% | 17.1216% |
| Official test | Mode confidence/recall95 | 90.6306% | 87.7494% | 96.4484% | 91.8935% | 16.4973% |

## Improvement over each experiment’s own binary baseline

Changes below are percentage points; they help describe the fusion contribution within each experiment but do not remove differences between evaluation sets.

| Change | Imported plain-AE + latent fusion | Recent confidence rule |
|---|---:|---:|
| F1 | +0.1673 pp | +0.1771 pp |
| Attack recall | +0.3180 pp | -0.0882 pp |
| False-positive rate | -0.0242 pp | -0.6243 pp |
| Accuracy | +0.2826 pp | +0.2320 pp |

The archived fusion recovered 269 attacks and lost 77: net **+192** detected attacks, with **-2** false positives. Exploits recall rose from 95.7356% to 96.2896%.

The recent rule recovered 0 and lost 40 attacks: net **-40** detected attacks, with **-231** false positives. The archived fusion is promising for recovering attacks; the recent rule primarily suppresses false alarms.

## Confusion matrices

Rows = actual; columns = predicted.

### Imported plain-AE + latent fusion

| | Normal | Attack |
|---|---:|---:|
| Normal | 7,879 | 380 |
| Attack | 2,372 | 58,012 |

### Recent confidence rule

| | Normal | Attack |
|---|---:|---:|
| Normal | 30,896 | 6,104 |
| Attack | 1,610 | 43,722 |

## Highest raw archived F1

`naive_or` at the 5% budget recorded 97.7552% F1, but its observed FPR was 8.6935%, exceeding its budget. It should not be described as the best budget-compliant run.

## Conclusion and next comparison

The archived plain-AE + latent method deserves a matched evaluation: it improved both recall and FPR relative to its own baseline. We cannot conclude that it beats the recent method from these saved metrics alone. Freeze the archived rule and score it on the same official test as a historical check; for a stronger comparison, train both approaches using identical grouped partitions, data access and selection objectives, then evaluate on a fresh holdout. In particular, do not evaluate the recent model on this archived internal holdout without checking training overlap.

No retraining, retuning or promotion was performed. Source artifacts remain unchanged. The recent official test has known training-feature overlap and repeated development use, documented in its original reports.

Reproduce: `.venv/bin/python experiments/seed42_comparison_20261007/compare.py`.

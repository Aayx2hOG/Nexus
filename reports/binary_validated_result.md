# Binary LightGBM: current validation winner

Reviewed 2026-10-03. Experiment: `models/lightgbm_validated_v1`.

## Verdict and scope

Trial 4 is the best of all 16 LightGBM candidates in this run for the declared
objective: minimize false-positive rate (FPR), subject to attack recall >=95%.
All 16 candidates meet that recall constraint on selection data; trial 4 has the
lowest FPR. It also beats the paired Random Forest baseline on that same data.
This is our current binary validation winner, **not a demonstrated best-ever
model on independent test data**. No final evaluation is recorded for this run.

The selection set contains 25,031 rows: 8,193 normal and 16,838 attack rows.
Selection results after a 16-candidate search may be optimistic. RF is a fixed
300-tree baseline and does not receive an equivalent tuning budget.

## Measured improvement over paired Random Forest

Both models use the same fitting and evaluation partitions, with thresholds
chosen on separate calibration rows. Percentage-point changes below are
LightGBM minus RF; they are not relative percentage changes.

| Metric | Paired RF | Selected LightGBM | Change |
|---|---:|---:|---:|
| Accuracy | 95.1620% | 95.5056% | +0.3436 pp |
| Balanced accuracy | 95.1506% | 95.6379% | +0.4872 pp |
| Attack precision | 97.5650% | 98.0079% | +0.4430 pp |
| Attack recall | 95.1835% | 95.2548% | +0.0713 pp |
| Attack F1 | 96.3595% | 96.6118% | +0.2522 pp |
| Normal-traffic FPR | 4.8822% | 3.9790% | -0.9032 pp |
| ROC-AUC | 0.993359 | 0.994262 | +0.000904 |
| Average precision | 0.996624 | 0.997176 | +0.000552 |
| False alerts / 1,000 normal flows | 48.82 | 39.79 | -9.03 |
| False positives | 400 | 326 | 74 fewer |
| Missed attacks | 811 | 799 | 12 fewer |

The false-positive reduction is **18.5% relative**: `(400 - 326) / 400`.
Total classification errors decrease from 1,211 to 1,125 (86 fewer).
Approximately 40 false alerts per 1,000 benign flows remains a meaningful
operational burden; this result alone does not establish deployment readiness.

Source: [validation_summary.json](../models/lightgbm_validated_v1/validation_summary.json).

## What changed and what the evidence supports

The revised workflow groups identical predictor rows so they cannot cross
partitions. Preprocessing is fitted only on model-fitting rows. Early stopping,
threshold calibration and model selection have separate partitions. After early
stopping, the model is refitted on fitting plus early-stopping rows, and its
threshold is then chosen without using selection labels. These changes improve
the credibility of evaluation; their individual effects on performance have not
been isolated.

The winning model uses balanced class weights (`weight_alpha=1.0`), 63 leaves,
minimum leaf samples 100, learning rate 0.03, L2 regularization 1.0, feature and
row subsampling 0.85, and 1,318 refit boosting rounds on 124,861 rows.

There is also a useful within-run LightGBM comparison. Trial 1 uses the same
base search settings but no class weighting, with its own selected boosting
rounds and calibrated threshold. Moving from trial 1 to trial 4 changes:

| Metric | Unweighted trial 1 | Balanced trial 4 | Change |
|---|---:|---:|---:|
| FPR | 4.1255% | 3.9790% | -0.1465 pp |
| Attack recall | 95.2429% | 95.2548% | +0.0119 pp |
| False positives | 338 | 326 | 12 fewer (3.55% relative) |
| Missed attacks | 801 | 799 | 2 fewer |
| Decision threshold | 0.693861 | 0.577693 | -0.116169 |

This supports the balanced training procedure on this split. It does not prove
the threshold change alone caused the gain: the fitted model and boosting
rounds also change.

Sources: [trial 1](../models/lightgbm_validated_v1/binary/trials/trial_0001.json),
[selected trial 4](../models/lightgbm_validated_v1/binary/selected_validation.json).

## Why the threshold changed

The earlier `lightgbm_binary_recall95` winner used threshold 0.721548; the new
winner uses 0.577693, a decrease of 0.143855 in score units. The earlier model
was unweighted. The new model gives normal rows weight 1.5684 and attack rows
weight 0.7340, which tends to lower attack scores. These raw model outputs are
not calibrated probabilities, so cutoffs across models are not directly
comparable confidence levels.

The new calibration procedure also uses a one-sided 95% Wilson lower bound on
recall, instead of requiring only observed recall >=95%. This adds a calibration
margin and tends to lower the threshold for a given model. Grouped partitions,
new preprocessing fits and refitting also change score distributions. The
individual contributions to the old-to-new threshold difference are unmeasured.

The algorithm minimizes calibration FPR subject to the recall requirement,
breaking ties by higher recall and then higher threshold. The Wilson margin
does not guarantee recall after model selection or under distribution shift.

## Historical improvement: keep it separate

The previous binary experiment did improve on the original LightGBM on the
same full official test set:

| Metric | Original LightGBM | Previous recall95 winner | Change |
|---|---:|---:|---:|
| FPR | 18.3649% | 16.3595% | -2.0054 pp |
| Attack recall | 97.0264% | 95.8903% | -1.1361 pp |
| Accuracy | 90.1096% | 90.3853% | +0.2757 pp |
| Attack F1 | 91.5275% | 91.6545% | +0.1270 pp |

That tradeoff removed 742 false positives but added 515 missed attacks, while
remaining above the 95% recall gate. Its ROC-AUC decreased from 0.984217 to
0.982302, so the improvement was specific to the operating objective.

Do **not** compare that 16.3595% test FPR with the new 3.9790% selection FPR
and call the difference an improvement. They are different populations under
different evaluation protocols.

Sources: [previous binary result](../experiments/lightgbm_binary_recall95/binary/selected_result.json)
and [original baseline metrics](../experiments/lightgbm_binary_recall95/binary/baseline_metrics.json).

## How to establish whether the new model beats the old model

1. Freeze the exact saved models, preprocessing and thresholds before scoring.
   Predeclare the decision rule: a candidate must meet 95% attack recall, then
   compare FPR. Report recall failures explicitly, along with all other metrics.
2. Score old and new models on exactly the same compatible-schema labeled
   holdout rows. Remove predictor overlap with the union of their development
   data, and preserve each model's own feature transformation and threshold.
   The new internal selection rows cannot test the old models fairly: those
   models used the official training data containing these rows.
3. The existing official test CSV can provide a **historical diagnostic**, but
   its results have already informed development. Evaluate all contenders on
   the same overlap-filtered subset; do not compare these numbers with older
   full-test metrics.
4. For independent confirmation, use genuinely unused labeled data that did
   not influence training, threshold choice or model development. Freeze the
   protocol before inspecting outcomes. Report paired uncertainty intervals,
   resampling duplicate groups together rather than treating copies as
   independent observations. Session/time dependence needs additional grouping
   when suitable identifiers exist.
5. Do not change thresholds based on those evaluation outcomes. Further tuning
   starts a new development cycle and needs a new independent confirmation set.

The existing evaluator can already score the **new LightGBM and paired RF** on
the historical test file, without refitting or changing thresholds:

```sh
.venv/bin/python training/evaluate_frozen_lightgbm.py \
  --experiment-dir models/lightgbm_validated_v1 \
  --evaluation-csv 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv' \
  --data-role historical
```

It verifies frozen hashes, removes development overlap, records the remaining
population, and writes `final_evaluation/audit.json` and `metrics.json`. It
refuses to overwrite an existing evaluation. For fresh data, substitute the
new CSV path and use `--data-role fresh-holdout`; the flag records a user
assertion, not proof of independence.

**This evaluator does not load the older LightGBM checkpoints.** A comparison
adapter is still needed to apply their saved preprocessor and thresholds to
the exact same retained rows. Running the command above alone establishes only
the new LightGBM-versus-RF comparison, not an old-versus-new LightGBM win.

No new training or predictive test evaluation was performed for this report.
Figures were checked against the saved JSON metrics and all 16 trial records.

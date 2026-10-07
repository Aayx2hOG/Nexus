# Nexus model configuration — Working v2

**Decision record updated:** 4 October 2026 (IST).  
**Latest recorded official-test evaluation:** 3 October 2026, 18:54 IST.  
**Scope:** model development, configuration, results, and the working setup selected for this stage.

## 1. Final working setup

`working_v2.md` is the consolidated reference for the current **LightGBM v2 + baseline autoencoder** setup and its evaluated **OR decision rule**. LightGBM remains the primary supervised detector; the autoencoder supplies complementary anomaly evidence. The combined configuration prioritizes recovering additional attacks, with a measured increase in false alerts.

| Component | Configuration selected for Working v2 |
| --- | --- |
| Primary model | Frozen binary LightGBM, selected trial 4 |
| LightGBM artifact | `experiments/lightgbm_validated_v2/binary/model.joblib` |
| LightGBM attack cutoff | Score **>= 0.5776925765603604** |
| Complementary model | Baseline normal-traffic autoencoder (AE) |
| AE artifact | `models/autoencoder/autoencoder.joblib` |
| AE attack cutoff | Mean squared reconstruction error **>= 0.0008786016260273755** |
| Combined rule | Attack if LightGBM **OR** AE predicts attack |
| Output labels | `0 = normal`, `1 = attack` |
| Primary task | Binary network intrusion detection using UNSW-NB15 flow features |
| Operational role | Analyst-facing detection; automatic traffic blocking is outside this setup |

The OR configuration is the combined working configuration documented here, **not an across-the-board performance winner**. On the recorded full official test, LightGBM alone has better accuracy, precision, F1, and FPR; OR has higher attack recall. Keep both results visible when presenting the decision.

This document does not promote a release bundle or change application serving. The repository's separate [model assessment](reports/current_model_assessment.md) favors selective fusion as a future recovery candidate; that policy is distinct from the frozen OR evaluation in this document.

## 2. Model work completed so far

The following is a development overview assembled from saved reports, configurations, and training workflows. A retained implementation establishes that a workflow exists; it does not establish that every possible run completed.

| Workstream | Work recorded in the repository | Outcome and place in Working v2 |
| --- | --- | --- |
| Data preparation | Numeric/categorical preprocessing, tree inputs, saved preprocessing artifacts, and normal-only AE scaling | Each model must use its matching saved transformation |
| Supervised baselines | Random Forest and LightGBM training and evaluation | LightGBM retained as primary; RF retained as a paired baseline |
| Binary threshold tuning | Recall-constrained threshold selection and the historical `lightgbm_binary_recall95` experiment | Established the recall/FPR operating-point tradeoff |
| Multiclass experiments | Macro-F1 tuning and controlled multiclass experiments across normal plus nine attack families | Auxiliary research; not the final binary decision rule |
| Validated LightGBM | Grouped partitions, isolated fitting/early stopping/calibration/selection, 16 candidates, fixed-round refit, frozen manifests | Trial 4 selected; exact v2 settings recorded below |
| Anomaly baselines | Autoencoder and Isolation Forest workflows, reconstruction-error evaluation, complementarity analysis | Baseline AE retained in the evaluated v2 combination |
| AE optimization | Separate search/freeze and evaluation commands; documented 36-candidate search plus one alpha sensitivity candidate | Workflow retained; no replacement AE winner is asserted here without its saved result |
| Unseen-family research | Leave-one-attack-family-out evaluation and plain/scaled/denoising AE, latent-distance, and Isolation Forest comparisons | Measures generalization and complementary recovery; does not prove zero-day detection |
| Fusion research | Learned fusion, naive OR, rank OR, and selective recovery; eight-seed assessment | Selective recovery is a separate research candidate; mixed and negative results retained |
| Working v2 benchmark | Frozen LightGBM v2, baseline AE, and OR scored on the same 82,332 official testing rows | Full comparison and confusion matrices preserved in the appendix |
| Application support | Existing LightGBM serving, TreeSHAP explanations, prediction logging, analyst dashboard | No v2 OR deployment is established by this documentation update |

Sources: [training workflows](training/README.md), [validated training](training/VALIDATED_TRAINING.md), [historical experiments](experiments/README.md), [fusion guide](training/FUSION_EXPERIMENTS.md), and [model assessment](reports/current_model_assessment.md).

### Historical supervised results

The earlier binary report records the following full official-test comparison:

| Model | Accuracy | Attack recall | Attack F1 | Normal FPR |
| --- | ---: | ---: | ---: | ---: |
| Original LightGBM | 90.1096% | 97.0264% | 91.5275% | 18.3649% |
| Historical recall95 winner | 90.3853% | 95.8903% | 91.6545% | 16.3595% |
| Working v2 LightGBM | 90.3986% | 96.5367% | 91.7164% | 17.1216% |

The earlier recall95 model reduced false positives while missing more attacks. V2 occupies another tradeoff: higher recall and higher FPR than that historical recall95 winner. These repeatedly inspected benchmark results are historical diagnostics, not independent confirmation of progress.

The retained multiclass summaries report **52.5171% macro-F1 / 72.7105% accuracy** for the macro-F1 experiment and **53.0383% macro-F1 / 72.3753% accuracy** for the controlled experiment. This illustrates why accuracy alone was insufficient for minority-family evaluation. These multiclass scores do not describe the binary v2 model.

Sources: [binary assessment](reports/binary_validated_result.md), [multiclass summary](experiments/lightgbm_multiclass_macrof1/summary.json), [controlled multiclass summary](experiments/lightgbm_controlled_multiclass/summary.json).

## 3. Dataset, features, and validation design

The v2 training audit records **175,341 official training rows**, **101,040 unique predictor rows**, and **74,301 duplicates beyond the first occurrence**. It identifies **1,772 conflicting-label groups**, containing **30,528 rows**. Labels are retained; identical predictors stay in the same partition. Exact-row grouping does not establish session or temporal independence.

There are **42 raw predictor columns**. `id`, `label`, and `attack_cat` are excluded from model inputs. `proto`, `service`, and `state` are categorical; the remaining predictors describe flow duration, packet/byte counts, rates, timing, TCP fields, and connection counts. Preserve the column order and transformation stored with the model.

| Partition | Rows | Purpose |
| --- | ---: | --- |
| Fit | 99,778 | Initial model and preprocessing fit |
| Early stopping | 25,083 | Choose boosting rounds |
| Calibration | 25,449 | Select the decision threshold |
| Selection | 25,031 | Compare candidates after threshold selection |
| Final refit: fit + early stopping | 124,861 | Fit selected-round models before calibration/selection scoring |

The model-selection objective is **minimum normal-traffic FPR subject to attack recall >=95%**. Calibration uses a one-sided 95% Wilson recall margin. This margin is not a guarantee after model selection or under distribution shift. The selection set is development evidence after searching candidates.

Sources: [v2 data audit](experiments/lightgbm_validated_v2/data_audit.json), [frozen manifest](experiments/lightgbm_validated_v2/manifest.json), [validated workflow](training/VALIDATED_TRAINING.md).

## 4. Exact LightGBM v2 configuration

| Parameter | Frozen value |
| --- | --- |
| Objective / boosting | `binary` / `gbdt` |
| Selected trial / candidates | `4` / `16` |
| Seed | `42` |
| Learning rate | `0.03` |
| Number of leaves | `63` |
| Maximum depth | `-1` |
| Minimum child samples | `100` |
| L1 / L2 regularization | `0.0` / `1.0` |
| Row subsampling / frequency | `0.85` / `1` |
| Feature subsampling | `0.85` |
| Weight exponent | `1.0` (balanced weighting) |
| Refit normal / attack class weights | `1.5684084913955534` / `0.7339928987960873` |
| Maximum search rounds / patience | `2500` / `100` |
| Selected refit rounds | `1318` |
| Early-stopping criterion | Log loss |
| Threads | `4` |
| Deterministic / force column-wise | `true` / `true` |
| Attack threshold | `0.5776925765603604` |

The model's raw probability output is a decision score; no calibrated-probability claim is made. This threshold-bearing bundle's `predict()` applies its saved cutoff. The v2 evaluator explicitly checks that `predict()` agrees with `predict_proba(X)[:, 1] >= decision_threshold_`.

### Paired internal selection comparison

| Metric | Random Forest | LightGBM v2 |
| --- | ---: | ---: |
| Accuracy | 95.1620% | 95.5056% |
| Attack precision | 97.5650% | 98.0079% |
| Attack recall | 95.1835% | 95.2548% |
| Attack F1 | 96.3595% | 96.6118% |
| Normal FPR | 4.8822% | 3.9790% |
| False positives | 400 | 326 |
| Missed attacks | 811 | 799 |

LightGBM removes 74 false positives and 12 missed attacks on these paired selection rows. RF is a fixed 300-tree comparator, not an equally tuned search. **Do not compare the 3.9790% selection FPR with the 17.1216% official-test FPR as if they measured the same population.**

Sources: [selected configuration](experiments/lightgbm_validated_v2/binary/selected_validation.json), [v2 validation summary](experiments/lightgbm_validated_v2/validation_summary.json).

## 5. Baseline autoencoder and combined decision

The baseline implementation uses a normal-only `MLPRegressor` autoencoder with reconstruction as its target. Its saved numeric scaler transforms numeric features; one-hot features remain unchanged. The anomaly score is per-row mean squared reconstruction error. The baseline training workflow selects a threshold to maximize attack recall subject to validation FPR <=10%; this is distinct from the later fusion research's 1%, 3%, and 5% budgets.

The recorded baseline cutoff is **0.0008786016260273755**. The exact trained hidden layers, epoch history, and optimizer settings must come from the saved `autoencoder_config.json`; they cannot be recovered from the evaluation JSON alone. That configuration and the AE checkpoint are absent from this checkout, so script defaults are not presented as verified trained settings.

```text
lgbm_attack = lightgbm_attack_score >= 0.5776925765603604
ae_attack   = mean_squared_reconstruction_error >= 0.0008786016260273755
final_attack = lgbm_attack OR ae_attack
```

| LightGBM | AE | Combined output |
| --- | --- | --- |
| Normal | Normal | Normal |
| Attack | Normal | Attack |
| Normal | Attack | Attack |
| Attack | Attack | Attack |

This rule preserves every LightGBM attack decision. On the recorded official test it recovers **635 of LightGBM's 1,570 missed attacks**, leaving **935 misses**, while adding **3,700 false positives**. Recall rises from **96.5367% to 97.9374%**; normal FPR rises from **17.1216% to 27.1216%**. That is about **5.83 extra false alerts per additional attack recovered** on this dataset.

Source: [v2 evaluation JSON](reports/official_testing_lightgbm_v2_ae_evaluation.json), [baseline AE implementation](training/train_autoencoder.py).

## 6. What the separate fusion research established

The existing assessment reports **960 result rows** across four scenarios, ten methods, three calibration budgets, and eight seeds. Scenarios are closed set and withheld Reconnaissance, Exploits, or DoS. These experiments use internal partitions of the official training CSV, not the full official testing population below.

At the nominal 3% calibration budget, selective fusion reports the following mean additional detections and false positives relative to paired LightGBM:

| Scenario | Additional attacks recovered | Lost LightGBM detections | Added false positives | Observed selective FPR |
| --- | ---: | ---: | ---: | ---: |
| Closed set | 1.63 | 0 | 0.25 | 3.089% |
| Reconnaissance withheld | 16.50 | 0 | 1.00 | 2.986% |
| Exploits withheld | 154.88 | 0 | 3.25 | 3.067% |
| DoS withheld | 6.88 | 0 | 1.13 | 3.046% |

Selective fusion preserves LightGBM positives, routes eligible negatives using supervised/anomaly scores, and applies a learned recovery cutoff calibrated against remaining benign false-positive allowance. It differs from unconditional OR. Exploits has the strongest reported recovery; closed-set gains are very small. Unrestricted learned fusion can lose baseline detections; standalone anomaly detectors were weaker than LightGBM at practical FPR budgets. Scaling and denoising did not establish universal improvements over plain AE.

These are reported historical research findings from the [eight-seed assessment](reports/current_model_assessment.md). Its underlying `artifacts/` suites are absent here and were not re-evaluated for this document. The results do not validate the exact v2 OR pair at a 3% FPR budget.

## 7. Decision boundaries and remaining work

- **Fixed for this working record:** the v2 LightGBM artifact, baseline AE identity, saved cutoffs, OR rule, and full official-test comparison.
- **Retained comparison:** LightGBM alone, because it has substantially fewer false positives and higher F1 on the same benchmark.
- **Research candidate:** selective recovery, pending matched evaluation and integration. Its results must retain their own splits, budgets, and checkpoints.
- **Auxiliary work:** multiclass family prediction and unknown rejection do not form part of the final binary OR rule.
- **Next evidence needed:** fresh confirmation data, anomaly-input ablations, explicit recovery-budget allocation, and measured inference latency/memory.
- **Serving milestone:** export/verify the intended release bundle and integrate the chosen policy; writing this file does not complete that step.

The official test has already informed development and includes **8,541 rows with predictors also present in development**, retained for this full-benchmark evaluation. It is not an untouched holdout. No zero-day, production-readiness, or signature-IDS superiority claim follows from these results.

## 8. Reproducibility and artifact availability

The v2 manifest records the original invocation:

```sh
.venv/bin/python training/train_validated_lightgbm.py \
  --task binary \
  --split-indices data/processed/split_indices.npz \
  --trials 16 --max-rounds 2500 --patience 100 \
  --rf-trees 300 --n-jobs 4 --minimum-recall 0.95 \
  --experiment-dir experiments/lightgbm_validated_v2
```

This is provenance, not an instruction to overwrite the frozen directory. Any new training run requires a new output directory and the original compatible inputs. The recorded versions are LightGBM `4.7.0`, scikit-learn `1.9.1`, pandas `3.0.6`, and NumPy `2.5.3`; these are manifest values, not a check of the current environment.

The historical evaluator is:

```sh
.venv/bin/python training/evaluate_official_testing_v2.py
```

**The current evaluator writes directly to `working_v2.md`. Running it unchanged would replace this consolidated decision record with its generated metric report.** Preserve this document or redirect that output before a future run. No evaluator code was changed for this documentation task.

| Required item | Availability when this document was updated |
| --- | --- |
| V2 LightGBM checkpoint and selected configuration | Present under `experiments/lightgbm_validated_v2/` |
| V2 manifest, audit, and validation summary | Present |
| Historical evaluation JSON | Present under `reports/` |
| Baseline AE checkpoint and config | Absent at the recorded `models/autoencoder/` paths |
| Saved processed inputs/preprocessor | Absent at the required `data/processed/` paths |
| Eight-seed research artifact suites | Absent at the documented `artifacts/` paths |

The historical evaluation JSON records that frozen hashes, feature schema, labels, and AE cached inputs passed checks when that evaluation ran. This update checks saved documentation and metrics; it does not rerun inference or certify absent artifacts.

## 9. Preserved official-test evaluation

The original Working v2 metric report follows, including its complete classification reports. Its reproduction command is subject to the availability and overwrite notes above.

### Official testing data evaluation — LightGBM v2

Evaluated: 2026-10-03T18:54:23.817368+05:30.

Dataset: `data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv`. All 82,332 rows: 37,000 normal and 45,332 attack. Labels: 0 = normal, 1 = attack.

**Scope:** Evaluation uses the full official testing split with frozen models and saved thresholds; no retraining or tuning was performed. The LightGBM manifest states that existing official test results informed development, so these are historical test metrics rather than fresh independent confirmation.

Rows with predictors also present in development: 8,541; retained for the full official benchmark.

LightGBM: `experiments/lightgbm_validated_v2/binary/model.joblib`, attack probability threshold `>= 0.5776925765603604`.

Baseline AE: `models/autoencoder/autoencoder.joblib`, mean squared reconstruction error threshold `>= 0.0008786016260273755`. Inputs use the saved preprocessor and AE numeric scaler.

**Combined rule:** predict attack if either model predicts attack (OR).

### Comparison

| Model | Accuracy | Attack precision | Attack recall | Attack F1 | Normal FPR |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.903986 | 0.873545 | 0.965367 | 0.917164 | 0.171216 |
| Baseline AE | 0.835277 | 0.868579 | 0.825774 | 0.846636 | 0.153081 |
| Combined OR | 0.866759 | 0.815642 | 0.979374 | 0.890040 | 0.271216 |

### LightGBM

Confusion matrix: rows = actual, columns = predicted.

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 30665 | 6335 |
| Attack | 1570 | 43762 |

Classification report:

```text
              precision    recall  f1-score   support

      normal   0.951295  0.828784  0.885824     37000
      attack   0.873545  0.965367  0.917164     45332

    accuracy                       0.903986     82332
   macro avg   0.912420  0.897075  0.901494     82332
weighted avg   0.908486  0.903986  0.903079     82332
```

### Baseline AE

Confusion matrix: rows = actual, columns = predicted.

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 31336 | 5664 |
| Attack | 7898 | 37434 |

Classification report:

```text
              precision    recall  f1-score   support

      normal   0.798695  0.846919  0.822100     37000
      attack   0.868579  0.825774  0.846636     45332

    accuracy                       0.835277     82332
   macro avg   0.833637  0.836347  0.834368     82332
weighted avg   0.837173  0.835277  0.835610     82332
```

### Combined OR

Confusion matrix: rows = actual, columns = predicted.

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 26965 | 10035 |
| Attack | 935 | 44397 |

Classification report:

```text
              precision    recall  f1-score   support

      normal   0.966487  0.728784  0.830971     37000
      attack   0.815642  0.979374  0.890040     45332

    accuracy                       0.866759     82332
   macro avg   0.891064  0.854079  0.860506     82332
weighted avg   0.883432  0.866759  0.863495     82332
```

OR fusion detects 635 additional attacks and adds 3,700 false positives compared with LightGBM alone.

### Reproducibility

Run from the repository root:

```sh
.venv/bin/python training/evaluate_official_testing_v2.py
```

Frozen manifest, all recorded artifact/source hashes, and LightGBM feature schema verified; raw labels and all AE input rows match cached testing data.

Full metrics and SHA-256 artifact fingerprints: [evaluation JSON](reports/official_testing_lightgbm_v2_ae_evaluation.json).

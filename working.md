# Official testing data evaluation

Evaluated: 2026-10-03T16:36:53.768214+05:30.

Dataset: `data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv`. All 82,332 rows: 37,000 normal and 45,332 attack. Labels: 0 = normal, 1 = attack.

**Scope:** Evaluation uses the full official testing split with frozen models and saved thresholds; no retraining or tuning was performed. The LightGBM manifest states that existing official test results informed development, so these are historical test metrics rather than fresh independent confirmation.

LightGBM: `experiments/lightgbm_validated_v/binary/model.joblib`, attack probability threshold `>= 0.5776925765603604`.

Baseline AE: `models/autoencoder/autoencoder.joblib`, mean squared reconstruction error threshold `>= 0.0008786016260273755`. Inputs use the saved preprocessor and AE numeric scaler.

**Combined rule:** predict attack if either model predicts attack (OR).

## Comparison

| Model | Accuracy | Attack precision | Attack recall | Attack F1 | Normal FPR |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.903986 | 0.873545 | 0.965367 | 0.917164 | 0.171216 |
| Baseline AE | 0.835277 | 0.868579 | 0.825774 | 0.846636 | 0.153081 |
| Combined OR | 0.866759 | 0.815642 | 0.979374 | 0.890040 | 0.271216 |

## LightGBM

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

## Baseline AE

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

## Combined OR

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

## Reproducibility

Run from the repository root:

```sh
.venv/bin/python training/evaluate_official_testing_models.py
```

LightGBM manifest hash and feature schema verified; raw labels and all AE input rows match cached testing data.

Full metrics and SHA-256 artifact fingerprints: [evaluation JSON](reports/official_testing_lightgbm_ae_evaluation.json).

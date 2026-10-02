# Model Evaluation Results

This report compares the results recorded in:

- `models/random_forest_metrics.json`
- `reports/lightgbm_metrics.json`

Both models were evaluated on the same 82,332-row test set using 194 features. The training set contained 175,341 rows.

## Overall Metrics

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 score | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 87.11% | 85.80% | 81.66% | **98.76%** | 89.40% | 98.04% |
| LightGBM | **90.11%** | **89.33%** | **86.62%** | 97.03% | **91.53%** | **98.42%** |

LightGBM produced the stronger overall result. Compared with Random Forest, it improved accuracy by 3.00 percentage points, balanced accuracy by 3.53 points, precision by 4.95 points, and F1 score by 2.12 points. Random Forest achieved the higher attack recall, by 1.74 points.

## Confusion Matrices

Rows represent actual classes and columns represent predicted classes, in the order `normal`, `attack`.

| Model | True normal | False attack alarm | Missed attack | Detected attack |
|---|---:|---:|---:|---:|
| Random Forest | 26,948 | 10,052 | **561** | **44,771** |
| LightGBM | **30,205** | **6,795** | 1,348 | 43,984 |

LightGBM generated 3,257 fewer false attack alarms, reducing the false-positive rate from 27.17% to 18.36%. This came with 787 more missed attacks: its attack false-negative rate was 2.97%, compared with 1.24% for Random Forest.

## Per-Class Results

| Model | Class | Precision | Recall | F1 score | Support |
|---|---|---:|---:|---:|---:|
| Random Forest | Normal | **97.96%** | 72.83% | 83.55% | 37,000 |
| Random Forest | Attack | 81.66% | **98.76%** | 89.40% | 45,332 |
| LightGBM | Normal | 95.73% | **81.64%** | **88.12%** | 37,000 |
| LightGBM | Attack | **86.62%** | 97.03% | **91.53%** | 45,332 |

## Conclusion

LightGBM is the better balanced model and the preferred default based on accuracy, balanced accuracy, precision, F1 score, and ROC AUC. Its largest practical advantage is the substantial reduction in false attack alarms.

Random Forest remains preferable when minimizing missed attacks is the overriding priority. It detected 98.76% of attacks and missed only 561 attack samples, but this sensitivity resulted in considerably more normal traffic being classified as an attack.

## Training Summary

| Model | Main configuration |
|---|---|
| Random Forest | 300 trees, `class_weight=balanced_subsample`, `max_features=sqrt`, `random_state=42` |
| LightGBM | 1,000 boosting iterations, 31 leaves, learning rate 0.05, `class_weight=balanced`, `random_state=42` |

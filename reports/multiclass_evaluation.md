# Multiclass Model Evaluation

This report summarizes the multiclass evaluation recorded in
[`multiclass_evaluation_metrics.json`](./multiclass_evaluation_metrics.json).
Both models were evaluated on the same 82,332-row test set across the
`attack_cat` classes.

## Overall Metrics

| Model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro ROC-AUC | Weighted ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 75.08% | 51.54% | 54.73% | 51.54% | 49.21% | **78.48%** | 94.28% | 96.22% |
| LightGBM | 69.32% | **63.07%** | 53.47% | **63.07%** | **51.89%** | 74.61% | **95.72%** | **97.06%** |

Random Forest achieves higher overall accuracy and weighted F1, largely because
it performs better on the largest `Normal` and `Generic` classes. LightGBM has
better balanced accuracy, macro recall, macro F1, and multiclass ROC-AUC,
indicating stronger performance across the class distribution as a whole.

## Per-Class Metrics

### Random Forest

| Class | Precision | Recall | F1 score | Support |
|---|---:|---:|---:|---:|
| Normal | 95.97% | 76.90% | **85.38%** | 37,000 |
| Analysis | 2.07% | 2.66% | 2.33% | 677 |
| Backdoor | 3.47% | 29.85% | 6.22% | 583 |
| DoS | 29.07% | 21.67% | 24.83% | 4,089 |
| Exploits | 73.21% | 67.52% | 70.25% | 11,132 |
| Fuzzers | 30.05% | 57.64% | 39.51% | 6,062 |
| Generic | **99.90%** | 96.68% | **98.26%** | 18,871 |
| Reconnaissance | 92.94% | 79.49% | **85.69%** | 3,496 |
| Shellcode | 40.63% | 64.81% | 49.95% | 378 |
| Worms | 80.00% | 18.18% | 29.63% | 44 |

### LightGBM

| Class | Precision | Recall | F1 score | Support |
|---|---:|---:|---:|---:|
| Normal | **98.44%** | 62.90% | 76.76% | 37,000 |
| Analysis | 2.78% | 12.41% | 4.54% | 677 |
| Backdoor | 5.51% | **64.49%** | 10.15% | 583 |
| DoS | **40.31%** | 19.83% | **26.59%** | 4,089 |
| Exploits | **81.09%** | 61.96% | 70.24% | 11,132 |
| Fuzzers | 26.23% | **65.79%** | 37.51% | 6,062 |
| Generic | 99.82% | **97.10%** | **98.44%** | 18,871 |
| Reconnaissance | 87.67% | **84.21%** | **85.91%** | 3,496 |
| Shellcode | 25.50% | **91.53%** | 39.88% | 378 |
| Worms | 67.39% | **70.45%** | **68.89%** | 44 |

## Confusion Matrices

Rows are the actual classes and columns are the predicted classes. Class order
for both matrices is:

```text
Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers,
Generic, Reconnaissance, Shellcode, Worms
```

The complete matrices are retained in
[`multiclass_evaluation_metrics.json`](./multiclass_evaluation_metrics.json).
The main errors are confusion among the smaller attack families, especially
`Analysis`, `Backdoor`, and `DoS`. Both models classify `Generic` reliably.

### Random Forest

| Actual \ Predicted | Normal | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode | Worms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 28,452 | 284 | 0 | 14 | 691 | 7,418 | 6 | 28 | 107 | 0 |
| Analysis | 3 | 18 | 254 | 284 | 57 | 61 | 0 | 0 | 0 | 0 |
| Backdoor | 2 | 18 | 174 | 304 | 17 | 65 | 0 | 0 | 3 | 0 |
| DoS | 32 | 220 | 1,917 | 886 | 825 | 140 | 5 | 27 | 37 | 0 |
| Exploits | 106 | 244 | 1,932 | 837 | 7,516 | 294 | 5 | 136 | 60 | 2 |
| Fuzzers | 1,019 | 59 | 479 | 591 | 290 | 3,494 | 2 | 11 | 117 | 0 |
| Generic | 9 | 0 | 7 | 72 | 463 | 59 | 18,245 | 5 | 11 | 0 |
| Reconnaissance | 11 | 28 | 245 | 54 | 321 | 35 | 0 | 2,779 | 23 | 0 |
| Shellcode | 11 | 0 | 0 | 6 | 54 | 58 | 0 | 4 | 245 | 0 |
| Worms | 1 | 0 | 0 | 0 | 32 | 2 | 1 | 0 | 0 | 8 |

### LightGBM

| Actual \ Predicted | Normal | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode | Worms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal | 23,274 | 1,961 | 8 | 76 | 431 | 10,932 | 3 | 11 | 302 | 2 |
| Analysis | 32 | 84 | 438 | 100 | 23 | 0 | 0 | 0 | 0 | 0 |
| Backdoor | 1 | 83 | 376 | 108 | 7 | 1 | 0 | 0 | 7 | 0 |
| DoS | 20 | 289 | 2,312 | 811 | 517 | 43 | 8 | 32 | 56 | 1 |
| Exploits | 53 | 396 | 2,493 | 562 | 6,897 | 158 | 17 | 364 | 187 | 5 |
| Fuzzers | 253 | 175 | 889 | 211 | 174 | 3,988 | 3 | 5 | 364 | 0 |
| Generic | 3 | 10 | 20 | 113 | 318 | 49 | 18,324 | 0 | 29 | 5 |
| Reconnaissance | 4 | 28 | 287 | 27 | 125 | 14 | 1 | 2,944 | 65 | 1 |
| Shellcode | 3 | 0 | 1 | 4 | 4 | 17 | 0 | 2 | 346 | 1 |
| Worms | 0 | 0 | 0 | 0 | 9 | 2 | 1 | 0 | 1 | 31 |

## Conclusions

- **Random Forest** is preferable when overall accuracy and performance on the
  majority classes are the priority.
- **LightGBM** is preferable when balanced performance across attack categories
  and detection recall are more important.
- Both models struggle with `Analysis`, which has very low precision and F1.
- LightGBM substantially improves recall for `Backdoor`, `Shellcode`, and
  `Worms`, but this often reduces precision for those classes.
- The strong ROC-AUC values should be interpreted alongside the lower macro F1:
  ranking quality is high, but selecting the final class label remains
  difficult for minority classes.

# Autoencoder vs Isolation Forest Validation Metrics

This report compares the frozen validation metrics for the autoencoder and
Isolation Forest anomaly detectors. Both models were evaluated on the same
validation split:

- **Validation rows:** 35,069
- **Normal rows:** 11,200
- **Attack rows:** 23,869
- **Confusion-matrix order:** `[[true negative, false positive], [false negative, true positive]]`

## Overall ranking metrics

| Metric | Autoencoder | Isolation Forest | Difference (AE - IF) | Better result |
|---|---:|---:|---:|---|
| ROC-AUC | 0.911720 | 0.832244 | 0.079476 | Autoencoder |
| PR-AUC | 0.961543 | 0.905891 | 0.055652 | Autoencoder |

The autoencoder separates normal and attack traffic better on the
threshold-independent validation metrics, with ROC-AUC higher by 0.079476 and
PR-AUC higher by 0.055652.

## Operating-point comparison

The following tables compare metrics at the same false-positive-rate limits.
Small differences from the requested FPR limit are caused by selecting a
threshold from discrete validation scores.

### 5% false-positive-rate limit

| Metric | Autoencoder | Isolation Forest | Difference (AE - IF) | Better result |
|---|---:|---:|---:|---|
| False-positive rate | 0.050000 | 0.050000 | 0.000000 | Tie |
| Specificity | 0.950000 | 0.950000 | 0.000000 | Tie |
| Precision | 0.969516 | 0.940653 | 0.028863 | Autoencoder |
| Recall | 0.746156 | 0.371863 | 0.374293 | Autoencoder |
| F1 | 0.843296 | 0.533013 | 0.310283 | Autoencoder |
| Balanced accuracy | 0.848078 | 0.660932 | 0.187147 | Autoencoder |

| Confusion matrix | Autoencoder | Isolation Forest |
|---|---|---|
| Counts | `[[10640, 560], [6059, 17810]]` | `[[10640, 560], [14993, 8876]]` |

### 10% false-positive-rate limit — selected operating point

| Metric | Autoencoder | Isolation Forest | Difference (AE - IF) | Better result |
|---|---:|---:|---:|---|
| False-positive rate | 0.099554 | 0.099821 | -0.000268 | Autoencoder |
| Specificity | 0.900446 | 0.900179 | 0.000268 | Autoencoder |
| Precision | 0.944558 | 0.911635 | 0.032923 | Autoencoder |
| Recall | 0.795844 | 0.483221 | 0.312623 | Autoencoder |
| F1 | 0.863847 | 0.631637 | 0.232211 | Autoencoder |
| Balanced accuracy | 0.848145 | 0.691700 | 0.156445 | Autoencoder |

| Confusion matrix | Autoencoder | Isolation Forest |
|---|---|---|
| Counts | `[[10085, 1115], [4873, 18996]]` | `[[10082, 1118], [12335, 11534]]` |

The 10% FPR point is the selected operating point for both detectors. At
nearly identical false-positive rates, the autoencoder detects 7,462 more
attacks and improves recall by 0.312623.

### 15% false-positive-rate limit

| Metric | Autoencoder | Isolation Forest | Difference (AE - IF) | Better result |
|---|---:|---:|---:|---|
| False-positive rate | 0.150000 | 0.149911 | 0.000089 | Isolation Forest |
| Specificity | 0.850000 | 0.850089 | -0.000089 | Isolation Forest |
| Precision | 0.921421 | 0.890944 | 0.030476 | Autoencoder |
| Recall | 0.824836 | 0.575014 | 0.249822 | Autoencoder |
| F1 | 0.870458 | 0.698936 | 0.171521 | Autoencoder |
| Balanced accuracy | 0.837462 | 0.712507 | 0.124956 | Autoencoder |

| Confusion matrix | Autoencoder | Isolation Forest |
|---|---|---|
| Counts | `[[9521, 1679], [4181, 19688]]` | `[[9520, 1680], [10144, 13725]]` |

## Conclusion

The autoencoder is the stronger validation model across all three matched
false-positive-rate operating points. At the selected 10% FPR point, it
achieves:

- **Recall:** 0.795844 vs 0.483221 for Isolation Forest
- **F1:** 0.863847 vs 0.631637
- **Balanced accuracy:** 0.848145 vs 0.691700
- **Precision:** 0.944558 vs 0.911635

Isolation Forest does not outperform the autoencoder on any reported
validation metric at the matched operating points. These results support
using the autoencoder as the preferred standalone detector for this
validation split, while retaining Isolation Forest as a complementary model
candidate for fusion or additional analysis.

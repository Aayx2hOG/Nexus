# Baseline AE + LightGBM learned stacking

Selected on development data: logistic_C0.1. Six predeclared candidates; recall target 95%.

Meta-model training and threshold calibration use disjoint grouped halves of calibration data. Base detectors stay frozen.

```
         name    tn   fp   fn    tp   recall      fpr  precision       f1  accuracy  recovered  lost  added_fp  removed_fp
     LightGBM 30665 6335 1570 43762 0.965367 0.171216   0.873545 0.917164  0.903986          0     0         0           0
logistic_C0.1 30623 6377 1550 43782 0.965808 0.172351   0.872864 0.916987  0.903719         20     0        42           0
```

Exploratory only: selection/test previously used, one seed, known official-test overlap. The AE is the saved reconstructed baseline, not the unavailable historical checkpoint.

Reproduce: `.venv/bin/python training/experiment_stacked_ae_fusion.py --output-dir <new-directory>`.

Outcome: retain LightGBM for the minimum-FPR objective. Stacking selection FPR is 4.0522% versus 3.9790% for LightGBM; both meet 95% selection recall. On the reused official test, stacking recovers 20 attacks, loses zero, and adds 42 false positives. F1 decreases from 91.7164% to 91.6987%. No serving configuration was changed.

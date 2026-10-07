# Mode confidence — AE calibration FPR 5%

Load this directory with `nexus.fusion_model.load_fusion_model`. The package includes all fitted preprocessing and both base estimators. See `config.json` for exact cutoffs, `metrics.json` for test results, and `verification.json` for full-test prediction parity.

AE calibration FPR is not a system/test FPR guarantee. Input is a DataFrame of the 42 raw flow features; extra label/id columns are ignored.

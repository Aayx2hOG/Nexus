# Model research

The active workflow keeps LightGBM as the supervised baseline and measures the
incremental value of anomaly-assisted recovery.

| Workflow | Entry point | Instructions |
| --- | --- | --- |
| Controlled model/fusion experiments | `run_novelty_experiment.py` | [Complete command guide](FUSION_EXPERIMENTS.md) |
| Aggregate CSVs, seed statistics, six plots | `summarize_anomaly_results.py` | [Reporting commands](FUSION_EXPERIMENTS.md#aggregate-csvs-and-statistical-summaries) |
| Frozen binary checkpoint training/evaluation | `train_validated_lightgbm.py`, `evaluate_frozen_lightgbm.py` | [Validated training](VALIDATED_TRAINING.md) |
| Release export for the existing app | `export_release_bundle.py` | [Local demo](../docs/LOCAL_DEMO.md) |

Preprocessing runs inside the current experiment workflow on isolated fitting
rows. No separate whole-dataset preprocessing is required. Keep outputs under
Git-ignored `artifacts/`; every run must use a new directory. Model fitting,
calibration and evaluation run in the foreground with visible progress.

See the [current assessment](../reports/current_model_assessment.md) for the
eight-seed results and limitations. The optional attack-type classifier is a
conditional offline diagnostic, not a complete unknown-attack classifier.

## Historical code

Older training, tuning, preprocessing and official-test scripts remain for
reproduction and compatibility. The frozen validated runner imports some of
these helpers and records source hashes, so removing them would break existing
experiments. They are not the recommended entry point for new fusion research.
Use each script's `--help` and its existing frozen manifest when reproducing a run.

Historical generated reports were archived locally beneath
`artifacts/archive/cleanup_20261003/`. Earlier official-test results are historical
diagnostics, not independent confirmation of the new fusion policy.

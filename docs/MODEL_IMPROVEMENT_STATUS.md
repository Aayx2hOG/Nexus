# Current model and cleanup status

Reviewed 2026-10-05.

Keep serving `artifacts/bundles/v1.0.0/`. Its frozen source checkpoint is
`models/lightgbm_validated_v1/binary/model.joblib` (LightGBM trial 4).
The decision threshold is 0.5776925765603604.

Selection accuracy is 95.5056%, attack precision 98.0079%, attack recall
95.2548%, attack F1 96.6118%, and normal-traffic FPR 3.9790%.
The objective is minimum FPR subject to attack recall >=95%. These are
selection results, not independent test confirmation or production estimates.
See [the full result](../reports/binary_validated_result.md).

The historical overlap-filtered diagnostic gave 89.706% accuracy, 95.944%
recall, 17.012% FPR and 76.125% Fuzzers recall. It is a different population
and must not be compared directly with the selection metrics.

## Candidate outcome

TTL removal and Fuzzers weighting did not qualify on development selection.
Hard-normal plus Fuzzers 2x qualified on development but regressed historically:
33 fewer false alarms and 238 additional missed attacks. It was not promoted.
The eight-configuration robustness run completed; its saved summary recorded
`recommended_candidate: null`, with no eligible replacement. No new training
or predictive benchmark was run during cleanup.

## Cleanup

Removed obsolete standalone binary/multiclass LightGBM, Random Forest and
legacy autoencoder artifacts, including their associated JSON files. Removed
rejected binary-improvement, hard-normal and robustness candidate directories,
completed diagnostic outputs, the old report archive, superseded v1 plots,
six obsolete standalone evaluation/threshold scripts, and Python/tool caches.
This removed approximately 1.82 GB of local files.

Retained the complete frozen validated experiment, release bundle, active
fusion research and its evidence, source datasets, and regression tests.
The frozen Random Forest comparator and historical training helpers remain
because the manifest verifies their hashes and the frozen evaluator needs them.
Do not remove these files individually or rewrite the frozen manifest.

Model binaries and release bundles are Git-ignored. A GitHub push includes
source and documentation, not the local serving model. Copy the complete
release bundle separately when setting up another machine, as described in
[the local demo](LOCAL_DEMO.md).

## Verification after cleanup

Frozen experiment artifact/source hashes verified; the serving bundle loaded
successfully. Regression suite: 247 passed, 1 failed. The fusion training/reuse
test reports differing `partition_sha256` values despite matching predictions;
its runner already had local edits before cleanup and was left unchanged.
`git diff --check` passed.

# Recommended UNSW-NB15 training workflow

Run from the repository root:

```sh
.venv/bin/python training/train_validated_lightgbm.py \
  --task both --trials 16 --max-rounds 2500 --patience 100 \
  --rf-trees 300 --n-jobs 4 --minimum-recall 0.95 \
  --experiment-dir experiments/lightgbm_validated_new
```

The output directory must be new. Omit `--experiment-dir` to generate a timestamped
name. No preprocessing command is needed. The command reads only the official
**training CSV**, never the testing CSV or previous test metrics. `--task binary`
focuses on the challenge's main normal-versus-attack objective.

This runs 16 candidates **per task**, with an early-stopping fit followed by a
fixed-round refit: 64 LightGBM fits and two 300-tree RF fits for `--task both`.
There is no time cap; runtime depends on the machine and convergence. Use
`--trials 5` for just the initial weighting comparisons. Model artifacts, especially
Random Forest, can take substantial disk space. No score improvement is guaranteed.

## Evaluation foundation

- Predictors exclude `id`, `label`, and `attack_cat`; binary labels must agree with
  the family labels. This runner is for the UNSW-NB15 schema, not a generic loader
  for NSL-KDD or CICIDS2017.
- Identical predictor rows, including conflicting labels, stay together. Groups
  are stratified by their majority label and assigned to fitting (~57%), early
  stopping (~14%), calibration (~14%), and selection (~14%). Percentages refer
  approximately to groups; row fractions vary. Split indices, class counts,
  data/source hashes and library versions are recorded.
- Conflicting labels are retained and counted, not corrected using evaluation
  outcomes. Duplicates cannot leak between partitions, but still affect training
  and row-weighted metrics. Identical-row grouping is not session-level or
  time-based grouping. The available split files do not establish those properties.
- The initial encoder learns only from fitting rows. After selecting the number
  of rounds on early-stopping data, a new encoder and model fit on fitting plus
  early-stopping rows. Calibration and selection rows never enter either fit.
- RF trains on the exact same refit rows and one-hot vocabulary. Its numeric
  imputer fits on those rows only; LightGBM handles numeric missingness itself.
  RF uses 300 trees and balanced-subsample weights by default. It is a fixed
  baseline, not an equally tuned competitor.
- Both binary models select thresholds on calibration data, then are compared
  on selection data using minimum FPR subject to recall >=95%. A pointwise Wilson
  margin is used on calibration by default; this is not a recall guarantee under
  model selection or distribution shift. `--recall-confidence 0` disables it.
  An infeasible selection constraint is explicitly reported, never called a win.
- Multiclass selection maximizes macro-F1, with balanced accuracy and accuracy as
  tie-breakers. The search starts with the previous 0.55 weighting exponent and
  compares unweighted/soft/full class weights, then bounded randomized tree and
  regularization settings. Complete fixed-round refits avoid time-cap confounding.
- Threshold-bearing saved bundles preserve the binary decision in `predict()`.
  Their raw probability outputs are **not** claimed calibrated probabilities.

Inspect `validation_summary.json` for the paired comparison, `data_audit.json`
for duplicates/conflicts/counts, and each task's `selected_validation.json` and
`trials/` for all candidate details. The bundles are `<task>/model.joblib` and
`<task>/random_forest.joblib`. Training ends with a frozen manifest.

Validation results after searching are selection estimates, not independent final
scores. Do not describe previous official-test experiments as an untouched test.
Repeat predetermined seeds to assess split sensitivity if resources permit; do
not select the best seed and present its score as the overall result.

## Separate evaluation, after freezing choices

For a historical diagnostic on the existing official test file:

```sh
.venv/bin/python training/evaluate_frozen_lightgbm.py \
  --experiment-dir experiments/lightgbm_validated_v1 \
  --evaluation-csv 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv' \
  --data-role historical
```

This checks frozen artifact/source hashes, rejects incomplete experiments and
refuses to overwrite an existing `final_evaluation/`. Keep the matching code
version available for evaluation. It removes rows whose predictor hashes match
**any development partition**, then scores frozen RF and LightGBM without changing
thresholds or fitting anything. The exclusion count and remaining class counts
are reported. This estimates performance on the non-overlapping subset, **not**
the complete official benchmark. Internal evaluation duplicates are retained and
counted. If a required class disappears, that task is explicitly not scored
because its complete AUC comparison would be undefined.

Metrics include precision, recall, FPR, ROC-AUC, F1, confusion matrices, class
reports, binary average precision and false alerts per 1,000 benign flows.
Multiclass also reports macro/weighted F1 and normal-traffic false-positive rate.

For independent confirmation, supply genuinely unused, compatible-schema labeled
data and `--data-role fresh-holdout`. That flag records your assertion; software
cannot verify that humans have never inspected the data. New attack labels need
an open-set evaluation protocol, not this closed-set evaluator. Comparing against
older full-test scores after excluding overlaps is not a valid improvement claim.

## Source and artifact compatibility

The current runner imports helpers from `compare_lightgbm.py`,
`tune_lightgbm_fast.py`, `tune_lightgbm.py`, and the training utility modules.
These historical modules remain required. Saved bundles also depend on the
classes in `fast_lightgbm_models.py`.

Frozen manifests record the exact training source hashes. Preserve those sources
when evaluating an existing checkpoint; do not rewrite historical hashes to
bypass verification. Legacy training scripts remain for baseline reproduction
and source verification. Their command-line options are available through
`--help`; use this validated workflow for new model selection.

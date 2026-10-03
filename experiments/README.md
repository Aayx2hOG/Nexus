# LightGBM experiments

These are historical workflows and provenance retained for reproduction.
For current model research, start with the
[fusion experiment guide](../training/FUSION_EXPERIMENTS.md) and
[eight-seed assessment](../reports/current_model_assessment.md).

Run from the repository root with the existing `data/processed/` inputs and baseline
`models/lightgbm_metrics.json` and `models/multiclass_lightgbm_metrics.json`:

```sh
.venv/bin/python training/tune_lightgbm.py \
  --task both \
  --trials 30 \
  --max-rounds 3000 \
  --patience 100 \
  --minimum-recall 0.95 \
  --n-jobs -1 \
  --experiment-dir experiments/lightgbm_recall95
```

The output directory must not already exist. Omit `--experiment-dir` to create a
unique timestamped directory. Use `--task binary` or `--task multiclass` to run one
search. `--trials` is **per task**, so the command above fits 60 models. The first
three trials compare balanced, unweighted and square-root-balanced weighting;
remaining trials are reproducible random combinations of tree, regularization,
sampling and weighting parameters. For a broader search, set `--trials 60` before
running, rather than repeatedly selecting settings based on test results.

If dependencies are missing, install them into the project environment:

```sh
.venv/bin/python -m pip install -r requirements.txt
```

## Objectives and evaluation

- Binary: minimize validation false-positive rate subject to recall >= 95%.
  Ties prefer higher recall, then F1. Each trial searches all distinct probability
  thresholds, correctly handling tied scores. The threshold is saved on the
  estimator as `decision_threshold_` and in JSON. `training_utils` evaluation
  applies it automatically. **Direct calls to LightGBM's `predict()` do not use
  this custom attribute**; for inference use
  `model.predict_proba(X)[:, 1] >= model.decision_threshold_`.
- Multiclass: maximize validation macro F1, breaking ties by balanced accuracy
  and accuracy. This may trade overall accuracy for minority-class performance.
- The official training set is split into 70% fitting, 15% early stopping and 15%
  selection. Early stopping uses log loss, while trial selection uses the objectives
  above. Threshold selection does not reuse the early-stopping subset.
- Each selected model is frozen and evaluated once on the official test set.
  It is not refitted: refitting could change the probability scale and invalidate
  the selected threshold. Existing models and evaluation reports are not replaced.
- The 95% constraint is empirical on selection data, not a statistical guarantee
  on unseen traffic. A test result below 95% cannot qualify as a binary improvement.
- Existing preprocessing was fitted on the whole official training set, including
  internal validation rows. These are not fully fold-isolated preprocessing results.
  The runner does not perform a duplicate/group leakage audit. Small classes such
  as Worms have few validation examples, so class metrics remain uncertain.

## Outputs

At the experiment root:

- `manifest.json`: invocation, arguments, versions, data SHA-256 hashes and limitations.
- `summary.json`: selected test metrics and baseline comparison for both tasks.

In each `binary/` or `multiclass/` subfolder:

- `trials/trial_XXXX.json`: every trial's full validation metrics, confusion matrix,
  classification report, exact model/search arguments, early-stopping scores and rounds.
- `all_trial_metrics.json`: combined trial results.
- `split_indices.npz`, `split_class_counts.json`: reproducible split membership/counts.
- `baseline_metrics.json`: snapshot of the current baseline.
- `selected_validation.json`: selection recorded before test predictions.
- `selected_result.json`: selected arguments, validation and complete test metrics.
- `model.joblib`: frozen selected model, including the binary threshold attribute.
- `results.md`: complete comparison, metric changes, arguments and full test metrics.
- `outperforming_result.md`: written **only** if the selected test result has lower
  binary FPR with recall >= the requested minimum, or higher multiclass macro F1.
  These are objective-specific improvements, not claims that every metric improves.

Only validation metrics are computed for rejected trials; scoring every trial on
test data would encourage test-set selection. No winning result is fabricated if
none beats the baseline. An interrupted search retains completed trial JSON files;
restart with a fresh directory (automatic resume is not implemented).
# Leave-one-attack-family-out

`training/leave_one_attack_family_out.py` runs a leakage-controlled simulated
unseen-attack-family generalization experiment. By default it holds out
Fuzzers, Exploits, DoS, Reconnaissance, Analysis, Backdoor, and Shellcode one
at a time. Add `--include-generic` to run the larger optional Generic case.

```bash
python training/leave_one_attack_family_out.py
```

Results are written beneath `experiments/leave_one_attack_family_out/`; no
production model or preprocessing artifacts are loaded or overwritten.

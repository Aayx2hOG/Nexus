# Model training

The scripts in this directory train binary intrusion classifiers from the sparse
tree-model arrays created by the preprocessing pipeline.

From the repository root, install the training dependencies and run either model:

```sh
python3 -m pip install -r requirements.txt
python3 training/train_random_forest.py
python3 training/train_lightgbm.py
python3 training/train_isolation_forest.py
python3 training/evaluation_isolation_forest.py
python3 training/train_autoencoder.py
python3 training/evaluate_autoencoder.py
python3 training/compare_anomaly_detectors.py
python3 training/selective_fusion.py
python3 training/evaluate_models.py
```

### Controlled autoencoder optimization

The optimization is intentionally split into two commands. The first command
uses only the official training artifacts, runs 36 architecture/learning-rate/
batch-size candidates plus one controlled alpha sensitivity candidate, and
freezes the validation winner without opening the official test arrays:

```sh
python3 training/optimize_autoencoder.py
```

Only after that command completes, evaluate the single frozen winner once:

```sh
python3 training/evaluate_optimized_autoencoder.py
```

Both commands write only under `experiments/autoencoder_optimization/`; the
existing `models/autoencoder/` baseline is read for comparison and is never
overwritten. The search command validates the expected baseline metrics before
training and fails if they differ.

By default, both scripts read `data/processed/X_{train,test}_tree.npz` and
`data/processed/y_{train,test}.npy`. Fitted estimators and JSON evaluation
reports are written to `artifacts/models/` so generated models do not mix with
processed input data.

Use `--help` to see tuning options. For example:

```sh
python3 training/train_random_forest.py --n-estimators 500 --max-depth 30
python3 training/train_lightgbm.py --learning-rate 0.03 --num-leaves 63
```

LightGBM chooses the number of boosting rounds with a stratified validation
split taken only from the training set, then refits on all training rows. The
official test set is used once for the final metrics.

Isolation Forest is the unsupervised anomaly-detection baseline, not a known-
attack classifier. It is fitted only on normal traffic from an internal portion
of the official training split. Validation anomaly scores select a threshold;
the frozen threshold is then evaluated once on every row in the official test
split. Higher `-model.decision_function(X)` values mean more anomalous traffic.
An anomalous result indicates behavior different from learned normal traffic,
not proof of a zero-day attack or an instruction to block the traffic.

The autoencoder uses the same normal-only fit split and labelled validation
threshold protocol as Isolation Forest. Numeric features are standardized using
only the normal fit rows, while one-hot features are retained. Reconstruction
mean squared error is the anomaly score. After both detectors have been
evaluated, `compare_anomaly_detectors.py` writes a side-by-side metric table to
`reports/anomaly_detector_comparison.csv`.

`selective_fusion.py` runs a separate, leakage-free calibration experiment from
the raw official splits. It first makes a stratified 80/20 split of the official
training rows (random state 42). The imputer, one-hot encoder, fresh LightGBM,
fresh Random Forest, and fresh Isolation Forest are fitted using only the 80%
model-training subset; Isolation Forest sees only its normal rows. LightGBM's
boosting-round selection uses a further inner split of that 80%, followed by a
refit on the complete model-training subset.

Fusion candidates are evaluated only on the untouched 20% calibration subset.
After the rule is frozen, the official test set is transformed and scored once
for final reporting. Experimental models, preprocessing, split indices, the
pre-test frozen rule, candidate tables, and results are written under
`experiments/selective_fusion_leakage_free/`; production artifacts are neither
loaded nor overwritten. Use a new `--output-dir` to retain multiple runs.

The evaluation script loads both trained models and prints accuracy, balanced
accuracy, precision, recall, F1, false-positive rate, specificity, ROC-AUC, and
the confusion matrix. It also saves the complete classification report to
`reports/evaluation_metrics.json` by default. It checks `artifacts/models/`,
`models/`, and `data/processed/` in that order. To select a different report
location explicitly, use:

```sh
python3 training/evaluate_models.py --model-dir data/processed --output reports/custom_metrics.json
```

## Multiclass attack classification

The multiclass pipeline predicts `attack_cat` (Normal plus each attack family)
while reusing the same leakage-free feature matrices. Prepare the targets, train
both models, and evaluate them with:

```sh
python3 preprocessing/prepare_multiclass_targets.py
python3 training/train_multiclass_random_forest.py
python3 training/train_multiclass_lightgbm.py
python3 training/evaluate_multiclass_models.py
```

Target preparation writes `y_{train,test}_multiclass.npy` and
`multiclass_labels.json` to `data/processed/`. The fitted models are named
`multiclass_random_forest.joblib` and `multiclass_lightgbm.joblib`, so they do
not overwrite the binary models. The combined report defaults to
`reports/multiclass_evaluation_metrics.json` and includes macro and weighted
metrics, one-vs-rest multiclass ROC-AUC, per-class metrics, and confusion
matrices.

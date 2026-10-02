# Model training

The scripts in this directory train binary intrusion classifiers from the sparse
tree-model arrays created by the preprocessing pipeline.

From the repository root, install the training dependencies and run either model:

```sh
python3 -m pip install -r requirements.txt
python3 training/train_random_forest.py
python3 training/train_lightgbm.py
python3 training/evaluate_models.py
```

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

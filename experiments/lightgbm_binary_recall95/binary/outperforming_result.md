# Binary LightGBM experiment

Objective: Lower false-positive rate with recall >= 95.0%.

**Outperforms current baseline: YES.**

Selection used validation only. Test results below did not select the model.
The recall constraint is empirical on validation; test/deployment recall may differ.
Models are saved without refitting, preserving the validated probability scale.

| Metric | Current baseline | Selected test result | Change |
|---|---:|---:|---:|
| accuracy | 0.901096 | 0.903853 | +0.002757 |
| balanced_accuracy | 0.893308 | 0.897654 | +0.004347 |
| precision | 0.866185 | 0.877771 | +0.011587 |
| recall | 0.970264 | 0.958903 | -0.011361 |
| f1 | 0.915275 | 0.916545 | +0.001270 |
| false_positive_rate | 0.183649 | 0.163595 | -0.020054 |
| specificity | 0.816351 | 0.836405 | +0.020054 |
| roc_auc | 0.984217 | 0.982302 | -0.001916 |

## Selected model and exact arguments

```json
{
  "trial_id": 17,
  "task": "binary",
  "search_arguments": {
    "weight_alpha": 0.0,
    "subsample": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "num_leaves": 63,
    "min_child_samples": 100,
    "max_depth": -1,
    "learning_rate": 0.03,
    "colsample_bytree": 0.85
  },
  "model_parameters": {
    "boosting_type": "gbdt",
    "class_weight": null,
    "colsample_bytree": 0.85,
    "importance_type": "split",
    "learning_rate": 0.03,
    "max_depth": -1,
    "min_child_samples": 100,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 3000,
    "n_jobs": -1,
    "num_leaves": 63,
    "objective": "binary",
    "random_state": 42,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.85,
    "subsample_for_bin": 200000,
    "subsample_freq": 1,
    "verbosity": -1
  },
  "best_iteration": 1539,
  "decision_threshold": 0.7215475587378909,
  "validation_metrics": {
    "decision_threshold": 0.7215475587378909,
    "accuracy": 0.9555547106683903,
    "balanced_accuracy": 0.9585668932973703,
    "precision": 0.9839204118225461,
    "recall": 0.9502290246899787,
    "f1": 0.9667812793043676,
    "false_positive_rate": 0.033095238095238094,
    "specificity": 0.9669047619047619,
    "roc_auc": 0.9947274812338074,
    "confusion_matrix": [
      [
        8122,
        278
      ],
      [
        891,
        17011
      ]
    ],
    "classification_report": {
      "normal": {
        "precision": 0.9011427937423722,
        "recall": 0.9669047619047619,
        "f1-score": 0.9328662493539309,
        "support": 8400.0
      },
      "attack": {
        "precision": 0.9839204118225461,
        "recall": 0.9502290246899787,
        "f1-score": 0.9667812793043676,
        "support": 17902.0
      },
      "accuracy": 0.9555547106683903,
      "macro avg": {
        "precision": 0.9425316027824591,
        "recall": 0.9585668932973703,
        "f1-score": 0.9498237643291493,
        "support": 26302.0
      },
      "weighted avg": {
        "precision": 0.9574839434219126,
        "recall": 0.9555547106683903,
        "f1-score": 0.955949926115117,
        "support": 26302.0
      }
    }
  },
  "early_stopping_scores": {
    "valid_0": {
      "binary_logloss": 0.08525942420236012
    }
  },
  "fit_rows": 122738,
  "early_stopping_rows": 26301,
  "selection_rows": 26302
}
```

## Complete test metrics

```json
{
  "decision_threshold": 0.7215475587378909,
  "accuracy": 0.9038526939707525,
  "balanced_accuracy": 0.8976543042203944,
  "precision": 0.8777714954969509,
  "recall": 0.9589032030353833,
  "f1": 0.9165454277099543,
  "false_positive_rate": 0.1635945945945946,
  "specificity": 0.8364054054054054,
  "roc_auc": 0.9823015652089927,
  "confusion_matrix": [
    [
      30947,
      6053
    ],
    [
      1863,
      43469
    ]
  ],
  "classification_report": {
    "normal": {
      "precision": 0.9432185309356903,
      "recall": 0.8364054054054054,
      "f1-score": 0.8866065033662799,
      "support": 37000.0
    },
    "attack": {
      "precision": 0.8777714954969509,
      "recall": 0.9589032030353833,
      "f1-score": 0.9165454277099543,
      "support": 45332.0
    },
    "accuracy": 0.9038526939707525,
    "macro avg": {
      "precision": 0.9104950132163205,
      "recall": 0.8976543042203944,
      "f1-score": 0.901575965538117,
      "support": 82332.0
    },
    "weighted avg": {
      "precision": 0.9071833925871874,
      "recall": 0.9038526939707525,
      "f1-score": 0.9030908754008163,
      "support": 82332.0
    }
  }
}
```

## Run arguments

```json
{
  "task": "binary",
  "data_dir": "/home/aayush/projects/Nexus/data/processed",
  "baseline_dir": "/home/aayush/projects/Nexus/models",
  "experiment_dir": "experiments/lightgbm_binary_recall95",
  "trials": 30,
  "max_rounds": 3000,
  "patience": 100,
  "minimum_recall": 0.95,
  "selection_size": 0.15,
  "early_stopping_size": 0.15,
  "seed": 42,
  "n_jobs": -1
}
```

See manifest.json for data hashes, versions, command and preprocessing limitations.
See split_indices.npz and trials/ for split membership and every validation result.

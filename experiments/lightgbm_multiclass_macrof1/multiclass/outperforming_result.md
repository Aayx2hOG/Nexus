# Multiclass LightGBM experiment

Objective: Higher macro F1.

**Outperforms current baseline: YES.**

Selection used validation only. Test results below did not select the model.
The recall constraint is empirical on validation; test/deployment recall may differ.
Models are saved without refitting, preserving the validated probability scale.

| Metric | Current baseline | Selected test result | Change |
|---|---:|---:|---:|
| accuracy | 0.692222 | 0.727105 | +0.034883 |
| balanced_accuracy | 0.632245 | 0.593627 | -0.038618 |
| precision_macro | 0.534869 | 0.522742 | -0.012127 |
| recall_macro | 0.632245 | 0.593627 | -0.038618 |
| f1_macro | 0.518255 | 0.525171 | +0.006916 |
| f1_weighted | 0.745300 | 0.766547 | +0.021248 |
| roc_auc_ovr_macro | 0.957146 | 0.961300 | +0.004154 |
| roc_auc_ovr_weighted | 0.970391 | 0.973028 | +0.002637 |

## Selected model and exact arguments

```json
{
  "trial_id": 3,
  "task": "multiclass",
  "search_arguments": {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "weight_alpha": 0.5
  },
  "model_parameters": {
    "boosting_type": "gbdt",
    "class_weight": {
      "0": 0.6742271214178339,
      "1": 3.567674580893078,
      "2": 3.818683362056719,
      "3": 1.4407183981479141,
      "4": 0.8731190063194053,
      "5": 1.18318351116239,
      "6": 0.7977562884474996,
      "7": 1.557802827758284,
      "8": 4.74037455504117,
      "9": 13.99357100470982
    },
    "colsample_bytree": 0.8,
    "importance_type": "split",
    "learning_rate": 0.05,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "n_estimators": 3000,
    "n_jobs": -1,
    "num_leaves": 31,
    "objective": "multiclass",
    "random_state": 42,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.8,
    "subsample_for_bin": 200000,
    "subsample_freq": 1,
    "verbosity": -1
  },
  "best_iteration": 296,
  "decision_threshold": null,
  "validation_metrics": {
    "accuracy": 0.8178845715154741,
    "balanced_accuracy": 0.6633479379466882,
    "precision_macro": 0.7055295955262586,
    "recall_macro": 0.6633479379466882,
    "f1_macro": 0.6552016914124789,
    "f1_weighted": 0.8271906759907753,
    "roc_auc_ovr_macro": 0.9644340795287569,
    "roc_auc_ovr_weighted": 0.9784529579622497,
    "class_names": [
      "Normal",
      "Analysis",
      "Backdoor",
      "DoS",
      "Exploits",
      "Fuzzers",
      "Generic",
      "Reconnaissance",
      "Shellcode",
      "Worms"
    ],
    "confusion_matrix": [
      [
        7491,
        52,
        0,
        5,
        47,
        792,
        1,
        3,
        9,
        0
      ],
      [
        18,
        67,
        13,
        180,
        22,
        0,
        0,
        0,
        0,
        0
      ],
      [
        1,
        14,
        32,
        174,
        38,
        0,
        0,
        1,
        2,
        0
      ],
      [
        5,
        6,
        1,
        1352,
        436,
        14,
        4,
        8,
        14,
        0
      ],
      [
        11,
        16,
        6,
        1741,
        3022,
        59,
        5,
        115,
        29,
        5
      ],
      [
        144,
        0,
        0,
        179,
        64,
        2312,
        0,
        4,
        25,
        0
      ],
      [
        1,
        0,
        2,
        44,
        56,
        5,
        5890,
        0,
        2,
        0
      ],
      [
        1,
        1,
        0,
        241,
        126,
        4,
        0,
        1200,
        1,
        0
      ],
      [
        0,
        0,
        0,
        4,
        17,
        12,
        0,
        3,
        133,
        1
      ],
      [
        0,
        0,
        0,
        1,
        3,
        2,
        0,
        0,
        0,
        13
      ]
    ],
    "classification_report": {
      "Normal": {
        "precision": 0.97640771637122,
        "recall": 0.8917857142857143,
        "f1-score": 0.9321801891488303,
        "support": 8400.0
      },
      "Analysis": {
        "precision": 0.42948717948717946,
        "recall": 0.22333333333333333,
        "f1-score": 0.29385964912280704,
        "support": 300.0
      },
      "Backdoor": {
        "precision": 0.5925925925925926,
        "recall": 0.12213740458015267,
        "f1-score": 0.20253164556962025,
        "support": 262.0
      },
      "DoS": {
        "precision": 0.3448099974496302,
        "recall": 0.7347826086956522,
        "f1-score": 0.46936295781982296,
        "support": 1840.0
      },
      "Exploits": {
        "precision": 0.7888279822500652,
        "recall": 0.6033140347374726,
        "f1-score": 0.683710407239819,
        "support": 5009.0
      },
      "Fuzzers": {
        "precision": 0.7225,
        "recall": 0.8475073313782991,
        "f1-score": 0.7800269905533064,
        "support": 2728.0
      },
      "Generic": {
        "precision": 0.9983050847457627,
        "recall": 0.9816666666666667,
        "f1-score": 0.9899159663865547,
        "support": 6000.0
      },
      "Reconnaissance": {
        "precision": 0.8995502248875562,
        "recall": 0.7623888182973316,
        "f1-score": 0.8253094910591472,
        "support": 1574.0
      },
      "Shellcode": {
        "precision": 0.6186046511627907,
        "recall": 0.7823529411764706,
        "f1-score": 0.6909090909090909,
        "support": 170.0
      },
      "Worms": {
        "precision": 0.6842105263157895,
        "recall": 0.6842105263157895,
        "f1-score": 0.6842105263157895,
        "support": 19.0
      },
      "accuracy": 0.8178845715154741,
      "macro avg": {
        "precision": 0.7055295955262586,
        "recall": 0.6633479379466882,
        "f1-score": 0.6552016914124789,
        "support": 26302.0
      },
      "weighted avg": {
        "precision": 0.8579760224380968,
        "recall": 0.8178845715154741,
        "f1-score": 0.8271906759907753,
        "support": 26302.0
      }
    }
  },
  "early_stopping_scores": {
    "valid_0": {
      "multi_logloss": 0.4400546337435839
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
  "accuracy": 0.7271048923869212,
  "balanced_accuracy": 0.5936272478610067,
  "precision_macro": 0.522742197692394,
  "recall_macro": 0.5936272478610067,
  "f1_macro": 0.5251706244049551,
  "f1_weighted": 0.7665471891691793,
  "roc_auc_ovr_macro": 0.9613001355240552,
  "roc_auc_ovr_weighted": 0.9730277168206992,
  "class_names": [
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms"
  ],
  "confusion_matrix": [
    [
      25103,
      1302,
      0,
      62,
      631,
      9643,
      3,
      11,
      245,
      0
    ],
    [
      28,
      49,
      186,
      296,
      118,
      0,
      0,
      0,
      0,
      0
    ],
    [
      2,
      49,
      166,
      254,
      103,
      1,
      0,
      0,
      8,
      0
    ],
    [
      28,
      391,
      911,
      1551,
      1059,
      54,
      8,
      37,
      49,
      1
    ],
    [
      63,
      439,
      1032,
      1422,
      7593,
      147,
      15,
      278,
      138,
      5
    ],
    [
      459,
      138,
      351,
      565,
      422,
      3856,
      2,
      8,
      260,
      1
    ],
    [
      7,
      0,
      6,
      67,
      391,
      51,
      18312,
      3,
      29,
      5
    ],
    [
      5,
      45,
      113,
      138,
      239,
      14,
      1,
      2884,
      56,
      1
    ],
    [
      3,
      0,
      2,
      2,
      16,
      26,
      0,
      2,
      326,
      1
    ],
    [
      0,
      0,
      0,
      1,
      16,
      0,
      1,
      0,
      2,
      24
    ]
  ],
  "classification_report": {
    "Normal": {
      "precision": 0.9768464471943342,
      "recall": 0.6784594594594595,
      "f1-score": 0.8007591948706497,
      "support": 37000.0
    },
    "Analysis": {
      "precision": 0.020306672192291753,
      "recall": 0.0723781388478582,
      "f1-score": 0.03171521035598705,
      "support": 677.0
    },
    "Backdoor": {
      "precision": 0.059992771955186125,
      "recall": 0.2847341337907376,
      "f1-score": 0.0991044776119403,
      "support": 583.0
    },
    "DoS": {
      "precision": 0.35589720055071133,
      "recall": 0.3793103448275862,
      "f1-score": 0.367230969574997,
      "support": 4089.0
    },
    "Exploits": {
      "precision": 0.7171326029467322,
      "recall": 0.6820876751706791,
      "f1-score": 0.699171270718232,
      "support": 11132.0
    },
    "Fuzzers": {
      "precision": 0.27958236658932717,
      "recall": 0.6360936984493566,
      "f1-score": 0.3884355797320439,
      "support": 6062.0
    },
    "Generic": {
      "precision": 0.9983644095518482,
      "recall": 0.9703778284139685,
      "f1-score": 0.9841721978878349,
      "support": 18871.0
    },
    "Reconnaissance": {
      "precision": 0.8948184920881167,
      "recall": 0.8249427917620137,
      "f1-score": 0.8584610805179342,
      "support": 3496.0
    },
    "Shellcode": {
      "precision": 0.29290206648697215,
      "recall": 0.8624338624338624,
      "f1-score": 0.43729040912139505,
      "support": 378.0
    },
    "Worms": {
      "precision": 0.631578947368421,
      "recall": 0.5454545454545454,
      "f1-score": 0.5853658536585366,
      "support": 44.0
    },
    "accuracy": 0.7271048923869212,
    "macro avg": {
      "precision": 0.522742197692394,
      "recall": 0.5936272478610067,
      "f1-score": 0.5251706244049551,
      "support": 82332.0
    },
    "weighted avg": {
      "precision": 0.8433194957057167,
      "recall": 0.7271048923869212,
      "f1-score": 0.7665471891691793,
      "support": 82332.0
    }
  }
}
```

## Run arguments

```json
{
  "task": "multiclass",
  "data_dir": "/home/aayush/projects/Nexus/data/processed",
  "baseline_dir": "/home/aayush/projects/Nexus/models",
  "experiment_dir": "experiments/lightgbm_multiclass_macrof1",
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

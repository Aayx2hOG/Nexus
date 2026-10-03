"""Evaluate frozen LightGBM and baseline AE; write root working.md. No fitting."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from train_autoencoder import make_autoencoder_features, reconstruction_errors
from train_validated_lightgbm import canonical_features
from tune_lightgbm_fast import read_data

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = ROOT / 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv'
    model_path = ROOT / 'models/lightgbm_validated_v1/binary/model.joblib'
    ae_path = ROOT / 'models/autoencoder/autoencoder.joblib'
    config_path = ae_path.with_name('autoencoder_config.json')
    preprocessor_path = ROOT / 'data/processed/preprocessor.joblib'
    raw = pd.read_csv(raw_path)
    x, multi, _ = read_data(raw_path)
    x = canonical_features(x)
    y = raw.label.to_numpy(dtype=int)
    np.testing.assert_array_equal(y, (multi > 0).astype(int))
    np.testing.assert_array_equal(y, np.load(ROOT / 'data/processed/y_train.npy'))
    model = joblib.load(model_path)
    assert list(x.columns) == model.features.native.columns
    np.testing.assert_array_equal(model.classes_, [0, 1])
    manifest = json.loads((model_path.parent.parent / 'manifest.json').read_text())
    assert digest(model_path) == manifest['artifact_hashes']['binary/model.joblib']
    p = model.predict(x).astype(int)
    np.testing.assert_array_equal(p, (model.predict_proba(x)[:, 1] >= model.decision_threshold_).astype(int))

    # Rebuild AE inputs from these exact raw rows with the saved preprocessing.
    prep = joblib.load(preprocessor_path)
    numeric = prep['numeric_imputer'].transform(raw[prep['numerical_columns']]).astype(np.float32)
    categorical = raw[prep['categorical_columns']].fillna('__MISSING__').astype(str)
    tree = sparse.hstack([sparse.csr_matrix(numeric), prep['tree_encoder'].transform(categorical)], format='csr')
    cached = sparse.load_npz(ROOT / 'data/processed/X_train_tree.npz')
    assert tree.shape == cached.shape and (tree != cached).nnz == 0
    ae = joblib.load(ae_path)
    config = json.loads(config_path.read_text())
    features = make_autoencoder_features(tree, config['numerical_feature_count'], ae['numeric_scaler'])
    assert features.shape[1] == config['feature_count']
    with threadpool_limits(limits=4):
        scores = reconstruction_errors(ae['model'], features, 2048)
    assert np.isfinite(scores).all()
    a = (scores >= config['selected_threshold']).astype(int)
    predictions = {'LightGBM': p, 'Baseline AE': a, 'Combined OR': p | a}
    results = {}
    for name, pred in predictions.items():
        cm = confusion_matrix(y, pred, labels=[0, 1])
        assert cm.sum() == len(y)
        results[name] = {'confusion_matrix': cm.tolist(), 'classification_report': classification_report(y, pred, labels=[0, 1], target_names=['normal', 'attack'], output_dict=True, zero_division=0)}
    record = {
        'evaluated_at': datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),
        'data_role': 'full_official_training_development_data',
        'artifacts': {str(path.relative_to(ROOT)): digest(path) for path in [raw_path, model_path, ae_path, config_path, preprocessor_path]},
        'rows': len(y), 'normal_rows': int((y == 0).sum()), 'attack_rows': int((y == 1).sum()),
        'thresholds': {'lightgbm': float(model.decision_threshold_), 'baseline_ae': config['selected_threshold']},
        'fusion_rule': 'attack if LightGBM OR baseline AE predicts attack',
        'checks': 'LightGBM manifest hash and feature schema verified; raw labels and all AE input rows match cached training data.',
        'results': results,
    }
    output = ROOT / 'reports/official_training_lightgbm_ae_evaluation.json'
    output.write_text(json.dumps(record, indent=2) + '\n')
    lines = ['# Official training data evaluation', '', f"Evaluated: {record['evaluated_at']}.", '',
        f'Dataset: `{raw_path.relative_to(ROOT)}`. All {len(y):,} rows: {(y == 0).sum():,} normal and {(y == 1).sum():,} attack. Labels: 0 = normal, 1 = attack.', '',
        '**Scope:** These are development-data metrics, not independent holdout results. This split includes fitting and validation/threshold-selection data. Models and thresholds were kept frozen; no retraining or tuning was performed.', '',
        f"LightGBM: `{model_path.relative_to(ROOT)}`, attack probability threshold `>= {model.decision_threshold_}`.", '',
        f"Baseline AE: `{ae_path.relative_to(ROOT)}`, mean squared reconstruction error threshold `>= {config['selected_threshold']}`. Inputs use the saved preprocessor and AE numeric scaler.", '',
        '**Combined rule:** predict attack if either model predicts attack (OR).', '',
        '## Comparison', '', '| Model | Accuracy | Attack precision | Attack recall | Attack F1 | Normal FPR |', '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        report = r['classification_report']; attack = report['attack']; cm = r['confusion_matrix']
        lines.append(f"| {name} | {report['accuracy']:.6f} | {attack['precision']:.6f} | {attack['recall']:.6f} | {attack['f1-score']:.6f} | {cm[0][1] / sum(cm[0]):.6f} |")
    for name, pred in predictions.items():
        cm = results[name]['confusion_matrix']
        lines += ['', f'## {name}', '', 'Confusion matrix: rows = actual, columns = predicted.', '', '| Actual / Predicted | Normal | Attack |', '|---|---:|---:|', f'| Normal | {cm[0][0]} | {cm[0][1]} |', f'| Attack | {cm[1][0]} | {cm[1][1]} |', '', 'Classification report:', '', '```text', classification_report(y, pred, labels=[0, 1], target_names=['normal', 'attack'], digits=6, zero_division=0).rstrip(), '```']
    rescued = int(((y == 1) & (p == 0) & (a == 1)).sum())
    added_fp = int(((y == 0) & (p == 0) & (a == 1)).sum())
    lines += ['', f'OR fusion detects {rescued:,} additional attacks and adds {added_fp:,} false positives compared with LightGBM alone.', '', '## Reproducibility', '', 'Run from the repository root:', '', '```sh', '.venv/bin/python training/evaluate_official_training_models.py', '```', '', record['checks'], '', 'Full metrics and SHA-256 artifact fingerprints: [evaluation JSON](reports/official_training_lightgbm_ae_evaluation.json).', '']
    (ROOT / 'working.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

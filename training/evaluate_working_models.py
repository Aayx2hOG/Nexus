"""Recompute working.md metrics from frozen model.joblib and saved baseline AE scores.

Run from the repository root. Writes the JSON evaluation record; does not tune models.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
sys.path.insert(0, str(Path('training').resolve()))
from tune_lightgbm_fast import read_data
from train_validated_lightgbm import canonical_features

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(y, p, s=None):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    r = {k: float(f(y, p)) for k, f in [('accuracy', accuracy_score), ('balanced_accuracy', balanced_accuracy_score), ('precision', precision_score), ('recall', recall_score), ('f1', f1_score)]}
    r.update(specificity=float(tn / (tn + fp)), false_positive_rate=float(fp / (tn + fp)), confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]], classification_report=classification_report(y, p, target_names=['normal', 'attack'], output_dict=True))
    if s is not None:
        r.update(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))
    return r
model_path = 'models/lightgbm_validated_v1/binary/model.joblib'
raw_path = 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv'
ae_path = 'models/autoencoder/autoencoder_test_scores.csv'
m = joblib.load(model_path)
x, multi, names = read_data(raw_path)
x = canonical_features(x)
y = (multi > 0).astype(int)
assert list(x.columns) == m.features.native.columns
assert np.array_equal(m.classes_, [0, 1])
assert np.array_equal(y, np.load('data/processed/y_test.npy'))
s = m.predict_proba(x)[:, 1]
p = m.predict(x)
assert np.array_equal(p, (s >= m.decision_threshold_).astype(int))
ae = pd.read_csv(ae_path)
raw = pd.read_csv(raw_path)
assert np.array_equal(ae.true_label, y)
assert np.array_equal(ae.attack_cat.astype(str).str.strip(), raw.attack_cat.astype(str).str.strip())
a = ae.anomaly_prediction.to_numpy(dtype=int)
at = json.loads(Path('models/autoencoder/autoencoder_config.json').read_text())
threshold = float(at['selected_threshold'])
assert np.array_equal(a, (ae.anomaly_score >= threshold).astype(int))
c = p | a
results = {'lightgbm': metrics(y, p, s), 'baseline_autoencoder': metrics(y, a, ae.anomaly_score), 'combined_or': metrics(y, c)}
manifest = json.loads(Path('models/lightgbm_validated_v1/manifest.json').read_text())
sha = digest(model_path)
expected = manifest['artifact_hashes'].get('binary/model.joblib')
overlap = {label: {f'lightgbm_{i}_ae_{j}': int(np.sum((y == value) & (p == i) & (a == j))) for i in (0, 1) for j in (0, 1)} for label, value in [('actual_normal', 0), ('actual_attacks', 1)]}
r = {'evaluation_date': datetime.now(ZoneInfo('Asia/Calcutta')).date().isoformat(), 'data_role': 'full_official_test_historical', 'evaluation_command': '.venv/bin/python training/evaluate_working_models.py', 'dataset': {'raw_test_path': raw_path, 'raw_test_sha256': digest(raw_path), 'test_records': len(y), 'normal_records': int(sum(y == 0)), 'attack_records': int(sum(y == 1)), 'raw_and_processed_labels_aligned': True, 'ae_labels_and_attack_categories_aligned': True}, 'lightgbm_artifact': {'path': model_path, 'sha256': sha, 'raw_feature_count': x.shape[1], 'transformed_feature_count': m.estimator.n_features_in_, 'decision_threshold': m.decision_threshold_, 'validated_manifest_expected_sha256': expected, 'matches_validated_manifest': sha == expected}, 'baseline_ae_artifact': {'scores_path': ae_path, 'scores_sha256': digest(ae_path), 'decision_threshold': threshold}, 'fusion_rule': 'combined_attack = lightgbm_attack OR baseline_ae_attack', 'results': results, 'overlap': overlap}
Path('reports/working_lightgbm_ae_evaluation.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps(r, indent=2))

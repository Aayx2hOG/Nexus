"""Frozen baseline AE + LightGBM stacking; no base-detector retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from compare_working_v2_fusion import metrics
from train_autoencoder import make_autoencoder_features, reconstruction_errors
from train_validated_lightgbm import canonical_features, load_split_indices

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/working_v2_baseline_fusion_20261005'
FROZEN = ROOT / 'experiments/lightgbm_validated_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(p, a, interaction):
    logit = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))
    error = np.log1p(a)
    columns = [logit, error]
    if interaction:
        columns.append(logit * error)
    return np.column_stack(columns)


def threshold(y, score):
    fpr, recall, cutoffs = roc_curve(y, score, drop_intermediate=False)
    eligible = np.flatnonzero(recall >= .95)
    best = min(eligible, key=lambda i: (fpr[i], -recall[i], -cutoffs[i]))
    return float(cutoffs[best])


def main(out):
    out.mkdir(exist_ok=False)
    def dump(name, value):
        (out/name).write_text(json.dumps(value, indent=2) + '\n')
    source_hashes = json.loads((SOURCE/'artifact_hashes.json').read_text())
    for name, expected in source_hashes.items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    manifest = json.loads((FROZEN/'manifest.json').read_text())
    for name, expected in manifest['artifact_hashes'].items():
        if digest(FROZEN/name) != expected:
            raise ValueError(f'Frozen hash mismatch: {name}')
    path = ROOT/'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv'
    assert digest(path) == manifest['train_sha256']
    raw = pd.read_csv(path)
    x = canonical_features(raw.drop(columns=['id', 'label', 'attack_cat']))
    y = raw.label.to_numpy(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    split = load_split_indices(FROZEN/'split_indices.npz', y, groups)
    cal = split['calibration']
    left, right = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=2026).split(cal, groups=groups[cal]))
    fit, tune, sel = cal[left], cal[right], split['selection']
    assert not set(groups[fit]) & set(groups[tune])
    np.savez_compressed(out/'split_indices.npz', meta_fit=fit, threshold=tune, selection=sel)
    dump('protocol.json', dict(methods=['logistic', 'logistic_interaction'], C=[.1, 1., 10.], seed=2026,
        objective='Minimum selection FPR subject to recall >= 0.95; tie higher recall then F1',
        meta_fit='Grouped half of existing calibration partition', threshold='Other grouped half',
        base_detectors_retrained=False, source=str(SOURCE.relative_to(ROOT)),
        limitations=['Previously used selection and official test; exploratory development comparison.',
                    'Reconstructed baseline-architecture AE, not missing historical AE.',
                    'Single seed; official test has known training predictor overlap.']))
    ae = joblib.load(SOURCE/'autoencoder.joblib')
    lgb = joblib.load(FROZEN/'binary/model.joblib')
    subset = raw.iloc[cal]
    nums, cats = ae['numerical_columns'], ae['categorical_columns']
    tree = sparse.hstack([sparse.csr_matrix(ae['imputer'].transform(subset[nums]).astype(np.float32)),
        ae['encoder'].transform(subset[cats].fillna('__MISSING__').astype(str))], format='csr')
    a = reconstruction_errors(ae['model'], make_autoencoder_features(tree, len(nums), ae['numeric_scaler']), 2048)
    p = lgb.predict_proba(x.iloc[cal])[:, 1]
    saved = np.load(SOURCE/'scores.npz')
    sy, sp, sa = [saved['selection_'+k] for k in ['y', 'lightgbm', 'ae']]
    assert np.array_equal(sy, y[sel])
    assert np.allclose(sp, lgb.predict_proba(x.iloc[sel])[:, 1])
    assert np.isfinite(a).all() and np.isfinite(p).all()
    rows, models = [], {}
    for interaction in [False, True]:
        for c in [.1, 1., 10.]:
            name = f'logistic{"_interaction" if interaction else ""}_C{c}'
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=2026))
            model.fit(features(p[left], a[left], interaction), y[fit])
            cutoff = threshold(y[tune], model.predict_proba(features(p[right], a[right], interaction))[:, 1])
            pred = model.predict_proba(features(sp, sa, interaction))[:, 1] >= cutoff
            rows.append(dict(name=name, interaction=interaction, C=c, threshold=cutoff, **metrics(sy, pred)))
            models[name] = model
    pd.DataFrame(rows).to_csv(out/'selection.csv', index=False)
    eligible = [r for r in rows if r['recall'] >= .95]
    winner = min(eligible, key=lambda r: (r['fpr'], -r['recall'], -r['f1'])) if eligible else None
    dump('frozen_selection.json', dict(winner=winner, baseline=metrics(sy, sp >= lgb.decision_threshold_)))
    if winner is None:
        print('No stacking candidate meets selection recall >=95%.')
        return
    joblib.dump(dict(model=models[winner['name']], rule=winner), out/'stacker.joblib')
    # Access test arrays only after freezing the selection winner.
    ty, tp, ta = [saved['test_'+k] for k in ['y', 'lightgbm', 'ae']]
    baseline = tp >= lgb.decision_threshold_
    stacked = models[winner['name']].predict_proba(features(tp, ta, winner['interaction']))[:, 1] >= winner['threshold']
    results = []
    for name, pred in [('LightGBM', baseline), (winner['name'], stacked)]:
        results.append(dict(name=name, **metrics(ty, pred), recovered=int(((ty==1)&~baseline&pred).sum()),
            lost=int(((ty==1)&baseline&~pred).sum()), added_fp=int(((ty==0)&~baseline&pred).sum()),
            removed_fp=int(((ty==0)&baseline&~pred).sum())))
    frame = pd.DataFrame(results)
    frame.to_csv(out/'official_test.csv', index=False)
    np.savez_compressed(out/'test_decisions.npz', labels=ty, baseline=baseline, stacked=stacked)
    dump('audit.json', dict(source_hashes=source_hashes, train_sha256=digest(path),
         meta_fit_rows=len(fit), threshold_rows=len(tune), selection_rows=len(sel),
         grouped_splits_verified=True, selection_lightgbm_scores_reproduced=True,
         script_sha256=digest(Path(__file__))))
    report = ['# Baseline AE + LightGBM learned stacking', '',
        'Selected on development data: '+winner['name']+'. Six predeclared candidates; recall target 95%.', '',
        'Meta-model training and threshold calibration use disjoint grouped halves of calibration data. Base detectors stay frozen.', '',
        '```', frame.to_string(index=False), '```', '',
        'Exploratory only: selection/test previously used, one seed, known official-test overlap. The AE is the saved reconstructed baseline, not the unavailable historical checkpoint.', '',
        'Reproduce: `.venv/bin/python training/experiment_stacked_ae_fusion.py --output-dir <new-directory>`.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    dump('artifact_hashes.json', {p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print('SELECTION:', winner)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        main(args.output_dir)

"""Retrain baseline AE on grouped v2 splits and select fusion before test scoring."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from train_autoencoder import fit_autoencoder, make_autoencoder_features, reconstruction_errors
from train_isolation_forest import select_operating_point
from train_validated_lightgbm import canonical_features, load_split_indices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/working_v2_baseline_fusion_20261005'
RAW = ROOT / 'data/raw/CSV_Files/Training and Testing Sets'
FROZEN = ROOT / 'experiments/lightgbm_validated_v2'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def metrics(y, pred):
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, recall=tp/(tp+fn), fpr=fp/(fp+tn),
                precision=tp/max(tp+fp, 1), f1=2*tp/max(2*tp+fp+fn, 1), accuracy=(tn+tp)/len(y))


def main():
    OUT.mkdir(exist_ok=False)
    manifest = json.loads((FROZEN/'manifest.json').read_text())
    for name, expected in manifest['artifact_hashes'].items():
        assert hashlib.sha256((FROZEN/name).read_bytes()).hexdigest() == expected, name
    train_path = RAW/'UNSW_NB15_training-set.csv'
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest['train_sha256']
    raw = pd.read_csv(train_path)
    x = canonical_features(raw.drop(columns=['id', 'label', 'attack_cat']))
    y = raw.label.to_numpy(int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    split = load_split_indices(FROZEN/'split_indices.npz', y, groups)
    fit, early, cal, sel = [split[k] for k in ('fit', 'early', 'calibration', 'selection')]
    normal = fit[y[fit] == 0]
    cats = ['proto', 'service', 'state']
    nums = [c for c in x if c not in cats]
    imputer = SimpleImputer().fit(raw.iloc[normal][nums])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32).fit(raw.iloc[normal][cats].fillna('__MISSING__').astype(str))
    def transform(frame):
        return sparse.hstack([sparse.csr_matrix(imputer.transform(frame[nums]).astype(np.float32)), encoder.transform(frame[cats].fillna('__MISSING__').astype(str))], format='csr')
    tree = transform(raw)
    scaler = StandardScaler().fit(tree[normal, :len(nums)].toarray())
    features = make_autoencoder_features(tree, len(nums), scaler)
    ae, history, epoch = fit_autoencoder(features[normal], features[early[y[early] == 0]], (128, 32, 128), 20, 512, .001, 4, 42)
    ae_scores = reconstruction_errors(ae, features, 2048)
    ae_threshold = select_operating_point(y[cal], ae_scores[cal], .10)['threshold']
    lgb = joblib.load(FROZEN/'binary/model.joblib')
    p = lgb.predict_proba(x)[:, 1]
    cutoff = float(lgb.decision_threshold_)
    reference = np.sort(ae_scores[normal])
    def rank(a):
        return np.searchsorted(reference, a, side='right') / len(reference)
    candidates = [{'name': 'LightGBM', 'kind': 'lgb'}, {'name': 'Baseline AE', 'kind': 'ae'}, {'name': 'OR', 'kind': 'or'}, {'name': 'AND', 'kind': 'and'}]
    for weight in (.1, .25, .5, .75, .9):
        scores = (1-weight)*p[cal] + weight*rank(ae_scores[cal])
        fpr, recall, thresholds = roc_curve(y[cal], scores, drop_intermediate=False)
        eligible = np.flatnonzero(recall >= .95)
        best = min(eligible, key=lambda i: (fpr[i], -recall[i], -thresholds[i]))
        candidates.append(dict(name=f'Weighted rank {weight}', kind='weighted', weight=weight, threshold=float(thresholds[best])))
    for gate in (0, .1, .25, .4, .5):
        for quantile in (.90, .95, .97, .99, .995):
            candidates.append(dict(name=f'Selective gate={gate} q={quantile}', kind='selective', gate=gate, threshold=float(np.quantile(ae_scores[cal[y[cal] == 0]], quantile))))
    def predict(c, prob, a):
        lp, ap = prob >= cutoff, a >= ae_threshold
        if c['kind'] == 'lgb': return lp
        if c['kind'] == 'ae': return ap
        if c['kind'] == 'or': return lp | ap
        if c['kind'] == 'and': return lp & ap
        if c['kind'] == 'weighted': return (1-c['weight'])*prob + c['weight']*rank(a) >= c['threshold']
        return lp | ((prob >= c['gate']) & (a >= c['threshold']))
    validation = [dict(**c, **metrics(y[sel], predict(c, p[sel], ae_scores[sel]))) for c in candidates]
    eligible = [c for c in validation if c['recall'] >= .95]
    winner = min(eligible, key=lambda c: (c['fpr'], -c['recall'], -c['f1']))
    fusion_winner = min([c for c in eligible if c['kind'] not in ('lgb', 'ae')], key=lambda c: (c['fpr'], -c['recall'], -c['f1']))
    dump('frozen_selection.json', dict(winner=winner, fusion_winner=fusion_winner, candidates=candidates, objective='Minimum selection FPR subject to recall >= 0.95; ties higher recall then F1'))
    pd.DataFrame(validation).to_csv(OUT/'selection.csv', index=False)
    joblib.dump(dict(model=ae, numeric_scaler=scaler, imputer=imputer, encoder=encoder, numerical_columns=nums, categorical_columns=cats, rank_reference=reference), OUT/'autoencoder.joblib')
    dump('autoencoder_config.json', dict(hidden_layers=[128,32,128], epochs=20, patience=4, batch_size=512, learning_rate=.001, seed=42, best_epoch=epoch, history=history, selected_threshold=ae_threshold, fit_normal_rows=len(normal), feature_count=features.shape[1], protocol='Normal-only grouped fit, independent early stopping, calibration threshold at FPR <= 10%; preprocessing fitted only on normal fit rows. Architecture defaults retained; not an exact reconstruction of missing historical AE.'))
    # Test is first loaded only after selection and model artifacts are saved.
    test = pd.read_csv(RAW/'UNSW_NB15_testing-set.csv')
    xt = canonical_features(test.drop(columns=['id','label','attack_cat']))
    yt = test.label.to_numpy(int)
    pt = lgb.predict_proba(xt)[:, 1]
    at = reconstruction_errors(ae, make_autoencoder_features(transform(test), len(nums), scaler), 2048)
    assert np.isfinite(at).all() and np.isfinite(pt).all()
    chosen = candidates[:4] + [c for c in candidates[4:] if c['name'] == fusion_winner['name']]
    results = []
    base = pt >= cutoff
    for c in chosen:
        pred = predict(c, pt, at)
        results.append(dict(name=c['name'], **metrics(yt, pred), recovered=int(((yt==1)&~base&pred).sum()), lost=int(((yt==1)&base&~pred).sum()), added_fp=int(((yt==0)&~base&pred).sum())))
    pd.DataFrame(results).to_csv(OUT/'official_test.csv', index=False)
    np.savez_compressed(OUT/'scores.npz', test_y=yt, test_lightgbm=pt, test_ae=at, selection_y=y[sel], selection_lightgbm=p[sel], selection_ae=ae_scores[sel])
    historical = json.loads((ROOT/'reports/official_testing_lightgbm_v2_ae_evaluation.json').read_text())
    assert [[results[0]['tn'],results[0]['fp']],[results[0]['fn'],results[0]['tp']]] == historical['results']['LightGBM']['confusion_matrix']
    dump('audit.json', dict(lightgbm_hash_verified=True, historical_lightgbm_matrix_reproduced=True, grouped_partitions_verified=True, official_test_overlap_rows=int(np.isin(pd.util.hash_pandas_object(xt,index=False).to_numpy(), groups).sum()), lightgbm_threshold=cutoff, limitations=['Official test and selection set previously used in development; no fresh holdout claim.', 'AE architecture matches script defaults; historical config unavailable.', 'Normal-only preprocessing refitted to avoid fitting on calibration/selection rows.', 'Single seed; empirical recall constraint, not a confidence bound.']))
    print('SELECTION WINNER', winner, flush=True)
    print('BEST FUSION', fusion_winner, flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

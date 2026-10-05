"""Read-only audit of saved decisions; no fitting, calibration, or model inference."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from anomaly_detection_models import benign_rank
from complementary_fusion import SelectiveFusion
from summarize_anomaly_results import enrich

SOURCE = ROOT / 'experiments/fusion_ablation_20261005T033313Z'
OUT = Path(__file__).resolve().parent
METHODS = ['lightgbm', 'fusion_control', 'selective_fusion', 'fusion_ae_plain_latent',
           'selective_fusion_primary_0.75', 'selective_fusion_primary_0.5']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


protocol = json.loads((SOURCE / 'protocol.json').read_text())
original = enrich(pd.read_csv(SOURCE / 'comparison.csv'))
raw_path = Path(protocol['train_csv'])
require(sha(raw_path) == protocol['data_sha256'], 'Source dataset hash mismatch')
raw = pd.read_csv(raw_path, usecols=['label', 'attack_cat'])
raw_families = raw.attack_cat.astype(str).str.strip().to_numpy()
raw_labels = raw.label.to_numpy()
rows, audits = [], []
for family in ['Exploits', 'Reconnaissance', 'DoS', 'none']:
    for seed in protocol['seeds']:
        directory = SOURCE / family / f'seed_{seed}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        require(manifest['status'] == 'complete' and manifest['seed'] == seed
                and manifest['held_family'] == family, 'Manifest scenario mismatch')
        require(manifest['data_sha256'] == protocol['data_sha256'], 'Dataset mismatch')
        for name, expected in manifest['artifact_hashes'].items():
            require(sha(directory / name) == expected, f'Artifact hash mismatch: {directory/name}')
        with np.load(directory / 'split_indices.npz') as archive:
            split = {key: archive[key] for key in archive.files}
        combined = np.concatenate(list(split.values()))
        require(np.array_equal(np.sort(combined), np.arange(len(raw))), 'Partitions overlap/incomplete')
        for part, ids in split.items():
            if family != 'none' and part != 'evaluation':
                require(not np.any(raw_families[ids] == family), 'Held family leaked into development')
        policy = json.loads((directory / 'calibration.json').read_text())
        require(policy['partition'] == 'calibration' and not policy['uses_evaluation_labels'],
                'Calibration metadata invalid')
        saved = joblib.load(directory / 'models.joblib')
        require('plain' in saved['anomaly_models'] and 'denoising' in saved['anomaly_models'],
                'Required saved AE missing')
        require(tuple(saved['fusion_ablations']['fusion_ae_plain_latent']['columns']) ==
                ('lightgbm', 'ae_plain', 'ae_latent'), 'Wrong fusion representation')
        require(saved['selective_policies'] == policy['selective_policies'], 'Policy copy mismatch')
        require(saved['thresholds'] == policy['thresholds'], 'Threshold copy mismatch')
        with np.load(directory / 'evaluation_scores.npz') as scores, np.load(
                directory / 'evaluation_decisions.npz') as decisions:
            ids = split['evaluation']
            require(np.array_equal(scores['row_indices'], ids) and
                    np.array_equal(decisions['row_indices'], ids), 'Evaluation row mismatch')
            y, families = scores['labels'], scores['families']
            require(np.array_equal(y, raw_labels[ids]) and
                    np.array_equal(families, raw_families[ids]), 'Labels/families misaligned')
            attack, benign = y == 1, y == 0
            held = families == family if family != 'none' else np.zeros(len(y), dtype=bool)
            rank = benign_rank(saved['rank_references']['ae_denoising'], scores['ae_denoising'])
            for budget in protocol['fpr_budgets']:
                baseline = decisions[f'lightgbm__{budget}']
                require(np.array_equal(baseline, decisions[f'fusion_control__{budget}']),
                        'Logistic control differs from baseline')
                for method in METHODS:
                    prediction = decisions[f'{method}__{budget}']
                    if method.startswith('selective_fusion'):
                        key = str(budget) if method == 'selective_fusion' else f'{method}__{budget}'
                        frozen = SelectiveFusion(**policy['selective_policies'][key])
                        reproduced = frozen.predict(scores['lightgbm'], scores['learned_fusion'], rank)
                        require(frozen.calibration_false_positives <=
                                int(np.floor(budget * frozen.calibration_rows)), 'Calibration cap violated')
                    else:
                        reproduced = scores[method] >= policy['thresholds'][str(budget)][method]
                    require(np.array_equal(prediction, reproduced), 'Frozen decision mismatch')
                    recovered = attack & ~baseline & prediction
                    lost = attack & baseline & ~prediction
                    added = benign & ~baseline & prediction
                    removed = benign & baseline & ~prediction
                    result = dict(held_family=family, seed=seed, budget=budget, model=method,
                        recall=float(prediction[attack].mean()),
                        withheld_family_recall=float(prediction[held].mean()) if held.any() else np.nan,
                        fpr=float(prediction[benign].mean()),
                        precision=float((prediction & attack).sum()/prediction.sum()) if prediction.any() else 0.,
                        recovered_lightgbm_misses=int(recovered.sum()),
                        lost_lightgbm_detections=int(lost.sum()),
                        net_recovered_attacks=int(recovered.sum()-lost.sum()),
                        additional_false_positives=int(added.sum()),
                        net_additional_false_positives=int(added.sum()-removed.sum()),
                        held_recovered=int((held & ~baseline & prediction).sum()),
                        held_lost=int((held & baseline & ~prediction).sum()),
                        held_net_gain=int((held & prediction).sum()-(held & baseline).sum()),
                        changed_decisions=int((prediction != baseline).sum()),
                        recall_delta=float(prediction[attack].mean()-baseline[attack].mean()),
                        held_recall_delta=float(prediction[held].mean()-baseline[held].mean()) if held.any() else np.nan,
                        fpr_delta=float(prediction[benign].mean()-baseline[benign].mean()))
                    match = original[(original.held_family == family) & (original.seed == seed)
                                     & (original.budget == budget) & (original.model == method)]
                    require(len(match) == 1, 'Missing/duplicate result')
                    for key in result:
                        if key in match.columns and isinstance(result[key], (int, float)):
                            require(np.isclose(result[key], match.iloc[0][key], equal_nan=True,
                                               rtol=1e-10, atol=1e-12), f'Metric mismatch: {key}')
                    if method == 'selective_fusion':
                        require(not np.any(baseline & ~prediction), 'Selective baseline alert lost')
                    rows.append(result)
        audits.append(dict(family=family, seed=seed, manifest_sha256=sha(directory/'manifest.json'),
                           verified_artifacts=list(manifest['artifact_hashes'])))
        del saved
results = pd.DataFrame(rows)
results.to_csv(OUT/'per_seed.csv', index=False, mode='x')
keys = ['held_family', 'budget', 'model']
measures = [key for key in results.columns if key not in keys + ['seed']]
summary = results.groupby(keys, sort=False)[measures].agg(['mean', 'std', 'min', 'max'])
summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
summary.to_csv(OUT/'summary.csv', mode='x')
with (OUT/'audit.json').open('x') as stream:
    json.dump(dict(source=str(SOURCE), script_sha256=sha(Path(__file__)),
                   comparison_sha256=sha(SOURCE/'comparison.csv'), runs=audits,
                   verified_rows=len(results), training=False, recalibration=False,
                   model_inference=False, limitations='Internal overlapping development holdouts; '
                   'saved scores were audited, not recomputed from detector inference.'), stream, indent=2)
print(f'PASS: {len(audits)} saved runs, {len(results)} metric rows; hashes, partitions, frozen decisions, CSV metrics verified.')
print('All 36 logistic-control comparisons identical; all 36 original selective policies preserve baseline alerts.')
print('Means: held recall; net attacks; losses; gross/net FPs; held net gain')
for (family,budget,method), row in summary.iterrows():
    if method in ['selective_fusion','fusion_ae_plain_latent'] or (family=='Exploits' and budget==.01):
        values=[row[f'{key}_mean'] for key in ['withheld_family_recall','net_recovered_attacks',
                 'lost_lightgbm_detections','additional_false_positives','net_additional_false_positives','held_net_gain']]
        print(family,budget,method, ' '.join(f'{v:.3f}' for v in values))

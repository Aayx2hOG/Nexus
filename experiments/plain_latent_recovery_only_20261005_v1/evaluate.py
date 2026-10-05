"""Frozen archive-only OR comparison. No training, inference or calibration."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'experiments/fusion_ablation_20261005T033313Z'
OUT = Path(__file__).resolve().parent
stage = sys.argv[1]
if stage == 'initial':
    cases = {'Exploits':[.01]}
elif stage == 'extension':
    cases = {'Exploits':[.03,.05], 'Reconnaissance':[.01,.03,.05],
             'DoS':[.01,.03,.05], 'none':[.01,.03,.05]}
else:
    raise ValueError('Stage must be initial or extension')
output = OUT/stage
output.mkdir(exist_ok=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


rows, provenance, paired = [], [], []
for scenario, budgets in cases.items():
    for seed in [42,123,2026]:
        directory=SOURCE/scenario/f'seed_{seed}'
        manifest=json.loads((directory/'manifest.json').read_text())
        require(manifest['status']=='complete' and manifest['held_family']==scenario
                and manifest['seed']==seed, 'Manifest mismatch')
        hashes={}
        for name in ['evaluation_scores.npz','evaluation_decisions.npz','calibration.json','metrics.json']:
            hashes[name]=digest(directory/name)
            require(hashes[name]==manifest['artifact_hashes'][name],f'Hash mismatch: {name}')
        policy=json.loads((directory/'calibration.json').read_text())
        metrics=json.loads((directory/'metrics.json').read_text())
        require(policy['partition']=='calibration' and not policy['uses_evaluation_labels'],
                'Calibration provenance invalid')
        with np.load(directory/'evaluation_scores.npz') as scores, np.load(directory/'evaluation_decisions.npz') as decisions:
            require(np.array_equal(scores['row_indices'],decisions['row_indices']), 'Misaligned rows')
            labels=scores['labels']; attack=labels==1; benign=labels==0
            held=scores['families']==scenario
            for budget in budgets:
                baseline=decisions[f'lightgbm__{budget}']
                selective=decisions[f'selective_fusion__{budget}']
                thresholds=policy['thresholds'][str(budget)]
                require('fusion_ae_plain_latent' in thresholds, 'No frozen plain+latent threshold; stop')
                require(np.array_equal(baseline,scores['lightgbm']>=thresholds['lightgbm']),
                        'Baseline threshold mismatch')
                recovery=scores['fusion_ae_plain_latent']>=thresholds['fusion_ae_plain_latent']
                require(np.array_equal(recovery,decisions[f'fusion_ae_plain_latent__{budget}']),
                        'Frozen fusion threshold mismatch')
                candidate=baseline | ((~baseline) & recovery)
                for method,prediction in [('lightgbm',baseline),('selective_fusion',selective),
                                           ('plain_latent_recovery_only',candidate)]:
                    require(not np.any(baseline & ~prediction), 'Baseline alert lost')
                    recovered=int((attack & ~baseline & prediction).sum())
                    lost=int((attack & baseline & ~prediction).sum())
                    added=int((benign & ~baseline & prediction).sum())
                    removed=int((benign & baseline & ~prediction).sum())
                    result=dict(scenario=scenario,budget=budget,seed=seed,method=method,
                        recovered=recovered,lost_baseline=lost,net_attack_gain=recovered-lost,
                        added_fps_gross=added,removed_fps=removed,net_fps=added-removed,
                        held_recall=float(prediction[held].mean()) if held.any() else np.nan,
                        overall_recall=float(prediction[attack].mean()),
                        observed_fpr=float(prediction[benign].mean()),
                        precision=float((attack & prediction).sum()/prediction.sum()) if prediction.any() else 0.,
                        held_recovered=int((held & ~baseline & prediction).sum()),
                        evaluation_attack_rows=int(attack.sum()),evaluation_benign_rows=int(benign.sum()),
                        frozen_plain_latent_threshold=thresholds['fusion_ae_plain_latent'],
                        frozen_lightgbm_threshold=thresholds['lightgbm'])
                    historical=next(r for r in metrics if r['model']==
                        ('fusion_ae_plain_latent' if method=='plain_latent_recovery_only' else method)
                        and r['budget']==budget)
                    require(recovered==historical['recovered_lightgbm_misses'] and
                            added==historical['additional_false_positives'], 'Recovery/FP count mismatch')
                    rows.append(result)
                paired.append(dict(scenario=scenario,budget=budget,seed=seed,
                    extra_attacks_vs_selective=int((attack & candidate).sum()-(attack & selective).sum()),
                    extra_fps_vs_selective=int((benign & candidate).sum()-(benign & selective).sum()),
                    candidate_only_attack_detections=int((attack & candidate & ~selective).sum()),
                    selective_only_attack_detections=int((attack & selective & ~candidate).sum())))
        provenance.append(dict(scenario=scenario,seed=seed,hashes=hashes))
results=pd.DataFrame(rows)
results.to_csv(output/'per_seed.csv',index=False,mode='x')
measures=['recovered','lost_baseline','net_attack_gain','added_fps_gross','removed_fps','net_fps',
          'held_recall','overall_recall','observed_fpr','precision','held_recovered']
summary=results.groupby(['scenario','budget','method'])[measures].agg(['mean','std','min','max'])
summary.columns=[f'{metric}_{stat}' for metric,stat in summary.columns]
summary.to_csv(output/'summary.csv',mode='x')
pd.DataFrame(paired).to_csv(output/'versus_selective.csv',index=False,mode='x')
with (output/'audit.json').open('x') as stream:
    json.dump(dict(inputs=provenance,script_sha256=digest(Path(__file__)),stage=stage,
        rule='lightgbm_alert | ((~lightgbm_alert) & (archived_plain_latent_score >= frozen_plain_latent_threshold))',
        threshold_source='Existing full-detector plain-AE+latent calibration threshold, reused unchanged',
        union_calibration_budget_guarantee=False,training=False,recalibration=False,inference=False),stream,indent=2)
if stage=='initial':
    print(results[['seed','method',*measures[:6],'held_recall','overall_recall','observed_fpr','precision']].round(6).to_string(index=False))
    print('SUMMARY (mean, sample SD, min, max):')
    print(summary[[f'{m}_{s}' for m in ['recovered','added_fps_gross'] for s in ['mean','std','min','max']]].round(3).to_string())
    print('PAIRED:',paired)
else:
    print('Extension complete:',len(rows),'rows; zero lost baseline alerts in all comparisons.')

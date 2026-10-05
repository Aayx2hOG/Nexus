"""Characterize saved decision changes. Reads archives only; never fits or infers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/fusion_ablation_20261005T033313Z'
OUT = Path(__file__).resolve().parent
METHODS = ['fusion_ae_plain_latent', 'selective_fusion']
COHORTS = ['recovered_attacks', 'lost_attacks', 'removed_false_alarms', 'added_false_alarms',
           'persistent_attack_misses', 'retained_attack_detections',
           'persistent_false_alarms', 'retained_benign_negatives']
SCORES = ['lightgbm', 'ae_plain', 'ae_latent', 'ae_denoising',
          'fusion_ae_plain_latent', 'learned_fusion']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


counts, quantiles, families_out, bins, slices, inputs = [], [], [], [], [], []
row_ids = {}
for scenario in ['Exploits', 'Reconnaissance', 'DoS', 'none']:
    for seed in [42, 123, 2026]:
        directory = SOURCE / scenario / f'seed_{seed}'
        manifest = json.loads((directory/'manifest.json').read_text())
        check(manifest['status']=='complete' and manifest['held_family']==scenario
              and manifest['seed']==seed, 'Manifest mismatch')
        hashes = {}
        for file in ['evaluation_scores.npz','evaluation_decisions.npz','calibration.json','metrics.json']:
            hashes[file]=sha(directory/file)
            check(hashes[file]==manifest['artifact_hashes'][file], f'Hash mismatch: {directory/file}')
        hashes['reconnaissance_diagnostics.json']=sha(directory/'reconnaissance_diagnostics.json')
        inputs.append(dict(scenario=scenario,seed=seed,hashes=hashes))
        calibration=json.loads((directory/'calibration.json').read_text())
        metrics=json.loads((directory/'metrics.json').read_text())
        with np.load(directory/'evaluation_scores.npz') as archive, np.load(directory/'evaluation_decisions.npz') as decisions:
            check(np.array_equal(archive['row_indices'],decisions['row_indices']), 'Rows differ')
            ids,y,family=archive['row_indices'],archive['labels'],archive['families']
            scores={key:archive[key] for key in SCORES}
            normal,attack=y==0,y==1
            for key in ['ae_plain','ae_latent','ae_denoising']:
                reference=np.sort(scores[key][normal])
                scores[key+'_benign_eval_percentile']=100*np.searchsorted(reference,scores[key],side='right')/len(reference)
            for budget in [.01,.03,.05]:
                baseline=decisions[f'lightgbm__{budget}']
                tree_cutoff=calibration['thresholds'][str(budget)]['lightgbm']
                tree_ratio=scores['lightgbm']/tree_cutoff
                for method in METHODS:
                    prediction=decisions[f'{method}__{budget}']
                    metadata=dict(scenario=scenario,seed=seed,budget=budget,method=method)
                    cohort_masks=[attack & ~baseline & prediction, attack & baseline & ~prediction,
                        normal & baseline & ~prediction, normal & ~baseline & prediction,
                        attack & ~baseline & ~prediction, attack & baseline & prediction,
                        normal & baseline & prediction, normal & ~baseline & ~prediction]
                    check(np.all(np.sum(cohort_masks,axis=0)==1), 'Cohorts not exclusive/exhaustive')
                    fusion_key='learned_fusion' if method=='selective_fusion' else method
                    cutoff=(calibration['selective_policies'][str(budget)]['recovery_threshold']
                            if method=='selective_fusion' else calibration['thresholds'][str(budget)][method])
                    current_scores={**scores,'tree_cutoff_ratio':tree_ratio,
                                    'fusion_threshold_margin':scores[fusion_key]-cutoff}
                    result={**metadata,'tree_threshold':tree_cutoff,'fusion_threshold':cutoff}
                    for cohort,mask in zip(COHORTS,cohort_masks,strict=True):
                        result[cohort]=int(mask.sum())
                        if cohort in COHORTS[:4]:
                            row_ids[f'{scenario}__{seed}__{budget}__{method}__{cohort}']=ids[mask]
                        for score,values in current_scores.items():
                            q=np.quantile(values[mask],[0,.1,.25,.5,.75,.9,1]) if mask.any() else [np.nan]*7
                            quantiles.append({**metadata,'cohort':cohort,'score':score,'n':int(mask.sum()),
                                **dict(zip(['min','p10','p25','median','p75','p90','max'],q,strict=True))})
                        if cohort in ['recovered_attacks','lost_attacks','persistent_attack_misses','retained_attack_detections']:
                            for label in np.unique(family[attack]):
                                families_out.append({**metadata,'cohort':cohort,'attack_family':label,
                                    'count':int((mask & (family==label)).sum())})
                        boundaries=[0,.25,.5,.75,1,1.25,np.inf]
                        for lo,hi in zip(boundaries[:-1],boundaries[1:],strict=True):
                            bins.append({**metadata,'cohort':cohort,'tree_ratio_low':lo,'tree_ratio_high':hi,
                                'count':int((mask & (tree_ratio>=lo) & (tree_ratio<hi)).sum())})
                    result['net_attack_gain']=result['recovered_attacks']-result['lost_attacks']
                    result['net_added_false_alarms']=result['added_false_alarms']-result['removed_false_alarms']
                    expected=next(r for r in metrics if r['model']==method and r['budget']==budget)
                    for key,original in [('recovered_attacks','recovered_lightgbm_misses'),
                        ('lost_attacks','lost_lightgbm_detections'),('added_false_alarms','additional_false_positives'),
                        ('removed_false_alarms','removed_false_positives'),('net_attack_gain','net_recovered_attacks')]:
                        check(result[key]==expected[original],f'Counts disagree: {key}')
                    counts.append(result)
        diagnostics=json.loads((directory/'reconnaissance_diagnostics.json').read_text())
        lookup={(r['slice'],r['cohort'],r['method']):r for r in diagnostics}
        for r in diagnostics:
            method,budget_string=r['method'].rsplit('__',1)
            if method not in METHODS:
                continue
            base=lookup[r['slice'],r['cohort'],f'lightgbm__{budget_string}']
            slices.append(dict(scenario=scenario,seed=seed,budget=float(budget_string),method=method,
                slice=r['slice'],cohort=r['cohort'],rows=r['rows'],recovered=r['recovered'],lost=r['lost'],
                net_attack_gain=r['recovered']-r['lost'],attack_misses=r['attack_misses'],
                baseline_attack_misses=base['attack_misses'],benign_false_positives=r['benign_false_positives'],
                baseline_benign_false_positives=base['benign_false_positives'],
                net_added_false_alarms=r['benign_false_positives']-base['benign_false_positives']))
for name,data in [('counts',counts),('score_quantiles',quantiles),('attack_families',families_out),
                  ('tree_score_bins',bins),('archived_slices',slices)]:
    pd.DataFrame(data).to_csv(OUT/f'{name}.csv',mode='x',index=False)
with (OUT/'changed_row_ids.npz').open('xb') as f:
    np.savez_compressed(f,**row_ids)
with (OUT/'audit.json').open('x') as f:
    json.dump(dict(inputs=inputs,script_sha256=sha(Path(__file__)),comparisons=len(counts),
        raw_data_used=False,training=False,inference=False,recalibration=False,
        percentile_definition='Descriptive rank against archived benign EVALUATION scores; not calibration',
        limitations='No row-level proto/service/state available; saved slices cover Reconnaissance and benign only, overlap, and lack gross benign transition counts.'),f,indent=2)
print('Validated and characterized',len(counts),'method/budget/seed comparisons across 12 saved runs.')
print(pd.DataFrame(counts).query("scenario == 'Exploits' and budget == .01")[[
    'seed','method','recovered_attacks','lost_attacks','net_attack_gain','added_false_alarms','removed_false_alarms','net_added_false_alarms']].to_string(index=False))

# MIXED — HOLD

The frozen plain-AE + latent recovery-only union has a consistent recovery advantage,
but a materially higher gross false-alert cost than existing selective fusion.
It is promising as a recovery mechanism, not established as an acceptable-budget replacement.

## Exact candidate and provenance

candidate = lightgbm_alert | ((~lightgbm_alert) & plain_ae_latent_recovery)

Recovery is the archived plain-AE + latent score passing its existing frozen full-detector
calibration threshold. That pass/fail decision was verified against the archived decision
array. No new threshold, eligibility gate, model fit, recalibration or inference was used.
The full-detector threshold was not originally calibrated for this OR union: it supplies
no total-FPR guarantee for the candidate. Budget labels identify the source policy budget.
Archived selective decisions are the comparator; candidate does not replace or modify them.

Exploits 1% was evaluated and reported first. Only then was the same frozen comparison
extended to Exploits 3%/5% and Reconnaissance, DoS and none at 1%/3%/5%, all three
existing seeds. Artifact hashes, row alignment, baseline thresholds, archived recovery
threshold decisions and historical recovery/FP counts were checked. The rule preserved
EVERY baseline alert, both attack and benign, in all 36 cases. No false alarms were removed.

## Exploits 1% per seed

scenario,budget,seed,method,recovered,lost_baseline,net_attack_gain,added_fps_gross,removed_fps,net_fps,held_recall,overall_recall,observed_fpr,precision,held_recovered,evaluation_attack_rows,evaluation_benign_rows,frozen_plain_latent_threshold,frozen_lightgbm_threshold
Exploits,0.01,42,lightgbm,0,0,0,0,0,0,0.8706315694906118,0.8814255431902491,0.0082334423053638,0.9987240111086092,0,60384,8259,0.9212093740011216,0.884398675506917
Exploits,0.01,42,selective_fusion,84,0,84,1,0,1,0.8726080316233942,0.8828166401695814,0.0083545223392662,0.9987073083912548,66,60384,8259,0.9212093740011216,0.884398675506917
Exploits,0.01,42,plain_latent_recovery_only,1494,0,1494,22,0,22,0.904800407270985,0.9061671966083732,0.0108972030512168,0.998357903955627,1141,60384,8259,0.9212093740011216,0.884398675506917
Exploits,0.01,123,lightgbm,0,0,0,0,0,0,0.8440990626778068,0.8615265687493698,0.0081730769230769,0.9986753418786768,0,59506,8320,0.933232510422766,0.8910744251915685
Exploits,0.01,123,selective_fusion,9,0,9,0,0,0,0.8443386338454167,0.8616778140019494,0.0081730769230769,0.9986755740802056,8,59506,8320,0.933232510422766,0.8910744251915685
Exploits,0.01,123,plain_latent_recovery_only,1602,0,1602,14,0,14,0.8860240170095529,0.8884482237085336,0.0098557692307692,0.9984513692162418,1400,59506,8320,0.933232510422766,0.8910744251915685
Exploits,0.01,2026,lightgbm,0,0,0,0,0,0,0.8159195040876831,0.8353363770645897,0.0081077858590676,0.9986354697596018,0,59576,8387,0.9258005624678938,0.912272708011708
Exploits,0.01,2026,selective_fusion,1587,0,1587,9,0,9,0.859761027760309,0.8619746206526118,0.0091808751639442,0.9985028193661288,1464,59576,8387,0.9258005624678938,0.912272708011708
Exploits,0.01,2026,plain_latent_recovery_only,3081,0,3081,21,0,21,0.8838978228970144,0.8870518329528669,0.0106116609037796,0.9983187244975064,2270,59576,8387,0.9258005624678938,0.912272708011708

## Exploits 1% three-seed summaries

scenario,budget,method,recovered_mean,recovered_std,recovered_min,recovered_max,lost_baseline_mean,lost_baseline_std,lost_baseline_min,lost_baseline_max,net_attack_gain_mean,net_attack_gain_std,net_attack_gain_min,net_attack_gain_max,added_fps_gross_mean,added_fps_gross_std,added_fps_gross_min,added_fps_gross_max,removed_fps_mean,removed_fps_std,removed_fps_min,removed_fps_max,net_fps_mean,net_fps_std,net_fps_min,net_fps_max,held_recall_mean,held_recall_std,held_recall_min,held_recall_max,overall_recall_mean,overall_recall_std,overall_recall_min,overall_recall_max,observed_fpr_mean,observed_fpr_std,observed_fpr_min,observed_fpr_max,precision_mean,precision_std,precision_min,precision_max,held_recovered_mean,held_recovered_std,held_recovered_min,held_recovered_max
Exploits,0.01,lightgbm,0.0,0.0,0,0,0.0,0.0,0,0,0.0,0.0,0,0,0.0,0.0,0,0,0.0,0.0,0,0,0.0,0.0,0,0,0.8435500454187005,0.027360164292757557,0.8159195040876831,0.8706315694906118,0.8594294963347363,0.023116035409944745,0.8353363770645897,0.8814255431902491,0.008171435029169458,6.28443114820468e-05,0.008107785859067605,0.008233442305363846,0.9986782742489626,4.434345180109374e-05,0.9986354697596018,0.9987240111086092,0.0,0.0,0,0
Exploits,0.01,plain_latent_recovery_only,2059.0,886.723745029984,1494,3081,0.0,0.0,0,0,2059.0,886.723745029984,1494,3081,19.0,4.358898943540674,14,22,0.0,0.0,0,0,19.0,4.358898943540674,14,22,0.8915740823925175,0.0115035615221603,0.8838978228970144,0.904800407270985,0.8938890844232579,0.0106560548942061,0.8870518329528669,0.9061671966083731,0.010454877728588581,0.0005381280736128536,0.00985576923076923,0.010897203051216855,0.998375999223125,6.814861980599361e-05,0.9983187244975065,0.9984513692162418,1603.6666666666667,591.4138426967476,1141,2270
Exploits,0.01,selective_fusion,560.0,890.1982925168976,9,1587,0.0,0.0,0,0,560.0,890.1982925168976,9,1587,3.3333333333333335,4.932882862316247,0,9,0.0,0.0,0,0,3.3333333333333335,4.932882862316247,0,9,0.8589025644097067,0.014154237251636831,0.8443386338454167,0.8726080316233942,0.8688230249413809,0.012119734894508363,0.8616778140019494,0.8828166401695814,0.008569491475429127,0.0005371900228976263,0.008173076923076924,0.0091808751639442,0.9986285672791965,0.0001100507603566942,0.9985028193661287,0.9987073083912547,512.6666666666666,824.3890667235545,8,1464

Candidate net attack gains are 1494, 1602, 3081, versus selective's 84, 9, 1587.
Compared with selective, incremental gains are 1410, 1593, 1494, costing 21, 14,
12 extra benign alerts. Thus the incremental benefit is not confined to one seed.
The candidate is not a superset of selective: in seed 2026 it misses 166 attacks
that selective recovers, while detecting 1660 attacks selective misses. The guarantee
applies to LightGBM alerts, not selective-only alerts.

Candidate gross additional FPs equal net additional FPs: 22, 14, 21. Unlike the
unrestricted detector, none are offset by removing baseline false alarms.
Observed candidate FPR is 1.090%, 0.986%, 1.061%; two seeds exceed 1%.
Candidate mean held recall is 89.157% versus 85.890% selective and 84.355% baseline.
Candidate mean precision is 99.838%, versus 99.863% selective; internal benchmark
prevalence means these precision values are not estimates for deployment traffic.

## All frozen comparisons: means across three seeds

|Scenario|Source budget|Method|Recovered/net attacks|Gross/net added FPs|Observed FPR|
|---|---:|---|---:|---:|---:|
|Exploits|1%|selective_fusion|560.0|3.3|0.857%|
|Exploits|1%|plain_latent_recovery_only|2059.0|19.0|1.045%|
|Exploits|3%|selective_fusion|194.3|2.7|2.731%|
|Exploits|3%|plain_latent_recovery_only|606.7|22.3|2.967%|
|Exploits|5%|selective_fusion|162.0|3.3|4.743%|
|Exploits|5%|plain_latent_recovery_only|393.3|24.3|4.995%|
|Reconnaissance|1%|selective_fusion|62.3|2.0|0.952%|
|Reconnaissance|1%|plain_latent_recovery_only|2016.3|23.0|1.204%|
|Reconnaissance|3%|selective_fusion|24.3|2.0|2.970%|
|Reconnaissance|3%|plain_latent_recovery_only|268.7|15.0|3.127%|
|Reconnaissance|5%|selective_fusion|4.0|0.7|5.037%|
|Reconnaissance|5%|plain_latent_recovery_only|93.3|25.3|5.334%|
|DoS|1%|selective_fusion|3.0|0.0|1.188%|
|DoS|1%|plain_latent_recovery_only|599.7|14.3|1.360%|
|DoS|3%|selective_fusion|5.3|0.7|2.922%|
|DoS|3%|plain_latent_recovery_only|124.7|23.7|3.199%|
|DoS|5%|selective_fusion|10.7|4.0|5.038%|
|DoS|5%|plain_latent_recovery_only|92.7|20.3|5.234%|
|none|1%|selective_fusion|4.3|2.0|1.144%|
|none|1%|plain_latent_recovery_only|66.3|15.7|1.308%|
|none|3%|selective_fusion|2.0|0.0|3.049%|
|none|3%|plain_latent_recovery_only|29.3|11.0|3.181%|
|none|5%|selective_fusion|0.3|1.0|5.091%|
|none|5%|plain_latent_recovery_only|39.0|20.7|5.328%|

The candidate gains more attacks in all 36 comparisons, but its evaluation FPR exceeds
the source nominal budget in 28/36 versus 13/36 for selective. Both are empirical
holdout observations, not calibration guarantees or independent statistical trials.
Extra recovery is much smaller in scenario none: means 66.3/29.3/39.0 attacks for
15.7/11.0/20.7 added FPs at 1%/3%/5%, versus selective's 4.3/2.0/0.3 attacks for
2.0/0.0/1.0 added FPs. No universal utility advantage is established.

## Recommendation

Do not train yet. Specify the acceptable incremental false-alert allowance and whether
the source nominal budget is a hard evaluation cap before promoting the recovery-only
rule. Existing evidence supports further archived analysis, but does not determine that
operational tradeoff. Do not select a new threshold using these evaluation outcomes.

## Outputs and execution

initial/ and extension/ each contain per_seed.csv (all requested metrics), summary.csv
(mean, sample SD with ddof=1, minimum and maximum), versus_selective.csv (paired
incremental gains/costs and nonoverlapping attack detections), and audit.json.
The script evaluate.py is isolated here; no model or experiment-runner code changed.
All outputs were exclusive-create in a new directory; historical artifacts are untouched.

Executed in the foreground, with single-thread limits:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/plain_latent_recovery_only_20261005_v1/evaluate.py initial
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/plain_latent_recovery_only_20261005_v1/evaluate.py extension

Additional work comprised targeted source/metadata inspection and pandas aggregation
of these output CSVs. No training, recalibration, threshold or architecture search,
raw-data inference, background process, or full experiment sweep occurred.
These are overlapping internal development holdouts, not external confirmation.

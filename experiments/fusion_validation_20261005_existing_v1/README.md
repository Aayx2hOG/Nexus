# Existing-artifact validation: HOLD on further training

Source: ../fusion_ablation_20261005T033313Z

The existing functionality demonstrates useful anomaly-assisted recovery on withheld
Exploits. This is not yet a consistent case for additional training across scenarios.
Keep LightGBM primary and retain original selective fusion as a conservative candidate.
Investigate saved loss cohorts and seed variation before training again.

## What was validated

- 12 existing scenario/seed runs; 216 rows covering six methods and three budgets.
- Every artifact hash listed by each manifest, including checkpoint and score archives.
- Dataset hash, exhaustive/disjoint row partitions, withheld-family exclusion from
  development partitions, and evaluation labels/row alignment against the source CSV.
- Existing trained plain/denoising AE entries and plain-AE + latent fusion configuration.
- Matching saved policy/threshold copies in calibration JSON and checkpoint bundles.
- Saved decisions reproduced from existing score arrays and frozen thresholds/policies.
- Required metrics recomputed from archived decisions and checked against comparison.csv.
- Paired partition, calibration-condition, and seed checks through the existing enrich helper.
- Exact LightGBM/control decision equality in all 36 comparisons; original selective
  fusion preserves all baseline alerts in all 36 comparisons.

No detector inference was repeated. This does not independently verify that archived
scores reproduce from raw inputs and checkpoint inference, or audit preprocessing fit
history. Hashes establish consistency with saved manifests, not external authenticity.

## Outputs

per_seed.csv contains overall/held recall, FPR, precision, gross recovery, baseline
losses, net attack gain, gross/net additional FPs, held-family recovery/loss/net gain,
and decision-change counts, plus paired recall/FPR deltas.
summary.csv gives the mean, sample SD (ddof=1), minimum, and maximum of every measure
for each scenario/budget/method. Held-family recall is blank for scenario none;
held-family counts in that scenario are zero because no held family exists.
audit.json records input identifiers and checks. validate.py is the exact audit code.

## Interpretation

Anomaly methods differ from the score-only control and recover additional held-family
attacks. The control itself adds nothing in this run. This supports useful anomaly
information in these fitted models, without implying universal improvement.

At the 1% Exploits calibration budget, original selective fusion averages 560 net
additional attacks (SD 890.20; range 9–1587), zero baseline losses, and 3.33 gross/net
additional FPs. Most recovery occurs in seed 2026. Held-family recall averages 85.89%
versus LightGBM's 84.36%. Observed FPR averages 0.857% versus 0.817%.

Plain-AE + latent fusion yields positive net attack and held-family gains for Exploits
in all nine existing seed/budget comparisons. At 1%, it averages 1640 net attacks
(SD 824.46; range 1163–2592), after losing 419 baseline detections and recovering 2059.
It adds 19 gross FPs but removes enough baseline FPs that net additional FPs are 1.67.
Thus its low net FP cost should not be mistaken for few newly alerted benign rows.
It loses net attacks on average for Reconnaissance at 3%/5%, DoS at 1%, and none at
1%/3%. No universal replacement or preferred operating point is established.

Both predefined budget allocations produce lower net attack gains than original
selective fusion in 24 of 36 paired comparisons and higher gains in 12. Each causes
negative net attack gain in Exploits seed 42 at 1% and 5%; a positive aggregate mean
hides those regressions. Reallocation has no preservation guarantee.

Calibration budgets are empirical calibration caps, not evaluation guarantees:
81 of the 216 audited method/budget rows exceed their nominal budget on evaluation.
These counts include the identical baseline/control rows, not independent trials.
All findings are internal development holdouts. Seeds are overlapping experimental
splits, not independent deployments or external confirmation. No significance claim
or operational recovery/FP utility weighting is made.

## Execution

Foreground, single-thread-limited archive audit:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/fusion_validation_20261005_existing_v1/validate.py

The output files are exclusive-create; rerunning in the same directory refuses to
overwrite them. A separate read-only pandas aggregation inspected the resulting
summaries and per-seed tradeoffs. No training, recalibration, threshold/architecture
search, new detector inference, full experiment sweep, or background process ran.

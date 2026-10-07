# Nexus: demonstrable edge and next improvements

The strongest pitch is: **Nexus measures what an extra detector catches, what it
loses, and how much analyst workload it creates.** The working product serves
LightGBM with explanations, durable alerts, analyst feedback and non-alert review;
independent confirmation and production promotion remain pending.

The product uses a frozen release threshold; operators do not tune thresholds in the dashboard. Operational alerts
remain driven by the serving model. Prediction retries do not create duplicate alerts and changed
inputs are rejected under an existing flow/bundle identity. Research artifact
reuse copies the original split archive so file-hash provenance remains stable.
No benchmark detector was retrained or promoted.

## What is differentiated, and what is not

| Comparison | Defensible edge | Evidence boundary |
| --- | --- | --- |
| A classifier accuracy dashboard | Explain decisions, audit non-alerts, expose detection regressions and gross false-alert cost, and export replay evidence | Metrics are historical evidence, not an operator threshold control |
| LightGBM alone | Offline selective fusion can recover additional attacks while preserving every baseline alert by construction | Strongest for withheld Exploits; improvement is small elsewhere and variable across seeds |
| Naive detector OR | Explicit accounting of the combined false-alert budget and preserved baseline detections | Calibration budgets do not guarantee future/evaluation FPR |
| Established IDS platforms | Potential complementary flow-scoring and recovery layer with reproducible paired experiments | No same-traffic signature-IDS benchmark establishes superiority |

Do not present anomaly detection, SHAP, threshold sliders, or classifier/AE
ensembles as new inventions. Suricata already emits alerts, anomalies and protocol
metadata ([official EVE documentation](https://docs.suricata.io/en/suricata-8.0.3/output/eve/eve-json-output.html)).
The contribution is the tested recovery policy and transparent accounting of
benefits, regressions and operational costs.

## Current research evidence takes precedence over older plans

The [eight-seed assessment](../reports/current_model_assessment.md) and README
contain an earlier research stage. Subsequent local reports already include the
LightGBM-score-only control and plain-AE/latent ablations:

- [Existing-artifact validation](../experiments/fusion_validation_20261005_existing_v1/README.md):
  12 scenario/seed runs and 216 method/budget rows audited. The score-only control
  matched baseline decisions in all 36 comparisons. Selective preserved baseline
  alerts. This supports complementary anomaly information in these fitted runs.
- [Decision-change cohorts](../experiments/fusion_cohort_audit_20261005_v1/REPORT.md):
  at Exploits 1%, seed 2026 contributes 94.5% of summed selective recovery across
  three seeds. Unrestricted fusion loses some Fuzzers and Shellcode detections;
  net false-positive counts conceal gross added and removed alerts.
- [Plain-AE/latent recovery-only candidate](../experiments/plain_latent_recovery_only_20261005_v1/REPORT.md):
  the frozen union preserves baseline alerts in 36 cases but costs more false
  alerts. Exploits 1% mean recovery is 2,059 attacks with 19 added benign alerts;
  two of three seeds exceed the nominal 1% source budget. Status remains HOLD.

These are overlapping internal experiments, not independent deployments or unique
attacks summed across seeds. Do not mix these three-seed results with the earlier
eight-seed table. Saved score validation is not raw-input inference parity.

## Priorities with concrete acceptance gates

| Priority | Area | Improvement | Completion evidence |
| --- | --- | --- | --- |
| P0 | Working build | Package frozen model, preprocessing and manifests with a reproducible launch/preflight script | A second machine starts API/UI, readiness passes, mixed traffic produces persisted alerts |
| P0 | Model | Independently confirm the frozen recovery policy on a newly reserved split or compatible external corpus | Paired baseline/fusion counts, family recall, gross FP cost, confidence intervals; no threshold tuning on confirmation data |
| P1 | Model | Calibrate the total recovery-only union on development calibration data, rather than reusing a full-detector cutoff | Combined calibration budget checked; held evaluation violations and zero baseline losses reported separately |
| P1 | Model | Investigate recovered/lost cohorts and seed instability before more architectures | Explain Exploits gains and family regressions; rerun only predefined comparisons; retain negative results |
| P1 | Backend | Enforce non-alert feedback idempotency/content conflicts and revision checks; return only latest review per flow | Conflicting retries fail; concurrent reviews cannot overwrite; sample pagination has no duplicate rows |
| P1 | Backend | Move CPU inference off the async event loop and bound concurrent inference/SHAP work | Readiness remains responsive during batches; measured p50/p95/p99 latency and memory at declared concurrency |
| P1 | Data integrity | Replace replay's row-count heuristic with dataset-hash and split-manifest validation; add replay run IDs | Wrong file/split fails closed; reports can select an exact immutable cohort without mixing demo runs |
| P1 | Operations | Add request/bundle IDs, inference and DB timings, failure counters, drift reference statistics and delayed-label monitoring | Reproducible load/shift replay; distribution shift distinguished from confirmed concept drift |
| P2 | Security/product | Replace the local shared token with user identity, tenant isolation and role permissions | Analyst actions attributable to actual users; cross-tenant reads/writes denied |
| P2 | Scalability | PostgreSQL for concurrent writes; separate durable ingestion from bounded inference workers; transactional outbox for notifications | Restart/retry tests show no lost acknowledged flows or duplicate alerts; measured backlog recovery |
| P2 | Adoption | Validated flow extractor and SIEM integration with explicit feature units and missing-value policy | Same traffic scored against a signature baseline; incompatible features rejected, not guessed |

Threshold exploration must not become automatic production tuning. Threshold
selection needs separate validation data; see the [scikit-learn threshold-tuning
example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html).
The new lab deliberately has no “deploy this threshold” action.

## System design direction

Keep the single-process API and SQLite for the local demonstration. Before adding
services, measure inference time, SHAP time, write contention and queue depth.
Only introduce the next component when that measurement justifies it.

A future scaled pipeline is: compatible flow adapter → validated durable ingestion
→ bounded inference workers with immutable bundles → atomic predictions/alerts
and notification outbox → analyst API/dashboard/SIEM. A separate offline path
curates labels, trains candidates, checks evaluation gates and supports human
promotion/rollback. Training never belongs in the prediction request path.

The current evidence view aggregates all rows for one bundle in SQL. Large deployments
will need run/time filters, a bundle-oriented index and cached cohort aggregates.
`last_sequence` is a useful prediction watermark, not an immutable snapshot ID;
late truth labels can change the same cohort's metrics. Exported JSON records the
aggregate evidence at the moment of export.

## Eight-minute panel walkthrough

| Time | Show | Rubric coverage |
| --- | --- | --- |
| 0:00–0:45 | Readiness, bundle version, problem: recover misses without flooding analysts | Problem fit & coverage (15) |
| 0:45–2:15 | Submit actual CSV flows, open a persisted alert, explain signed TreeSHAP contributions | Working demonstration (25) |
| 2:15–3:00 | Submit analyst feedback and inspect a non-alert flow | Demonstration depth (15) |
| 3:00–4:30 | Model evaluation: fixed policy, recovered attacks vs extra FPs, family coverage | Working demonstration and depth (40 combined) |
| 4:30–5:30 | Offline selective-fusion evidence, preserved detections, one mixed/negative outcome | Technical implementation (20) |
| 5:30–6:30 | Walk code: inference, transaction retry handling, policy aggregation and regression tests | Technical implementation (20) |
| 6:30–7:15 | Integration and measured scaling plan; name current limits | Product potential & scalability (15) |
| 7:15–8:00 | Each teammate explains their actual contribution and answers one question in their area | Team effort & clarity (10) |

Use real contribution history for team ownership; do not invent assignments.
Rehearse backend failure and missing-data states. Live website
probes estimate/fix some flow inputs; use dataset flows as the reliable model demo.

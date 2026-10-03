# Nexus: current model assessment

Reviewed 2026-10-03. Scope: model research and existing local results, not frontend/backend integration.

## 1. Current scenario and verdict

**Keep LightGBM as the primary detector. Keep selective fusion as the conservative
recovery candidate. Retain unrestricted learned fusion as a research comparator,
not a universal replacement for LightGBM.**

The strongest defensible result is **additional Exploits detection while preserving
all baseline detections and adding few false positives**. DoS and Reconnaissance
gains from selective fusion are much smaller. In the closed-set `none` scenario,
the extra model complexity currently buys almost no additional detection.

We have made a meaningful improvement in how anomaly information is used and
evaluated. We have not established a generally superior IDS, reliable zero-day
detection, or superiority over Snort, Suricata, or Zeek.

### Evidence used

- **960 result rows:** four scenarios × ten methods × three budgets × eight seeds.
- Seeds: 7, 21, 42, 100, 123, 314, 1337, 2026.
- Original three seeds come from [v2 reuse results](../artifacts/anomaly_comparison_v2/comparison.csv);
  five additional seeds come from [new-seed results](../artifacts/fusion_extra_seeds_v2/comparison.csv).
- Main source: [eight-seed comparison](../artifacts/fusion_eight_seed_statistics_v2/comparison.csv),
  [statistical summary](../artifacts/fusion_eight_seed_statistics_v2/summary.csv),
  and [source manifest](../artifacts/fusion_eight_seed_statistics_v2/analysis_manifest.json).
- Verified source CSV hashes, eight seeds per comparison cell, and completion
  markers for all 32 scenario/seed runs. All 324 original-method result rows
  retain their v1 recall, FPR, recovery, and lost-detection values in v2.
- The current `fusion_report_v2` derives from the **three-seed** suite. Its plots
  must not be presented as eight-seed figures. This document uses the combined eight seeds.

The runs use internal development holdouts from the official training CSV.
Withheld-family rows are excluded from model development, but we are now using
their evaluation results to choose the next research direction. Further changes
need independent confirmation; these results are not a fresh final benchmark.

## 2. What the latest results actually show

All figures below are means across eight seeds, not totals of unique attacks.
`none` means ordinary closed-set evaluation. Overall recall includes all evaluation
attack families; held-family recall measures only the excluded family. These must
not be confused when claiming recovery of unfamiliar attacks.

### Detection at the nominal 3% calibration budget

| Scenario | Method | Precision | Overall recall | Observed FPR | Held-family recall |
| --- | --- | --- | --- | --- | --- |
| none | lightgbm | 98.49% | 93.17% | 3.086% | — |
| none | learned_fusion | 98.53% | 93.03% | 3.001% | — |
| none | selective_fusion | 98.49% | 93.18% | 3.089% | — |
| Reconnaissance | lightgbm | 99.46% | 92.73% | 2.974% | 85.96% |
| Reconnaissance | learned_fusion | 99.46% | 92.54% | 3.003% | 84.64% |
| Reconnaissance | selective_fusion | 99.46% | 92.77% | 2.986% | 86.07% |
| Exploits | lightgbm | 99.55% | 93.88% | 3.029% | 93.75% |
| Exploits | learned_fusion | 99.55% | 94.33% | 3.048% | 94.54% |
| Exploits | selective_fusion | 99.55% | 94.14% | 3.067% | 94.19% |
| DoS | lightgbm | 99.41% | 95.50% | 3.033% | 97.60% |
| DoS | learned_fusion | 99.41% | 95.60% | 3.054% | 97.82% |
| DoS | selective_fusion | 99.41% | 95.52% | 3.046% | 97.63% |

Precision is high in these attack-heavy evaluation populations. It does not
predict precision on a production network with a much lower attack prevalence.
About 3% FPR still means roughly 30 false alerts per 1,000 benign flows before
any alert grouping. Calibration budgets are not guaranteed evaluation FPR caps.

### Recovery and its cost at 3%

Gross recovered attacks alone can hide serious regressions. **Net recovered
attacks = recovered LightGBM misses − lost LightGBM detections.** Gross additional
FPs and net additional FPs also answer different questions.

| Scenario | Method | Recovered | Lost | Net attacks | Gross added FPs | Net added FPs |
| --- | --- | --- | --- | --- | --- | --- |
| none | learned_fusion | 32.12 | 56.88 | -24.75 | 11.88 | -7.12 |
| none | selective_fusion | 1.62 | 0.00 | +1.62 | 0.25 | +0.25 |
| none | naive_or | 57.25 | 0.00 | +57.25 | 224.88 | +224.88 |
| Reconnaissance | learned_fusion | 231.88 | 330.38 | -98.50 | 18.88 | +2.50 |
| Reconnaissance | selective_fusion | 16.50 | 0.00 | +16.50 | 1.00 | +1.00 |
| Reconnaissance | naive_or | 328.25 | 0.00 | +328.25 | 237.62 | +237.62 |
| Exploits | learned_fusion | 520.00 | 251.00 | +269.00 | 24.12 | +1.62 |
| Exploits | selective_fusion | 154.88 | 0.00 | +154.88 | 3.25 | +3.25 |
| Exploits | naive_or | 890.12 | 0.00 | +890.12 | 233.38 | +233.38 |
| DoS | learned_fusion | 103.00 | 60.38 | +42.62 | 18.38 | +1.75 |
| DoS | selective_fusion | 6.88 | 0.00 | +6.88 | 1.12 | +1.12 |
| DoS | naive_or | 136.00 | 0.00 | +136.00 | 227.88 | +227.88 |

Key interpretation:

- **Exploits:** selective fusion adds 154.88 detections per seed with zero losses
  and 3.25 added FPs. The pooled gross recovery/added-FP ratio is **47.65**.
  Unrestricted fusion adds more net detections on average, but can lose many
  baseline detections in individual seeds.
- **DoS:** selective fusion adds 6.88 detections for 1.13 added FPs. This is a
  modest improvement, not a breakthrough. Unrestricted fusion has a larger mean gain.
- **Reconnaissance:** unrestricted fusion loses more detections than it recovers
  at 3%. Selective fusion avoids that regression and adds 16.50 detections for
  1.00 added FP; the held-family recall improvement is only about 0.11 percentage points.
- **Closed set:** unrestricted fusion reduces recall in every seed at 3%.
  Selective fusion adds only 1.63 detections on average. LightGBM alone remains
  the simpler choice when the extra recovery is not worth the inference cost.
- **Naive OR:** overall recall increases, but observed FPR is around 5.8% at a
  nominal 3% per-detector budget. This is not a fair win over a detector near 3% FPR.

### Budget tradeoff

Each cell is **held-family recall / observed benign FPR**. Budgets were calibrated
without the held-out family. No operating point below should be chosen using the
identity or labels of a future unknown attack.

| Held family | Budget | LightGBM | Learned fusion | Selective fusion |
| --- | --- | --- | --- | --- |
| Reconnaissance | 1% | 57.81% / 0.995% | 60.27% / 0.999% | 58.04% / 1.011% |
| Reconnaissance | 3% | 85.96% / 2.974% | 84.64% / 3.003% | 86.07% / 2.986% |
| Reconnaissance | 5% | 94.47% / 4.989% | 93.69% / 4.964% | 94.49% / 5.006% |
| Exploits | 1% | 85.68% / 0.940% | 86.49% / 0.959% | 87.07% / 0.980% |
| Exploits | 3% | 93.75% / 3.029% | 94.54% / 3.048% | 94.19% / 3.067% |
| Exploits | 5% | 95.58% / 5.071% | 96.26% / 5.018% | 95.84% / 5.096% |
| DoS | 1% | 91.32% / 1.089% | 90.94% / 1.016% | 91.39% / 1.095% |
| DoS | 3% | 97.60% / 3.033% | 97.82% / 3.054% | 97.63% / 3.046% |
| DoS | 5% | 98.45% / 4.994% | 98.55% / 4.983% | 98.47% / 5.026% |

Exploits is the most convincing selective-recovery case, including at 1%:
488.25 additional attacks per seed with zero losses and 3.38 added FPs on average.
For Reconnaissance, unrestricted fusion helps at 1% but hurts at 3% and 5%.
For DoS, unrestricted fusion hurts at 1%. There is **no universal best fusion
method or budget** in these results. Higher recall at 5% is purchased with a
larger false-alert allowance.

## 3. What we improved—and what we did not

### Demonstrated improvements

1. **Prediction preservation:** selective fusion loses zero LightGBM detections
   in all 96 scenario/seed/budget configurations. Preservation is also enforced
   by the decision rule; it is not an independent statistical discovery.
2. **Selective use of anomaly evidence:** confidently benign-looking traffic
   can still enter recovery when its anomaly score is suspicious; ordinary
   negatives are not blindly OR-ed into alerts.
3. **Explicit FP accounting:** all 96 saved selective calibration policies meet
   their empirical calibration allowance. Evaluation failures are still reported.
4. **Stronger evidence:** eight seeds replace conclusions drawn from only three;
   the report exposes lost detections, gross/net FP changes and paired effects.
5. **Reproducibility:** saved models/splits are reused with hash checks, policies
   are recorded, and old comparison results remain reproducible.

We did **not** improve the original LightGBM checkpoint or retrain the three-seed
fusion model into a stronger model. The key new method is the conditional recovery
policy; extra seeds strengthen measurement rather than automatically improving a model.

### Variance and budget reliability

Net attack gains below are mean ± sample SD, followed by their observed range.
Positive-gain seeds and seeds meeting the evaluation FPR budget are separate counts;
they do not imply the same seeds satisfied both criteria.

| Scenario | Method | Net attack gain ± SD | Range | Positive gain | FPR ≤3% |
| --- | --- | --- | --- | --- | --- |
| none | learned_fusion | -24.75 ± 14.07 | -48 to -7 | 0/8 | 4/8 |
| none | selective_fusion | 1.62 ± 1.92 | 0 to 5 | 5/8 | 3/8 |
| Reconnaissance | learned_fusion | -98.50 ± 133.25 | -292 to 60 | 3/8 | 4/8 |
| Reconnaissance | selective_fusion | 16.50 ± 19.35 | 0 to 59 | 7/8 | 6/8 |
| Exploits | learned_fusion | 269.00 ± 513.27 | -730 to 852 | 6/8 | 3/8 |
| Exploits | selective_fusion | 154.88 ± 217.73 | 1 to 572 | 8/8 | 4/8 |
| DoS | learned_fusion | 42.62 ± 50.89 | -28 to 140 | 7/8 | 5/8 |
| DoS | selective_fusion | 6.88 ± 8.43 | 0 to 26 | 6/8 | 5/8 |

Example: unrestricted fusion's Exploits gain ranges from **−730 to +852** attacks
per seed. Its positive mean must not be described as consistent improvement.
Selective Exploits recovery is positive in all eight seeds, but its size also
varies substantially. Only four of those eight seeds meet the nominal 3%
evaluation cap; the paired LightGBM baseline also meets it in four.

No significance claim is justified here. Seeds share a dataset and, in the
withheld scenario, repeat evaluation of the same held-family rows. They are not
eight independent deployments or eight independent external test sets.

## 4. Our contribution and defensible novelty

The contribution is **a reproducible, false-alert-budgeted recovery layer around
a strong supervised IDS model**, with measured benefits and regressions under
withheld-family evaluation. The emphasis is on whether anomaly information adds
useful detections beyond LightGBM, rather than whether an autoencoder can classify
the entire dataset by itself.

The practical differentiation consists of:

- preserving the primary model's alert decisions;
- calibrating the recovery channel against its remaining benign FP allowance;
- admitting both uncertain negatives and strongly anomalous negatives;
- reporting gross recovery, lost detections, net recovery and FP cost together;
- testing excluded families and repeated seeds with isolated development partitions.

These are defensible engineering/research contributions **within this project**.
They are not proof that conditional fusion is a new algorithm or a first in the
literature. Autoencoder-based network detection already exists, for example in
[Kitsune (NDSS 2018)](https://arxiv.org/abs/1802.09089). Explanation-guided anomaly
analysis and false-positive reduction also have prior work, including
[DeepAID (CCS 2021)](https://arxiv.org/abs/2109.11495).

A supported presentation sentence is:

> Nexus uses anomaly evidence to recover attacks a strong LightGBM detector
> misses, while preserving its existing alerts and explicitly measuring the
> extra false-alert cost. The strongest current evidence is on withheld Exploits.

Do not say “we detect zero days,” “we beat signature IDSs,” or “fusion always
improves detection.” None follows from these experiments.

## 5. What should improve next, in priority order

### A. Allocate a small recovery budget deliberately

**88 of 96 configurations have zero remaining calibration FP slots** after
LightGBM consumes the allowance; six have one slot, one has two, and one has three.
This explains why the current selective policy is conservative. With zero
headroom it can recover only candidates above the routed benign calibration scores.

Test a predeclared allocation of the total budget between the supervised and
recovery branches. Compare against both the original full-budget LightGBM and
the stricter supervised branch. A stricter baseline may lose original detections:
do not retain the current zero-loss claim relative to the old baseline in that case.
Choose allocations on development data, never on the final evaluation family.

### B. Establish that anomaly inputs—not just rescaling LightGBM—cause the gain

The current logistic fusion contains LightGBM score, reconstruction error and
latent distance. Add a **LightGBM-score-only logistic control**, then compare
LightGBM + reconstruction, LightGBM + latent distance, and the full combination
using identical partitions and budgets. This is the most important missing
ablation before attributing the improvement specifically to anomaly information.

Consider a recovery-focused learner trained on known-attack misses and benign
negatives from the fusion partition. Keep calibration disjoint and report any
sampling/weighting changes. Its benefit is a hypothesis, not a promised improvement.

### C. Investigate the actual Reconnaissance failure slices

Reading the eight-seed decision archives at 3% shows:

| Reconnaissance feature slice | Rows per seed | LightGBM recall | Learned fusion | Selective fusion |
| --- | ---: | ---: | ---: | ---: |
| `proto=tcp` | 5,100 | 74.82% | 71.93% | 75.05% |
| `proto=udp` | 3,586 | 96.58% | 96.73% | 96.58% |
| `service=-` | 8,788 | 83.35% | 81.76% | 83.48% |
| `service=http` | 1,603 | 99.36% | 99.44% | 99.38% |
| `state=FIN` | 5,095 | 74.82% | 71.93% | 75.05% |

Slices overlap; do not add their row counts. These are post-hoc evaluation
diagnostics, not independent tuning data. TCP/FIN and unspecified-service traffic
deserve examination of score distributions, reconstruction residuals, and benign
lookalikes in an appropriately isolated development experiment.

Auxiliary ground-truth/event files contain real Reconnaissance subcategories,
but the benchmark CSV lacks row-level subtype labels and reliable connection/time
join keys. **Subtype-level performance is not supported by the current benchmark.**
Do not rename the feature slices “scan subtypes.”

### D. Test anomaly representations on complementarity

At 3%, standalone recall averaged equally across the four scenarios is:

| Anomaly method | Overall recall | Observed FPR |
| --- | --- | --- |
| ae_plain | 65.97% | 2.815% |
| ae_scaled | 30.77% | 3.225% |
| ae_denoising | 33.47% | 3.206% |
| ae_latent | 21.49% | 3.088% |
| isolation_forest | 24.67% | 2.964% |

The plain AE is substantially stronger standalone than the scaled/denoising
variants. Therefore the preprocessing/denoising changes cannot be advertised as
an established improvement. However, standalone recall does not settle which
score best complements LightGBM. Compare plain-AE fusion with denoising-AE fusion
before removing either input. Avoid another broad architecture search first.

### E. Strengthen confirmation and operating-point reliability

Predeclare the objective, method and budget before evaluating new confirmation
data. Consider a conservative calibration margin or an appropriate one-sided
false-positive bound, with explicit sampling assumptions; exact-row grouping
does not make correlated network flows independent. Report the recall cost.

Report fixed-operating-point partial ROC behavior, per-family recall, inference
latency and memory as well as the existing precision/recall/F1/AUC results.
Current learned fusion does not consistently improve ROC-AUC; the evidence is
about specific operating points, not universal ranking superiority. Discuss
drift as an unmeasured deployment risk until time-aware or fresh-traffic evidence
exists. Do not optimize additional seeds and then present their best run as final.

## 6. What should be removed or de-emphasized

| Item | Recommendation | Reason |
| --- | --- | --- |
| Naive OR as the main detector | Remove from the candidate deployment path; retain as a negative control | Its recall gains spend substantially more FPR. |
| Rank OR as the preferred fusion | De-prioritize; retain its result row | At 3% it loses net attacks on average in every scenario. |
| Standalone AE/Isolation Forest replacing LightGBM | Remove from the main project story | Their practical-budget recall is much weaker. |
| “Denoising/scaling improved the AE” | Remove the claim until supported | Plain AE currently has stronger standalone recall. |
| “Learned fusion wins everywhere” | Remove the claim | Closed-set and Reconnaissance regressions are visible. |
| Huge model/threshold searches | Pause | Missing ablations and budget allocation are more informative now. |
| Best-seed or gross-recovery-only reporting | Remove | It hides instability, losses and FP cost. |
| Unsupported zero-day, subtype and commercial-superiority claims | Remove | No corresponding evaluation exists. |

Do not delete checkpoints, splits, old result files or baseline implementations.
Keep them as reproducibility evidence. Simplify the active research shortlist to
LightGBM, unrestricted learned fusion, and selective recovery; keep the other
methods as ablations rather than ten competing product features.

## 7. What edge do we have over existing programs?

Separate a **measured internal advantage** from a **potential product role**.

| Comparator | What already exists | Defensible Nexus position | What remains unproven |
| --- | --- | --- | --- |
| LightGBM alone in this workspace | Strong supervised detection | Selective recovery adds detections with zero removed baseline alerts; strongest on Exploits | Net operational value after latency, drift and false-alert cost |
| Naive OR in this workspace | Simple anomaly-assisted recovery | Far fewer extra false alerts; explicit combined calibration accounting | Dominance at exactly equal observed external-test FPR |
| Snort | Rule-based IDS/IPS detection; rules support alert decisions | A possible complementary learned detector for patterns not represented by configured rules | Any attack missed by Snort but caught by Nexus; no paired Snort evaluation exists |
| Suricata | Alerts, anomalies, flow/protocol records and structured EVE output | A possible complementary ML scoring layer with explicit recovery metrics | Better coverage, speed, reliability or false-alert rate |
| Zeek | Programmable network analysis and notice policies for unusual/suspicious activity | A possible model consuming compatible extracted features | Superior anomaly detection or enterprise readiness |

The external-tool descriptions are grounded in the official
[Snort rule guide](https://docs.snort.org/rules/),
[Suricata EVE documentation](https://docs.suricata.io/en/suricata-8.0.5/output/eve/eve-json-output.html),
and [Zeek notice framework](https://docs.zeek.org/en/master/frameworks/notice.html).
Existing tools are not fairly described as incapable of anomaly detection or
analyst-facing alerts. “We alert instead of blocking” matches the challenge,
but is not exclusive to Nexus.

Our strongest current edge over a basic hackathon classifier demo is an auditable
answer to **“what did the second detector catch, what did it break, and how many
false alerts did it cost?”** This is an advantage in demonstrated methodology,
not a benchmark victory over mature security programs.

To establish a competitive edge later, run the same labeled traffic through a
specified signature/rule configuration and Nexus, reconcile feature extraction
and timestamps, and compare additional detections at a common alert budget.
The current flow CSV experiments cannot substitute for that test.

## 8. Decision now

Preserve the current checkpoints and results. Treat **selective recovery on
Exploits** as the strongest demonstrated candidate, while retaining the full
negative and mixed results in the presentation. Next, run the score-only and
plain-AE fusion ablations, then test deliberate budget allocation on development
data. Require confirmation before promoting a policy or claiming a general gain.

No new training, threshold search, or expensive evaluation was run to write this
assessment. Only existing CSVs, policy files and saved decision archives were read.

Source comparison SHA-256: `a4456766c442ba2e6f41f03c05daf681c67da23ffe666a0a8b3df80bb251f1ad`.

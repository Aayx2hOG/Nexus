# Archived decision-change cohorts

## Scope and checks

Twelve existing runs, three budgets (1%, 3%, 5%), and two methods: Plain-AE + latent
and original selective fusion (72 comparisons). No model changes. Only saved
scores, decisions, calibration metadata, metrics, manifests, and previously saved
slice diagnostics were read. No raw dataset, checkpoint loading, detector inference,
training, recalibration, or threshold search was used.

Score/decision/calibration/metric hashes were checked against manifests. Row IDs were
aligned and the eight attack/benign transition cohorts checked to partition every
row exactly once. Changed-cohort counts match saved metrics. Diagnostic-file hashes
are recorded for provenance; those files were not included in original manifest hashes.

## Exploits scenario at 1%: seed-level changes

Counts cover all attack families unless explicitly labelled Exploits-only.

| Seed | Method | Recovered attacks | Lost attacks | Net attack gain | Added false alarms | Removed false alarms | Net added false alarms |
|---|---|---:|---:|---:|---:|---:|---:|
|42|Plain AE + latent|1494|329|1165|22|16|6|
|123|Plain AE + latent|1602|439|1163|14|15|-1|
|2026|Plain AE + latent|3081|489|2592|21|21|0|
|42|Selective|84|0|84|1|0|1|
|123|Selective|9|0|9|0|0|0|
|2026|Selective|1587|0|1587|9|0|9|

Exploits-only recovered/lost: Plain AE + latent = 1141/185, 1400/265,
2270/268, yielding net 956, 1135, 2002. Selective = 66/0, 8/0, 1464/0.
Exploits represents 76.4%, 87.4%, and 73.7% of plain-fusion recovered attacks.
Plain fusion also has net Fuzzers losses of 50, 79, and 49, plus Shellcode losses
of 10, 10, and 11 with no Shellcode recovery in these comparisons.

Baseline attack misses are 7160, 8240, 9810. Plain fusion recovers 20.87%, 19.44%,
31.41% of those misses; selective recovers 1.17%, 0.11%, 16.18%. Seed 2026 contributes
94.5% of selective's summed recovery and 52.7% of plain fusion's summed net gain.
These descriptive sums are not counts of unique attacks across seeds.

## What distinguishes recovery and regression

Plain fusion recovered attacks have median LightGBM scores .862, .852, .895,
versus frozen cutoffs .88440, .89107, .91227. Between 98.2% and 99.0% of recovered
attacks have scores between 75% and 100% of their cutoff. Most recovery therefore
comes from relatively high-scoring baseline misses, rather than very low tree scores.
Persistent misses have much lower median tree scores: .681, .674, .686.

Lost attacks sit just above the tree cutoff: median score/cutoff ratios 1.026,
1.026, 1.031. Their plain reconstruction errors are comparatively ordinary:
median benign-evaluation percentiles 38.7, 50.5, 47.5 versus 98.9, 97.9, 97.4 for
recovered attacks. Their latent percentiles are 69.6, 74.7, 68.9 versus 94.4,
97.1, 96.2 for recovered attacks. These are descriptive ranks among saved benign
EVALUATION scores, not fitted calibration percentiles or new decision thresholds.

Raw plain reconstruction-error medians for recovered attacks are .076, .083, .045;
for lost attacks they are approximately .003, .005, .003. Raw latent-distance
medians are 28.2, 44.2, 26.8 for recovery versus 8.5, 9.0, 7.6 for losses.
Raw score scales are seed/model-specific; percentiles aid descriptive comparison.

Added benign alerts resemble recovered attacks: median plain-error percentiles
99.4, 98.2, 96.1 and latent percentiles 93.5, 93.7, 98.5, with tree scores just
below the baseline cutoff. Removed benign alerts resemble lost attacks: median
plain-error percentiles 54.2, 52.5, 33.1 and latent percentiles 70.9, 74.4, 69.0,
with tree scores just above it. The observed reordering helps on high-anomaly
misses, but low-anomaly attacks can be suppressed along with false alarms.
Net FP changes alone conceal this alert churn: seed 2026 adds and removes 21 each.

Selective has no lost attacks or removed false alarms, because baseline alerts are
preserved. Its recovered attacks have median latent percentiles 98.5, 99.7, 98.2.
The saved recovery cutoffs are .95995, .98908, .96434; seed 123 has the highest
numeric cutoff and only nine recoveries. Different fitted scores/partitions mean
this is descriptive evidence, not a causal explanation of the seed effect.

## Protocol/service/state evidence and its limits

There is no archived row-level protocol/service/state mapping. Thus the Exploits
recovery and loss cohorts cannot be assigned these features without additional
metadata, which was intentionally not read from raw data. The saved diagnostic
slices cover Reconnaissance-labelled attacks and benign rows only, and overlap.
They are not attack subtypes, and their counts must not be added across slices.

In the Exploits 1% evaluation's benign slices, plain fusion's net added false alarms:

|Seed|proto=tcp|proto=udp|service=-|service=http|state=FIN|
|---|---:|---:|---:|---:|---:|
|42|5|1|1|5|5|
|123|0|-1|-8|4|0|
|2026|-1|1|-16|16|-1|

HTTP contributes positive net benign alert changes in every seed, while service=-
reductions offset these in seeds 123/2026. The archives do not identify gross
added/removed benign transitions inside each slice. For selective, which removes
no alerts, saved net slice changes equal added false alarms: seed 42 adds one
matching TCP/FIN/service=-; seed 123 adds none; seed 2026 adds eight TCP/FIN, one
UDP, six HTTP, and one service=- (overlapping categories, not additive).

For the withheld Reconnaissance scenario, plain fusion's higher-budget regressions
concentrate in the TCP/FIN and service=- diagnostic slices. Across seeds at 3%,
TCP and FIN each show 230 recoveries versus 869 losses (net -639), and service=-
shows 265 versus 892 (net -627). At 5%, corresponding net changes are -352 and
-350. HTTP has small positive net changes (+4 at 3%, +2 at 5%). These sums describe
repeated holdouts; they are not independent or deduplicated populations.

## Other scenarios and budgets

Plain fusion has positive net attack gain in all nine Exploits seed/budget cases.
At 3%, gains are 360, 476, 644; at 5%, 192, 410, 371. Original selective is also
positive in all nine, but its 3% gains (10, 1, 572) again concentrate in seed 2026.

Reconnaissance at 1% has positive overall net gain in every seed, but most of the
summed gain is in known-family Exploits (+2374) and DoS (+1628), versus +670 in
held Reconnaissance. At 3%, the held-family net loss is 623 across seeds despite
other-family gains; at 5%, all seeds have negative total net gain (range -159 to -92).
Held Reconnaissance loses 348 net detections at 5%.

DoS at 1% is strongly seed-dependent: recovered/lost = 1261/169 (seed 42),
200/2360 (123), 338/114 (2026). Net changes are +1092, -2160, +224; the negative
mean is dominated by seed 123. At 3% and 5%, all seeds have positive net gain.

With no held family, plain fusion loses net attacks in every seed at 1% and 3%.
At 1%, pooled Fuzzers net loss is 270 of 293 total net losses across seeds.
At 5%, net gains are mixed (range -23 to +56). Original selective remains lossless
across every scenario/budget, but gains outside Exploits are often small or zero.

## Outputs and interpretation

- counts.csv: all eight transition cohorts, frozen cutoffs and net changes per seed.
- changed_row_ids.npz: exact source row indices for each of the four changed cohorts;
  keys encode scenario, seed, budget, method and cohort.
- score_quantiles.csv: min/p10/p25/median/p75/p90/max for changed and unchanged cohorts.
- attack_families.csv: recovery/loss and unchanged-cohort family counts.
- tree_score_bins.csv: counts in predefined score/cutoff-ratio bins; no thresholds selected.
- archived_slices.csv: existing overlapping diagnostic changes by seed.
- audit.json and analyze.py: provenance and exact executed analysis.

The successful pattern is high anomaly evidence on already plausible LightGBM
misses. Regression occurs when low anomaly evidence suppresses valid baseline
alerts, with recurring Fuzzers/Shellcode costs and budget-dependent Reconnaissance
losses. This is an association from saved model outputs, not proof of a causal
mechanism or generalization. Seeds are internal, overlapping development holdouts.
No statistical significance, external confirmation, or universal superiority is claimed.

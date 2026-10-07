> Historical workflow: unrelated scripts were moved to
> [the source archive](../experiments/archived_training_20261007/README.md).
> For the active final models, use [configs.md](configs.md).

# Complementary fusion: execution guide

Run from `/home/aayush/projects/Nexus`. Every command is foreground. Nothing is
automatically launched. Existing output directories cause an error; use a new
version name when repeating a command. No original CSV, split, or model is overwritten.

## Setup, data preparation, and validation

```sh
.venv/bin/python -m pip install -e '.[dev,training]'
.venv/bin/python -m pytest -q tests/test_anomaly_detection_models.py
.venv/bin/python training/run_novelty_experiment.py \
  --check-data --output-dir artifacts/fusion_data_check_v2
```

The raw training CSV is already present at
`data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv`.
No download or external preprocessing is required. `--check-data` validates
labels and every requested grouped split, without fitting any model.
Preprocessing fits inside the runner on the proper fitting partition; do not
preprocess the whole dataset first. The official testing CSV is never opened.

## Recommended first run: reuse, calibrate, and evaluate one scenario

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_single_v2 \
  --model learned_fusion selective_fusion naive_or rank_or \
  --held-family DoS --seed 42 --budget 0.03 \
  --uncertain-lower-ratio 0.5 --suspicious-quantile 0.99 --min-fusion-score 0.0
```

LightGBM is always included as a paired baseline. `--model learned_fusion` alone
also works. The singular flags are aliases of the original plural CLI flags.
Model selection controls reported methods; prerequisite detectors are retained.

`--reuse-dir` verifies the training data hash, model/split hashes, and exact split
membership, then reuses the existing fitted preprocessing, LightGBM, anomaly
models, and logistic fusion. It performs no training. It calibrates on the same
benign calibration rows and scores the evaluation partition. The source is
recorded in `reused_from.json`. Gate settings are not optimized on evaluation.

## Complete evaluation using the already-trained suite (recommended)

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/anomaly_comparison_v2 \
  --families none Reconnaissance Exploits DoS \
  --seeds 42 123 2026 --fpr-budgets 0.01 0.03 0.05
```

This preserves all nine original comparisons and adds `selective_fusion`. It
does not rerun the optional attack-type classifier; its original reports remain
in the source suite. Use fresh training below if new attack-type reports are needed.

## Fresh training: LightGBM, anomaly models, fusion, calibration, evaluation

These stages are intentionally one ordered command, not independent scripts
with potentially mismatched preprocessing/splits. It trains LightGBM, trains the
three AEs and Isolation Forest, fits fusion on its disjoint partition, calibrates
cutoffs on benign calibration rows, and finally reports evaluation metrics.
AE normal-validation loss and LightGBM early-stopping progress are visible.

One full-length fresh scenario:

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --output-dir artifacts/fusion_training_single_v2 \
  --families DoS --seeds 42 --fpr-budgets 0.03 \
  --epochs 40 --patience 5 --max-rounds 1500 --n-jobs 4 --attack-types
```

Full fresh suite (an alternative to reusing v1, not a prerequisite):

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --output-dir artifacts/anomaly_fresh_v2 \
  --families none Reconnaissance Exploits DoS \
  --seeds 42 123 2026 --fpr-budgets 0.01 0.03 0.05 \
  --epochs 40 --patience 5 --max-rounds 1500 --n-jobs 4 --attack-types
```

Optional short training smoke run, not evidence for model selection:

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --output-dir artifacts/fusion_smoke_v2 \
  --families DoS --seeds 42 --fpr-budgets 0.03 \
  --epochs 2 --patience 1 --max-rounds 20 --n-jobs 4
```

## Individual budgets and held-out families

These are alternatives to the complete reuse command. Do not aggregate both
these and that complete suite: their scenario/seed/model rows overlap.

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_budget01_v2 \
  --families none Reconnaissance Exploits DoS --seeds 42 123 2026 --budget 0.01

.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_budget03_v2 \
  --families none Reconnaissance Exploits DoS --seeds 42 123 2026 --budget 0.03

.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_budget05_v2 \
  --families none Reconnaissance Exploits DoS --seeds 42 123 2026 --budget 0.05
```

All withheld families, excluding the closed-set scenario:

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --reuse-dir artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_held_families_v2 \
  --families Reconnaissance Exploits DoS --seeds 42 123 2026 \
  --fpr-budgets 0.01 0.03 0.05
```

## Additional seeds

Additional seeds need training because v1 has only 42, 123, and 2026. Declare the
seeds before examining their results; keep the gate settings fixed across runs.

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --output-dir artifacts/fusion_extra_seeds_v2 \
  --families none Reconnaissance Exploits DoS \
  --seeds 7 21 100 314 1337 --fpr-budgets 0.01 0.03 0.05 \
  --epochs 40 --patience 5 --max-rounds 1500 --n-jobs 4
```

## Aggregate CSVs and statistical summaries

Every training/reuse suite already writes `comparison.csv` and `summary.csv`.
The reporting command can regenerate and combine disjoint suites. It also
writes paired per-seed fusion-versus-LightGBM deltas. It rejects duplicate rows.

```sh
.venv/bin/python training/summarize_anomaly_results.py \
  --input-dirs artifacts/anomaly_comparison_v2 \
  --output-dir artifacts/fusion_statistics_v2
```

After running additional seeds, aggregate all eight:

```sh
.venv/bin/python training/summarize_anomaly_results.py \
  --input-dirs artifacts/anomaly_comparison_v2 artifacts/fusion_extra_seeds_v2 \
  --output-dir artifacts/fusion_eight_seed_statistics_v2
```

If instead you ran individual budgets, aggregate those three disjoint suites:

```sh
.venv/bin/python training/summarize_anomaly_results.py \
  --input-dirs artifacts/fusion_budget01_v2 artifacts/fusion_budget03_v2 artifacts/fusion_budget05_v2 \
  --output-dir artifacts/fusion_budget_statistics_v2
```

## Plots and Reconnaissance investigation

Generate all six plots, summaries, and Reconnaissance diagnostics from v2:

```sh
.venv/bin/python training/summarize_anomaly_results.py \
  --input-dirs artifacts/anomaly_comparison_v2 \
  --output-dir artifacts/fusion_report_v2 --plots --reconnaissance
```

The same reporting code supports the existing v1 CSVs without retraining:

```sh
.venv/bin/python training/summarize_anomaly_results.py \
  --input-dirs artifacts/anomaly_comparison_v1 \
  --output-dir artifacts/fusion_report_v1 --plots --reconnaissance
```

Use `MPLCONFIGDIR=/tmp/nexus-matplotlib` before a plotting command only if your
environment prevents Matplotlib from writing its default cache. It does not
change output paths.

## Outputs

| Location under the selected output directory | Contents |
| --- | --- |
| `protocol.json`, `run.log` | Declared configuration, progress and Python warnings/errors |
| `<family>/seed_<seed>/models.joblib` | Frozen fitted models, preprocessors, fusion and policies |
| `<family>/seed_<seed>/*_history.json` | AE validation/checkpoint history for freshly trained runs; best checkpoint is in `models.joblib` |
| `<family>/seed_<seed>/split_indices.npz` | Exact data partitions |
| `<family>/seed_<seed>/calibration.json` | Cutoffs, gates, FP headroom and calibration FPR accounting |
| `<family>/seed_<seed>/metrics.json` | All evaluation metrics, including recovered and lost attacks |
| `<family>/seed_<seed>/evaluation_scores.npz`, `evaluation_decisions.npz` | Scores, row identities and actual budget-specific decisions |
| `<family>/seed_<seed>/manifest.json`, `reused_from.json` | Hashes, library versions, limitations and reuse provenance |
| `comparison.csv`, `summary.csv`, `completed.json` | Per-seed results, descriptive statistics, completion marker |
| Report directory: `paired_fusion_vs_lightgbm.csv` | Paired gains/losses and FPR/recall deltas |
| Report directory: `reconnaissance_metadata.json`, `reconnaissance_slices.csv` | Available subtype metadata and protocol/service/state diagnostics |
| Report directory: `plots/` | Six figures, each PNG and SVG |

Native library output remains visible on the console; `run.log` captures Python
streams and registered LightGBM callback output, not arbitrary native file-descriptor writes.
No separate background log collector is started.

## Calibration, research interpretation, and current findings

The original `learned_fusion` remains the primary comparator. Its logistic model
and feature inputs are unchanged; `--fusion-c` exposes its regularization for
predeclared training experiments. `selective_fusion` is an additional recovery
policy, not a claim of a better measured model before the new results exist.

Selective fusion preserves every LightGBM positive. A negative is eligible when
its score is at least `uncertain_lower_ratio * LightGBM cutoff`, or its AE score
is at/above the `suspicious_quantile` benign-reference percentile. The latter
keeps confidently missed but anomalous attacks eligible. These are score gates,
not calibrated probability/confidence claims. Eligible negatives must also pass
the learned-fusion recovery cutoff and optional minimum fusion score.

The policy counts LightGBM false positives on **benign calibration rows** and
spends only the remaining integer FP allowance. With no headroom, it can still
recover attacks scoring above all routed benign calibration rows. It cannot
lower LightGBM's FPR or lose its detections. Evaluation FPR may exceed the cap;
`evaluation_budget_met` reports that honestly. The same rows are not used to
train fusion and calibrate cutoffs. Withheld families enter neither step.

`recovered_per_additional_fp` divides by **gross** additional FPs, not net FPs.
When the denominator is zero it is JSON null / CSV empty, with an explicit
status. Reports include a pooled ratio, counts of defined ratios and zero-FP
seeds, mean/SD/min/max and seed counts. One-seed SD stays undefined. Seed
variation is descriptive: repeats share evaluation samples, particularly the
withheld family, and no significance or independent-sample CI is claimed.

The existing v1 CSV reveals why both recovery and loss matter. At 3%, learned
fusion recovers 97.67 DoS attacks but loses 55.67, and recovers 671.33 Exploits
attacks but loses 424.67 on average. Reconnaissance's net change is negative.
These are existing historical development results; do not tune the new gate
settings by repeatedly checking the same evaluation labels.

Auxiliary UNSW ground-truth/event files do contain Reconnaissance subcategories.
The benchmark CSV has no subtype labels, source/destination addresses/ports or
timestamps needed for a reliable join. The report records actual auxiliary
categories but **does not invent row-level subtype accuracy**. Protocol, service,
and state slices are diagnostics, not attack subtypes, and never enter calibration.

Recommended order: setup/tests → data check → one reuse experiment → complete
reuse evaluation → reports/plots → optional extra-seed training → combined summary.
Fresh training and the separate-budget commands are alternatives, not required
repetitions of the complete reuse evaluation.

### Opt-in matched fusion ablations and allocation study

`run_novelty_experiment.py` now accepts `--fusion-ablations`. This fits the
LightGBM-score logistic control, LightGBM + latent distance, and (for each
requested reconstruction representation) LightGBM + reconstruction and LightGBM
+ reconstruction + latent distance. All use the same disjoint fusion rows,
seed, logistic C, benign calibration rows, budgets, and frozen evaluation rows.
The control uses the original LightGBM probability; reconstruction and latent
scores retain the existing log1p transform and fusion-partition standardization.

Use `--fusion-representations ae_plain ae_denoising` to include both saved AE
representations (optionally `ae_scaled`). Latent distance remains the existing
denoising AE latent-distance score. With `--reuse-dir`, detector checkpoints,
source hashes, seed, and exact grouped partitions are verified; missing required
checkpoints fail rather than triggering training. Opt-in logistic controls are
fitted on the fusion partition even in reuse mode. Without reuse, the existing
runner trains detectors, so do not invoke it merely to inspect these options.

`--primary-budget-fractions 1 .75 .5` declares allocations in advance; it does
not select a winner. Fraction 1 retains the original policy and spends only
remaining headroom. Smaller fractions independently cap primary benign FPs at
floor(N * budget * fraction) and recovery FPs at
floor(N * budget * (1 - fraction)); unused primary allowance is not reassigned.
Ties are treated conservatively. All recovery/loss and additional-FP metrics
remain relative to the original **full-budget** LightGBM threshold. Reduced
primary shares can lose baseline detections and carry no preservation guarantee.
Allocations currently apply to the existing selective learned-fusion branch;
the matched logistic controls are reported as unrestricted comparators.

`--reconnaissance-diagnostics` writes score quantiles, detection recovery/loss,
benign false positives, and attack misses for the overlapping proto=tcp,
proto=udp, service=-, service=http, and state=FIN slices. Benign rows matching
each feature slice provide FP context. These are diagnostics, not attack subtypes
or disjoint populations. Empty score distributions are null.

Results include partition/data identifiers, detector checkpoint identity,
calibration/evaluation policies, method columns, and allocation. The manifest
records method conditions and exact output checkpoint/source hashes; a reused
detector retains its originating checkpoint identity despite reserialization.
JSON writes are exclusive, and suite output directories must be new. The
summarizer rejects mixed provenance, mismatched paired partitions/conditions,
and unequal seed sets for new results; fully historical tables remain readable.
Existing per-seed metrics and mean, sample SD, minimum, and maximum summaries
are retained. No historical results are regenerated by these additions.

These switches enable future experiments; they do not establish that any
representation, allocation, or fusion method improves detection.

### Calibration evidence and frozen-boundary audit

Future explicitly authorized novelty runs also write `calibration_evidence.npz`.
This captures the calibration arrays already in memory: row indices, labels,
family labels, all available detector/fusion scores, the denoising anomaly rank
against benign fusion-partition references, and route eligibility for every saved
selective policy. Embedded metadata records the seed, held family, budgets,
frozen policies, dataset identifier, partition hash, and exact output checkpoint
hash. The run manifest includes the evidence archive hash. Archive creation is
exclusive and does not change detector scores, thresholds, or decisions.

The standalone auditor loads no models and has no training, inference,
recalibration, or regeneration path. Run it on one **new evidence-bearing run**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  training/calibration_evidence.py \
  --run-dir /path/to/new-suite/Exploits/seed_42 \
  --policy-key 0.01
```

Output is JSON on stdout. `--output-json /path/to/new-report.json` optionally
writes a new file and refuses overwrites. For an allocated policy, use its saved
key, such as `selective_fusion_primary_0.75__0.01`.

The auditor checks archive/policy/split hashes and matches seed, scenario, data,
and checkpoint identifiers to the manifest. It verifies row alignment, disjoint
partitions, held-family exclusion, route masks, and recorded calibration FP counts.
It does not load or independently rehash the checkpoint, re-open the source CSV,
or establish preprocessing provenance beyond recorded identifiers and row checks.

For all benign and routed benign observations separately, it reports fixed score
quantiles, above/equal/passing counts, the maximum and its tie count, distance to
the next distinct score, nearest scores below/above the frozen cutoff, and ties
at the immediately preceding floating-point value. The last count matters because
`nextafter` can put a cutoff strictly above a tied boundary, leaving zero scores
exactly equal to the cutoff. The default local window is an explicit absolute
score distance of `1e-4`; `--window` changes only that descriptive count, never
the policy. It does not select a window or threshold based on outcomes.

Primary calibration FPs, recovery FPs, policy recovery allowance, and unused
allowance are separate fields. Evaluation FPs are explicitly not analyzed.
Zero remaining allowance is not itself evidence that the cutoff is inappropriate,
and these summaries do not establish causality for differences across seeds.

The original October 5 runs lack this archive and will fail with a missing-evidence
error. This change does not backfill them. Reconstructing historical calibration
scores would require separately authorized inference, which this auditor cannot do.

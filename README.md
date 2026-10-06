# Nexus

**Recover attacks a strong supervised detector misses, while measuring the extra false-alert cost.**

Nexus is a network-intrusion detection research project for the challenge
*Catch the Attack the Signatures Miss*. It uses UNSW-NB15 flow features to label
traffic normal or attack and support SOC review. Its current research combines
LightGBM with complementary anomaly scores through learned and selective fusion.
Alerts go to analysts; the project does not automatically block traffic.

The local application already serves LightGBM predictions with TreeSHAP
explanations, prediction logging and an analyst dashboard. **Selective fusion now supports opt-in shadow serving with frozen-checkpoint parity
verification. It never creates live alerts; independent confirmation is deferred.**

## Documentation

For the maintained documentation index, see
[docs/README.md](docs/README.md). It links to the local demo, API contract,
shadow-fusion guide, research status, roadmap and the repository-specific
training, report, experiment and frontend guides.

| Start here | Contents |
| --- | --- |
| [Serving model and cleanup status](docs/MODEL_IMPROVEMENT_STATUS.md) | Selected checkpoint, metrics, candidate outcomes and retained dependencies |
| [Current model assessment](reports/current_model_assessment.md) | Eight-seed results, failures, research contribution and next experiments |
| [Experiment commands](training/FUSION_EXPERIMENTS.md) | Reuse, training, calibration, evaluation, extra seeds and plots |
| [Training workflows](training/README.md) | Active entry points and historical-code boundaries |
| [Shadow fusion](docs/SHADOW_FUSION.md) | Frozen candidate export, parity, activation and evidence |
| [Local demo](docs/LOCAL_DEMO.md) | Backend, frontend and historical dataset replay |
| [Demo edge and improvement roadmap](docs/PROJECT_ROADMAP.md) | Policy Lab, latest research caveats, priorities and judging walkthrough |
| [API contract](docs/api.md) | Prediction, alert and analyst-feedback endpoints |

## Teacher demonstration: main model and shadow fusion

Both model bundles, the frontend build and the curated demo files must exist
locally. They are already prepared in the development workspace; on a fresh clone,
follow [local setup](docs/LOCAL_DEMO.md) and [shadow export](docs/SHADOW_FUSION.md).
The paths below use this workspace. Leave terminals 1 and 2 running.

**Terminal 1 — start the API with live LightGBM and optional shadow fusion:**

```sh
cd /home/aayush/projects/Nexus
export NEXUS_BUNDLE_VERSION=v1.0.0
export NEXUS_SHADOW_BUNDLE_DIR=artifacts/shadow_bundles/selective-exploits-s42-b03-v2
export NEXUS_DATABASE_PATH="artifacts/teacher-demo-$(date +%Y%m%d-%H%M%S).sqlite3"
export NEXUS_API_TOKEN="test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
export NEXUS_REVIEWER_ID=local-analyst
.venv/bin/uvicorn nexus.api:app --host 127.0.0.1 --port 8000
```

The timestamped database isolates this demonstration from previous runs.
The token above is a local demo credential and must match in each terminal.

**Terminal 2 — start the already-built dashboard:**

```sh
cd /home/aayush/projects/Nexus/web
export NEXUS_BACKEND_URL=http://127.0.0.1:8000
export NEXUS_API_TOKEN="test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
npm run start -- --hostname 127.0.0.1 -p 3000
```

If the frontend build is missing or its source has changed, run
`npm run build -- --webpack` in `web/` before starting it.
Open [Traffic & Ingestion](http://127.0.0.1:3000/traffic), load the mixed CSV preset
and click **Run Detection**. Open [Alerts](http://127.0.0.1:3000/alerts) and show an
alert's feature explanations. Operational alerts come from the live LightGBM.

**Terminal 3 — submit the curated shadow examples and their separate truth labels:**

```sh
cd /home/aayush/projects/Nexus
export NEXUS_API_TOKEN="test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

curl --fail-with-body -sS http://127.0.0.1:8000/api/v1/predictions \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/shadow_demo_v2/predictions.json \
  -o /tmp/nexus-teacher-predictions.json

curl --fail-with-body -sS http://127.0.0.1:8000/api/v1/replay/truth \
  -H "Authorization: Bearer $NEXUS_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/shadow_demo_v2/truth.json
```

Open [Shadow Fusion](http://127.0.0.1:3000/shadow) and click **Refresh evidence**.
For these 15 labeled examples, **Compared with its research baseline** shows
3 recovered attacks, 3 added false positives and 0 lost baseline detections;
3 persistent attack misses remain. The mixed CSV UI preset does not attach truth,
so those earlier rows contribute to coverage/disagreement, not these labeled metrics.
These are deliberately selected examples, not representative accuracy results.
Expand flow evidence to show routing, reconstruction error, anomaly rank and the
fusion cutoff. Press **Ctrl+C** in terminals 1 and 2 when finished.

### Where the shadow model and results live

| Item | Location |
| --- | --- |
| Inference, routing and bundle verification | [src/nexus/shadow.py](src/nexus/shadow.py) |
| Frozen model and policy | `artifacts/shadow_bundles/selective-exploits-s42-b03-v2/models.joblib` and `manifest.json` |
| Full raw-input parity evidence | `artifacts/shadow_bundles/selective-exploits-s42-b03-v2/parity.json`: 68,643 rows, zero decision mismatches |
| Dashboard and authenticated results API | `/shadow`; `GET /api/v1/shadow`; `GET /api/v1/shadow/predictions` |
| Persisted runtime results | `shadow_predictions` table in `NEXUS_DATABASE_PATH`; the commands above create `artifacts/teacher-demo-<timestamp>.sqlite3` |
| Curated cases and expected results | `artifacts/shadow_demo_v2/evidence.json`, `predictions.json`, `truth.json` |
| Original candidate evaluation | [Exploits / seed 42 metrics](experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42/metrics.json), filtering budget `0.03` and model `selective_fusion` |
| Reproduction, failure behavior and verification | [Shadow fusion guide](docs/SHADOW_FUSION.md) |

For that **single frozen research run**, overall attack recall moves from
93.8875% to 93.9040%: 10 additional attacks detected, zero lost detections and
6 additional false positives. Observed benign FPR moves from 2.5064% to 2.5790%.
These are internal development-evaluation results against the research LightGBM,
not against the live release and not independent confirmation. Parity measures
faithful implementation, not model improvement. Local bundles and runtime databases
are Git-ignored; committing code does not upload those artifacts.

## Where we stand

The eight-seed findings below describe the earlier selective-fusion suite. Later
three-seed score-only and AE/latent ablations are already complete; see the
[updated assessment and roadmap](docs/PROJECT_ROADMAP.md#current-research-evidence-takes-precedence-over-older-plans)
before treating the older “next experiments” list as pending work.

The latest evaluation contains **960 result rows**: four scenarios, ten methods,
three calibration FPR budgets and eight seeds. The scenarios are closed-set
(`none`) and withheld Reconnaissance, Exploits and DoS. Seeds are
7, 21, 42, 100, 123, 314, 1337 and 2026.

At the **3% calibration FPR budget**, selective fusion produces the following
means across eight seeds. Recovery counts cover all evaluation attacks in each
scenario; they are not counts exclusively from the withheld family.

| Scenario | LightGBM recall | Selective recall | Selective observed FPR | Additional attacks recovered | Lost LightGBM detections | Additional false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Closed set | 93.169% | 93.178% | 3.089% | 1.63 | 0 | 0.25 |
| Reconnaissance withheld | 92.734% | 92.767% | 2.986% | 16.50 | 0 | 1.00 |
| Exploits withheld | 93.882% | 94.141% | 3.067% | 154.88 | 0 | 3.25 |
| DoS withheld | 95.504% | 95.519% | 3.046% | 6.88 | 0 | 1.13 |

The strongest current case is **Exploits recovery**. At 3%, its withheld-family
recall increases from 93.746% to 94.189%; the overall recovery/added-FP ratio is
47.65 when pooling the repeated-seed counts. These repetitions are not unique
attacks from independent deployments.

Selective fusion preserved every baseline detection in all **96**
scenario/seed/budget configurations. This is enforced by its decision rule.
It does not guarantee zero extra false positives: only four of eight Exploits
runs met the nominal 3% evaluation cap, as did the paired LightGBM baseline.

The experiments also exposed approaches that should not be promoted:

- Unrestricted learned fusion has larger mean gains in some settings, but loses
  detections in others. At 3%, its mean net gain is +269 Exploits detections,
  versus -98.50 for Reconnaissance and -24.75 in the closed-set scenario.
- Naive OR raises mean FPR to roughly 5.8% when each detector receives a 3% budget.
- Standalone anomaly detectors have substantially weaker recall than LightGBM
  at practical FPR budgets. Denoising and scaling did not establish a universal
  improvement over the plain AE.
- Selective recovery is small for DoS and almost negligible for closed-set traffic.

These findings improve the model-selection evidence and identify a useful
recovery policy. They do not establish a universal accuracy improvement,
production readiness, or better runtime performance.

Source: [model assessment and evidence links](reports/current_model_assessment.md).
The older [validated binary result](reports/binary_validated_result.md) belongs
to a different experiment and must not be compared directly with these figures.

## Our approach

```mermaid
flowchart TD
    A[Flow features] --> B[LightGBM]
    A --> C[Normal-only anomaly models]
    B --> D{LightGBM alert?}
    D -->|Yes| E[Preserve alert]
    D -->|No| F{Uncertain score or suspicious anomaly?}
    C --> F
    F -->|No| G[Keep normal decision]
    F -->|Yes| H[Learned fusion score]
    B --> H
    C --> H
    H --> I{Pass calibrated recovery cutoff?}
    I -->|Yes| E
    I -->|No| G
```

This diagram describes the selective-fusion candidate, now available in opt-in shadow mode. Its operating
policy preserves LightGBM positives and considers uncertain or anomalous
negatives for recovery. Score gates are configurable; they are not calibrated
probabilities or proof that a flow is malicious.

1. **Keep related rows together.** Identical predictor rows share a partition.
   Groups containing a withheld family are excluded from all development stages.
2. **Separate development stages.** Remaining groups are allocated to fitting
   (55%), early stopping (10%), fusion fitting (10%), calibration (10%) and
   evaluation (15%). Row fractions vary with group size.
3. **Fit complementary detectors.** LightGBM sees fitting rows; anomaly models
   learn from benign fitting rows. Compare plain, scaled and denoising AEs,
   latent covariance distance, and Isolation Forest.
4. **Learn fusion on unseen fitting data.** Logistic fusion uses the LightGBM
   score, log reconstruction error and log latent distance on its disjoint partition.
5. **Calibrate the combined decision.** Thresholds use benign calibration rows
   at budgets of 1%, 3% and 5%. Selective recovery can spend only the allowance
   remaining after the preserved LightGBM alerts.
6. **Report both gains and costs.** Precision, recall, F1, FPR, ROC-AUC, PR-AUC,
   held-family recall, recovered/lost detections and gross/net added FPs accompany
   seed means, SDs and paired differences. Plots use the saved experiment CSVs.

The original nine comparisons remain reproducible alongside `selective_fusion`.
There is no automatic winner selection, model promotion or deployment.

## Contribution and comparison with existing projects

Our contribution is a **reproducible, budget-calibrated recovery layer** around a
strong supervised detector. The central question is: *what does the second
model catch, what does it break, and how many false alerts does it cost?*

This is an engineering and experimental contribution. Combining classifiers and
autoencoders, conditional routing, and anomaly detection are not new inventions.
A score-only fusion control is still needed to isolate how much of the gain
comes specifically from anomaly information.

| Comparison | What we can support |
| --- | --- |
| LightGBM alone, on our paired splits | Selective fusion recovers additional attacks without removing baseline alerts; Exploits is the strongest result. |
| Naive OR, on our paired splits | Selective recovery adds far fewer false positives, with a smaller recall increase. No dominance at identical external-test FPR is established. |
| A basic classifier demo | Nexus exposes leakage controls, held-family tests, seed variation and the cost of complementary detection, alongside an existing analyst-review application. |
| Snort, Suricata and Zeek | A potential complementary ML component. We have not run a paired benchmark demonstrating superiority over these tools. |

[Snort supports rule-based IDS/IPS detection](https://docs.snort.org/rules/).
[Suricata already produces alerts and anomaly events](https://docs.suricata.io/en/suricata-8.0.5/output/eve/eve-json-output.html),
and [Zeek supports programmable notice policies](https://docs.zeek.org/en/master/frameworks/notice.html).
Anomaly handling and analyst-facing alerts are not exclusive to Nexus.
Demonstrating signature-bypass coverage requires a common traffic corpus and an
actual signature-IDS baseline; withheld-family flow classification is not that test.

## Research foundations

| Paper | Insight used and scope |
| --- | --- |
| [UNSW-NB15 — Moustafa & Slay, 2015](https://doi.org/10.1109/MILCIS.2015.7348942) | Benchmark foundation for the current flow-feature experiments. |
| [LightGBM — Ke et al., NIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Gradient-boosted tree foundation for our primary supervised detector. |
| [Kitsune — Mirsky et al., NDSS 2018](https://arxiv.org/abs/1802.09089) | Motivation for normal-behavior learning with autoencoders. Our flow-level models do not reproduce Kitsune's online packet-feature ensemble. |
| [Dos and Don'ts of ML in Computer Security — Arp et al., USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/arp) | Evaluation guidance: leakage, sampling bias, misleading metrics and unsupported security claims. |
| [Autoencoders for Anomaly Detection are Unreliable — Bouman & Heskes, 2025 preprint](https://arxiv.org/abs/2501.13864) | Reconstruction error can fail to distinguish anomalies; motivates comparing scores and measuring complementary value rather than assuming AE superiority. |
| [DeepAID — Han et al., CCS 2021](https://arxiv.org/abs/2109.11495) | Inspiration for investigating anomaly explanations and false positives. DeepAID's interpreter is not implemented; current TreeSHAP explains LightGBM. |

These references inform the design. Their published results are not results
achieved by Nexus, and we do not claim to outperform their systems.

## Reproduce the model research

Python 3.11+ and the local UNSW-NB15 training CSV are required. Models, datasets,
split indices, raw results and release bundles remain Git-ignored and local.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
.venv/bin/python -m pytest -q tests/test_anomaly_detection_models.py
```

Start a new experiment directory; existing directories are never overwritten:

```sh
.venv/bin/python training/run_novelty_experiment.py \
  --output-dir artifacts/fusion_next_run \
  --families none Reconnaissance Exploits DoS \
  --seeds 42 123 2026 --fpr-budgets 0.01 0.03 0.05
```

This trains and evaluates the models. If the local v1 suite exists, use
`--reuse-dir artifacts/anomaly_comparison_v1` to reuse its matching seeds and
fitted models. Full commands for all eight seeds, calibration, reporting and
plots are in the [execution guide](training/FUSION_EXPERIMENTS.md).
Everything runs in the foreground. No experiment starts automatically.

## Limitations and next steps

- Internal development holdouts are not independent external confirmation.
  Repeated seeds share data; no statistical-significance claim is made.
- A calibration FPR cap can be exceeded on evaluation or drifting traffic.
  In 88/96 configurations, LightGBM already consumed all calibration FP slots.
- Exact-row grouping does not establish session or temporal independence.
- High benchmark precision need not transfer to low-prevalence production traffic.
- Reconnaissance subtype labels cannot reliably be joined to benchmark rows.
  TCP/FIN and service slices are diagnostics, not invented attack subtypes.
- Attack-type classification exists as an optional offline diagnostic; unknown
  rejection and end-to-end family classification remain unfinished.

Next model work: a LightGBM-score-only fusion control, plain-AE versus denoising-AE
fusion ablations, explicit budget allocation, and independent confirmation.
Measure latency and drift before claiming operational improvement. Shadow fusion integration is implemented; production promotion and independent
confirmation remain separate milestones.

## Repository layout

| Path | Role |
| --- | --- |
| `training/` | Current experiment runner, anomaly/fusion models, reporting and required historical training helpers |
| `tests/` | Model invariants, API behavior and integration checks |
| `reports/` | Curated research assessments; raw generated reports stay local |
| `models/` | Local frozen checkpoints, including `lightgbm_validated_v1` |
| `artifacts/` | Local experiment suites, aggregate CSVs, plots, logs and release bundles |
| `experiments/` | Retained historical LightGBM provenance needed by the binary report |
| `src/nexus/`, `web/` | Existing LightGBM-serving API and SOC console |
| `docs/` | API contract and local demo instructions |

For the application, use the [local demo guide](docs/LOCAL_DEMO.md). Dataset replay
is historical traffic replay, not live packet capture or independent evaluation.

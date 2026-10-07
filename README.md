# Nexus

Nexus is a network intrusion-detection project that combines **LightGBM** with a
**normal-only baseline autoencoder (AE)** to classify UNSW-NB15 flow records as
normal or attack. The experiments measure attack recall, missed attacks and
false alarms to make the detection tradeoff explicit.

Two final experimental configurations are exported under `models/`:

- **Uncertainty-band four-mode, AE 10%** — the higher-recall configuration.
- **Mode confidence, AE 5%** — the configuration with fewer false alarms.

Both share the same trained LightGBM and baseline AE, with different frozen
decision rules. The local API and SOC dashboard still serve the separate
LightGBM v2 bundle; the exported fusion models have **not** been integrated into
application serving. The application supports analyst review, not automatic
traffic blocking.

## Start here

| Reference | Contents |
|---|---|
| [Model artifacts and configs](artifacts/config.md) | Final model files, training settings and decision configurations |
| [Complete training configuration](training/configs.md) | Dataset partitions, preprocessing, hyperparameters and reproduction commands |
| [Working model record](working_v2.md) | Exact rules, confusion matrices and loading instructions |
| [Training directory](training/README.md) | Retained scripts and their dependencies |
| [AE FPR sweep](experiments/ae_fpr_5_10_20261007/REPORT.md) | Results for the 5% and 10% AE calibration targets |
| [Local demo](docs/LOCAL_DEMO.md) | LightGBM-serving backend, frontend and recorded traffic replay |
| [API contract](docs/api.md) | Prediction, alert and analyst-feedback endpoints |

## Final model packages

| Configuration | Package | Machine-readable config |
|---|---|---|
| Uncertainty-band four-mode, AE 10% | [models/uncertainty_band_ae10](models/uncertainty_band_ae10/) | [config.json](models/uncertainty_band_ae10/config.json) |
| Mode confidence, AE 5% | [models/mode_confidence_ae05](models/mode_confidence_ae05/) | [config.json](models/mode_confidence_ae05/config.json) |

The local application serves LightGBM predictions with TreeSHAP explanations,
prediction logging and an analyst dashboard. Selective fusion also supports
opt-in shadow serving with frozen-checkpoint parity verification; it never
creates live alerts.

For the maintained documentation index, see [docs/README.md](docs/README.md).
It links to the local demo, API contract, shadow-fusion guide, research status,
roadmap, training, reports and frontend guides.

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

### Export contents

Each package contains:

- `weights.joblib`: fitted LightGBM, AE and all required preprocessing.
- `config.json`: exact feature schema, thresholds, training settings and library versions.
- `metrics.json`: official-test results.
- `manifest.json`: SHA-256 checks for weights, config and metrics.
- `verification.json`: verification against the experiment's saved predictions.
- `README.md`: package overview.

Each export reproduced **all 82,332 official-test predictions exactly**, with
zero mismatches. Inference uses [the package loader](src/nexus/fusion_model.py)
and does not depend on experiment files or training-module imports.

Model binaries and raw datasets are local, Git-ignored artifacts. A fresh clone
needs the complete model directories, including `weights.joblib`, copied from
the trained workspace, or a new training/export run. JSON configs alone are
not trained models.

## Results on the same official test set

All rows below use the same **82,332 test records**: **37,000 normal** and
**45,332 attacks**. Attack is the positive class.

| Model | Attack recall | Missed attacks | False alarms | System FPR | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary-only LightGBM | 96.5367% | 1,570 | 6,335 | 17.1216% | 87.3545% | 91.7164% | 90.3986% |
| Uncertainty-band four-mode, AE 10% | **97.2029%** | **1,268** | 6,039 | 16.3216% | 87.9468% | 92.3435% | 91.1250% |
| Mode confidence, AE 5% | 95.9543% | 1,834 | **5,343** | **14.4405%** | **89.0604%** | **92.3789%** | **91.2829%** |

Compared with binary-only, uncertainty-band fusion detects **302 more attacks
net** and produces **296 fewer false alarms**. Mode confidence misses **264
additional attacks** while removing **992 false alarms**. The two exported
configurations therefore represent different operating tradeoffs; the highest
F1 is not automatically the preferred security policy.

The AE's **5% and 10% settings are calibration FPR targets**, not limits on
combined-system test FPR. The recent AE alone measured 10.31% and 16.45% test
FPR at those respective settings. See the [full sweep report](experiments/ae_fpr_5_10_20261007/REPORT.md)
for all evaluated rules and confusion matrices.

## How the selected rules work

LightGBM produces a raw attack score `p`. The AE reconstructs the flow features
and produces mean squared reconstruction error `e`. Three frozen error
boundaries divide AE scores into **no anomaly, low, mid and high**. Equality
enters the higher mode. The base binary decision is attack when
`p >= 0.5776925765603604`.

The selective-fusion candidate is also available in opt-in shadow mode. Its
operating policy preserves LightGBM positives and considers uncertain or
anomalous negatives for recovery. Score gates are configurable; they are not
calibrated probabilities or proof that a flow is malicious.

| AE mode | Uncertainty-band four-mode, AE 10% | Mode confidence, AE 5% |
|---|---|---|
| No anomaly | Inside the score band: normal | Attack if `p >= 0.65` |
| Low | Inside the score band: follow binary | Attack if `p >= 0.5776925765603604` |
| Mid or high | Inside the score band: attack | Attack if `p >= 0.5776925765603604` |

For uncertainty-band fusion, the band is **`0.10 <= p <= 0.65`**, including both
endpoints. Outside that band, preserve the binary prediction. For mode
confidence, a score below the applicable cutoff means normal.

The configurations use different AE boundaries. Their exact values are saved
in the package configs and [working_v2.md](working_v2.md). These are anomaly
score bands, not ground-truth attack severity classes. LightGBM's score is not
a verified live attack probability.

## Load a final model

From the repository root, install the project and training dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,training]'
```

For existing serialized packages, match the library versions recorded in their
configs. The verified export environment used Python 3.12.3; the project's
minimum Python version is 3.11.

```python
import pandas as pd
from nexus.fusion_model import load_fusion_model

flows = pd.read_csv(
    "data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
)

recall_model = load_fusion_model("models/uncertainty_band_ae10")
fewer_alerts_model = load_fusion_model("models/mode_confidence_ae05")

labels = recall_model.predict(flows)  # 0 = normal, 1 = attack
print(fewer_alerts_model.predict_details(flows).head())
```

Supply a DataFrame containing the **42 raw feature columns** listed in each
config. Do not scale or one-hot encode inputs yourself. Extra columns such as
`id`, `label` and `attack_cat` are ignored; missing required features are rejected.

`predict_details()` returns the final prediction, LightGBM attack score,
binary prediction, AE reconstruction error and anomaly mode.

## Training and reproduction

The final pipeline uses grouped partitions of the official training CSV.
Identical predictor rows stay in the same partition. LightGBM is refitted on
124,861 fit/early rows. The baseline AE and its preprocessing learn from
31,861 normal fit rows; AE FPR thresholds use 8,002 normal calibration rows.

| Stage | Retained script |
|---|---|
| Select LightGBM hyperparameters | `training/train_validated_lightgbm.py` |
| Train the final base detectors | `training/experiment_four_mode_ae.py`, using `train_autoencoder.py` |
| Explore confidence cutoffs and uncertainty bands | `training/experiment_confidence_ae.py`, `experiment_uncertainty_residuals.py` |
| Calibrate AE 5%/10% thresholds and compare rules | `training/experiment_ae_fpr_sweep.py` |
| Export and verify final packages | `training/export_selected_fusion_models.py` |

Exact settings and new-output commands are in [training/configs.md](training/configs.md).
The retained scripts preserve their historical source-artifact paths; choosing
a new output directory does not automatically redirect downstream inputs.
Existing final package directories are not overwritten by the exporter.

Run focused model checks with:

```sh
.venv/bin/python -m pytest -q \
  tests/test_autoencoder.py tests/test_fusion_model.py tests/test_ae_fpr_sweep.py
```

The active `training/` directory covers this pipeline, shared dependencies,
serving-bundle checks and selective-fusion shadow support. The 26 unrelated research
scripts and their research-only tests are preserved in the
[historical source archive](experiments/archived_training_20261007/README.md).
Older reports describe their own models, splits and operating points, not the
current selected configurations.

## Local application

The FastAPI backend and Next.js SOC console provide LightGBM predictions,
TreeSHAP explanations, prediction logging and analyst feedback. After installing
backend and frontend dependencies, the local launcher is:

```sh
.venv/bin/python scripts/dev.py
```

Follow the [local demo guide](docs/LOCAL_DEMO.md) for bundle requirements, ports,
authentication and frontend setup. Its serving bundle is separate from the two
fusion packages above. Dataset replay is recorded traffic replay, not live
packet capture.

Shadow fusion integration is implemented; independent confirmation and production
promotion remain separate milestones. See the [roadmap](docs/PROJECT_ROADMAP.md)
for completed ablations and remaining work.

## Repository layout

| Path | Role |
|---|---|
| `models/` | The two selected model packages and their exact configs |
| `training/` | Final-model training, calibration, export and required helpers |
| `artifacts/config.md` | Artifact-folder reference for models and training settings |
| `artifacts/bundles/` | Separate application-serving bundles |
| `experiments/` | Saved experiments, metrics, decisions and archived research source |
| `reports/` | Historical assessments and supporting reports |
| `preprocessing/`, `data/` | Dataset preparation and local raw/processed data |
| `src/nexus/` | Model loader, API, inference, logging and analyst workflow |
| `web/` | SOC dashboard |
| `tests/` | Active model, API and integration tests |
| `docs/` | Application setup and API documentation |

## Earlier selective-fusion research

The eight-seed findings describe the earlier selective-fusion suite. Later
three-seed score-only and AE/latent ablations are already complete; see the
[updated assessment and roadmap](docs/PROJECT_ROADMAP.md#current-research-evidence-takes-precedence-over-older-plans)
before treating the older “next experiments” list as pending work.

That evaluation contains **960 result rows**: four scenarios, ten methods,
three calibration FPR budgets and eight seeds. The scenarios are closed-set
(`none`) and withheld Reconnaissance, Exploits and DoS. Seeds are
7, 21, 42, 100, 123, 314, 1337 and 2026.

## Evaluation limits

The official test has been used repeatedly during development and includes
**8,541 rows with predictors also present in the training CSV**. The final
configuration choices reflect that development history; these results are not
fresh-holdout confirmation. Exact-row grouping does not establish session or
temporal independence, and calibration FPR need not transfer to new traffic.

Earlier withheld-family and multi-seed studies remain historical research.
They do not establish zero-day detection or superiority over signature-based
IDS tools. The next deployment steps are confirmation on new data, operational
false-alarm and latency assessment, and explicit integration of a selected
fusion package into the application.

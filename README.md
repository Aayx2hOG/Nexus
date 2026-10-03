# Nexus

Network intrusion detection research with validated LightGBM training, real-model TreeSHAP inference, all-prediction logging, an analyst-review API, and a Next.js SOC console.

## Project overview

Nexus investigates normal-versus-attack detection on UNSW-NB15 and provides an
end-to-end operational pipeline for real-time model inference and intrusion alert review.
The implemented research workflow compares LightGBM with Random Forest using duplicate-grouped
partitions, separate threshold calibration, and a frozen evaluation procedure.
The operational system connects the cryptographically sealed LightGBM release bundle to
a FastAPI backend, an all-prediction audit log, an explainability engine (TreeSHAP),
and a Next.js analyst triage console.

| Documentation | Contents |
| --- | --- |
| [Binary results and improvement](reports/binary_validated_result.md) | Measured gains, threshold rationale, comparison limits, and evidence links. |
| [Reproducible training and evaluation](training/VALIDATED_TRAINING.md) | Dataset handling, commands, split methodology, and frozen evaluation. |
| [API contract](docs/api.md) | Implemented endpoints, validation, alert persistence, and analyst feedback. |

Raw metrics, split audits, and manifests remain under `experiments/` and
`reports/` as supporting evidence. Older experiment summaries are retained as
JSON rather than duplicate Markdown reports.

## Project contribution and progress

Nexus focuses on making intrusion-detection results reproducible and actionable
through recall-constrained classification, traceable evaluation, and an
analyst-review API. Its contribution is an end-to-end vertical slice: connecting
a validated LightGBM detector to an operational API with TreeSHAP explanations,
all-prediction logging, an analyst review queue, and a Next.js SOC console.

The cited research provides datasets, algorithms, and evaluation guidance.
Nexus builds on those foundations with an explicit false-alert objective and
a persistent review workflow. This is an engineering contribution, not a claim
of a new learning algorithm or superiority over the published systems. The
measured improvement is against our paired Random Forest baseline, not against
the papers' reported results.

| Capability | Completed work | Remaining work |
| --- | --- | --- |
| Known-attack detection | LightGBM/RF training and binary candidate selection; 18.5% fewer false positives than paired RF on selection data. Real LightGBM inference (bundle v1.0.0) with TreeSHAP feature attributions. | Compare frozen old/new checkpoints on common evaluation rows and confirm on fresh data. |
| Reproducibility | Duplicate-grouped partitions, fitting-only preprocessing, separate threshold calibration, saved audits, and frozen source/model hashes. | Assess split sensitivity and session/time dependence where metadata permits. |
| Analyst review | Flow validation, persistent alerts, TreeSHAP signed explanations, all-prediction logging, Next.js SOC console, missed-attack sampling, and dataset replay streaming. | Multi-user authentication, role-based permissions, and WebSocket notifications. |
| Unknown-attack detection | Research direction and evaluation requirements documented. | Implement and compare normal-only autoencoder and Isolation Forest candidates using held-out attack families and a common false-alert budget. |

The intended extension combines known-attack classification with complementary
anomaly detection and analyst review. Its value must be demonstrated by comparing
the classifier alone with each added detector on identical evaluation data,
including the combined alert rule's false-positive rate. A hybrid design alone
does not establish research novelty or prove zero-day detection.

## Model research

Use the [validated training workflow](training/VALIDATED_TRAINING.md) for new runs.
The [latest binary report](reports/binary_validated_result.md) documents the current
validation winner: 95.25% attack recall and 3.98% false-positive rate, with 18.5%
fewer false positives than the paired Random Forest. Independent confirmation
and comparison with older checkpoints remain outstanding.

Install research and development dependencies with:

```sh
python -m pip install -e '.[dev,training]'
```

The validated runner reads the raw training CSV and performs its own preprocessing.
Scripts in `preprocessing/` reproduce the older feature arrays and are not required
for this workflow. Historical training modules remain because the current runner
imports their helpers and the frozen experiment verifies their source hashes.
Datasets and model binaries stay local.

Autoencoders, Isolation Forest, and novelty detection are future research directions.
Real-model API inference (v1.0.0) with TreeSHAP attribution is fully operational.

## Research foundations

These references explain the dataset and methods used in Nexus, guide evaluation,
and identify possible extensions. Nexus is not a reproduction of every cited
system, and their published results are not results achieved by this project.

| Paper | Relevance to Nexus |
| --- | --- |
| Moustafa & Slay (MILCIS 2015), [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://doi.org/10.1109/MILCIS.2015.7348942) | Source of the benchmark used for binary and attack-family classification. See the [official dataset description](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for its construction and features. |
| Ke et al. (NIPS 2017), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Algorithmic foundation of the implemented LightGBM classifier; it does not establish intrusion-detection performance on our splits. |
| Arp et al. (USENIX Security 2022), [Dos and Don'ts of Machine Learning in Computer Security](https://www.usenix.org/conference/usenixsecurity22/presentation/arp) | Evaluation guidance on leakage, sampling bias, experimental design, and misleading performance claims. |
| Sommer & Paxson (IEEE S&P 2010), [Outside the Closed World: On Using Machine Learning for Network Intrusion Detection](https://gangw.cs.illinois.edu/class/cs562/papers/Closed_World-IDS-sp10.pdf) | Explains the gap between benchmark anomaly detection and operational intrusion detection, including traffic diversity, evaluation difficulties, and the cost of errors. |
| Mirsky et al. (NDSS 2018), [Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection](https://arxiv.org/abs/1802.09089) | Reference for a possible autoencoder extension. Kitsune uses online packet-derived features and an ensemble; a future flow-CSV autoencoder here would be an adaptation, not a reproduction. **Not implemented.** |
| Liu, Ting & Zhou (ICDM 2008), [Isolation Forest](https://www.lamda.nju.edu.cn/publication/icdm08b.pdf) | Reference for a possible anomaly-detection baseline using random isolation trees. **Not implemented.** |

## Evaluation pitfalls and safeguards

The following project practices apply the evaluation concerns described by
[Arp et al.](https://www.usenix.org/conference/usenixsecurity22/presentation/arp)
and [Sommer & Paxson](https://gangw.cs.illinois.edu/class/cs562/papers/Closed_World-IDS-sp10.pdf).
They are not claims that the cited authors made these mistakes or that Nexus has
eliminated every source of bias.

| Pitfall | Current safeguard and remaining limitation |
| --- | --- |
| Target or preprocessing leakage | The validated runner excludes `id`, `label`, and `attack_cat` from predictors and fits preprocessing only on fitting/refit rows. Older experiments used weaker isolation and remain historical evidence. |
| Duplicate leakage and correlated traffic | Identical predictor rows stay in one partition; final evaluation removes development overlap. Exact-row grouping does not establish session or temporal independence. Conflicting labels are retained and reported. |
| Tuning on the test set or reporting the best search score as final performance | Threshold calibration and selection are separate; evaluation verifies frozen hashes. Current scores are selection estimates. The official test set has already informed development, so fresh data is still needed for independent confirmation. |
| Unfair comparisons | Current LightGBM and RF use the same partitions. RF has a smaller search budget; old full-test scores cannot be compared directly with new selection or overlap-filtered scores. |
| High accuracy hiding false alarms and missed attacks | Binary selection minimizes FPR subject to recall >=95%; reports include precision, recall, F1, AUC, confusion counts and false alerts per 1,000 normal flows. The current rate is about 40 per 1,000, and benchmark precision may change with deployment attack prevalence. |
| Treating a model score as a trustworthy attack probability | Threshold calibration chooses a decision cutoff; it does not calibrate probabilities. Class weighting changes score scales, and the recall margin does not guarantee performance under drift. |
| Claiming unknown-attack detection from a closed-set benchmark | No zero-day detection claim is made. A future anomaly experiment must exclude the held-out attack family from all fitting and selection, use a separate benign calibration set, and evaluate the combined detector's false-alert rate. An anomaly is not automatically an attack. |

Repeated predeclared splits, independent holdout evaluation, session/time-aware
testing where metadata permits, and deployment monitoring remain outstanding.
The [results report](reports/binary_validated_result.md) records the scope of the
current improvement claim.

## Running the complete system (Backend, Frontend & Traffic Replay)

Nexus consists of three operational components:
1. **FastAPI Backend (`src/nexus`)**: Serves real-time LightGBM predictions, computes TreeSHAP explanations, dual-logs predictions, and manages analyst feedback in SQLite.
2. **Next.js SOC Console (`web/`)**: Minimal, high-density dashboard for reviewing alerts, inspecting signed feature contributions, and auditing sampled non-alert flows.
3. **Dataset Replay Adapter (`nexus.replay`)**: Streams held-out evaluation traffic into the running system in real time.

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** (Node 20+ recommended) and `npm`

---

### Step 1: Environment & Python dependencies

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,training]'
```

*(Note for macOS Apple Silicon users: LightGBM requires OpenMP. If not present, run `brew install libomp`).*

---

### Step 2: Ensure the release bundle is present

The backend requires the cryptographically sealed release bundle (`artifacts/bundles/v1.0.0/`). Because bundles and model binaries are ignored by Git:
- **On a new machine / fresh clone**: Copy the `artifacts/bundles/v1.0.0/` directory into your project root, OR export it if local training outputs and datasets are present:

```sh
python -m training.export_release_bundle
```

This verifies source code and training partition SHA-256 hashes, then packages `model.joblib`, `preprocessor.joblib`, and `manifest.json`.

---

### Step 3: Start the Backend API (FastAPI)

Set your environment variables and start Uvicorn on port `8000`:

```sh
export NEXUS_BUNDLE_VERSION=v1.0.0
export NEXUS_DATABASE_PATH=data/nexus.sqlite3
export NEXUS_API_TOKEN="test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
export NEXUS_REVIEWER_ID=local-analyst

uvicorn nexus.api:app --host 127.0.0.1 --port 8000 --reload
```

> **Authentication Tip**: The Next.js frontend API proxy defaults to `test-token-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`. Using this token in the backend avoids `401 Unauthorized` errors. If you use a custom secret token, export the same `NEXUS_API_TOKEN` before launching the frontend or define it in `web/.env.local`.

Verify backend health:
- Readiness: `curl http://127.0.0.1:8000/health/ready`
- Interactive API Docs: <http://127.0.0.1:8000/docs>

---

### Step 4: Seed Demo Alerts (Recommended on Fresh Clone)

On a freshly cloned repository, the database is empty (`0 alerts`). To immediately populate the queue with sample alerts for testing the UI:

```sh
NEXUS_DATABASE_PATH=data/nexus.sqlite3 python -m nexus.demo
```

---

### Step 5: Start the Frontend Console (Next.js)

In a **second terminal**, start the Next.js development server:

```sh
cd web
npm install
npm run dev
```

For production builds:
```sh
npm run build
npm run start -- -p 3000
```

Open your browser to <http://localhost:3000> (renders in dark mode).

---

### Step 6: Ingest Traffic via Dataset Replay

In a **third terminal**, stream real network traffic from the frozen held-out evaluation split into the live system *(requires `data/raw/CSV_Files/.../UNSW_NB15_testing-set.csv` and `split_indices.npz`)*:

```sh
source .venv/bin/activate

# Stream 500 flows at 50 flows/second:
python -m nexus.replay --limit 500 --rate 50

# Or stream 1,000 flows from the attack partition:
python -m nexus.replay --offset 6880 --limit 1000 --rate 100
```

**How it works:**
1. Ground-truth labels (`label`, `attack_cat`) are strictly stripped before transmission so the model receives zero hints.
2. Batches are POSTed to `http://127.0.0.1:8000/api/v1/predictions`.
3. Flows scoring $\ge 0.577693$ trigger alerts with TreeSHAP explanations.
4. Ground-truth labels are stored separately in the `replay_truth` table for empirical evaluation.
5. Alerts appear live in the web dashboard!

---

### Step 7: Using the SOC Analyst Console

Navigate through the console at <http://localhost:3000>:
- **/alerts**: Triage queue of active intrusion alerts. Filter by verdict, inspect raw scores, and view high-level triage statistics.
- **/alerts/[id]**: Alert detail page with TreeSHAP signed feature contributions (red = pushed toward alert, green = pushed toward normal) and interactive analyst verdict submission (Confirmed Attack, False Positive, Needs Investigation).
- **/review-sample**: Missed-attack audit queue sampling non-alert traffic (scores $< 0.5777$) to catch low-and-slow stealth attacks.
- **/model**: Release bundle integrity, SHA-256 hashes, selection partition metrics, and empirical replay confusion matrix.
- Press <kbd>?</kbd> anywhere in the console to view keyboard navigation shortcuts.

---

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health/live` | Process health. |
| GET | `/health/ready` | Release bundle readiness and cryptographic SHA-256 verification. |
| GET | `/api/v1/schema` | Provisional flow JSON Schema. |
| POST | `/api/v1/flows/validate` | Validate a batch without storing or scoring it. |
| POST | `/api/v1/predictions` | Real LightGBM inference, TreeSHAP explanation, prediction logging, and alert creation. |
| GET | `/api/v1/alerts` | Paginate stored alerts; optional verdict and severity filters. |
| GET | `/api/v1/alerts/{alert_id}` | Retrieve an alert, TreeSHAP contributions, and review status. |
| GET | `/api/v1/alerts/{alert_id}/feedback` | Paginate append-only review history. |
| POST | `/api/v1/alerts/{alert_id}/feedback` | Record a review with retry and concurrency protection. |
| GET | `/api/v1/predictions/sample` | Stratified sample of non-alert flows for missed-attack auditing. |
| POST | `/api/v1/predictions/{flow_id}/feedback` | Record analyst verdict on sampled non-alert flows. |
| GET | `/api/v1/model` | Model architecture, decision threshold, SHA-256 hashes, and selection metrics. |
| GET | `/api/v1/stats` | High-level triage statistics and verdict breakdown. |
| GET | `/api/v1/replay/summary` | Empirical confusion matrix and false alert rate from replay truth. |

Alert and review endpoints require the configured bearer token. The server assigns the reviewer ID; clients cannot supply reviewer identity, timestamps, or training eligibility. Without a token configured, alert endpoints return 503; missing or incorrect credentials return 401.

See [the API contract](docs/api.md) for request limits, feedback examples, pagination, and error behaviour.

## Code layout

| Module | Responsibility |
| --- | --- |
| `api.py` | App creation and database startup. |
| `routes.py` | HTTP routing, authentication, prediction logging, and stats endpoints. |
| `transport.py` | Header, body size, JSON, and timeout checks. |
| `schemas.py` / `alerts.py` | Flow, alert, prediction, and review contracts. |
| `bundle.py` | Release bundle manifest loading and SHA-256 integrity verification. |
| `inference.py` | Preprocessing, LightGBM inference engine, and TreeSHAP attribution. |
| `storage.py` | Parameterized SQLite queries, atomic dual-logging, and replay truth. |
| `replay.py` | Dataset replay CLI streaming held-out evaluation traffic into the API. |
| `config.py` / `errors.py` | Configuration and consistent error responses. |
| `demo.py` | Explicit synthetic alert seeding. |
| `web/` | Next.js SOC analyst console (App Router, Tailwind CSS, TanStack Query). |

Database calls run in FastAPI's worker pool and use a separate connection per operation. SQLite writes serialize through transactions; review history and the alert's revision commit together.

### Verification & Testing

```sh
# Backend test suite:
pytest -q

# Code formatting & linting:
ruff check src tests
ruff format --check src tests

# Frontend build & typecheck:
cd web && npm run build
```

## Next milestones

- [x] Integrate evaluated release bundle for real predictions and all-prediction logging.
- [x] Build the dashboard around stored alert and feedback endpoints.
- [x] Stream held-out evaluation data with isolated ground truth.
- [ ] Multi-class attack family classifier (e.g. Exploits, DoS, Reconnaissance).
- [ ] Out-of-Distribution / novelty detection (Autoencoders or Isolation Forest).
- [ ] Live PCAP / network interface ingestion adapter.
- [ ] Multi-user identity, authentication, and role-based permissions.
- [ ] WebSocket notifications for real-time alert push.

Keep this development service on loopback. Multi-user identity/authorization, rate limiting, deployment hardening, backups, and full access auditing remain future work. Analyst reviews are recorded with provenance, but none automatically become training labels.

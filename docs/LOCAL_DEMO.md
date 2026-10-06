# Local LightGBM demo

This demo serves the existing LightGBM release. Selective fusion can run alongside it in opt-in [shadow mode](SHADOW_FUSION.md); live alerts remain LightGBM-only.
Run commands from the repository root unless a command changes directories.

Nexus consists of three operational components:
1. **FastAPI Backend (`src/nexus`)**: Serves real-time LightGBM predictions, computes TreeSHAP explanations, dual-logs predictions, and manages analyst feedback in SQLite.
2. **Next.js SOC Console (`web/`)**: Minimal, high-density dashboard for reviewing alerts, inspecting signed feature contributions, and auditing sampled non-alert flows.
3. **Dataset Replay Adapter (`nexus.replay`)**: Replays recorded dataset traffic into the running system.

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

The backend requires the hash-checked release bundle (`artifacts/bundles/v1.0.0/`). Because bundles and model binaries are ignored by Git:
- **On a new machine / fresh clone**: Copy the `artifacts/bundles/v1.0.0/` directory into your project root, OR export it if local training outputs and datasets are present:

```sh
python -m training.export_release_bundle
```

The exporter verifies the experiment checkpoint, selection record, and split hashes, then packages the saved winning estimator and its fitted preprocessing without retraining. Serving scores are tested against that frozen checkpoint.

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

### Step 4: Use real model results

Do not seed synthetic alerts for the presentation. After starting the frontend,
open **/traffic**, load the mixed CSV preset (or paste/upload your own 42-feature
CSV), and click **Run Detection**. Presets provide inputs; scores and decisions
are computed by the loaded model. Labels are excluded from prediction requests.
The website probe uses estimated and fixed network features and is experimental;
use CSV flows to demonstrate model inference.

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
npm run build -- --webpack
npm run start -- -p 3000
```

Open your browser to <http://localhost:3000> (renders in dark mode).

---

### Step 6: Ingest Traffic via Dataset Replay

In a **third terminal**, replay historical dataset rows into the local system *(requires `data/raw/CSV_Files/.../UNSW_NB15_testing-set.csv` and `split_indices.npz`)*:

```sh
source .venv/bin/activate

# Stream 500 flows at 50 flows/second:
python -m nexus.replay --limit 500 --rate 50

# Or replay 1,000 rows starting at offset 6,880 (not an attack-only partition):
python -m nexus.replay --offset 6880 --limit 1000 --rate 100
```

The default replay CSV is the historical official testing file. The current adapter
applies saved split indices only to a 175,341-row input; it does not enforce a
held-out partition for other sizes. Do not use demo replay as proof of split isolation.

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
- **/review-sample**: Missed-attack audit queue sampling non-alert traffic (scores $< 0.5777$) to review possible missed attacks.
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

See [the API contract](api.md) for request limits, feedback examples, pagination, and error behaviour.

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
| `replay.py` | Dataset replay CLI for historical traffic. |
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



## Integration verification — 2026-10-04

The local `v1.0.0` release packages the frozen LightGBM trial-4 binary validation
winner from `models/lightgbm_validated_v1/binary/model.joblib`, including its
fitted preprocessing and threshold `0.5776925765603604`. This is the winner for
minimum validation FPR subject to at least 95% recall, not a claim of universal
or independently tested superiority.

- Backend: **223 tests passed**, including API/checkpoint parity and persisted alerts.
- Frontend: production build and TypeScript passed with `npm run build -- --webpack`.
  The default Turbopack build was blocked by local-port restrictions in this environment.
- Local HTTP integration: actual frontend CSV parser → Next.js proxy → FastAPI →
  model → proxy response. Mixed preset: **100 flows, 59 alerts, 41 normal,
  94 distinct scores; maximum difference from the frozen checkpoint: 0.0**.
  These counts demonstrate integration, not accuracy on an independent test set.
- `/traffic`, `/model`, and `/alerts` returned HTTP 200. Visual browser interaction
  was not automated. Temporary verification servers and database were cleaned up.
- Changed Python files pass lint and formatting; repository-wide checks still
  report existing issues in unrelated files.

Restart the backend after exporting so it loads the updated artifacts. Start it
with `NEXUS_BUNDLE_VERSION=v1.0.0` and the same API token as the frontend, following
Step 3. For the presentation, use `/traffic` → mixed preset → Run Detection, then
open a resulting alert. The website probe includes estimated/fixed inputs and is
not a validated website security assessment.

## Policy Lab: demonstrate detection benefit versus analyst cost

Open **/policy** after starting the API and UI. The lab reads predictions and
separately recorded dataset truth for the serving bundle. The mixed CSV UI preset
alone does not record truth. For a labeled historical demonstration, run from the
repository root with the same token as the backend:

```sh
.venv/bin/python -m nexus.replay \
  --csv 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv' \
  --split-file models/lightgbm_validated_v1/split_indices.npz \
  --split selection --limit 500 --rate 50 \
  --url http://127.0.0.1:8000 --token "$NEXUS_API_TOKEN" \
  --db /tmp/nexus-unused-replay-truth.sqlite3
```

The CSV above is explicit so the existing adapter applies its selection indices
to the 175,341-row training file. It still uses a row-count heuristic rather than
a dataset hash check; this is historical demonstration evidence, not a new
independent evaluation. The `--db` path must not exist: this makes the adapter
submit truth via the authenticated API, avoiding accidental writes to a different
local database. No database at that path needs to be created.

1. Click **Refresh replay**. Check labeled and unlabeled counts.
2. Reset to the release threshold: hypothetical and recorded decisions should agree.
3. Lower the threshold: show recovered attacks alongside added false alerts.
4. Raise it: show lost detections and false alerts removed.
5. Inspect family coverage and lower assumed attack prevalence to illustrate
   why benchmark precision does not imply a manageable production alert queue.
6. Export evidence JSON and identify its bundle, threshold and cohort watermark.

All saved replay runs for the selected bundle contribute; repeated CLI executions
create new flow IDs. Use a fresh `NEXUS_DATABASE_PATH` when you need one isolated
presentation cohort. The lab does not deploy a threshold or execute fusion.
See [the roadmap and rubric walkthrough](PROJECT_ROADMAP.md) for the novelty pitch,
research caveats and next model/backend/system improvements.

### Policy Lab verification — 2026-10-06

- Full backend/research regression suite: **253 passed** (14 dependency/training warnings).
- Real-model integration: 100 mixed preset flows scored through the API, retried
  with identical results and no extra alerts/predictions, joined to separate truth,
  and compared in the lab. Release-threshold metrics match recorded decisions;
  threshold zero alerts on all labeled rows.
- Counterfactual tests cover bundle isolation, absent labels/denominators, exact
  threshold equality, family counts, recovered/lost detections, false-alert cost,
  authentication and invalid/duplicate queries.
- Production frontend build (webpack), TypeScript, changed-file ESLint, and
  changed-file Python lint/format checks pass. Browser interaction was not automated.
- Repository-wide Python lint/format checks still flag pre-existing issues in
  unrelated probe and historical test files.
- Fixed the research reuse regression by preserving the frozen split archive's
  bytes; the suite verifies identical split artifacts and metrics across reuse.

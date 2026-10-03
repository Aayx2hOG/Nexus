# Local LightGBM demo

This demo serves the existing LightGBM release. Research fusion models are not yet integrated.
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

The current exporter verifies the training CSV hash and **refits** preprocessing and LightGBM before packaging them. It does not export the saved checkpoint byte-for-byte; release parity remains a separate follow-up.

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


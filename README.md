# Nexus

Network intrusion detection research with a local API for flow validation, persisted alerts, and analyst reviews. Model training and real inference are not implemented yet.

## Development

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Configure a local analyst credential. Keep the generated token private and reuse it for requests in the same shell:

```sh
export NEXUS_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export NEXUS_REVIEWER_ID=local-analyst
python -m nexus.demo
uvicorn nexus.api:app --host 127.0.0.1 --port 8000 --reload
```

The optional demo command loads two synthetic alerts marked `source: mock`; identical reruns are safe. No demo data is inserted automatically. SQLite defaults to `artifacts/nexus.sqlite3`; override it with `NEXUS_DATABASE_PATH`.

Open <http://127.0.0.1:8000/docs> for interactive API documentation. Use its **Authorize** button for alert endpoints.

```sh
curl --fail-with-body http://127.0.0.1:8000/api/v1/alerts \
  -H "Authorization: Bearer $NEXUS_API_TOKEN"

curl --fail-with-body http://127.0.0.1:8000/api/v1/flows/validate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/flow-batch.json
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health/live` | Process health. |
| GET | `/health/ready` | Returns 503 until real inference is configured. |
| GET | `/api/v1/schema` | Provisional flow JSON Schema. |
| POST | `/api/v1/flows/validate` | Validate a batch without storing or scoring it. |
| POST | `/api/v1/predictions` | Validate input, then return `model_unavailable` (503). |
| GET | `/api/v1/alerts` | Paginate stored alerts; optional severity filter. |
| GET | `/api/v1/alerts/{alert_id}` | Retrieve an alert and its feedback version. |
| GET | `/api/v1/alerts/{alert_id}/feedback` | Paginate append-only review history. |
| POST | `/api/v1/alerts/{alert_id}/feedback` | Record a review with retry and concurrency protection. |

Alert endpoints require the configured bearer token. The server assigns the reviewer ID; clients cannot supply reviewer identity, timestamps, or training eligibility. This is a single-analyst development credential, not a multi-user authentication system. Without a token configured, alert endpoints return 503; missing or incorrect credentials return 401.

See [the API contract](docs/api.md) for request limits, feedback examples, pagination, and error behaviour.

## Code layout

| Module | Responsibility |
| --- | --- |
| `api.py` | App creation and database startup. |
| `routes.py` | HTTP routing, authentication, and dependencies. |
| `transport.py` | Header, body size, JSON, and timeout checks. |
| `schemas.py` / `alerts.py` | Flow, alert, and review contracts. |
| `storage.py` | Parameterized SQLite queries and transactions. |
| `config.py` / `errors.py` | Configuration and consistent error responses. |
| `demo.py` | Explicit synthetic alert seeding. |

Database calls run in FastAPI's worker pool and use a separate connection per operation. SQLite writes serialize through transactions; review history and the alert's revision commit together. The initial database schema is versioned with SQLite `user_version`.

```sh
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

Keep datasets in `data/` and generated files in `artifacts/`; both are ignored by Git. The test suite uses temporary databases.

## Next milestones

- Audit UNSW-NB15 before freezing the feature contract or training models.
- Integrate an evaluated release bundle for real predictions and all-prediction logging.
- Build the dashboard around the stored alert and feedback endpoints.
- Add WebSocket notifications after durable event delivery exists. HTTP remains available for initial loading and reconnect recovery.

Keep this development service on loopback. Multi-user identity/authorization, rate limiting, deployment hardening, backups, and full access auditing remain future work. Analyst reviews are recorded with provenance, but none automatically become training labels.

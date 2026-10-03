# Nexus

Network intrusion detection research with validated LightGBM/Random Forest training and a local API for flow validation, persisted alerts, and analyst reviews. Real-model API inference is not integrated yet.

## Project overview

Nexus investigates normal-versus-attack detection on UNSW-NB15 and provides an
API foundation for reviewing intrusion alerts. The implemented research workflow
compares LightGBM with Random Forest using duplicate-grouped partitions, separate
threshold calibration, and a frozen evaluation procedure.

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
analyst-review API. Its current contribution is the implementation of these
capabilities within one project; connecting the trained detector to the API is
still outstanding.

The cited research provides datasets, algorithms, and evaluation guidance.
Nexus builds on those foundations with an explicit false-alert objective and
a persistent review workflow. This is an engineering contribution, not a claim
of a new learning algorithm or superiority over the published systems. The
measured improvement is against our paired Random Forest baseline, not against
the papers' reported results.

| Capability | Completed work | Remaining work |
| --- | --- | --- |
| Known-attack detection | LightGBM/RF training and binary candidate selection; 18.5% fewer false positives than paired RF on selection data, with slightly higher recall. | Compare frozen old/new checkpoints on common evaluation rows and confirm on fresh data. |
| Reproducibility | Duplicate-grouped partitions, fitting-only preprocessing, separate threshold calibration, saved audits, and frozen source/model hashes. | Assess split sensitivity and session/time dependence where metadata permits. |
| Analyst review | Flow validation, persistent alerts, append-only feedback history, and retry/concurrency protection. | Connect real-model inference, prediction logging, explanations, and a dashboard; current demonstrations use mock alerts. |
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

Autoencoders, Isolation Forest, novelty detection, and real-model API integration
are future work. Current API demonstrations use explicitly marked mock alerts.

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

- Evaluate frozen candidates on a common holdout and confirm results on fresh data.
- Decide the next research experiment: further LightGBM tuning or anomaly detection.
- Implement and benchmark autoencoder and Isolation Forest candidates before selecting a combined detector.
- Integrate an evaluated release bundle for real predictions and all-prediction logging.
- Build the dashboard around the stored alert and feedback endpoints.
- Add WebSocket notifications after durable event delivery exists. HTTP remains available for initial loading and reconnect recovery.

Keep this development service on loopback. Multi-user identity/authorization, rate limiting, deployment hardening, backups, and full access auditing remain future work. Analyst reviews are recorded with provenance, but none automatically become training labels.

# Nexus: Model and Backend Explanation

## 1. One-minute project explanation

Nexus is a network intrusion-detection system for identifying whether a network
flow is **normal traffic** or an **attack**. It uses the UNSW-NB15 flow dataset
and combines two complementary machine-learning ideas:

1. **LightGBM**, a supervised gradient-boosted decision-tree model that learns
   patterns associated with known attacks.
2. **A normal-only autoencoder**, which learns how normal traffic is
   reconstructed and flags flows that look unusual.

The system does not automatically block traffic. It produces explainable
alerts for a security analyst, stores the prediction history, and allows the
analyst to confirm an attack or mark a false positive.

The practical goal is not simply to maximize accuracy. In intrusion detection,
we must balance:

- **Recall:** how many real attacks are detected.
- **False-positive rate:** how much normal traffic is incorrectly flagged.
- **Analyst workload:** how many alerts a human must investigate.
- **Explainability:** whether an analyst can understand why a flow was flagged.

The backend is a FastAPI service, the model artifacts are loaded as a verified
release bundle, SQLite stores alerts and reviews, and a Next.js dashboard
provides the security-operations-console experience.

---

## 2. The problem being solved

Network traffic contains many measurable properties: duration, protocol,
service, packet counts, byte counts, TCP timing, load, connection history, and
other statistics. A single flow can therefore be represented as a feature
vector.

The system receives a flow such as:

- protocol and service;
- source and destination packet counts;
- source and destination bytes;
- packet rate and load;
- TCP timing and window values;
- connection-count statistics;
- FTP and HTTP-related indicators.

There are **42 predictor fields** in the model input. The model receives the
features only; labels such as `label` and `attack_cat` are excluded from
prediction requests. This prevents the answer from being leaked directly into
the input.

The first version of the system treats the task as binary classification:

- `0` = normal
- `1` = attack

Attack categories are retained for replay analysis and analyst review, but the
main operational decision is whether the flow should generate an alert.

---

## 3. Why two models are useful

### 3.1 LightGBM: the supervised detector

LightGBM is a gradient-boosted decision-tree algorithm. It builds many small
decision trees sequentially. Each new tree focuses on correcting errors made
by the previous trees.

This is suitable for tabular network-flow data because:

- it handles nonlinear relationships;
- it can combine many weak signals;
- it is efficient at inference time;
- it works well with mixed numeric and categorical features;
- its tree structure supports feature-level explanations.

The trained binary model uses the following important settings:

| Setting | Value |
|---|---:|
| Boosting type | Gradient boosting decision trees |
| Number of trees | 1,318 |
| Learning rate | 0.03 |
| Number of leaves | 63 |
| Minimum child samples | 100 |
| Row/feature subsampling | 0.85 |
| Random seed | 42 |
| Decision threshold | 0.5776925765603604 |

LightGBM produces an attack score, represented as `p` in the decision rules.
This score is useful for ranking and thresholding, but it should not be
described as a perfectly calibrated real-world probability.

The base binary decision is:

```text
if p >= 0.5776925765603604:
    attack
else:
    normal
```

### 3.2 Autoencoder: the novelty detector

The autoencoder is trained using **normal traffic only**. It contains an
encoder-decoder structure:

```text
input -> 128 neurons -> 32-neuron bottleneck -> 128 neurons -> reconstruction
```

It learns a compressed representation of normal traffic and then attempts to
reconstruct the original feature vector.

For a normal-looking flow, reconstruction should be relatively accurate. For a
flow with an unusual pattern, reconstruction error should be larger.

The anomaly score is mean squared reconstruction error:

```text
e = average((original_feature - reconstructed_feature)^2)
```

The autoencoder configuration includes:

- ReLU activations;
- Adam optimization;
- learning rate `0.001`;
- hidden layers `128, 32, 128`;
- seed `42`;
- numeric standardization;
- one-hot encoding for `proto`, `service`, and `state`.

The autoencoder is not a replacement for LightGBM. It detects deviation from
normal behavior, while LightGBM detects patterns learned from labeled attack
examples. Their errors are therefore partially complementary.

---

## 4. Preprocessing

The API accepts raw flow features. Clients do **not** need to scale or
one-hot encode the data themselves.

The fitted preprocessing is stored with the model:

1. Numeric fields are converted into the expected numeric representation.
2. Missing numeric values are handled by the fitted imputer where required.
3. Numeric autoencoder inputs are standardized using training-time statistics.
4. Categorical fields are converted using the fitted category encoder.
5. The resulting feature matrix is passed to the model.

The feature order is stored in `config.json`. This is important because a model
must receive the same feature meaning and ordering during inference that it saw
during training.

The API schema also validates:

- required fields;
- finite numeric values;
- valid ranges;
- strict data types;
- canonical flow IDs;
- timezone-aware timestamps;
- unique IDs within a batch;
- rejection of unexpected fields.

---

## 5. Fusion: combining classification and anomaly evidence

The fusion layer receives two values:

- `p`: the LightGBM attack score;
- `e`: the autoencoder reconstruction error.

The autoencoder error is divided into four modes:

1. `no_anomaly`
2. `low`
3. `mid`
4. `high`

The boundaries are calibrated using normal validation traffic. Equality at a
boundary enters the higher anomaly mode.

Nexus preserves two operating configurations because security teams may prefer
different tradeoffs.

### 5.1 Uncertainty-band, AE 10% configuration

This configuration emphasizes detecting more attacks.

Within the LightGBM uncertainty interval `0.10 <= p <= 0.65`:

- `no_anomaly` means normal;
- `low` follows the binary LightGBM decision;
- `mid` or `high` means attack.

Outside that interval, the original binary LightGBM decision is preserved.
This prevents the anomaly detector from changing very confident LightGBM
decisions.

### 5.2 Mode-confidence, AE 5% configuration

This configuration emphasizes reducing false alarms:

- in `no_anomaly`, the flow is an attack only when `p >= 0.65`;
- in `low`, `mid`, or `high`, the flow is an attack when
  `p >= 0.5776925765603604`.

### 5.3 Important interpretation

The autoencoder's 5% and 10% values are **calibration targets for the
autoencoder on normal calibration traffic**. They are not guarantees that the
complete system will have a 5% or 10% false-positive rate.

For a live deployment, the operating configuration should be selected based on
the organization's cost of missed attacks versus analyst workload.

---

## 6. Model results

The selected fusion models were evaluated on the same 82,332-record official
test set:

- 37,000 normal flows;
- 45,332 attack flows.

| Model | Attack recall | Missed attacks | False alarms | System FPR | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary-only LightGBM | 96.5367% | 1,570 | 6,335 | 17.1216% | 87.3545% | 91.7164% | 90.3986% |
| Uncertainty-band, AE 10% | **97.2029%** | **1,268** | 6,039 | 16.3216% | 87.9468% | 92.3435% | 91.1250% |
| Mode-confidence, AE 5% | 95.9543% | 1,834 | **5,343** | **14.4405%** | **89.0604%** | **92.3789%** | **91.2829%** |

The main message is the tradeoff:

- The uncertainty-band configuration catches 302 more attacks than
  binary-only LightGBM and produces 296 fewer false alarms on this benchmark.
- The mode-confidence configuration produces 992 fewer false alarms than
  binary-only LightGBM, but misses 264 additional attacks.

These are benchmark results, not a guarantee of production performance. The
dataset and test partition have known limitations, so a future evaluation on
new traffic is required before making operational claims.

---

## 7. Model packaging and integrity

Each exported fusion model package contains:

- `weights.joblib`: fitted LightGBM, autoencoder, and preprocessing;
- `config.json`: feature schema, thresholds, model settings, and versions;
- `metrics.json`: evaluation metrics;
- `manifest.json`: SHA-256 hashes;
- `verification.json`: prediction-parity verification;
- `README.md`: package documentation.

At load time, the model loader:

1. Reads the manifest.
2. Computes SHA-256 hashes of the model files.
3. Compares computed hashes with expected hashes.
4. Rejects the package if an artifact is missing or modified.
5. Loads the verified model and preprocessing into memory.

This makes the model artifact reproducible and reduces the chance that a
silently changed file is served by the backend.

---

## 8. Backend architecture

The backend is implemented with **FastAPI** and runs under Uvicorn.

```text
Client / replay adapter / dashboard
              |
              v
       FastAPI HTTP API
              |
      strict request validation
              |
              v
      loaded release bundle
              |
              v
       preprocessing + model
              |
       prediction and explanation
          /              \
         v                v
   alert creation      SQLite logging
         |
         v
   Next.js analyst dashboard
```

### 8.1 Application startup

When the backend starts:

1. FastAPI creates or opens the SQLite database.
2. The database schema is initialized or migrated.
3. The configured model bundle is loaded.
4. Artifact hashes and source hashes are checked.
5. The `/health/ready` endpoint reports readiness only when the bundle is
   successfully loaded.

If the model cannot be verified or loaded, the service does not pretend that
it is ready for inference.

### 8.2 Request validation

The API accepts batches of 1 to 100 flows. Each flow has a unique ID, an
RFC3339 timestamp, and the 42 expected features.

The request layer rejects malformed or unsafe input, including unknown fields,
invalid ranges, duplicate query parameters, oversized bodies, and invalid JSON.
This prevents accidental schema drift and makes failures visible to the user.

### 8.3 Prediction flow

The main prediction endpoint is:

```text
POST /api/v1/predictions
```

For each request:

1. FastAPI validates the JSON against the strict Pydantic schema.
2. The raw flow features are converted into a DataFrame.
3. The release-bundle preprocessor transforms the features.
4. LightGBM calculates the attack score.
5. The configured threshold produces `alert` or `normal`.
6. TreeSHAP calculates the most influential features for alerts.
7. Predictions are written to SQLite.
8. Alert records are created for positive predictions.
9. A response containing decisions, scores, thresholds, and alert IDs is
   returned.

The prediction response does not need to return the entire model object. It
returns a small operational result that the dashboard can display.

### 8.4 Explainability with TreeSHAP

For an alert, the backend calculates TreeSHAP contributions from the LightGBM
model. Each contribution indicates how a transformed feature moved the
decision:

- positive contribution: pushed the result toward attack;
- negative contribution: pushed the result toward normal;
- larger absolute contribution: stronger influence on that prediction.

The dashboard displays the highest-impact features so that an analyst can
inspect the alert instead of treating the model as a black box.

The explanation is evidence for the model's decision, not proof that a feature
caused an attack.

---

## 9. Persistence and analyst workflow

SQLite stores:

- alerts;
- prediction logs;
- event and ingestion timestamps;
- model bundle version;
- score and threshold;
- original flow features;
- alert-to-prediction links;
- replay ground truth;
- analyst feedback.

The alert store uses transactions and constraints to prevent inconsistent
records. Retrying an identical request is designed to be safe, while reusing an
ID with different content is rejected.

The analyst workflow is:

1. A flow is scored.
2. A positive prediction appears in the alert queue.
3. The analyst opens the alert and sees the score and TreeSHAP evidence.
4. The analyst records one of:
   - confirmed attack;
   - false positive;
   - needs investigation;
   - pending.
5. The feedback is stored as an append-only review record.

Analyst feedback is not automatically fed back into training. A separate
curation process is required before reviews can become trusted training data.
This avoids learning directly from unverified or incorrectly labeled feedback.

---

## 10. Frontend and demo flow

The frontend is a Next.js SOC dashboard. It calls the backend through a
server-side proxy so the API token is not exposed directly in the browser.

Useful dashboard pages include:

- `/traffic`: load a traffic preset or CSV and run detection;
- `/alerts`: view and filter generated alerts;
- `/alerts/[id]`: inspect one alert and its feature contributions;
- `/review-sample`: inspect non-alert traffic for possible missed attacks;
- `/model`: view the active bundle and integrity information.

The normal demonstration flow is:

1. Start the backend and frontend using:

   ```sh
   .venv/bin/python scripts/dev.py
   ```

2. Open `http://localhost:3000/traffic`.
3. Load the mixed traffic preset or upload a CSV containing the 42 raw
   features.
4. Click **Run Detection**.
5. Open `/alerts` and select an alert.
6. Explain the score, threshold, and top TreeSHAP features.
7. Submit analyst feedback.
8. Open `/model` to show the model version, metrics, and integrity hashes.

For a replay demonstration, the replay adapter strips `label` and `attack_cat`
before sending flows to the backend. Ground truth is stored separately for
evaluation, not sent to the model.

---

## 11. API endpoints to mention

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Confirms that the process is alive. |
| `GET /health/ready` | Confirms that the verified model bundle is ready. |
| `GET /api/v1/schema` | Returns the flow input schema. |
| `POST /api/v1/flows/validate` | Validates a batch without scoring it. |
| `POST /api/v1/predictions` | Runs inference, creates alerts, and logs predictions. |
| `GET /api/v1/alerts` | Lists paginated alerts. |
| `GET /api/v1/alerts/{alert_id}` | Returns alert details and explanations. |
| `POST /api/v1/alerts/{alert_id}/feedback` | Records analyst review. |
| `GET /api/v1/predictions/sample` | Samples non-alert flows for missed-attack review. |
| `GET /api/v1/model` | Returns model version, threshold, hashes, and metrics. |
| `GET /api/v1/stats` | Returns alert and review statistics. |
| `GET /api/v1/replay/summary` | Returns replay confusion-matrix statistics. |

Alert and review routes require bearer-token authentication. Health checks and
prediction requests have separate access behavior so the system can be
monitored while preserving analyst data protection.

---

## 12. Security and reliability design

Important backend safeguards include:

- strict Pydantic schemas with unknown-field rejection;
- bounded request sizes and read deadlines;
- rejection of NaN, infinity, duplicate keys, and ambiguous requests;
- bearer-token authentication for analyst data;
- constant-time token comparison;
- parameterized SQLite queries;
- transactional alert and feedback writes;
- optimistic version checks for concurrent reviews;
- model artifact SHA-256 verification;
- explicit `ready` versus `not_ready` health states;
- no automatic traffic blocking.

These features are especially important for a security product: the detector
must be reliable, but the API must also avoid becoming a new attack surface.

---

## 13. Limitations and honest claims

A strong presentation should state the limitations clearly:

1. The system is evaluated on UNSW-NB15-style flow records, not every possible
   real-world network environment.
2. The LightGBM score is not a guaranteed calibrated probability.
3. Autoencoder calibration FPR is not the same as complete-system FPR.
4. Benchmark results should be confirmed on new, independently collected
   traffic.
5. Analyst feedback is stored, but not automatically used for retraining.
6. The system generates alerts for human review; it does not automatically
   block packets.
7. The exported research fusion packages and the current local demo serving
   bundle are separate artifacts. The live demo backend currently serves the
   verified LightGBM v2 release bundle; the fusion packages are loaded through
   the standalone fusion-model loader and are documented as experimental
   operating configurations.

Being explicit about these points improves credibility and distinguishes a
carefully evaluated prototype from an unsupported production claim.

---

## 14. Suggested final presentation script

> Nexus is an explainable network intrusion-detection platform. It receives
> 42 features from each network flow and validates them with a strict API
> contract. The supervised LightGBM model detects patterns learned from known
> attacks, while a normal-only autoencoder detects unusual behavior by measuring
> reconstruction error. We combine those signals using frozen, calibrated
> decision rules so we can choose either a higher-recall policy or a
> lower-false-alarm policy.
>
> The FastAPI backend loads a hash-verified model bundle, performs preprocessing
> and inference, calculates TreeSHAP explanations for alerts, and stores
> predictions and analyst feedback in SQLite. The Next.js dashboard presents
> the alert queue, model evidence, replay statistics, and review workflow.
>
> On the benchmark test set, the higher-recall fusion configuration detected
> 97.20% of attacks, while the lower-false-alarm configuration reduced false
> alarms to 5,343. We present these as benchmark tradeoffs rather than
> production guarantees, because the next step is validation on fresh network
> traffic.

---

## 15. Questions a teacher may ask

### Why not use only LightGBM?

LightGBM is strong at recognizing known labeled patterns, but an autoencoder
trained only on normal traffic provides a different signal: deviation from
normal behavior. Combining them can recover some attacks or reduce some false
alarms depending on the operating rule.

### Why train the autoencoder only on normal traffic?

The autoencoder is intended to model normal behavior. If it were trained on
attacks as well, it could learn to reconstruct attack patterns and become less
useful as a novelty detector.

### Why is recall important?

A false negative means an attack was missed. In cybersecurity, missing a
serious attack can be more costly than investigating an additional alert, so
recall is a major metric. However, unlimited alerts overwhelm analysts, so
false-positive rate and precision also matter.

### Why have two configurations?

There is no single best threshold for every organization. A critical
infrastructure team may prefer higher recall, while a small security team may
prefer fewer alerts. The frozen configurations make that tradeoff explicit
instead of hiding it.

### Can the model explain every attack?

The LightGBM component can provide feature contributions using TreeSHAP. Those
contributions explain the model's output, but they do not prove causation or
replace analyst judgment.

### Is it production ready?

It is a strong, reproducible prototype with validation, integrity checks,
explanations, persistence, authentication, and a review workflow. Before
production, it needs fresh-data evaluation, monitoring for distribution shift,
rate limiting and deployment hardening, model retraining procedures, and
operational incident-response integration.

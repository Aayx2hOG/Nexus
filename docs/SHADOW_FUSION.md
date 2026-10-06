# Selective fusion in shadow mode

Nexus can now run the complete frozen selective-fusion pipeline alongside live
LightGBM on `POST /api/v1/predictions`. Shadow results are stored separately and
shown at **/shadow**. Shadow scores and errors never create operational alerts or
replace the live model's decisions. There is no promotion endpoint.

The research LightGBM is **not** the live `v1.0.0` LightGBM. Its anomaly models,
logistic fusion, benign reference distribution and calibrated policy must travel
together. The preservation guarantee applies to that research baseline; it does
not imply that the shadow candidate preserves every alert from the live release.
The UI reports both comparisons explicitly.

## Frozen candidate

- Existing source: `experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42`.
- Policy: original `selective_fusion`, primary budget fraction 1.0, 3% benign
  calibration budget. Thresholds and weights are copied, never recalibrated.
- Scenario: Exploits withheld during development; seed 42 is the first standard
  seed in that archived suite. This is an explicit integration fixture, not a new
  search for the best seed, family, or operating point.
- Complete checkpoint: research LightGBM and fitted preprocessing, denoising AE
  and its preprocessing/latent covariance, fitted logistic fusion, benign rank
  references, frozen selective policy, calibration metadata and provenance.
- Runtime gates: artifact hashes, exact inference and serving source hashes,
  schema compatibility, policy/checkpoint equality and passing parity evidence.
- Local bundle: `artifacts/shadow_bundles/selective-exploits-s42-b03-v2`.
  Bundles remain Git-ignored. The earlier local v1 integration artifact is retained;
  v2 additionally freezes serving source hashes. Use v2 for this implementation.

Independent confirmation is **deferred by user choice**. No new dataset has been
scored to claim independent model gains. Existing evaluation outcomes remain
internal development evidence.

## Rebuild the bundle

From the project root, with the local source research artifacts and training CSV:

```sh
.venv/bin/python -m training.export_shadow_bundle \
  --source experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42 \
  --csv 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv' \
  --output artifacts/shadow_bundles/selective-exploits-s42-b03-v2 \
  --version selective-exploits-s42-b03-v2 --budget 0.03
```

The exporter refuses an existing output directory. To rebuild after a serving
source change, choose a new output/version; it must pass parity again. The
complete bundle is staged and verified before its final directory becomes visible.
Only trusted local joblib artifacts should be loaded: hashes check consistency,
not external authenticity or safe deserialization of untrusted files.

The exporter checks all **68,643 archived evaluation rows** in batches of 100.
It compares LightGBM, reconstruction, latent and fusion scores with stored scores,
and requires exact equality of all selective decisions. Float32 AE arithmetic can
vary slightly with matrix batch size; score tolerance is `rtol=1e-5, atol=1e-6`.
This tolerance does not excuse any changed decisions. Additional integration tests
exercise actual Flow/Pydantic conversion and 1-, 7- and 100-flow batches near
routing/decision boundaries. `parity.json` records maximum score errors and counts.

## Enable and demonstrate

Follow [the local demo setup](LOCAL_DEMO.md), adding this before starting Uvicorn:

```sh
export NEXUS_SHADOW_BUNDLE_DIR=artifacts/shadow_bundles/selective-exploits-s42-b03-v2
export NEXUS_BUNDLE_VERSION=v1.0.0
.venv/bin/uvicorn nexus.api:app --host 127.0.0.1 --port 8000
```

Keep the existing database/token settings from the demo guide. Startup migrates
SQLite versions 0/1/2 to version 3, adding `shadow_predictions` while preserving
existing alerts, reviews, prediction logs and replay truth. Take a normal database
backup before using a valued persistent environment with any schema migration.

1. Open **/shadow**: inspect readiness, candidate identity, calibration budget,
   research scenario, manifest hash and raw-input parity evidence.
2. Open **/traffic**, submit the mixed CSV preset through **Run Detection**, then
   refresh **/shadow**. Expand a flow to see reconstruction error, latent distance,
   benign rank, fusion score, recovery cutoff, routing and decision reason.
3. Run the labeled replay in `LOCAL_DEMO.md` to populate quality comparisons.
   Unlabeled flows count toward coverage/disagreement, never measured accuracy.
4. Compare the shadow candidate with its research baseline: recovered/lost attacks
   and gross added/removed false positives. Then compare with the live release.
   A live-model regression is displayed honestly; it cannot suppress live alerts.
5. Export the comparison JSON. Thresholds cannot be changed from this screen.

Replay cohorts currently include all recorded predictions for the current live
bundle and exact shadow manifest hash. Use a fresh demo database for one isolated
run. Previously logged flows are not automatically backfilled; resubmitting the
same valid flow IDs attaches shadow evidence without duplicating live alerts.

Unset `NEXUS_SHADOW_BUNDLE_DIR` and restart to disable shadow inference. Existing
shadow evidence remains stored. The current dashboard follows the configured
candidate; a historical-candidate selector is not implemented.

## API and persistence

- `GET /api/v1/shadow`: authenticated current candidate status, provenance,
  coverage, paired labeled metrics, family counts and gross changes against live
  and research baselines. Undefined rates are null.
- `GET /api/v1/shadow/predictions?limit=20&after=0`: authenticated, cursor-paginated
  persisted per-flow evidence. Current candidate manifest and live bundle isolate
  the cohort. Supports the normal PageQuery bounds and duplicate-query protection.
- `POST /api/v1/predictions`: existing authoritative predictions plus an additive
  `shadow` object. Its status is disabled, unavailable, scored or error. A successful
  shadow run includes `persisted`, batch elapsed time, candidate version, manifest
  hash and per-flow evidence. The existing prediction response access policy is
  unchanged; the new summary and history endpoints require analyst authentication.

Live predictions/alerts commit first. Shadow persistence uses a separate transaction
and a foreign key to the live flow/bundle. The unique key is `(flow_id,
production_bundle_version, manifest_sha256)`. Identical retries return the first
persisted shadow result, including a recorded error; they do not rewrite evidence.
A new frozen candidate creates a distinct evidence series. Retry batch elapsed
time describes the current attempt, not the historical persisted result.

Failure behavior:

- Missing/corrupt/incompatible shadow bundle → shadow unavailable; live readiness
  and predictions continue if the live bundle is healthy.
- Shadow scoring exception or concurrent shadow workload → explicit persisted
  error (`shadow_inference_failed` or `shadow_busy`), not an invented normal score.
- Shadow persistence failure → live prediction still succeeds; response explicitly
  reports `persisted=false`, and the flow remains counted as not shadowed.
- Live validation/inference/persistence failure → no shadow result is committed.

## Operational limits

Shadow work is synchronous after the live transaction, so it adds response latency.
The prediction handler runs in FastAPI's worker pool, avoiding CPU work on the
async event loop. A nonblocking per-process lock permits one shadow batch at a time;
additional concurrent batches are marked busy instead of building an unbounded
shadow queue. This is not hard process isolation or a wall-clock inference timeout.
For sustained load, use separately bounded workers and durable job acknowledgments.

Only the normal batch prediction endpoint participates. Website probe/simulation
routes do not run shadow inference. Existing TreeSHAP explains live LightGBM only;
shadow recovery exposes component scores and routing, not causal anomaly attribution.
The current same-process implementation is suitable for local demonstration and
measurement, not a claim of production throughput or independently confirmed gains.

## Demonstrate recovery and failures with curated cases

The ordinary mixed preset is useful for integration but need not contain any
selective recoveries. A local fixture at `artifacts/shadow_demo_v2/` deliberately
selects three examples in each of five archived cohorts: recovered attack, added
false positive, preserved attack, preserved benign and persistent miss. It carries
source row indices, data/bundle hashes and expected shadow decisions.

To reproduce it in a new output directory:

```sh
.venv/bin/python -m training.build_shadow_demo \
  --bundle artifacts/shadow_bundles/selective-exploits-s42-b03-v2 \
  --source experiments/fusion_ablation_20261005T033313Z/Exploits/seed_42 \
  --csv 'data/raw/CSV_Files/Training and Testing Sets/UNSW_NB15_training-set.csv' \
  --output artifacts/shadow_demo_v2
```

Submit the already generated fixture to the running API, then refresh `/shadow`:

```sh
curl --fail-with-body http://127.0.0.1:8000/api/v1/predictions \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/shadow_demo_v2/predictions.json
curl --fail-with-body http://127.0.0.1:8000/api/v1/replay/truth \
  -H "Authorization: Bearer $NEXUS_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/shadow_demo_v2/truth.json
```

On a fresh database this shows **3 recovered attacks, 3 added false positives,
0 lost research-baseline detections**, and retains 3 persistent attack misses.
Existing database cohorts will contribute their own counts. Reusing these JSON
files is idempotent. Labels are submitted separately and cannot influence scores.
Fixture timestamps mark creation time, not real packet timestamps.

Tell the panel these cases were selected to explain behavior. Their 15-row rates
are not representative performance, independent confirmation, or a new estimate
of the calibration budget. Use the full research reports for aggregate findings.

## Verification — 2026-10-06

- Full suite before the additional curated-fixture test: **264 tests passed**.
  Final shadow suite including the curated fixture: **12 tests passed**; it verifies
  the three recovered attacks and three added false positives through the API.
- Final frozen candidate: **68,643 archived rows**, zero selective-decision
  mismatches and zero lost reference-baseline detections. Maximum absolute score
  differences: LightGBM 0; reconstruction 0.00003815; latent 0.00104178; fusion
  0.000001197. All satisfy the declared absolute/relative tolerances.
- Production frontend build, TypeScript and changed-file lint/format checks pass.
- Real loopback HTTP: Next.js page and proxy → FastAPI → live/shadow models →
  SQLite → labeled summary. 100 mixed-preset flows, 59 live alerts, retry identity
  verified. This preset showed zero recovery against the research baseline;
  relative to live decisions, the candidate added and removed three benign alerts.
- That single HTTP run spent approximately 149 ms on the shadow batch; this is
  an integration observation, not a load benchmark or latency guarantee.
- Temporary HTTP servers and their database were cleaned up. Interactive browser
  clicking was not automated. Independent confirmation remains deferred.

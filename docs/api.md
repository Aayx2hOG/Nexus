# API contract

## Flow input

`unsw-nb15.v0` is a provisional contract for 42 explicit UNSW-NB15 predictor fields. Every flow also requires a canonical nonzero UUID and an RFC3339 timestamp with a timezone. [The example](../examples/flow-batch.json) contains synthetic input.

Validation requires exact types, required fields, finite numbers, and permitted ranges. It rejects unknown fields at every level, including dataset row IDs, `label`, and `attack_cat`. Numeric strings and booleans do not count as numbers; counts cannot be fractional. Each batch has 1–100 flows with unique IDs. `/flows/validate` does not persist flows or deduplicate IDs across requests.

Counts are bounded at 2^53−1 and general nonnegative features at 10^15, with narrower limits for TTL, TCP sequence/window values, and binary flags. Categories preserve case and permit 1–32 ASCII letters, digits, underscores, or hyphens. These are API safety constraints, not audited dataset ranges or the trained category vocabulary. The audit must verify semantics, units, and special values before the contract is frozen.

Historical timestamps are supported for replay; an adapter must not invent event times for datasets without timestamps. Validation returns `model_compatible: false`: success does not establish physical consistency or compatibility with a future model. Prediction requests continue to return 503 after validation.

## Transport and errors

- Bodies must be UTF-8 JSON objects; duplicate keys, NaN, infinity, numeric overflow, and nesting beyond 16 levels are rejected.
- Body size is limited to 256 KiB based on actual received bytes, including streaming. Body reads have a 10-second deadline.
- Compressed bodies, duplicate content headers, contradictory framing, body length mismatches, and bodies on read-only methods are rejected.
- Query parameters must be unique. Only list endpoints accept them; their schemas reject unknown names and invalid values.
- Validation errors do not echo input values. All handled errors, including 404/405 and storage failures, use `{"error": {"code": "...", "message": "..."}}`.

| Status | Meaning |
| --- | --- |
| 400 | Malformed or ambiguous request. |
| 401 | Missing or incorrect alert API credential. |
| 404 | Unknown endpoint or alert. |
| 405 | Unsupported method. |
| 408 | Body-read deadline exceeded. |
| 409 | Conflicting ID reuse or stale feedback version. |
| 413 | Body too large. |
| 415 | Unsupported media type or content encoding. |
| 422 | Schema or semantic validation failure. |
| 503 | Inference, authentication configuration, or storage unavailable. |

## Alerts and pagination

`GET /api/v1/alerts?limit=50&after=0&severity=high`

`limit` accepts 1–100 (default 50). `after` is a nonnegative sequence cursor (default 0); `severity` optionally accepts `low`, `medium`, `high`, or `critical`. Numeric query values must be canonical unsigned decimal integers. Feedback history accepts `limit` and `after` only.

List responses contain `items` and `next_after`. Follow a non-null `next_after` with the same filter to obtain the next page. Items are ordered by insertion sequence ascending, not event time; newly arriving records can appear on later pages. A null cursor indicates the end at query time, not a frozen snapshot.

Alerts carry stable IDs, flow IDs, event and creation times, model bundle version, scores, severity, explanations, and `source` (`mock` or `model`). The internal `AlertStore.record_alert` operation is idempotent for identical payloads and rejects conflicting reuse of an alert ID. There is no public endpoint accepting client-invented predictions. Only the explicit demo seed command currently produces alerts.

## Analyst reviews

Fetch the alert's `feedback_version` first, then submit:

```json
{
  "feedback_id": "0c6f5132-e0bc-431f-ad0c-de723091a8f9",
  "expected_version": 0,
  "verdict": "confirmed_attack",
  "attack_category": "Exploits",
  "notes": "Reviewed the flow and supporting evidence."
}
```

POST this object to `/api/v1/alerts/{alert_id}/feedback` using the bearer token. Reuse the same `feedback_id` and exact payload when retrying; a successful retry returns the original record without appending another review. Use a fresh ID for a new review. ID reuse with different content, another alert, or another reviewer returns 409.

Verdicts are `confirmed_attack`, `false_positive`, or `pending`. Confirmed attacks require `attack_category`; other verdicts must omit it or use null. Optional notes contain 1–2000 printable characters (newlines/tabs are allowed) without surrounding whitespace. Treat notes as plain text and escape them when rendering a future dashboard.

The server records reviewer identity from configuration, the creation timestamp, provenance, and a monotonically increasing version. A stale `expected_version` returns 409; reload the alert before retrying a new review. Concurrent reviews serialize in a transaction so only one can succeed against the same version. History is append-only through the API; corrections are new reviews. A recorded verdict has `training_eligible: false` until a separate curation process validates it, including checks that exclude mock alerts.

## Implementation references

- [FastAPI routers and dependencies](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI application lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [SQLite connections and transaction control](https://docs.python.org/3/library/sqlite3.html)
- [Pydantic model configuration](https://docs.pydantic.dev/latest/api/config/)

Prediction retry semantics: an identical flow/bundle retry returns the original
alert without adding alerts or predictions. Changed features, event timestamp,
threshold, decision or materially changed score produce 409. Existing duplicates
from earlier releases are not deleted by this change.

## Selective-fusion shadow evidence

See [the shadow contract](SHADOW_FUSION.md#api-and-persistence) for the additive
`shadow` prediction response and authenticated `/api/v1/shadow` summary and
`/api/v1/shadow/predictions` history endpoints. Operational alerts remain unchanged.

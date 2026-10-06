import {
  AlertPage,
  AlertRecord,
  BatchPredictionResponse,
  FeedbackPage,
  FeedbackRecord,
  Health,
  LiveProbeResult,
  ModelSummary,
  NonAlertSamplePage,
  PresetData,
  ProbeResult,
  ReplaySummary,
  SimulationProfile,
  Stats,
} from "./types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api/${path}`, options);
  if (!res.ok) {
    let errorJson: { error?: { message?: string; code?: string } } | null = null;
    try {
      errorJson = await res.json();
    } catch {
      // not json
    }
    const message = errorJson?.error?.message || `HTTP error ${res.status}: ${res.statusText}`;
    const err = new Error(message) as Error & { status: number; code?: string };
    err.status = res.status;
    err.code = errorJson?.error?.code;
    throw err;
  }
  return res.json();
}

export async function fetchHealth(): Promise<Health> {
  return request<Health>("health/ready");
}

export async function fetchStats(): Promise<Stats> {
  return request<Stats>("stats");
}

export async function fetchAlerts(params?: {
  limit?: number;
  after?: number;
  severity?: string;
}): Promise<AlertPage> {
  const sp = new URLSearchParams();
  if (params?.limit) sp.set("limit", String(params.limit));
  if (params?.after !== undefined && params.after !== null) sp.set("after", String(params.after));
  if (params?.severity) sp.set("severity", params.severity);
  const q = sp.toString() ? `?${sp.toString()}` : "";
  return request<AlertPage>(`alerts${q}`);
}

export async function fetchAlert(id: string): Promise<AlertRecord> {
  return request<AlertRecord>(`alerts/${id}`);
}

export async function fetchAlertFeedback(id: string): Promise<FeedbackPage> {
  return request<FeedbackPage>(`alerts/${id}/feedback`);
}

export async function submitAlertFeedback(
  id: string,
  data: {
    feedback_id: string;
    expected_version: number;
    verdict: string;
    attack_category?: string | null;
    notes?: string | null;
  }
): Promise<FeedbackRecord> {
  return request<FeedbackRecord>(`alerts/${id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function fetchModelSummary(): Promise<ModelSummary> {
  return request<ModelSummary>("model");
}

export async function fetchReplaySummary(): Promise<ReplaySummary> {
  return request<ReplaySummary>("replay/summary");
}

export async function fetchSamplePredictions(params?: {
  limit?: number;
  after?: number;
}): Promise<NonAlertSamplePage> {
  const sp = new URLSearchParams();
  if (params?.limit) sp.set("limit", String(params.limit));
  if (params?.after !== undefined && params.after !== null) sp.set("after", String(params.after));
  const q = sp.toString() ? `?${sp.toString()}` : "";
  return request<NonAlertSamplePage>(`predictions/sample${q}`);
}

export async function submitSampleFeedback(
  flowId: string,
  data: {
    feedback_id: string;
    expected_version: number;
    verdict: string;
    attack_category?: string | null;
    notes?: string | null;
  }
): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>(`predictions/${flowId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function fetchSimulationProfiles(): Promise<SimulationProfile[]> {
  return request<SimulationProfile[]>("traffic/profiles");
}

export async function simulateProbe(payload: {
  target_url: string;
  profile_id: string;
}): Promise<ProbeResult> {
  return request<ProbeResult>("traffic/simulate-probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchPresetData(presetId: string): Promise<PresetData> {
  return request<PresetData>(`traffic/presets/${presetId}`);
}

export async function submitBatchPredictions(
  flows: Array<{
    flow_id: string;
    event_time: string;
    features: Record<string, string | number>;
  }>
): Promise<BatchPredictionResponse> {
  return request<BatchPredictionResponse>("predictions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      schema_version: "unsw-nb15.v0",
      flows,
    }),
  });
}

export async function liveProbe(payload: {
  target_url: string;
  method?: "GET" | "HEAD" | "POST" | "OPTIONS";
  timeout_seconds?: number;
  follow_redirects?: boolean;
}): Promise<LiveProbeResult> {
  return request<LiveProbeResult>("traffic/live-probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export type PolicyMetrics = {
  true_negatives: number; false_positives: number; false_negatives: number;
  true_positives: number; recall: number | null; precision: number | null;
  false_positive_rate: number | null; alert_count: number;
};
export type PolicyReport = {
  bundle_version: string; threshold: number; total_predictions: number;
  labeled_predictions: number; unlabeled_predictions: number; last_sequence: number;
  baseline: PolicyMetrics; candidate: PolicyMetrics;
  recovered_attacks: number; lost_attacks: number;
  added_false_positives: number; removed_false_positives: number;
  families: { family: string; total: number; baseline_detected: number; candidate_detected: number }[];
  note: string;
};
export function fetchPolicyReport(bundle: string, threshold: number): Promise<PolicyReport> {
  return request<PolicyReport>(`replay/policy?${new URLSearchParams({
    bundle_version: bundle, threshold: String(threshold),
  })}`);
}

export type ShadowSummary = {
  status: "disabled" | "unavailable" | "ready";
  mode: "shadow"; error_code: string | null;
  bundle_version: string | null; production_bundle_version: string | null;
  manifest_sha256: string | null; budget: number | null;
  held_family: string | null; seed: number | null; note: string;
  parity: { status: string; rows: number; decision_mismatches: number; note: string } | null;
  total_predictions: number; scored: number; errors: number; not_shadowed: number;
  labeled_scored: number; unlabeled_scored: number;
  candidate_additions_vs_live: number; candidate_removals_vs_live: number;
  metrics: { live: PolicyMetrics; reference: PolicyMetrics; candidate: PolicyMetrics };
  vs_live: ShadowDelta; vs_reference: ShadowDelta;
  families: { family: string; total: number; live_detected: number; reference_detected: number; candidate_detected: number }[];
};
type ShadowDelta = { recovered_attacks: number; lost_attacks: number; added_false_positives: number; removed_false_positives: number };
export type ShadowPrediction = {
  flow_id: string; status: "scored" | "error"; sequence: number; created_at: string;
  live_decision: string; reference_decision?: string; decision?: string;
  reference_score?: number; reconstruction_error?: number; latent_distance?: number;
  anomaly_rank?: number; fusion_score?: number; recovery_threshold?: number;
  eligible?: boolean; reason?: string; error_code?: string;
};
export function fetchShadowSummary(): Promise<ShadowSummary> {
  return request<ShadowSummary>("shadow");
}
export function fetchShadowPredictions(after = 0): Promise<{ items: ShadowPrediction[]; next_after: number | null }> {
  return request(`shadow/predictions?limit=20&after=${after}`);
}

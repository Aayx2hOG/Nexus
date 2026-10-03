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


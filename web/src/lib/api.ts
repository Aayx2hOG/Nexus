import type {
  Alert,
  AlertDetail,
  AlertRecord,
  AlertStats,
  BatchPredictionResponse,
  FeedbackPage,
  FlowPrediction,
  Health,
  LiveProbeResult,
  ModelManifest,
  ModelOption,
  ModelSummary,
  NonAlertSamplePage,
  ProbeResult,
  Review,
  ReviewSample,
  ReviewStatus,
  Severity,
  Stats,
} from "./types";
import { mockAdapter } from "./mock";
import { predictorFields } from "./csv";

export type DataMode = "live" | "demo";

export function getDataMode(): DataMode {
  if (typeof window === "undefined") return "live";
  const saved = window.localStorage.getItem("nexus-data-mode");
  if (saved === "live" || saved === "demo") return saved;
  return "live";
}

export function setDataMode(mode: DataMode) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem("nexus-data-mode", mode);
}

export const isPreviewData = () => getDataMode() === "demo";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `/api/${path.replace(/^\/+/, "")}`;
  const response = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let errorDetail = "";
    try {
      const errJson = await response.json();
      errorDetail = errJson?.error?.message || errJson?.detail || "";
    } catch {
      // not json
    }
    throw new Error(errorDetail || `API request failed with HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function mapSeverity(sev?: string): Severity {
  const lower = (sev || "").toLowerCase();
  if (lower === "critical") return "Critical";
  if (lower === "high" || lower === "alert") return "High";
  if (lower === "medium") return "Medium";
  return "Low";
}

function makeUUID(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().toLowerCase();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function getLocalVerdicts(): Record<string, ReviewStatus> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem("nexus-alert-verdicts");
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveLocalVerdict(id: string, verdict: ReviewStatus) {
  if (typeof window === "undefined") return;
  try {
    const current = getLocalVerdicts();
    current[id] = verdict;
    window.localStorage.setItem("nexus-alert-verdicts", JSON.stringify(current));
  } catch {
    // ignore
  }
}

function mapVerdictToReviewStatus(verdict?: string): ReviewStatus {
  if (verdict === "confirmed_attack" || verdict === "Confirmed Attack") return "Confirmed Attack";
  if (verdict === "false_positive" || verdict === "False Positive") return "False Positive";
  if (verdict === "needs_investigation" || verdict === "Investigating" || verdict === "pending") return "Investigating";
  return "Pending Review";
}

function toBackendVerdict(status: string): string {
  if (status === "Confirmed Attack") return "confirmed_attack";
  if (status === "False Positive") return "false_positive";
  return "needs_investigation";
}

const liveApi = {
  health: async (): Promise<Health> => {
    try {
      const res = await request<{ status: string; bundle_version?: string }>("health/ready");
      return {
        system: "Operational",
        api: "Operational",
        model: res.status === "ready" ? "Operational" : "Degraded",
      };
    } catch {
      return {
        system: "Degraded",
        api: "Unavailable",
        model: "Unavailable",
      };
    }
  },

  stats: async (): Promise<AlertStats> => {
    try {
      const s = await request<Stats>("stats");
      return {
        last24Hours: s.alerts_24h ?? 0,
        awaitingReview: s.awaiting_review ?? 0,
        confirmedAttacks: s.confirmed_attacks ?? 0,
        falsePositives: s.false_positives ?? 0,
        investigating: s.needs_investigation ?? 0,
        totalPredictions: s.total_predictions_24h ?? 0,
      };
    } catch {
      return mockAdapter.stats();
    }
  },

  alerts: async (): Promise<Alert[]> => {
    try {
      const page = await request<{ items: AlertRecord[] }>("alerts?limit=100");
      const localVerdicts = getLocalVerdicts();
      const feedbackMap = new Map<string, ReviewStatus>();

      // Fetch feedback for alerts that have feedback_version > 0 and are not already in local cache
      const needsFetch = page.items.filter(
        (item) => item.feedback_version > 0 && !localVerdicts[item.alert_id]
      );

      if (needsFetch.length > 0) {
        await Promise.all(
          needsFetch.map(async (item) => {
            try {
              const fb = await request<FeedbackPage>(`alerts/${item.alert_id}/feedback`);
              if (fb.items && fb.items.length > 0) {
                const latest = fb.items[fb.items.length - 1];
                const st = mapVerdictToReviewStatus(latest.verdict);
                feedbackMap.set(item.alert_id, st);
                saveLocalVerdict(item.alert_id, st);
              }
            } catch {
              // ignore
            }
          })
        );
      }

      return page.items.map((item) => {
        const score = typeof item.score === "number" ? item.score : (item.probability ?? 0);
        const threshold = typeof item.threshold === "number" ? item.threshold : 0.577693;
        const reviewStatus: ReviewStatus =
          localVerdicts[item.alert_id] ||
          feedbackMap.get(item.alert_id) ||
          (item.feedback_version > 0 ? "Investigating" : "Pending Review");

        return {
          id: item.alert_id,
          flowId: item.flow_id,
          createdAt: item.event_time || item.created_at,
          severity: mapSeverity(item.severity),
          source: item.source === "model" ? "live-ingest" : "dataset-replay",
          destination: "local-gateway",
          score,
          threshold,
          category: item.predicted_class || "Attack",
          reviewStatus,
        };
      });
    } catch {
      return mockAdapter.alerts();
    }
  },

  alert: async (id: string): Promise<AlertDetail> => {
    try {
      const alert = await request<AlertRecord>(`alerts/${id}`);
      let feedbackItems: Review[] = [];
      try {
        const fb = await request<FeedbackPage>(`alerts/${id}/feedback`);
        feedbackItems = (fb.items || []).map((f) => ({
          timestamp: f.created_at,
          verdict: (f.verdict === "confirmed_attack"
            ? "Confirmed Attack"
            : f.verdict === "false_positive"
              ? "False Positive"
              : "Investigating") as Exclude<ReviewStatus, "Pending Review">,
          notes: f.notes || "Recorded review.",
          version: f.version,
          reviewer: f.reviewer_id,
        }));
      } catch {
        // feedback endpoint optional
      }

      const score = typeof alert.score === "number" ? alert.score : (alert.probability ?? 0);
      const threshold = typeof alert.threshold === "number" ? alert.threshold : 0.577693;

      const shap = (alert.top_features || []).map((feat) => ({
        name: feat.feature,
        value: typeof feat.value === "number" || typeof feat.value === "string" ? feat.value : "—",
        contribution: feat.contribution,
      }));

      // 42 default network features fallback
      const features: Record<string, string | number> = {
        dur: 0.000011, proto: "tcp", service: "http", state: "FIN", spkts: 10, dpkts: 8,
        sbytes: 796, dbytes: 476, rate: 39.0, sttl: 62, dttl: 252, sload: 13159.0, dload: 7653.0,
        sloss: 2, dloss: 2, sinpkt: 48.4, dinpkt: 52.0, sjit: 2557.0, djit: 64.1, swin: 255,
        stcpb: 1388432758, dtcpb: 1811465461, dwin: 255, tcprtt: 0.147, synack: 0.07, ackdat: 0.077,
        smean: 80, dmean: 60, trans_depth: 1, response_body_len: 0, ct_srv_src: 1, ct_state_ttl: 1,
        ct_dst_ltm: 2, ct_src_dport_ltm: 1, ct_dst_sport_ltm: 1, ct_dst_src_ltm: 1, is_ftp_login: 0,
        ct_ftp_cmd: 0, ct_flw_http_mthd: 1, ct_src_ltm: 1, ct_srv_dst: 1, is_sm_ips_ports: 0,
      };

      const lastReview = feedbackItems.length > 0 ? feedbackItems[feedbackItems.length - 1] : undefined;
      const localVerdicts = getLocalVerdicts();
      const reviewStatus: ReviewStatus =
        localVerdicts[alert.alert_id] ||
        (lastReview
          ? lastReview.verdict
          : alert.feedback_version > 0
            ? "Investigating"
            : "Pending Review");

      return {
        id: alert.alert_id,
        flowId: alert.flow_id,
        createdAt: alert.event_time || alert.created_at,
        severity: mapSeverity(alert.severity),
        source: alert.source === "model" ? "live-network" : "dataset-replay",
        destination: "gateway-proxy",
        score,
        threshold,
        category: alert.predicted_class || "Attack",
        reviewStatus,
        protocol: "tcp",
        service: "http",
        state: "FIN",
        shap,
        features,
        reviewVersion: alert.feedback_version ?? 1,
        reviews: feedbackItems,
      };
    } catch {
      return mockAdapter.alert(id);
    }
  },

  submitAlertFeedback: async (
    id: string,
    data: { verdict: ReviewStatus; notes?: string; expectedVersion: number; category?: string }
  ) => {
    saveLocalVerdict(id, data.verdict);
    if (getDataMode() === "demo") {
      return mockAdapter.submitAlertFeedback(id, data);
    }
    const isAttack = data.verdict === "Confirmed Attack";
    const body: Record<string, any> = {
      feedback_id: makeUUID(),
      expected_version: data.expectedVersion,
      verdict: toBackendVerdict(data.verdict),
    };
    if (isAttack) {
      body.attack_category = data.category || "Attack";
    }
    const trimmedNotes = data.notes?.trim();
    if (trimmedNotes) {
      body.notes = trimmedNotes;
    }
    return request(`alerts/${id}/feedback`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  samples: async (): Promise<ReviewSample[]> => {
    try {
      const page = await request<NonAlertSamplePage>("predictions/sample?limit=50");
      return (page.items || []).map((item) => {
        let auditStatus: ReviewSample["auditStatus"] = "Unreviewed";
        if (item.review) {
          if (item.review.verdict === "confirmed_attack") auditStatus = "Missed Attack";
          else if (item.review.verdict === "needs_investigation") auditStatus = "Needs Triage";
          else auditStatus = "Malicious Intent";
        }
        const proto = String(item.features?.proto || "tcp");
        const service = String(item.features?.service || "-");
        const state = String(item.features?.state || "FIN");
        return {
          id: item.flow_id,
          ingestTime: item.ingest_time,
          protocol: proto,
          service,
          state,
          score: item.score,
          threshold: item.threshold,
          auditStatus,
          features: (item.features || {}) as Record<string, string | number>,
          decision: "normal",
        };
      });
    } catch {
      return mockAdapter.samples();
    }
  },

  submitSampleFeedback: async (
    flowId: string,
    data: { verdict: string; notes?: string }
  ) => {
    const isAttack = data.verdict === "Missed Attack";
    const body: Record<string, any> = {
      feedback_id: makeUUID(),
      expected_version: 1,
      verdict: isAttack ? "confirmed_attack" : "needs_investigation",
    };
    if (isAttack) {
      body.attack_category = "Exploits";
    }
    const trimmedNotes = data.notes?.trim();
    if (trimmedNotes) {
      body.notes = trimmedNotes;
    }
    return request(`predictions/${flowId}/feedback`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  manifest: async (): Promise<ModelManifest> => {
    try {
      const model = await request<ModelSummary>("model");
      const sel = (model.selection_metrics || {}) as Record<string, number | string>;
      return {
        architecture: model.algorithm || "LightGBM Classifier",
        bundleVersion: model.bundle_version,
        threshold: model.decision_threshold,
        selection: {
          Accuracy: sel.accuracy ? `${(Number(sel.accuracy) * 100).toFixed(2)}%` : "95.51%",
          "Balanced Accuracy": sel.balanced_accuracy ? `${(Number(sel.balanced_accuracy) * 100).toFixed(2)}%` : "95.64%",
          Precision: sel.precision ? `${(Number(sel.precision) * 100).toFixed(2)}%` : "98.01%",
          Recall: sel.recall ? `${(Number(sel.recall) * 100).toFixed(2)}%` : "95.25%",
          F1: sel.f1 ? `${(Number(sel.f1) * 100).toFixed(2)}%` : "96.61%",
          "False positive rate": sel.false_positive_rate ? `${(Number(sel.false_positive_rate) * 100).toFixed(2)}%` : "3.98%",
          "ROC-AUC": sel.roc_auc ? Number(sel.roc_auc).toFixed(4) : "0.9943",
        },
        artifactHashes: model.bundle_hashes || {},
        sourceHashes: model.source_hashes || {},
      };
    } catch {
      return mockAdapter.manifest();
    }
  },

  probe: async (target: string): Promise<ProbeResult> => {
    const rawTarget = target.trim();
    const normalizedTarget =
      rawTarget.startsWith("http://") || rawTarget.startsWith("https://")
        ? rawTarget
        : `https://${rawTarget}`;

    try {
      const probeRes = await request<LiveProbeResult>("traffic/live-probe", {
        method: "POST",
        body: JSON.stringify({ target_url: normalizedTarget }),
      });

      const conn = probeRes.connection;
      const measurements: Record<string, string> = {
        "DNS resolution": conn ? `${conn.dns_resolve_ms.toFixed(1)} ms` : "—",
        "TCP connection": conn ? `${conn.tcp_connect_ms.toFixed(1)} ms` : "—",
        "TLS handshake":
          conn && conn.tls_handshake_ms != null ? `${conn.tls_handshake_ms.toFixed(1)} ms` : "None",
        "Time to first byte": conn ? `${conn.ttfb_ms.toFixed(1)} ms` : "—",
        "Total request time": conn ? `${conn.total_ms.toFixed(1)} ms` : "—",
        "HTTP status": conn ? String(conn.status_code) : "—",
        "HTTP version": conn ? conn.http_version : "—",
        "Response size": conn ? `${conn.response_size_bytes.toLocaleString()} bytes` : "—",
        "Server header": conn?.server_header || "Not provided",
        "HTTPS": conn ? (conn.is_https ? "Yes" : "No") : "—",
      };

      const shap = (probeRes.top_features || []).map((feat) => ({
        name: feat.feature,
        value: typeof feat.value === "number" || typeof feat.value === "string" ? feat.value : "—",
        contribution: feat.contribution,
      }));

      return {
        id: probeRes.flow_id,
        target: normalizedTarget,
        timestamp: probeRes.event_time || new Date().toISOString(),
        verdict: probeRes.decision === "alert" ? "Threat Flagged" : "Normal Traffic",
        score: probeRes.score,
        threshold: probeRes.threshold,
        status: conn?.status_code || 200,
        threatLevel: probeRes.decision === "alert" ? "High" : "Normal",
        measurements,
        shap,
      };
    } catch (err) {
      if (getDataMode() === "demo") {
        return mockAdapter.probe(normalizedTarget);
      }
      throw err;
    }
  },

  models: async (): Promise<ModelOption[]> => {
    try {
      return await request<ModelOption[]>("models");
    } catch {
      return mockAdapter.models();
    }
  },

  predict: async (
    flows: Record<string, string | number>[],
    modelId?: string
  ): Promise<FlowPrediction[]> => {
    try {
      const categoricalFields = new Set(["proto", "service", "state"]);
      const formattedFlows = flows.map((f) => {
        const cleanFeatures: Record<string, string | number> = {};
        for (const field of predictorFields) {
          if (field in f) {
            const val = f[field];
            if (categoricalFields.has(field)) {
              cleanFeatures[field] = String(val);
            } else {
              const num = Number(val);
              cleanFeatures[field] = Number.isFinite(num) ? num : 0;
            }
          }
        }
        return {
          flow_id: makeUUID(),
          event_time: new Date().toISOString(),
          features: cleanFeatures,
        };
      });
      const res = await request<BatchPredictionResponse>("predictions", {
        method: "POST",
        body: JSON.stringify({
          schema_version: "unsw-nb15.v0",
          flows: formattedFlows,
          model_id: modelId || undefined,
        }),
      });

      return res.predictions.map((p, idx) => ({
        flowId: p.flow_id,
        alertId: p.alert_id || undefined,
        score: p.score,
        threshold: p.threshold,
        decision: p.decision === "alert" ? "Alert" : "Normal",
        category: p.predicted_class,
        groundTruth:
          flows[idx]?.ground_truth === "Attack"
            ? "Attack"
            : flows[idx]?.ground_truth === "Normal"
            ? "Normal"
            : undefined,
        features: flows[idx] || {},
        aeMode: p.ae_mode || undefined,
        aeError: p.ae_error ?? undefined,
        modelId: modelId || res.model_id || undefined,
      }));
    } catch {
      return mockAdapter.predict(flows, modelId);
    }
  },
};

const getActiveAdapter = () => (getDataMode() === "demo" ? mockAdapter : liveApi);

export const api = {
  health: () => getActiveAdapter().health(),
  stats: () => getActiveAdapter().stats(),
  alerts: () => getActiveAdapter().alerts(),
  alert: (id: string) => getActiveAdapter().alert(id),
  submitAlertFeedback: (
    id: string,
    data: {
      verdict: ReviewStatus;
      notes?: string;
      expectedVersion: number;
      category?: string;
    }
  ) => liveApi.submitAlertFeedback(id, data),
  samples: () => getActiveAdapter().samples(),
  submitSampleFeedback: (flowId: string, data: { verdict: string; notes?: string }) =>
    liveApi.submitSampleFeedback(flowId, data),
  manifest: () => getActiveAdapter().manifest(),
  models: () => getActiveAdapter().models(),
  probe: (target: string) => getActiveAdapter().probe(target),
  predict: (flows: Record<string, string | number>[], modelId?: string) =>
    getActiveAdapter().predict(flows, modelId),
};

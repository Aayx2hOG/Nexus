export type Severity = "Critical" | "High" | "Medium" | "Low";
export type ReviewStatus =
  | "Confirmed Attack"
  | "False Positive"
  | "Investigating"
  | "Pending Review";

export interface ShapFeature {
  name: string;
  value: string | number;
  contribution: number;
}

export interface Alert {
  id: string;
  flowId: string;
  createdAt: string;
  severity: Severity;
  source: string;
  destination?: string;
  score: number;
  threshold: number;
  category: string;
  reviewStatus: ReviewStatus;
}

export interface Review {
  timestamp: string;
  verdict: Exclude<ReviewStatus, "Pending Review">;
  notes: string;
  version: number;
  reviewer?: string;
}

export interface AlertDetail extends Alert {
  protocol: string;
  service: string;
  state: string;
  shap: ShapFeature[];
  features: Record<string, string | number>;
  reviewVersion: number;
  reviews: Review[];
}

export interface AlertStats {
  last24Hours: number;
  awaitingReview: number;
  confirmedAttacks: number;
  falsePositives: number;
  investigating: number;
  totalPredictions: number;
}

export interface Health {
  system: "Operational" | "Unavailable" | "Degraded";
  api: "Operational" | "Unavailable" | "Degraded";
  model: "Operational" | "Unavailable" | "Degraded";
}

export interface ReviewSample {
  id: string;
  ingestTime: string;
  protocol: string;
  service: string;
  state: string;
  score: number;
  threshold: number;
  auditStatus: "Unreviewed" | "Needs Triage" | "Missed Attack" | "Malicious Intent";
  features: Record<string, string | number>;
  decision?: "normal";
  reviewVersion?: number;
}

export interface ProbeResult {
  id: string;
  target: string;
  timestamp: string;
  verdict: string;
  score: number;
  threshold: number;
  status: number;
  threatLevel: Severity | "Normal";
  measurements: Record<string, string>;
  shap: ShapFeature[];
}

export interface ModelManifest {
  architecture: string;
  bundleVersion: string;
  threshold: number;
  selection: Record<string, string>;
  artifactHashes: Record<string, string>;
  sourceHashes: Record<string, string>;
}

export interface ModelOption {
  id: string;
  name: string;
  description: string;
  architecture: string;
  decision_threshold: number;
  focus: string;
  ae_budget?: string | null;
  is_fusion: boolean;
}

export interface FlowPrediction {
  flowId: string;
  alertId?: string;
  score: number;
  threshold: number;
  decision: "Alert" | "Normal";
  category?: string;
  groundTruth?: "Attack" | "Normal";
  features: Record<string, string | number>;
  aeMode?: string;
  aeError?: number;
  modelId?: string;
}

/* Backend specific types */
export type BackendVerdict = "confirmed_attack" | "false_positive" | "needs_investigation" | "pending";

export interface FeatureContribution {
  feature: string;
  contribution: number;
  value?: number | string | null;
}

export interface AlertRecord {
  sequence: number;
  alert_id: string;
  flow_id: string;
  event_time: string;
  source: "mock" | "model";
  bundle_version: string;
  predicted_class: string;
  score?: number | null;
  threshold?: number | null;
  probability?: number | null;
  severity: string;
  top_features: FeatureContribution[];
  created_at: string;
  feedback_version: number;
}

export interface AlertPage {
  items: AlertRecord[];
  next_after: number | null;
}

export interface FeedbackRecord {
  sequence: number;
  feedback_id: string;
  alert_id: string;
  version: number;
  verdict: BackendVerdict;
  attack_category: string | null;
  notes: string | null;
  reviewer_id: string;
  created_at: string;
  provenance: string;
}

export interface FeedbackPage {
  items: FeedbackRecord[];
  next_after: number | null;
}

export interface Stats {
  alerts_24h: number;
  awaiting_review: number;
  confirmed_attacks: number;
  false_positives: number;
  needs_investigation: number;
  total_predictions_24h: number;
}

export interface ModelSummary {
  bundle_version: string;
  algorithm: string;
  decision_threshold: number;
  bundle_hashes: Record<string, string>;
  source_hashes: Record<string, string>;
  selection_metrics: Record<string, unknown>;
  metrics_note: string;
}

export interface SampleReview {
  verdict: BackendVerdict;
  attack_category?: string | null;
  notes?: string | null;
  reviewer_id: string;
  created_at: string;
}

export interface NonAlertSampleItem {
  sequence: number;
  flow_id: string;
  bundle_version: string;
  event_time?: string | null;
  ingest_time: string;
  features: Record<string, unknown>;
  score: number;
  threshold: number;
  decision: "normal";
  review?: SampleReview | null;
}

export interface NonAlertSamplePage {
  items: NonAlertSampleItem[];
  next_after: number | null;
}

export interface SimulationProfile {
  id: string;
  name: string;
  category: string;
  description: string;
  service: string;
  sttl: number;
  dttl: number;
  swin: number;
  dwin: number;
  state: string;
  proto: string;
  attack_category: string | null;
  risk_level: string;
}

export interface PresetData {
  preset_id: string;
  name: string;
  description: string;
  flow_count: number;
  schema_version: string;
  flows: Array<{
    flow_id: string;
    event_time: string;
    features: Record<string, string | number>;
    expected_class?: string;
  }>;
}

export interface BatchPredictionItem {
  flow_id: string;
  decision: "alert" | "normal";
  score: number;
  threshold: number;
  predicted_class?: string;
  alert_id?: string | null;
  top_features?: FeatureContribution[];
  ae_mode?: string | null;
  ae_error?: number | null;
}

export interface BatchPredictionResponse {
  schema_version?: string;
  evaluated_count?: number;
  alert_count: number;
  bundle_version: string;
  predictions: BatchPredictionItem[];
  model_id?: string | null;
}

export interface ConnectionMetrics {
  dns_resolve_ms: number;
  tcp_connect_ms: number;
  tls_handshake_ms: number | null;
  ttfb_ms: number;
  total_ms: number;
  request_size_bytes: number;
  response_size_bytes: number;
  response_header_bytes: number;
  response_body_bytes: number;
  status_code: number;
  http_version: string;
  num_redirects: number;
  server_header: string | null;
  content_type: string | null;
  is_https: boolean;
}

export interface LiveProbeResult {
  target_url: string;
  method?: string;
  flow_id: string;
  connection?: ConnectionMetrics;
  target_hit?: boolean;
  score: number;
  threshold: number;
  decision: "alert" | "normal";
  threat_level?: "CRITICAL" | "HIGH" | "MEDIUM" | "CLEAN";
  alert_id?: string | null;
  top_features: FeatureContribution[];
  recommended_action?: string;
  event_time?: string;
  extracted_features?: Record<string, any>;
  resolved_ip?: string | null;
  elapsed_ms?: number;
  predicted_class?: string;
  tls_version?: string | null;
  http_version?: string | null;
  status_code?: number | null;
  measurements?: Record<string, string | number>;
  features?: Record<string, string | number>;
}

export interface ReplaySummary {
  evaluated_flows: number;
  evaluation_alerts: number;
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  recall: number | null;
  precision: number | null;
  false_positive_rate: number | null;
  false_alerts_per_1000_normal: number | null;
  summary_note: string;
}

type PolicyMetrics = {
  true_negatives: number;
  false_positives: number;
  false_negatives: number;
  true_positives: number;
  recall: number | null;
  precision: number | null;
  false_positive_rate: number | null;
  alert_count: number;
};

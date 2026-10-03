export type Severity = "alert" | "low" | "medium" | "high" | "critical";
export type Verdict = "confirmed_attack" | "false_positive" | "needs_investigation" | "pending";

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
  severity: Severity;
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
  verdict: Verdict;
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
  verdict: Verdict;
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
  expected_verdict: string;
}

export interface ProbeResult {
  target_url: string;
  profile_id: string;
  profile_name: string;
  category: string;
  flow_id: string;
  target_hit: boolean;
  score: number;
  threshold: number;
  decision: "alert" | "normal";
  threat_level: "CRITICAL" | "HIGH" | "MEDIUM" | "CLEAN";
  alert_id?: string | null;
  top_features: FeatureContribution[];
  recommended_action: string;
  event_time: string;
}

export interface PresetData {
  preset_id: string;
  flow_count: number;
  csv_text: string;
  flows: Record<string, unknown>[];
}

export interface BatchPredictionItem {
  flow_id: string;
  score: number;
  threshold: number;
  decision: "alert" | "normal";
  alert_id?: string | null;
}

export interface BatchPredictionResponse {
  bundle_version: string;
  predictions: BatchPredictionItem[];
  alert_count: number;
  total_count: number;
}


export interface ReplaySummary {
  total_replayed: number;
  confusion_matrix: number[][];
  true_positives: number;
  false_positives: number;
  true_negatives: number;
  false_negatives: number;
  accuracy: number;
  recall: number;
  false_positive_rate: number;
  false_alerts_per_1000_normal: number;
  summary_note: string;
}

export interface Health {
  status: "alive" | "not_ready" | "ready";
  bundle_version?: string;
}

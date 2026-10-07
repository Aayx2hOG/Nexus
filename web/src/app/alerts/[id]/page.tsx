"use client";

import { use, useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertOctagon, ChevronRight, RefreshCw } from "lucide-react";
import {
  Button,
  EmptyState,
  ErrorState,
  InfoHelp,
  LoadingState,
  ScoreScale,
  Status,
  TechnicalValue,
} from "@/components/components";
import { api, isPreviewData } from "@/lib/api";
import type { AlertDetail, ReviewStatus } from "@/lib/types";

const statusTone = (status: ReviewStatus) => {
  if (status === "Confirmed Attack") return "danger";
  if (status === "False Positive") return "success";
  if (status === "Investigating") return "warning";
  return "neutral";
};

function ShapSection({ alert }: { alert: AlertDetail }) {
  const max = Math.max(...alert.shap.map((item) => Math.abs(item.contribution)), 0.01);
  return (
    <section className="content-section">
      <div className="section-heading">
        <div>
          <span className="section-index">TREESHAP EVIDENCE</span>
          <div className="heading-with-help">
            <h2>Why this flow was flagged</h2>
            <InfoHelp
              label="TreeSHAP evidence"
              text="TreeSHAP shows which network features influenced the model's decision and how strongly each feature contributed."
            />
          </div>
          <p>Features that pushed the prediction toward an alert or toward normal activity.</p>
        </div>
        <div className="legend">
          <span><i className="legend-up" /> Increased attack probability</span>
          <span><i className="legend-down" /> Reduced attack probability</span>
        </div>
      </div>
      <div className="shap-table">
        <div className="shap-head">
          <span>Feature</span><span>Value</span><span>Contribution</span>
        </div>
        {alert.shap.length === 0 ? (
          <div className="shap-empty">No TreeSHAP evidence was returned for this alert.</div>
        ) : (
          alert.shap.map((feature) => (
            <div className="shap-row" key={feature.name}>
              <strong className="mono">{feature.name}</strong>
              <span className="mono">{feature.value}</span>
              <div className="contribution">
                <div className="contribution-axis" />
                <div
                  className={`contribution-bar ${feature.contribution >= 0 ? "positive" : "negative"}`}
                  style={{ width: `${(Math.abs(feature.contribution) / max) * 48}%` }}
                />
                <span className="mono">{feature.contribution > 0 ? "+" : ""}{feature.contribution.toFixed(3)}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default function AlertDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: ["alert", id], queryFn: () => api.alert(id) });
  const [selectedVerdict, setSelectedVerdict] = useState<Exclude<ReviewStatus, "Pending Review"> | "">("");
  const [analystNotes, setAnalystNotes] = useState("");
  const [savedSuccess, setSavedSuccess] = useState(false);

  const reviewMutation = useMutation({
    mutationFn: async () => {
      if (!selectedVerdict || !alert) return;
      await api.submitAlertFeedback(alert.id, {
        verdict: selectedVerdict,
        notes: analystNotes,
        expectedVersion: alert.reviewVersion,
        category: alert.category,
      });
    },
    onSuccess: () => {
      setSavedSuccess(true);
      void queryClient.invalidateQueries({ queryKey: ["alert", id] });
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void queryClient.invalidateQueries({ queryKey: ["stats"] });
      router.push("/alerts");
    },
  });

  const alert = query.data;
  if (query.isLoading) return <div className="page"><LoadingState label="Loading incident evidence" /></div>;
  if (query.isError || !alert) return <div className="page"><ErrorState retry={() => void query.refetch()} /></div>;

  return (
    <div className="incident-page">
      <div className="incident-banner">
        <div className="incident-banner-top">
          <button className="back-link" onClick={() => router.push("/alerts")}>
            <ChevronRight size={15} /> Back to alert queue
          </button>
          <span className="incident-live"><i /> ACTIVE INCIDENT</span>
        </div>
        <div className="incident-title-row">
          <div>
            <span className="incident-label"><AlertOctagon size={16} /> INCIDENT</span>
            <h1 className="mono">{alert.id}</h1>
            <div className="incident-source mono">{alert.source}</div>
          </div>
          <div className="incident-statuses">
            <strong className={`incident-severity severity-${alert.severity.toLowerCase()}`}>{alert.severity.toUpperCase()}</strong>
            <span>{alert.reviewStatus}</span>
          </div>
        </div>
        <div className="incident-snapshot">
          <div>
            <span>MODEL SCORE <InfoHelp label="model score" text="A number showing how strongly the model considers this network flow suspicious. Higher scores indicate stronger suspicion." /></span>
            <strong className="mono">{alert.score.toFixed(4)}</strong>
          </div>
          <div>
            <span>THRESHOLD <InfoHelp label="decision threshold" text="The minimum score required for NEXUS to classify a network flow as an alert." /></span>
            <strong className="mono">{alert.threshold.toFixed(4)}</strong>
          </div>
          <div><span>PREDICTED TYPE</span><strong>{alert.category}</strong></div>
          <div>
            <span>FLOW ID <InfoHelp label="Network flow" text="A summarized record of communication between network endpoints, including timing, protocol, and transferred data." /></span>
            <strong className="mono">{alert.flowId}</strong>
          </div>
          <div><span>DETECTED</span><strong className="mono">{new Date(alert.createdAt).toLocaleString()}</strong></div>
        </div>
      </div>

      <div className="incident-content">
        <section className="content-section evidence-section">
          <span className="section-index">CONNECTION</span>
          <p className="section-explanation">The systems and network service involved in this flow.</p>
          <div className="evidence-grid">
            <div><span>SOURCE</span><strong className="mono">{alert.source}</strong></div>
            <div><span>DESTINATION</span><strong className="mono">{alert.destination ?? "Not provided"}</strong></div>
            <div><span>PROTOCOL</span><strong className="mono">{alert.protocol}</strong></div>
            <div><span>SERVICE / STATE</span><strong className="mono">{alert.service} / {alert.state}</strong></div>
          </div>
          <div className="score-comparison">
            <div>
              <div className="heading-with-help">
                <h2>Score against threshold</h2>
                <InfoHelp label="score comparison" text="The flow becomes an alert when its model score crosses the configured decision threshold." />
              </div>
              <p className="mono">+{(alert.score - alert.threshold).toFixed(4)}</p>
            </div>
            <ScoreScale score={alert.score} threshold={alert.threshold} />
          </div>
        </section>

        <ShapSection alert={alert} />

        <section className="content-section">
          <div className="feature-heading">
            <div>
              <span className="section-index">MODEL INPUT</span>
              <h2>42 features</h2>
              <p>The complete set of network measurements the model used to score this flow.</p>
            </div>
            <span className="feature-count mono">{Object.keys(alert.features).length} / 42</span>
          </div>
          <div className="feature-grid">
            {Object.entries(alert.features).map(([name, value]) => (
              <div key={name}><span className="mono">{name}</span><strong className="mono">{String(value)}</strong></div>
            ))}
          </div>
        </section>

        <div className="review-layout">
          <section className="content-section">
            <span className="section-index">RESPONSE</span>
            <h2>Recorded decision</h2>
            <p className="section-explanation">The current review state recorded by the analyst team.</p>
            <div className="recorded-decision">
              <Status tone={statusTone(alert.reviewStatus)}>{alert.reviewStatus}</Status>
              <div>
                <strong>
                  {alert.reviewStatus === "Confirmed Attack"
                    ? "An analyst determined that this activity represents an attack."
                    : alert.reviewStatus === "False Positive"
                      ? "An analyst determined that this is normal activity flagged incorrectly."
                      : alert.reviewStatus === "Investigating"
                        ? "The alert is still under active investigation."
                        : "No analyst decision has been recorded yet."}
                </strong>
                <span>Review version <TechnicalValue>{alert.reviewVersion}</TechnicalValue></span>
              </div>
            </div>

            <div className="demo-review">
              <div className="demo-review-heading">
                <span>SUBMIT ANALYST VERDICT</span>
                <small>Records decision in backend storage.</small>
              </div>
              <div className="segmented">
                {(["Confirmed Attack", "False Positive", "Investigating"] as const).map((option) => (
                  <button
                    key={option}
                    className={selectedVerdict === option ? "selected" : ""}
                    onClick={() => { setSelectedVerdict(option); setSavedSuccess(false); }}
                  >
                    {option}
                  </button>
                ))}
              </div>
              <label className="field">
                <span>ANALYST FINDINGS & NOTES (OPTIONAL)</span>
                <textarea
                  value={analystNotes}
                  onChange={(event) => { setAnalystNotes(event.target.value); setSavedSuccess(false); }}
                  rows={3}
                  placeholder="Optional: Record justification, mitigation, or forensic observations."
                />
              </label>
              <div className="demo-review-submit">
                {savedSuccess && <span className="status-success font-mono">Verdict recorded successfully.</span>}
                <Button
                  variant="primary"
                  disabled={!selectedVerdict || reviewMutation.isPending}
                  onClick={() => reviewMutation.mutate()}
                >
                  {reviewMutation.isPending ? <RefreshCw className="spin" size={15} /> : "Submit verdict"}
                </Button>
              </div>
            </div>
          </section>

          <section className="content-section">
            <span className="section-index">HISTORY</span>
            <h2>Review history</h2>
            <p className="section-explanation">Previous analyst decisions, notes, and review versions for this alert.</p>
            {alert.reviews.length === 0 ? (
              <EmptyState title="No prior decisions." detail="This incident has not been reviewed." />
            ) : (
              <div className="timeline">
                {alert.reviews.map((review: AlertDetail["reviews"][number]) => (
                  <div className="timeline-item" key={review.version}>
                    <i />
                    <div>
                      <div className="timeline-top">
                        <Status tone={statusTone(review.verdict)}>{review.verdict}</Status>
                        <TechnicalValue>v{review.version}</TechnicalValue>
                      </div>
                      <p>{review.notes}</p>
                      <small>
                        <TechnicalValue>{new Date(review.timestamp).toLocaleString()}</TechnicalValue>
                        {review.reviewer && <> · {review.reviewer}</>}
                      </small>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertOctagon, ChevronRight, RefreshCw, Search } from "lucide-react";
import {
  Button,
  EmptyState,
  ErrorState,
  InfoHelp,
  LoadingState,
  PageHeader,
  Status,
  TechnicalValue,
} from "@/components/components";
import { api } from "@/lib/api";
import type { ReviewSample } from "@/lib/types";

export default function ReviewSamplePage() {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: ["review-samples"], queryFn: api.samples, refetchInterval: 20_000 });
  const [selected, setSelected] = useState<string | null>(null);
  const [analystDecision, setAnalystDecision] = useState("");
  const [analystNotes, setAnalystNotes] = useState("");
  const [savedSuccess, setSavedSuccess] = useState(false);

  const sample: ReviewSample | undefined = query.data?.find((item) => item.id === selected);

  useEffect(() => {
    setAnalystDecision("");
    setAnalystNotes("");
    setSavedSuccess(false);
  }, [selected]);

  const submitFeedbackMutation = useMutation({
    mutationFn: async () => {
      if (!selected || !analystDecision) return;
      await api.submitSampleFeedback(selected, {
        verdict: analystDecision,
        notes: analystNotes,
      });
    },
    onSuccess: () => {
      setSavedSuccess(true);
      void queryClient.invalidateQueries({ queryKey: ["review-samples"] });
    },
  });

  return (
    <div className="page">
      <PageHeader
        eyebrow="MODEL AUDIT"
        title="Missed attack review"
        description="Check network flows NEXUS classified as normal and identify activity the primary detector may have missed."
        actions={<Button onClick={() => void query.refetch()}><RefreshCw size={15} /> Refresh</Button>}
      />
      <div className="audit-context">
        <AlertOctagon size={20} />
        <div>
          <strong>
            False negative audit queue{" "}
            <InfoHelp label="Missed attack review" text="A review area for checking flows classified as normal and identifying possible attacks the primary detector missed." />
          </strong>
          <p>These flows scored below the alert threshold. Reviewing them helps identify attacks that may have been classified as normal.</p>
        </div>
      </div>
      {query.isLoading ? <LoadingState /> : query.isError ? <ErrorState retry={() => void query.refetch()} /> : !query.data?.length ? (
        <EmptyState title="No flows require audit." detail="No normal predictions are queued for additional review." />
      ) : (
        <div className="audit-layout">
          <div className="audit-list">
            <div className="audit-list-head">
              <div>
                <div className="heading-with-help">
                  <h2>Normal prediction queue</h2>
                  <InfoHelp label="normal flow" text="Network activity that did not cross the model's alert threshold." />
                </div>
                <p>{query.data.length} flows available</p>
              </div>
              <Search size={17} />
            </div>
            {query.data.map((item) => (
              <button
                className={`sample-row ${selected === item.id ? "selected" : ""}`}
                key={item.id}
                onClick={() => setSelected(item.id)}
              >
                <div className="sample-top">
                  <TechnicalValue>{item.id}</TechnicalValue>
                  <Status tone={item.auditStatus === "Unreviewed" ? "neutral" : "warning"}>{item.auditStatus}</Status>
                </div>
                <div className="sample-data">
                  <span><small>PROTOCOL</small>{item.protocol}</span>
                  <span><small>SERVICE</small>{item.service}</span>
                  <span><small>SCORE</small><b className="mono">{item.score.toFixed(4)}</b></span>
                </div>
                <div className="sample-foot">
                  <TechnicalValue>{new Date(item.ingestTime).toLocaleString()}</TechnicalValue>
                  <ChevronRight size={15} />
                </div>
              </button>
            ))}
          </div>
          <div className="audit-detail">
            {!sample ? (
              <EmptyState title="Select a flow for review." detail="Open a normal prediction to inspect all predictor values." />
            ) : (
              <>
                <div className="audit-detail-head">
                  <div><span>FLOW INSPECTION</span><h2>{sample.id}</h2></div>
                  <Status tone="success">Primary decision: Normal</Status>
                </div>
                <div className="audit-score">
                  <div><span>MODEL SCORE</span><strong className="mono">{sample.score.toFixed(4)}</strong></div>
                  <div><span>THRESHOLD</span><strong className="mono">{sample.threshold.toFixed(4)}</strong></div>
                  <div><span>DELTA</span><strong className="mono">{(sample.score - sample.threshold).toFixed(4)}</strong></div>
                </div>
                <div className="feature-grid compact">
                  {Object.entries(sample.features).map(([name, value]) => (
                    <div key={name}><span className="mono">{name}</span><strong className="mono">{String(value)}</strong></div>
                  ))}
                </div>
                <div className="audit-decision">
                  <h3>AUDIT STATUS</h3>
                  <p>The current classification returned by the prediction review API.</p>
                  <Status tone={sample.auditStatus === "Unreviewed" ? "neutral" : "warning"}>{sample.auditStatus}</Status>
                  <div className="demo-review">
                    <div className="demo-review-heading">
                      <span>RECORD AUDIT VERDICT</span>
                      <small>Records audit finding to SQLite database.</small>
                    </div>
                    <div className="segmented">
                      {["Missed Attack", "Needs Triage", "Malicious Intent"].map((option) => (
                        <button
                          key={option}
                          className={analystDecision === option ? "selected" : ""}
                          onClick={() => { setAnalystDecision(option); setSavedSuccess(false); }}
                        >
                          {option}
                        </button>
                      ))}
                    </div>
                    <label className="field">
                      <span>AUDIT NOTES (OPTIONAL)</span>
                      <textarea
                        value={analystNotes}
                        onChange={(event) => { setAnalystNotes(event.target.value); setSavedSuccess(false); }}
                        rows={3}
                        placeholder="Optional: Record audit reasoning or missed attack details."
                      />
                    </label>
                    <div className="demo-review-submit">
                      {savedSuccess && <span className="status-success font-mono">Audit finding recorded.</span>}
                      <Button
                        variant="primary"
                        disabled={!analystDecision || submitFeedbackMutation.isPending}
                        onClick={() => submitFeedbackMutation.mutate()}
                      >
                        {submitFeedbackMutation.isPending ? <RefreshCw className="spin" size={15} /> : "Save audit review"}
                      </Button>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchAlert, fetchAlertFeedback, submitAlertFeedback } from "@/lib/api";
import { Verdict } from "@/lib/types";
import {
  ArrowLeft,
  AlertTriangle,
  History,
  Copy,
  Check,
} from "lucide-react";

const ATTACK_CATEGORIES = [
  "Generic",
  "Exploits",
  "Fuzzers",
  "Reconnaissance",
  "DoS",
  "Backdoor",
  "Analysis",
  "Worms",
  "Shellcode",
];

export default function AlertDetailPage() {
  const params = useParams();
  const alertId = params.id as string;
  const queryClient = useQueryClient();

  const [verdict, setVerdict] = useState<Verdict>("confirmed_attack");
  const [attackCategory, setAttackCategory] = useState<string>("Generic");
  const [notes, setNotes] = useState<string>("");
  const [conflictError, setConflictError] = useState<string | null>(null);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  // Fetch alert detail
  const {
    data: alert,
    isLoading: alertLoading,
    error: alertError,
    refetch: refetchAlert,
  } = useQuery({
    queryKey: ["alert", alertId],
    queryFn: () => fetchAlert(alertId),
  });

  // Fetch feedback history
  const { data: feedbackPage, refetch: refetchFeedback } = useQuery({
    queryKey: ["feedback", alertId],
    queryFn: () => fetchAlertFeedback(alertId),
  });

  const feedbackList = feedbackPage?.items || [];

  // Mutation for submitting review verdict
  const submitMutation = useMutation({
    mutationFn: async () => {
      setConflictError(null);
      return submitAlertFeedback(alertId, {
        feedback_id: crypto.randomUUID(),
        expected_version: alert?.feedback_version || 0,
        verdict,
        attack_category: verdict === "confirmed_attack" ? attackCategory : null,
        notes: notes.trim() ? notes.trim() : null,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alert", alertId] });
      queryClient.invalidateQueries({ queryKey: ["feedback", alertId] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      setNotes("");
    },
    onError: (err: unknown) => {
      const errorObj = err as { status?: number; code?: string; message?: string };
      if (
        errorObj.status === 409 ||
        errorObj.code === "feedback_conflict" ||
        errorObj.code === "stale_feedback"
      ) {
        setConflictError(
          "Concurrent update detected: Another review was recorded. Please refresh to load the latest state before submitting."
        );
      } else {
        setConflictError(errorObj.message || "Failed to submit verdict.");
      }
    },
  });

  // Keyboard shortcut listener for quick verdict selection (1, 2, 3)
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }
      if (e.key === "1") {
        e.preventDefault();
        setVerdict("confirmed_attack");
      } else if (e.key === "2") {
        e.preventDefault();
        setVerdict("false_positive");
      } else if (e.key === "3") {
        e.preventDefault();
        setVerdict("needs_investigation");
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  function copyToClipboard(text: string, field: string) {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  }

  if (alertLoading) {
    return (
      <div className="py-24 text-center text-xs text-[#939AA6]">
        Loading alert details...
      </div>
    );
  }

  if (alertError || !alert) {
    return (
      <div className="bg-[#13151A] border border-[#282C35] p-6 rounded-sm text-center">
        <h2 className="text-sm font-semibold text-[#F87171]">Alert Not Found</h2>
        <p className="text-xs text-[#939AA6] mt-1">
          The requested alert ID does not exist in the database.
        </p>
        <Link
          href="/alerts"
          className="inline-flex items-center space-x-1 text-xs text-[#F1F3F6] underline mt-4"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Alerts Queue</span>
        </Link>
      </div>
    );
  }

  const score = alert.score ?? alert.probability ?? 0.0;
  const threshold = alert.threshold ?? 0.5777;
  const isAboveThreshold = score >= threshold;

  return (
    <div className="space-y-6">
      {/* Top Breadcrumb & Actions */}
      <div className="flex items-center justify-between">
        <Link
          href="/alerts"
          className="inline-flex items-center space-x-1 text-xs text-[#939AA6] hover:text-[#F1F3F6]"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Queue</span>
        </Link>

        <div className="flex items-center space-x-2">
          {alert.source === "mock" && (
            <span className="px-2 py-0.5 text-[11px] font-semibold uppercase bg-[#2B1F0B] text-[#FBBF24] border border-[#5E4012] rounded-sm">
              MOCK ALERT
            </span>
          )}
          <span className="px-2 py-0.5 text-[11px] font-semibold uppercase bg-[#2A1316] text-[#F87171] border border-[#5C1D24] rounded-sm">
            {alert.severity}
          </span>
        </div>
      </div>

      {/* Main Alert Info Card */}
      <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 border-b border-[#282C35] pb-4">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-[#939AA6] font-mono">
              Alert Identification
            </div>
            <div className="flex items-center space-x-2 mt-1">
              <span className="text-sm font-mono font-semibold text-[#F1F3F6]">
                {alert.alert_id}
              </span>
              <button
                onClick={() => copyToClipboard(alert.alert_id, "alert_id")}
                className="text-[#939AA6] hover:text-[#F1F3F6]"
                title="Copy Alert ID"
              >
                {copiedField === "alert_id" ? (
                  <Check className="w-3.5 h-3.5 text-[#34D399]" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>

          <div className="flex items-center space-x-6 text-xs">
            <div>
              <div className="text-[11px] text-[#939AA6]">Created Timestamp</div>
              <div className="font-mono text-[#F1F3F6] tabular-nums mt-0.5">
                {alert.created_at}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#939AA6]">Bundle Version</div>
              <div className="font-mono text-[#F1F3F6] mt-0.5">
                {alert.bundle_version}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#939AA6]">Review Version</div>
              <div className="font-mono text-[#F1F3F6] mt-0.5">
                v{alert.feedback_version}
              </div>
            </div>
          </div>
        </div>

        {/* Score vs Threshold Visual Bar */}
        <div className="bg-[#181B21] border border-[#282C35] p-4 rounded-sm space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div>
              <span className="font-medium text-[#F1F3F6]">
                Model Score Output
              </span>
              <span className="text-[11px] text-[#939AA6] ml-2">
                (Raw model score & fixed decision threshold; not calibrated probability)
              </span>
            </div>
            <div className="font-mono text-xs tabular-nums">
              <span className="font-semibold text-[#F87171]">
                Score: {score.toFixed(6)}
              </span>
              <span className="text-[#616875] mx-2">|</span>
              <span className="text-[#939AA6]">
                Threshold: {threshold.toFixed(6)}
              </span>
            </div>
          </div>

          {/* Bar track */}
          <div className="relative h-4 bg-[#282C35] rounded-xs overflow-hidden">
            {/* Fill for score */}
            <div
              className={`h-full ${
                isAboveThreshold ? "bg-[#F87171]" : "bg-[#34D399]"
              }`}
              style={{ width: `${Math.min(Math.max(score * 100, 0), 100)}%` }}
            />
            {/* Threshold marker line */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-[#F1F3F6] z-10"
              style={{ left: `${threshold * 100}%` }}
              title={`Threshold: ${threshold.toFixed(4)}`}
            />
          </div>

          <div className="flex justify-between text-[10px] text-[#939AA6] font-mono">
            <span>0.0 (Normal)</span>
            <span>Threshold ({threshold.toFixed(4)})</span>
            <span>1.0 (Alert)</span>
          </div>
        </div>
      </div>

      {/* Grid: TreeSHAP Explanations + Analyst Verdict Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: TreeSHAP Feature Contributions */}
        <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
          <div>
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
              TreeSHAP Feature Contributions
            </h2>
            <p className="text-[11px] text-[#939AA6] mt-0.5">
              Top signed contributions driving score toward or away from alert
            </p>
          </div>

          {alert.top_features && alert.top_features.length > 0 ? (
            <div className="space-y-2 pt-2">
              {alert.top_features.map((feat, idx) => {
                const isPositive = feat.contribution > 0;
                // Maximum contribution scaling for bar width
                const maxContrib = Math.max(
                  ...alert.top_features.map((f) => Math.abs(f.contribution)),
                  0.1
                );
                const barWidth = Math.min(
                  (Math.abs(feat.contribution) / maxContrib) * 100,
                  100
                );

                return (
                  <div key={idx} className="space-y-1 text-xs">
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-[#F1F3F6]">
                          {feat.feature}
                        </span>
                        {feat.value !== null && feat.value !== undefined && (
                          <span className="text-[#939AA6] text-[10px]">
                            = {String(feat.value)}
                          </span>
                        )}
                      </div>
                      <span
                        className={`tabular-nums font-semibold ${
                          isPositive ? "text-[#F87171]" : "text-[#34D399]"
                        }`}
                      >
                        {isPositive ? "+" : ""}
                        {feat.contribution.toFixed(4)}
                      </span>
                    </div>

                    <div className="h-2 bg-[#181B21] border border-[#282C35] rounded-xs flex overflow-hidden">
                      <div
                        className={`h-full ${
                          isPositive ? "bg-[#F87171]" : "bg-[#34D399]"
                        }`}
                        style={{ width: `${barWidth}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[#939AA6]">
              No TreeSHAP explanation attributes available for this alert.
            </div>
          )}
        </div>

        {/* Right Column: Analyst Review Form */}
        <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                Record Analyst Verdict
              </h2>
              <p className="text-[11px] text-[#939AA6] mt-0.5">
                Current version: v{alert.feedback_version} | Use keys 1, 2, 3 to select
              </p>
            </div>
          </div>

          {conflictError && (
            <div className="bg-[#2A1316] border border-[#5C1D24] p-3 rounded-sm text-xs text-[#F87171] flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <div>{conflictError}</div>
                <button
                  onClick={() => {
                    refetchAlert();
                    refetchFeedback();
                    setConflictError(null);
                  }}
                  className="underline font-semibold mt-1"
                >
                  Reload Latest Version
                </button>
              </div>
            </div>
          )}

          <div className="space-y-3 pt-1">
            {/* Verdict Selection Buttons */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-medium text-[#939AA6]">
                Verdict Decision:
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setVerdict("confirmed_attack")}
                  className={`px-3 py-2 text-xs font-medium rounded-sm border transition-colors flex flex-col items-center justify-center space-y-1 ${
                    verdict === "confirmed_attack"
                      ? "bg-[#2A1316] border-[#F87171] text-[#F87171] font-semibold"
                      : "bg-[#181B21] border-[#282C35] text-[#939AA6] hover:bg-[#1D2027]"
                  }`}
                >
                  <span>Confirmed Attack</span>
                  <kbd className="text-[10px] opacity-75 font-mono">(1)</kbd>
                </button>

                <button
                  type="button"
                  onClick={() => setVerdict("false_positive")}
                  className={`px-3 py-2 text-xs font-medium rounded-sm border transition-colors flex flex-col items-center justify-center space-y-1 ${
                    verdict === "false_positive"
                      ? "bg-[#0C241B] border-[#34D399] text-[#34D399] font-semibold"
                      : "bg-[#181B21] border-[#282C35] text-[#939AA6] hover:bg-[#1D2027]"
                  }`}
                >
                  <span>False Positive</span>
                  <kbd className="text-[10px] opacity-75 font-mono">(2)</kbd>
                </button>

                <button
                  type="button"
                  onClick={() => setVerdict("needs_investigation")}
                  className={`px-3 py-2 text-xs font-medium rounded-sm border transition-colors flex flex-col items-center justify-center space-y-1 ${
                    verdict === "needs_investigation"
                      ? "bg-[#2B1F0B] border-[#FBBF24] text-[#FBBF24] font-semibold"
                      : "bg-[#181B21] border-[#282C35] text-[#939AA6] hover:bg-[#1D2027]"
                  }`}
                >
                  <span>Investigating</span>
                  <kbd className="text-[10px] opacity-75 font-mono">(3)</kbd>
                </button>
              </div>
            </div>

            {/* Attack Category dropdown (required when confirmed_attack) */}
            {verdict === "confirmed_attack" && (
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-[#939AA6]">
                  Attack Category:
                </label>
                <select
                  value={attackCategory}
                  onChange={(e) => setAttackCategory(e.target.value)}
                  className="w-full bg-[#1D2027] border border-[#282C35] rounded-sm px-2.5 py-1.5 text-xs text-[#F1F3F6] focus:outline-hidden"
                >
                  {ATTACK_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Notes textarea */}
            <div className="space-y-1">
              <label className="text-[11px] font-medium text-[#939AA6]">
                Analyst Triage Notes:
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Observed anomalous port scan patterns, high packet rate, verified benign scanner, etc."
                rows={3}
                className="w-full bg-[#1D2027] border border-[#282C35] rounded-sm p-2 text-xs text-[#F1F3F6] focus:outline-hidden resize-none placeholder:text-[#616875]"
              />
            </div>

            <button
              onClick={() => submitMutation.mutate()}
              disabled={submitMutation.isPending}
              className="w-full py-2 bg-[#F1F3F6] text-[#0A0B0D] text-xs font-medium rounded-sm hover:bg-white transition-colors disabled:opacity-50"
            >
              {submitMutation.isPending ? "Submitting Verdict..." : "Save Review Verdict"}
            </button>
          </div>

          {/* Feedback History Timeline */}
          {feedbackList.length > 0 && (
            <div className="pt-4 border-t border-[#282C35] space-y-3">
              <div className="flex items-center space-x-1.5 text-xs font-semibold text-[#F1F3F6]">
                <History className="w-3.5 h-3.5 text-[#939AA6]" />
                <span>Review History ({feedbackList.length})</span>
              </div>

              <div className="space-y-2 max-h-48 overflow-y-auto">
                {feedbackList.map((fb) => (
                  <div
                    key={fb.feedback_id}
                    className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-sm text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold text-[#F1F3F6]">
                        v{fb.version}: {fb.verdict.replace("_", " ").toUpperCase()}
                        {fb.attack_category ? ` (${fb.attack_category})` : ""}
                      </span>
                      <span className="text-[#939AA6] tabular-nums font-mono">
                        {fb.created_at}
                      </span>
                    </div>
                    {fb.notes && (
                      <p className="text-[11px] text-[#939AA6] italic">
                        &ldquo;{fb.notes}&rdquo;
                      </p>
                    )}
                    <div className="text-[10px] text-[#616875]">
                      Reviewer: {fb.reviewer_id}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

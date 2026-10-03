"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchSamplePredictions, submitSampleFeedback } from "@/lib/api";
import { NonAlertSampleItem, Verdict } from "@/lib/types";
import {
  AlertTriangle,
  ArrowRight,
  X,
  RefreshCw,
} from "lucide-react";

export default function ReviewSamplePage() {
  const queryClient = useQueryClient();
  const [selectedFlow, setSelectedFlow] = useState<NonAlertSampleItem | null>(null);
  const [pageAfter, setPageAfter] = useState<number | undefined>(undefined);
  const [pageHistory, setPageHistory] = useState<number[]>([]);

  // Verdict submission state
  const [verdict, setVerdict] = useState<Verdict>("false_positive");
  const [attackCategory, setAttackCategory] = useState<string>("Exploits");
  const [notes, setNotes] = useState<string>("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Fetch sample predictions
  const {
    data: samplePage,
    isLoading,
    isRefetching,
    refetch,
  } = useQuery({
    queryKey: ["samplePredictions", pageAfter],
    queryFn: () => fetchSamplePredictions({ limit: 25, after: pageAfter }),
  });

  const flows = samplePage?.items || [];

  const feedbackMutation = useMutation({
    mutationFn: async () => {
      if (!selectedFlow) return;
      setErrorMsg(null);
      return submitSampleFeedback(selectedFlow.flow_id, {
        feedback_id: crypto.randomUUID(),
        expected_version: 0,
        verdict,
        attack_category: verdict === "confirmed_attack" ? attackCategory : null,
        notes: notes.trim() ? notes.trim() : null,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["samplePredictions"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      setSelectedFlow(null);
      setNotes("");
    },
    onError: (err: unknown) => {
      const msg = err instanceof Error ? err.message : "Failed to submit review feedback.";
      setErrorMsg(msg);
    },
  });

  function handleNextPage() {
    if (samplePage?.next_after) {
      setPageHistory((prev) => [...prev, pageAfter || 0]);
      setPageAfter(samplePage.next_after);
    }
  }

  function handlePrevPage() {
    if (pageHistory.length > 0) {
      const prev = [...pageHistory];
      const last = prev.pop();
      setPageHistory(prev);
      setPageAfter(last === 0 ? undefined : last);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header & Description */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-[#1C1C1A]">
            Missed Attack & Normal Flow Audit
          </h1>
          <p className="text-xs text-[#6B6966] max-w-2xl mt-0.5">
            Sample of flows scored below the alert threshold (decision: normal) for
            analyst false-negative auditing and blind review.
          </p>
        </div>

        <button
          onClick={() => refetch()}
          disabled={isRefetching}
          className="flex items-center space-x-1 px-2.5 py-1 text-xs border border-[#E2E2DD] bg-white rounded-sm hover:bg-[#FAFAF8] text-[#1C1C1A] self-start sm:self-auto disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefetching ? "animate-spin" : ""}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Dense Table */}
      <div className="bg-white border border-[#E2E2DD] rounded-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#FAFAF8] border-b border-[#E2E2DD] text-[#6B6966] font-medium">
                <th className="py-2.5 px-3 w-12 text-center">#</th>
                <th className="py-2.5 px-3">Flow ID</th>
                <th className="py-2.5 px-3">Ingest Time</th>
                <th className="py-2.5 px-3">Protocol / Service / State</th>
                <th className="py-2.5 px-3">Score / Threshold</th>
                <th className="py-2.5 px-3">Decision</th>
                <th className="py-2.5 px-3">Audit Review</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E2E2DD]">
              {isLoading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#6B6966]">
                    Loading flow sample...
                  </td>
                </tr>
              ) : flows.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-[#6B6966]">
                    No normal flows currently in sample queue.
                  </td>
                </tr>
              ) : (
                flows.map((flow) => {
                  const feat = flow.features || {};
                  return (
                    <tr
                      key={flow.flow_id}
                      className="hover:bg-[#FAFAF8]/50 transition-colors"
                    >
                      <td className="py-2.5 px-3 text-center text-[#6B6966] font-mono text-[11px] tabular-nums">
                        {flow.sequence}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-[11px] text-[#1C1C1A]">
                        {flow.flow_id.slice(0, 8)}...{flow.flow_id.slice(-4)}
                      </td>

                      <td className="py-2.5 px-3 text-[#6B6966] tabular-nums whitespace-nowrap text-[11px]">
                        {flow.ingest_time.replace("T", " ").replace("Z", "")}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-[11px] text-[#1C1C1A]">
                        {String(feat.proto ?? "—")} / {String(feat.service ?? "—")} / {String(feat.state ?? "—")}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap font-mono text-[11px] tabular-nums">
                        <span className="text-[#047857]">{flow.score.toFixed(4)}</span>
                        <span className="text-[#6B6966]"> / </span>
                        <span className="text-[#6B6966]">{flow.threshold.toFixed(4)}</span>
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded-xs text-[10px] font-semibold uppercase bg-[#ECFDF5] text-[#047857] border border-[#6EE7B7]">
                          normal
                        </span>
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        {flow.review ? (
                          <span className="text-[#B91C1C] text-[11px] font-semibold flex items-center space-x-1">
                            <AlertTriangle className="w-3 h-3" />
                            <span>
                              {flow.review.verdict.replace("_", " ").toUpperCase()}
                            </span>
                          </span>
                        ) : (
                          <span className="text-[#6B6966] text-[11px]">
                            Not Reviewed
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <button
                          onClick={() => {
                            setSelectedFlow(flow);
                            setVerdict(flow.review?.verdict || "false_positive");
                            setNotes(flow.review?.notes || "");
                          }}
                          className="inline-flex items-center space-x-1 text-xs font-medium text-[#1C1C1A] hover:underline"
                        >
                          <span>Inspect</span>
                          <ArrowRight className="w-3 h-3" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination footer */}
        <div className="border-t border-[#E2E2DD] px-3 py-2 flex items-center justify-between text-xs text-[#6B6966]">
          <div>Showing {flows.length} flows</div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrevPage}
              disabled={pageHistory.length === 0}
              className="px-2.5 py-1 border border-[#E2E2DD] rounded-sm hover:bg-[#FAFAF8] disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={handleNextPage}
              disabled={!samplePage?.next_after}
              className="px-2.5 py-1 border border-[#E2E2DD] rounded-sm hover:bg-[#FAFAF8] disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Review Modal for Selected Flow */}
      {selectedFlow && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-xs p-4">
          <div className="bg-white border border-[#E2E2DD] rounded-sm shadow-md max-w-3xl w-full max-h-[90vh] flex flex-col">
            <div className="p-4 border-b border-[#E2E2DD] flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-[#1C1C1A]">
                  Audit Review: Flow {selectedFlow.flow_id}
                </h3>
                <p className="text-[11px] text-[#6B6966]">
                  Score: {selectedFlow.score.toFixed(6)} | Threshold: {selectedFlow.threshold.toFixed(6)}
                </p>
              </div>
              <button
                onClick={() => setSelectedFlow(null)}
                className="text-[#6B6966] hover:text-[#1C1C1A] p-1 rounded-sm"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 overflow-y-auto space-y-5 flex-1">
              {errorMsg && (
                <div className="bg-[#FEF2F2] border border-[#FCA5A5] p-2.5 rounded-sm text-xs text-[#B91C1C]">
                  {errorMsg}
                </div>
              )}

              {/* Flow Features Grid */}
              <div className="space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-[#1C1C1A]">
                  Raw Flow Attributes (42 Predictor Features)
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 text-xs font-mono bg-[#FAFAF8] border border-[#E2E2DD] p-3 rounded-sm">
                  {Object.entries(selectedFlow.features || {}).map(([key, val]) => (
                    <div key={key} className="space-y-0.5">
                      <div className="text-[10px] text-[#6B6966]">{key}</div>
                      <div className="text-[#1C1C1A] font-medium truncate">
                        {String(val)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Analyst Verdict Form */}
              <div className="space-y-3 border-t border-[#E2E2DD] pt-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-[#1C1C1A]">
                  Submit Missed Attack / Benign Audit Verdict
                </h4>

                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => setVerdict("false_positive")}
                    className={`px-3 py-2 text-xs font-medium rounded-sm border ${
                      verdict === "false_positive"
                        ? "bg-[#FEF2F2] border-[#B91C1C] text-[#B91C1C] font-semibold"
                        : "bg-[#FAFAF8] border-[#E2E2DD] text-[#6B6966]"
                    }`}
                  >
                    Missed Attack (FN)
                  </button>

                  <button
                    type="button"
                    onClick={() => setVerdict("confirmed_attack")}
                    className={`px-3 py-2 text-xs font-medium rounded-sm border ${
                      verdict === "confirmed_attack"
                        ? "bg-[#FEF2F2] border-[#B91C1C] text-[#B91C1C] font-semibold"
                        : "bg-[#FAFAF8] border-[#E2E2DD] text-[#6B6966]"
                    }`}
                  >
                    Malicious Intent
                  </button>

                  <button
                    type="button"
                    onClick={() => setVerdict("pending")}
                    className={`px-3 py-2 text-xs font-medium rounded-sm border ${
                      verdict === "pending"
                        ? "bg-[#FFFBEB] border-[#B45309] text-[#B45309] font-semibold"
                        : "bg-[#FAFAF8] border-[#E2E2DD] text-[#6B6966]"
                    }`}
                  >
                    Needs Triage
                  </button>
                </div>

                {verdict === "confirmed_attack" && (
                  <div className="space-y-1">
                    <label className="text-[11px] font-medium text-[#6B6966]">
                      Attack Category:
                    </label>
                    <select
                      value={attackCategory}
                      onChange={(e) => setAttackCategory(e.target.value)}
                      className="w-full bg-[#FAFAF8] border border-[#E2E2DD] rounded-sm px-2.5 py-1.5 text-xs text-[#1C1C1A] focus:outline-hidden"
                    >
                      {[
                        "Exploits",
                        "Generic",
                        "Fuzzers",
                        "Reconnaissance",
                        "DoS",
                        "Backdoor",
                        "Analysis",
                        "Worms",
                        "Shellcode",
                      ].map((cat) => (
                        <option key={cat} value={cat}>
                          {cat}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                <div className="space-y-1">
                  <label className="text-[11px] font-medium text-[#6B6966]">
                    Analyst Notes:
                  </label>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Rationale for missed attack verdict or auditing comments..."
                    rows={3}
                    className="w-full bg-[#FAFAF8] border border-[#E2E2DD] rounded-sm p-2 text-xs text-[#1C1C1A] focus:outline-hidden resize-none"
                  />
                </div>
              </div>
            </div>

            <div className="p-4 border-t border-[#E2E2DD] flex justify-end space-x-2">
              <button
                onClick={() => setSelectedFlow(null)}
                className="px-3 py-1.5 border border-[#E2E2DD] rounded-sm text-xs text-[#6B6966] hover:bg-[#FAFAF8]"
              >
                Cancel
              </button>
              <button
                onClick={() => feedbackMutation.mutate()}
                disabled={feedbackMutation.isPending}
                className="px-4 py-1.5 bg-[#1C1C1A] text-white text-xs font-medium rounded-sm hover:bg-[#333] disabled:opacity-50"
              >
                {feedbackMutation.isPending ? "Recording..." : "Save Audit Verdict"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchModelSummary, fetchReplaySummary } from "@/lib/api";
import {
  Hash,
  Database,
  BarChart2,
} from "lucide-react";

export default function ModelPage() {
  const { data: model, isLoading: modelLoading } = useQuery({
    queryKey: ["modelSummary"],
    queryFn: fetchModelSummary,
  });

  const { data: replay, isLoading: replayLoading } = useQuery({
    queryKey: ["replaySummary"],
    queryFn: fetchReplaySummary,
  });

  const metrics = model?.selection_metrics || {};

  return (
    <div className="space-y-6">
      {/* Page Title & Status */}
      <div>
        <h1 className="text-lg font-semibold tracking-tight text-[#F1F3F6]">
          Model Architecture, Hashes & Empirical Performance
        </h1>
        <p className="text-xs text-[#939AA6] mt-0.5">
          Release bundle verification, cryptographic provenance, and held-out partition evaluation.
        </p>
      </div>

      {/* Overview Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
          <div className="text-[11px] font-medium text-[#939AA6]">Model Architecture</div>
          <div className="text-sm font-semibold text-[#F1F3F6]">
            {modelLoading ? "—" : model?.algorithm || "LightGBM Classifier"}
          </div>
          <div className="text-[10px] text-[#939AA6]">UNSW-NB15 Validated Model</div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
          <div className="text-[11px] font-medium text-[#939AA6]">Bundle Version</div>
          <div className="text-sm font-semibold font-mono text-[#F1F3F6]">
            {modelLoading ? "—" : model?.bundle_version}
          </div>
          <div className="text-[10px] text-[#939AA6]">Cryptographically sealed</div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
          <div className="text-[11px] font-medium text-[#939AA6]">Decision Threshold</div>
          <div className="text-sm font-semibold font-mono text-[#F87171] tabular-nums">
            {modelLoading ? "—" : model?.decision_threshold.toFixed(6)}
          </div>
          <div className="text-[10px] text-[#939AA6]">Single operational decision boundary</div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
          <div className="text-[11px] font-medium text-[#939AA6]">Selection Accuracy</div>
          <div className="text-sm font-semibold font-mono text-[#34D399] tabular-nums">
            {modelLoading
              ? "—"
              : typeof metrics.accuracy === "number"
              ? `${(metrics.accuracy * 100).toFixed(2)}%`
              : "95.51%"}
          </div>
          <div className="text-[10px] text-[#939AA6]">Held-out selection partition</div>
        </div>
      </div>

      {/* Grid: Selection Metrics vs Replay Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Frozen Selection Partition Metrics */}
        <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
          <div className="flex items-center space-x-2">
            <BarChart2 className="w-4 h-4 text-[#F1F3F6]" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
              Selection Partition Metrics (Validation Estimate)
            </h2>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
              <div className="text-[10px] text-[#939AA6]">Recall / Detection Rate</div>
              <div className="text-base font-semibold font-mono tabular-nums text-[#F1F3F6] mt-1">
                {typeof metrics.recall === "number"
                  ? `${(metrics.recall * 100).toFixed(2)}%`
                  : "95.25%"}
              </div>
            </div>

            <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
              <div className="text-[10px] text-[#939AA6]">False Positive Rate (FPR)</div>
              <div className="text-base font-semibold font-mono tabular-nums text-[#F1F3F6] mt-1">
                {typeof metrics.fpr === "number"
                  ? `${(metrics.fpr * 100).toFixed(2)}%`
                  : "3.98%"}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-[11px] font-medium text-[#939AA6]">
              Validation Confusion Matrix (25,031 flows):
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-center border border-[#282C35] text-xs font-mono">
                <thead>
                  <tr className="bg-[#181B21] text-[#939AA6]">
                    <th className="p-2 border-r border-[#282C35]">Actual \ Pred</th>
                    <th className="p-2 border-r border-[#282C35]">Pred Normal (0)</th>
                    <th className="p-2">Pred Alert (1)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#282C35]">
                  <tr>
                    <td className="p-2 font-medium bg-[#181B21] border-r border-[#282C35] text-left text-[#F1F3F6]">
                      Actual Normal (0)
                    </td>
                    <td className="p-2 border-r border-[#282C35] text-[#34D399] font-semibold">
                      7,867 (TN)
                    </td>
                    <td className="p-2 text-[#FBBF24] font-semibold">
                      326 (FP)
                    </td>
                  </tr>
                  <tr>
                    <td className="p-2 font-medium bg-[#181B21] border-r border-[#282C35] text-left text-[#F1F3F6]">
                      Actual Attack (1)
                    </td>
                    <td className="p-2 border-r border-[#282C35] text-[#F87171] font-semibold">
                      799 (FN)
                    </td>
                    <td className="p-2 text-[#34D399] font-semibold">
                      16,039 (TP)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right: Empirical Replay Performance */}
        <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-[#F1F3F6]" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
              Live Replay Performance (replay_truth)
            </h2>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
              <div className="text-[10px] text-[#939AA6]">Replayed Flows Count</div>
              <div className="text-base font-semibold font-mono tabular-nums text-[#F1F3F6] mt-1">
                {replayLoading ? "—" : replay?.total_replayed.toLocaleString() || "0"}
              </div>
            </div>

            <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
              <div className="text-[10px] text-[#939AA6]">False Alerts / 1k Normal</div>
              <div className="text-base font-semibold font-mono tabular-nums text-[#FBBF24] mt-1">
                {replayLoading
                  ? "—"
                  : replay?.false_alerts_per_1000_normal !== undefined
                  ? `${replay.false_alerts_per_1000_normal.toFixed(1)} / 1,000`
                  : "0.0 / 1,000"}
              </div>
            </div>
          </div>

          {replay && replay.total_replayed > 0 ? (
            <div className="space-y-2">
              <div className="text-[11px] font-medium text-[#939AA6]">
                Empirical Replay Matrix ({replay.total_replayed} flows evaluated):
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-center border border-[#282C35] text-xs font-mono">
                  <thead>
                    <tr className="bg-[#181B21] text-[#939AA6]">
                      <th className="p-2 border-r border-[#282C35]">Actual \ Pred</th>
                      <th className="p-2 border-r border-[#282C35]">Pred Normal</th>
                      <th className="p-2">Pred Alert</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#282C35]">
                    <tr>
                      <td className="p-2 font-medium bg-[#181B21] border-r border-[#282C35] text-left text-[#F1F3F6]">
                        Actual Normal
                      </td>
                      <td className="p-2 border-r border-[#282C35] text-[#34D399] font-semibold">
                        {replay.true_negatives}
                      </td>
                      <td className="p-2 text-[#FBBF24] font-semibold">
                        {replay.false_positives}
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2 font-medium bg-[#181B21] border-r border-[#282C35] text-left text-[#F1F3F6]">
                        Actual Attack
                      </td>
                      <td className="p-2 border-r border-[#282C35] text-[#F87171] font-semibold">
                        {replay.false_negatives}
                      </td>
                      <td className="p-2 text-[#34D399] font-semibold">
                        {replay.true_positives}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[#939AA6]">
              Run the dataset replay CLI (<code className="font-mono bg-[#181B21] px-1 py-0.5 border border-[#282C35] rounded-xs text-[#F1F3F6]">python -m nexus.replay</code>) to generate live empirical evaluation statistics.
            </div>
          )}
        </div>
      </div>

      {/* Cryptographic Provenance & SHA-256 Hashes */}
      <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
        <div className="flex items-center space-x-2">
          <Hash className="w-4 h-4 text-[#F1F3F6]" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
            Cryptographic SHA-256 Hashes & Bundle Provenance
          </h2>
        </div>

        <div className="divide-y divide-[#282C35] border border-[#282C35] rounded-sm text-xs font-mono">
          <div className="p-3 bg-[#181B21] font-semibold text-[#F1F3F6]">
            Release Artifacts ({model?.bundle_version || "v1.0.0"}):
          </div>

          {model?.bundle_hashes &&
            Object.entries(model.bundle_hashes).map(([file, hash]) => (
              <div key={file} className="p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <span className="text-[#F1F3F6] font-medium">{file}</span>
                <span className="text-[11px] text-[#939AA6] break-all">{hash}</span>
              </div>
            ))}

          <div className="p-3 bg-[#181B21] font-semibold text-[#F1F3F6]">
            Source Code & Training Partition Integrity:
          </div>

          {model?.source_hashes &&
            Object.entries(model.source_hashes).map(([file, hash]) => (
              <div key={file} className="p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <span className="text-[#F1F3F6] font-medium">{file}</span>
                <span className="text-[11px] text-[#939AA6] break-all">{hash}</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

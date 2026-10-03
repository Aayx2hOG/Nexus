"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  fetchPresetData,
  submitBatchPredictions,
  fetchModelSummary,
  liveProbe,
} from "@/lib/api";
import { parseCSVText, chunkFlows, ParsedFlow } from "@/lib/csv";
import { LiveProbeResult } from "@/lib/types";
import {
  UploadCloud,
  FileSpreadsheet,
  AlertTriangle,
  Play,
  ExternalLink,
  ChevronRight,
  BarChart2,
} from "lucide-react";

interface FlowResultItem {
  flow_id: string;
  score: number;
  threshold: number;
  decision: "alert" | "normal";
  alert_id?: string | null;
  features: Record<string, string | number>;
  ground_truth?: {
    label?: number;
    attack_cat?: string;
  };
}

export default function TrafficPage() {
  const [activeTab, setActiveTab] = useState<"csv" | "probe">("csv");

  // Query Model threshold
  const { data: model } = useQuery({
    queryKey: ["modelSummary"],
    queryFn: fetchModelSummary,
  });

  // --- CSV Ingestion State ---
  const [csvRawText, setCsvRawText] = useState("");
  const [parsedFlows, setParsedFlows] = useState<ParsedFlow[]>([]);
  const [parseErrors, setParseErrors] = useState<string[]>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState<{ current: number; total: number } | null>(null);
  const [flowResults, setFlowResults] = useState<FlowResultItem[]>([]);
  const [filterVerdict, setFilterVerdict] = useState<"all" | "alert" | "normal">("all");

  // --- Live Website Probe State ---
  const [targetUrl, setTargetUrl] = useState("https://");
  const [probeMethod, setProbeMethod] = useState<"GET" | "HEAD" | "POST" | "OPTIONS">("GET");
  const [probeTimeout, setProbeTimeout] = useState(10);
  const [followRedirects, setFollowRedirects] = useState(true);
  const [isProbing, setIsProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<LiveProbeResult | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [probeHistory, setProbeHistory] = useState<LiveProbeResult[]>([]);

  // --- CSV Ingestion Handlers ---
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      setCsvRawText(text);
      processParsedText(text, file.name);
    };
    reader.readAsText(file);
  };

  const processParsedText = (text: string, name?: string) => {
    setFlowResults([]);
    const res = parseCSVText(text);
    setParsedFlows(res.flows);
    setParseErrors(res.errors);
    setHasGroundTruth(res.hasGroundTruth);
    if (name) setFileName(name);
  };

  const handleLoadPreset = async (presetId: string, label: string) => {
    try {
      setIsProcessing(true);
      const data = await fetchPresetData(presetId);
      setCsvRawText(data.csv_text);
      processParsedText(data.csv_text, `${label}.csv`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load preset.";
      setParseErrors([message]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunDetection = async () => {
    if (parsedFlows.length === 0) return;
    setIsProcessing(true);
    setFlowResults([]);
    const batches = chunkFlows(parsedFlows, 100);
    const accumulated: FlowResultItem[] = [];

    try {
      for (let i = 0; i < batches.length; i++) {
        setProcessProgress({ current: i + 1, total: batches.length });
        const batchPayload = batches[i].map((f) => ({
          flow_id: f.flow_id,
          event_time: f.event_time,
          features: f.features,
        }));

        const res = await submitBatchPredictions(batchPayload);

        // Map back with ground truth
        res.predictions.forEach((pred, pIdx) => {
          const originalFlow = batches[i][pIdx];
          accumulated.push({
            flow_id: pred.flow_id,
            score: pred.score,
            threshold: pred.threshold,
            decision: pred.decision,
            alert_id: pred.alert_id,
            features: originalFlow.features,
            ground_truth: originalFlow.ground_truth,
          });
        });

        setFlowResults([...accumulated]);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Prediction request failed.";
      setParseErrors([message]);
    } finally {
      setIsProcessing(false);
      setProcessProgress(null);
    }
  };

  // --- Live Probe Handler ---
  const handleRunLiveProbe = async () => {
    if (!targetUrl.trim() || targetUrl.trim() === "https://" || targetUrl.trim() === "http://") return;
    setIsProbing(true);
    setProbeError(null);
    setProbeResult(null);

    try {
      const res = await liveProbe({
        target_url: targetUrl.trim(),
        method: probeMethod,
        timeout_seconds: probeTimeout,
        follow_redirects: followRedirects,
      });
      setProbeResult(res);
      setProbeHistory(prev => [res, ...prev].slice(0, 20)); // Keep last 20
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Live probe failed.";
      setProbeError(message);
    } finally {
      setIsProbing(false);
    }
  };

  // --- Derived Statistics for CSV Results ---
  const totalAnalyzed = flowResults.length;
  const alertCount = flowResults.filter((r) => r.decision === "alert").length;
  const normalCount = flowResults.filter((r) => r.decision === "normal").length;
  const attackRatio = totalAnalyzed > 0 ? ((alertCount / totalAnalyzed) * 100).toFixed(1) : "0.0";

  // Confusion matrix if ground truth is available
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  if (hasGroundTruth && totalAnalyzed > 0) {
    flowResults.forEach((r) => {
      const actualAttack = r.ground_truth?.label === 1;
      const predAttack = r.decision === "alert";
      if (actualAttack && predAttack) tp++;
      else if (!actualAttack && predAttack) fp++;
      else if (!actualAttack && !predAttack) tn++;
      else if (actualAttack && !predAttack) fn++;
    });
  }
  const evalAccuracy = totalAnalyzed > 0 ? (((tp + tn) / totalAnalyzed) * 100).toFixed(2) : "—";
  const evalRecall = tp + fn > 0 ? ((tp / (tp + fn)) * 100).toFixed(2) : "—";
  const evalPrecision = tp + fp > 0 ? ((tp / (tp + fp)) * 100).toFixed(2) : "—";

  const filteredResults = flowResults.filter((r) => {
    if (filterVerdict === "alert") return r.decision === "alert";
    if (filterVerdict === "normal") return r.decision === "normal";
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Top Banner & Title */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-[#F1F3F6] flex items-center space-x-2">
            <span>Traffic Ingestion & Website Probe</span>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 bg-[#181B21] border border-[#282C35] text-[#34D399] rounded-sm">
              Live Engine
            </span>
          </h1>
          <p className="text-xs text-[#939AA6] mt-0.5">
            Upload flow CSVs for model triage, or probe any real website to analyze its network behavior with ML.
          </p>
        </div>

        {/* Tab Toggle Switcher */}
        <div className="inline-flex p-1 bg-[#13151A] border border-[#282C35] rounded-sm text-xs">
          <button
            onClick={() => setActiveTab("csv")}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-xs transition-colors font-medium ${
              activeTab === "csv"
                ? "bg-[#1D2027] text-[#F1F3F6] border border-[#3B414E] shadow-xs"
                : "text-[#939AA6] hover:text-[#F1F3F6]"
            }`}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            <span>CSV Flow Ingestion</span>
          </button>
          <button
            onClick={() => setActiveTab("probe")}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-xs transition-colors font-medium ${
              activeTab === "probe"
                ? "bg-[#1D2027] text-[#F1F3F6] border border-[#3B414E] shadow-xs"
                : "text-[#939AA6] hover:text-[#F1F3F6]"
            }`}
          >
            <span>Live Website Probe</span>
          </button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* TAB 1: CSV Flow Ingestion & Attack Detector                    */}
      {/* ============================================================== */}
      {activeTab === "csv" && (
        <div className="space-y-6">
          {/* Ingestion Control Box */}
          <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6] flex items-center space-x-2">
                  <UploadCloud className="w-4 h-4 text-[#939AA6]" />
                  <span>Upload Network Flow Dataset (UNSW-NB15 Format)</span>
                </h2>
                <p className="text-[11px] text-[#939AA6] mt-0.5">
                  Accepts standard 42-feature CSVs. Ground truth labels (if present) are automatically isolated for empirical evaluation.
                </p>
              </div>

              {/* Sample Preset Buttons */}
              <div className="flex items-center space-x-2 flex-wrap gap-y-2">
                <span className="text-[10px] uppercase font-mono text-[#939AA6]">Load Presets:</span>
                <button
                  type="button"
                  onClick={() => handleLoadPreset("normal_50", "50_normal_flows")}
                  disabled={isProcessing}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#181B21] border border-[#282C35] text-[#34D399] hover:bg-[#1D2027] hover:border-[#34D399]/40 rounded-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  50 Normal Flows
                </button>
                <button
                  type="button"
                  onClick={() => handleLoadPreset("attack_50", "50_attack_flows")}
                  disabled={isProcessing}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#181B21] border border-[#282C35] text-[#F87171] hover:bg-[#1D2027] hover:border-[#F87171]/40 rounded-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  50 Attack Flows
                </button>
                <button
                  type="button"
                  onClick={() => handleLoadPreset("mixed_100", "100_mixed_flows")}
                  disabled={isProcessing}
                  className="px-2.5 py-1 text-[11px] font-mono bg-[#181B21] border border-[#282C35] text-[#FBBF24] hover:bg-[#1D2027] hover:border-[#FBBF24]/40 rounded-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  100 Mixed Flows
                </button>
              </div>
            </div>

            {/* Drag & Drop File Zone */}
            <div className="border border-dashed border-[#282C35] hover:border-[#3B414E] bg-[#181B21]/50 rounded-sm p-6 text-center transition-colors">
              <input
                type="file"
                id="csvFileInput"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
              />
              <label
                htmlFor="csvFileInput"
                className="cursor-pointer flex flex-col items-center justify-center space-y-2"
              >
                <FileSpreadsheet className="w-8 h-8 text-[#939AA6]" />
                <div className="text-xs font-medium text-[#F1F3F6]">
                  {fileName ? (
                    <span className="font-mono text-[#34D399]">{fileName}</span>
                  ) : (
                    <span>Click to browse or drag and drop your CSV file here</span>
                  )}
                </div>
                <div className="text-[10px] text-[#939AA6]">
                  Supports standard CSV containing the 42 flow features (dur, proto, service, sbytes, dbytes, sttl, etc.)
                </div>
              </label>
            </div>

            {/* Parse Warnings / Errors */}
            {parseErrors.length > 0 && (
              <div className="p-3 bg-[#7F1D1D]/20 border border-[#F87171]/40 rounded-sm space-y-1">
                <div className="text-xs font-semibold text-[#F87171] flex items-center space-x-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Validation Warnings ({parseErrors.length})</span>
                </div>
                <ul className="text-[11px] font-mono text-[#FCA5A5] list-disc list-inside space-y-0.5">
                  {parseErrors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Bar */}
            {parsedFlows.length > 0 && (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 border-t border-[#282C35]">
                <div className="text-xs text-[#939AA6]">
                  Parsed <span className="font-mono font-semibold text-[#F1F3F6]">{parsedFlows.length}</span> valid flow records.{" "}
                  {hasGroundTruth && (
                    <span className="text-[#34D399]">Ground-truth labels detected (evaluation scorecard enabled).</span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleRunDetection}
                  disabled={isProcessing}
                  className="flex items-center justify-center space-x-2 px-4 py-2 text-xs font-semibold bg-[#F1F3F6] text-[#0A0B0D] hover:bg-white rounded-sm transition-colors cursor-pointer disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isProcessing ? "Processing Batches..." : `Run Detection on ${parsedFlows.length} Flows`}</span>
                </button>
              </div>
            )}

            {/* Progress Bar */}
            {processProgress && (
              <div className="space-y-1.5 pt-2">
                <div className="flex justify-between text-[11px] font-mono text-[#939AA6]">
                  <span>Processing batch {processProgress.current} of {processProgress.total}...</span>
                  <span>{Math.round((processProgress.current / processProgress.total) * 100)}%</span>
                </div>
                <div className="w-full bg-[#181B21] h-1.5 rounded-full overflow-hidden border border-[#282C35]">
                  <div
                    className="bg-[#34D399] h-full transition-all duration-300"
                    style={{ width: `${(processProgress.current / processProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Results Analytics & Scorecard */}
          {flowResults.length > 0 && (
            <div className="space-y-6">
              {/* Summary KPIs */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
                  <div className="text-[11px] font-medium text-[#939AA6]">Total Flows Processed</div>
                  <div className="text-lg font-bold font-mono text-[#F1F3F6] tabular-nums">
                    {totalAnalyzed.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-[#939AA6]">Model: LightGBM ({model?.bundle_version || "v2.0.0"})</div>
                </div>

                <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
                  <div className="text-[11px] font-medium text-[#939AA6]">Attacks Flagged (Alerts)</div>
                  <div className="text-lg font-bold font-mono text-[#F87171] tabular-nums">
                    {alertCount.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-[#F87171]/80">Decision score &ge; {model?.decision_threshold.toFixed(4) || "0.7675"}</div>
                </div>

                <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
                  <div className="text-[11px] font-medium text-[#939AA6]">Normal Flows</div>
                  <div className="text-lg font-bold font-mono text-[#34D399] tabular-nums">
                    {normalCount.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-[#34D399]/80">Below operational decision boundary</div>
                </div>

                <div className="bg-[#13151A] border border-[#282C35] p-4 rounded-sm space-y-1">
                  <div className="text-[11px] font-medium text-[#939AA6]">Attack Ratio</div>
                  <div className="text-lg font-bold font-mono text-[#FBBF24] tabular-nums">
                    {attackRatio}%
                  </div>
                  <div className="text-[10px] text-[#939AA6]">Proportion of flagged suspicious flows</div>
                </div>
              </div>

              {/* Optional Ground Truth Scorecard */}
              {hasGroundTruth && (
                <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
                  <div className="flex items-center space-x-2">
                    <BarChart2 className="w-4 h-4 text-[#34D399]" />
                    <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                      Empirical Evaluation Scorecard (Against Ground Truth)
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
                    <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
                      <div className="text-[10px] text-[#939AA6]">Empirical Accuracy</div>
                      <div className="text-base font-semibold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                        {evalAccuracy}%
                      </div>
                    </div>
                    <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
                      <div className="text-[10px] text-[#939AA6]">Detection Recall</div>
                      <div className="text-base font-semibold font-mono text-[#34D399] mt-1 tabular-nums">
                        {evalRecall}%
                      </div>
                    </div>
                    <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
                      <div className="text-[10px] text-[#939AA6]">Attack Precision</div>
                      <div className="text-base font-semibold font-mono text-[#34D399] mt-1 tabular-nums">
                        {evalPrecision}%
                      </div>
                    </div>
                    <div className="p-3 bg-[#181B21] border border-[#282C35] rounded-sm">
                      <div className="text-[10px] text-[#939AA6]">Confusion Matrix</div>
                      <div className="text-xs font-mono text-[#939AA6] mt-1">
                        TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Interactive Results Table */}
              <div className="bg-[#13151A] border border-[#282C35] rounded-sm overflow-hidden">
                <div className="p-4 border-b border-[#282C35] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-[#F1F3F6] uppercase tracking-wider">
                      Processed Flow Records ({filteredResults.length})
                    </span>
                  </div>

                  {/* Filter Verdict buttons */}
                  <div className="flex items-center space-x-1 text-xs">
                    <button
                      onClick={() => setFilterVerdict("all")}
                      className={`px-2.5 py-1 rounded-xs transition-colors font-medium ${
                        filterVerdict === "all"
                          ? "bg-[#1D2027] text-[#F1F3F6] border border-[#3B414E]"
                          : "text-[#939AA6] hover:text-[#F1F3F6]"
                      }`}
                    >
                      All ({flowResults.length})
                    </button>
                    <button
                      onClick={() => setFilterVerdict("alert")}
                      className={`px-2.5 py-1 rounded-xs transition-colors font-medium ${
                        filterVerdict === "alert"
                          ? "bg-[#7F1D1D]/30 text-[#F87171] border border-[#F87171]/40"
                          : "text-[#939AA6] hover:text-[#F87171]"
                      }`}
                    >
                      Alerts Only ({alertCount})
                    </button>
                    <button
                      onClick={() => setFilterVerdict("normal")}
                      className={`px-2.5 py-1 rounded-xs transition-colors font-medium ${
                        filterVerdict === "normal"
                          ? "bg-[#064E3B]/30 text-[#34D399] border border-[#34D399]/40"
                          : "text-[#939AA6] hover:text-[#34D399]"
                      }`}
                    >
                      Normal Only ({normalCount})
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-[#181B21] text-[#939AA6] uppercase font-mono text-[10px] border-b border-[#282C35]">
                      <tr>
                        <th className="py-2.5 px-3">#</th>
                        <th className="py-2.5 px-3">Flow ID</th>
                        <th className="py-2.5 px-3">Protocol / Srv</th>
                        <th className="py-2.5 px-3">Attack Probability</th>
                        <th className="py-2.5 px-3">Decision</th>
                        {hasGroundTruth && <th className="py-2.5 px-3">Ground Truth</th>}
                        <th className="py-2.5 px-3 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#282C35] font-mono">
                      {filteredResults.slice(0, 100).map((row, idx) => {
                        const isAlert = row.decision === "alert";
                        const pct = (row.score * 100).toFixed(1);
                        return (
                          <tr key={row.flow_id} className="hover:bg-[#181B21]/50 transition-colors">
                            <td className="py-2.5 px-3 text-[#939AA6]">{idx + 1}</td>
                            <td className="py-2.5 px-3 text-[#F1F3F6] font-medium">
                              {row.flow_id.slice(0, 8)}...
                            </td>
                            <td className="py-2.5 px-3 text-[#939AA6]">
                              {String(row.features.proto).toUpperCase()} / {String(row.features.service)}
                            </td>
                            <td className="py-2.5 px-3">
                              <div className="flex items-center space-x-2">
                                <div className="w-16 bg-[#181B21] h-1.5 rounded-full overflow-hidden border border-[#282C35]">
                                  <div
                                    className={`h-full ${isAlert ? "bg-[#F87171]" : "bg-[#34D399]"}`}
                                    style={{ width: `${Math.min(100, Math.max(0, row.score * 100))}%` }}
                                  />
                                </div>
                                <span className={isAlert ? "text-[#F87171] font-semibold" : "text-[#34D399]"}>
                                  {pct}%
                                </span>
                              </div>
                            </td>
                            <td className="py-2.5 px-3">
                              {isAlert ? (
                                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-[#7F1D1D]/30 border border-[#F87171]/40 text-[#F87171] rounded-xs">
                                  ALERT
                                </span>
                              ) : (
                                <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-[#064E3B]/30 border border-[#34D399]/40 text-[#34D399] rounded-xs">
                                  NORMAL
                                </span>
                              )}
                            </td>
                            {hasGroundTruth && (
                              <td className="py-2.5 px-3 text-[#939AA6]">
                                {row.ground_truth?.label === 1 ? (
                                  <span className="text-[#FBBF24]">Attack ({row.ground_truth.attack_cat || "1"})</span>
                                ) : (
                                  <span className="text-[#939AA6]">Normal (0)</span>
                                )}
                              </td>
                            )}
                            <td className="py-2.5 px-3 text-right">
                              {row.alert_id ? (
                                <Link
                                  href={`/alerts/${row.alert_id}`}
                                  className="inline-flex items-center space-x-1 text-[#38BDF8] hover:text-[#7DD3FC] hover:underline"
                                >
                                  <span>View Alert</span>
                                  <ExternalLink className="w-3 h-3" />
                                </Link>
                              ) : (
                                <span className="text-[#555C68]">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 2: Live Website Probe (REAL HTTP CONNECTION)               */}
      {/* ============================================================== */}
      {activeTab === "probe" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Probe Configuration */}
          <div className="lg:col-span-5 space-y-5">
            <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                  Live Website Probe
                </h2>
                <span className="text-[10px] font-mono text-[#939AA6]">
                  HTTP Diagnostics
                </span>
              </div>
              <p className="text-[11px] text-[#939AA6]">
                Makes an actual HTTP connection to the target. Measures real DNS resolution, TCP handshake, 
                TLS negotiation, response timing, and payload sizes — then feeds those real metrics into the ML model.
              </p>

              {/* URL Input */}
              <div>
                <label className="block text-[11px] font-medium text-[#939AA6] mb-1">
                  Target URL
                </label>
                <input
                  type="text"
                  value={targetUrl}
                  onChange={(e) => setTargetUrl(e.target.value)}
                  placeholder="https://example.com"
                  onKeyDown={(e) => e.key === "Enter" && handleRunLiveProbe()}
                  className="w-full bg-[#181B21] border border-[#282C35] focus:border-[#38BDF8] text-[#F1F3F6] text-xs font-mono px-3 py-2.5 rounded-xs outline-hidden transition-colors"
                />
              </div>

              {/* Method & Timeout Row */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] font-medium text-[#939AA6] mb-1">
                    HTTP Method
                  </label>
                  <select
                    value={probeMethod}
                    onChange={(e) => setProbeMethod(e.target.value as "GET" | "HEAD" | "POST" | "OPTIONS")}
                    className="w-full bg-[#181B21] border border-[#282C35] text-[#F1F3F6] text-xs font-mono px-3 py-2 rounded-xs outline-hidden"
                  >
                    <option value="GET">GET</option>
                    <option value="HEAD">HEAD</option>
                    <option value="POST">POST</option>
                    <option value="OPTIONS">OPTIONS</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[11px] font-medium text-[#939AA6] mb-1">
                    Timeout (seconds)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={probeTimeout}
                    onChange={(e) => setProbeTimeout(Number(e.target.value))}
                    className="w-full bg-[#181B21] border border-[#282C35] text-[#F1F3F6] text-xs font-mono px-3 py-2 rounded-xs outline-hidden"
                  />
                </div>
              </div>

              {/* Follow Redirects Toggle */}
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="followRedirects"
                  checked={followRedirects}
                  onChange={(e) => setFollowRedirects(e.target.checked)}
                  className="rounded-xs border-[#282C35] bg-[#181B21]"
                />
                <label htmlFor="followRedirects" className="text-[11px] text-[#939AA6]">
                  Follow HTTP redirects (3xx)
                </label>
              </div>

              {/* Run Probe Button */}
              <button
                type="button"
                onClick={handleRunLiveProbe}
                disabled={isProbing || !targetUrl.trim() || targetUrl.trim() === "https://" || targetUrl.trim() === "http://"}
                className="w-full py-2.5 px-4 bg-[#38BDF8] hover:bg-[#7DD3FC] text-[#0A0B0D] text-xs font-bold uppercase tracking-wider rounded-sm transition-colors cursor-pointer flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProbing ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-[#0A0B0D]/30 border-t-[#0A0B0D] rounded-full animate-spin" />
                    <span>Connecting & Probing...</span>
                  </>
                ) : (
                  <span>Probe Target Website</span>
                )}
              </button>

              {probeError && (
                <div className="p-3 bg-[#7F1D1D]/20 border border-[#F87171]/40 rounded-sm text-xs text-[#F87171]">
                  <div className="flex items-center space-x-1.5 mb-1 font-semibold">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>Probe Failed</span>
                  </div>
                  {probeError}
                </div>
              )}
            </div>

            {/* Probe History */}
            {probeHistory.length > 1 && (
              <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#F1F3F6]">
                    Probe History ({probeHistory.length})
                  </span>
                </div>
                <div className="space-y-1.5 max-h-64 overflow-y-auto">
                  {probeHistory.map((h, i) => (
                    <button
                      key={`${h.flow_id}-${i}`}
                      onClick={() => setProbeResult(h)}
                      className={`w-full text-left p-2.5 rounded-xs border transition-colors text-xs ${
                        probeResult?.flow_id === h.flow_id
                          ? "bg-[#1D2027] border-[#3B414E]"
                          : "bg-[#181B21] border-[#282C35] hover:border-[#3B414E]"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[#F1F3F6] truncate max-w-[200px]">
                          {h.target_url.replace(/^https?:\/\//, "")}
                        </span>
                        <span className={`text-[9px] uppercase font-mono font-semibold px-1.5 py-0.5 rounded-xs ${
                          h.target_hit
                            ? "bg-[#7F1D1D]/40 text-[#F87171] border border-[#F87171]/30"
                            : "bg-[#064E3B]/40 text-[#34D399] border border-[#34D399]/30"
                        }`}>
                          {h.target_hit ? "HIT" : "CLEAN"}
                        </span>
                      </div>
                      <div className="flex items-center space-x-3 mt-1 text-[10px] text-[#939AA6]">
                        <span>{h.method}</span>
                        <span>{h.connection.total_ms.toFixed(0)}ms</span>
                        <span>Score: {(h.score * 100).toFixed(1)}%</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Live Probe Results */}
          <div className="lg:col-span-7 space-y-5">
            {probeResult ? (
              <div className="space-y-4">
                {/* BIG VERDICT BANNER */}
                <div
                  className={`p-5 border rounded-sm space-y-3 transition-all ${
                    probeResult.target_hit
                      ? "bg-[#7F1D1D]/20 border-[#F87171]"
                      : "bg-[#064E3B]/20 border-[#34D399]"
                  }`}
                >
                  <div>
                    <div
                      className={`text-sm font-bold tracking-tight uppercase font-mono ${
                        probeResult.target_hit ? "text-[#F87171]" : "text-[#34D399]"
                      }`}
                    >
                      {probeResult.target_hit ? "SUSPICIOUS ACTIVITY DETECTED" : "TARGET SECURE / ALL CLEAR"}
                    </div>
                    <div className="text-xs text-[#939AA6] mt-1">
                      <code className="font-mono text-[#F1F3F6]">{probeResult.method} {probeResult.target_url}</code>
                    </div>
                  </div>

                  {/* Score + Threat Level */}
                  <div className="grid grid-cols-2 gap-3 pt-2 border-t border-[#282C35]/60 text-xs">
                    <div className="p-3 bg-[#13151A]/80 border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Attack Probability</div>
                      <div
                        className={`text-lg font-bold font-mono mt-0.5 tabular-nums ${
                          probeResult.target_hit ? "text-[#F87171]" : "text-[#34D399]"
                        }`}
                      >
                        {(probeResult.score * 100).toFixed(2)}%
                      </div>
                      <div className="text-[9px] text-[#939AA6]">
                        Threshold: {(probeResult.threshold * 100).toFixed(2)}%
                      </div>
                    </div>

                    <div className="p-3 bg-[#13151A]/80 border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Threat Rating</div>
                      <div
                        className={`text-lg font-bold font-mono mt-0.5 ${
                          probeResult.threat_level === "CRITICAL" ? "text-[#F87171]" :
                          probeResult.threat_level === "HIGH" ? "text-[#FB923C]" :
                          probeResult.threat_level === "MEDIUM" ? "text-[#FBBF24]" :
                          "text-[#34D399]"
                        }`}
                      >
                        {probeResult.threat_level}
                      </div>
                      <div className="text-[9px] text-[#939AA6]">Verdict: {probeResult.decision.toUpperCase()}</div>
                    </div>
                  </div>
                </div>

                {/* Real Connection Metrics */}
                <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                      Real Connection Metrics
                    </h3>
                    <span className="text-[10px] font-mono text-[#939AA6]">
                      Raw Telemetry
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">DNS Resolve</div>
                      <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                        {probeResult.connection.dns_resolve_ms.toFixed(1)}ms
                      </div>
                    </div>
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">TCP Connect</div>
                      <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                        {probeResult.connection.tcp_connect_ms.toFixed(1)}ms
                      </div>
                    </div>
                    {probeResult.connection.tls_handshake_ms !== null && (
                      <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                        <div className="text-[10px] text-[#939AA6]">TLS Handshake</div>
                        <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                          {probeResult.connection.tls_handshake_ms.toFixed(1)}ms
                        </div>
                      </div>
                    )}
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Time to First Byte</div>
                      <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                        {probeResult.connection.ttfb_ms.toFixed(1)}ms
                      </div>
                    </div>
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Total Time</div>
                      <div className="text-sm font-bold font-mono text-[#38BDF8] mt-1 tabular-nums">
                        {probeResult.connection.total_ms.toFixed(1)}ms
                      </div>
                    </div>
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Response Size</div>
                      <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1 tabular-nums">
                        {(probeResult.connection.response_body_bytes / 1024).toFixed(1)} KB
                      </div>
                    </div>
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">HTTP Status</div>
                      <div className={`text-sm font-bold font-mono mt-1 tabular-nums ${
                        probeResult.connection.status_code >= 400 ? "text-[#F87171]" :
                        probeResult.connection.status_code >= 300 ? "text-[#FBBF24]" :
                        "text-[#34D399]"
                      }`}>
                        {probeResult.connection.status_code}
                      </div>
                    </div>
                    <div className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs">
                      <div className="text-[10px] text-[#939AA6]">Protocol</div>
                      <div className="text-sm font-bold font-mono text-[#F1F3F6] mt-1">
                        {probeResult.connection.http_version} {probeResult.connection.is_https ? "(HTTPS)" : ""}
                      </div>
                    </div>
                  </div>

                  {/* Server Info */}
                  <div className="flex items-center space-x-4 text-[11px] text-[#939AA6] pt-2 border-t border-[#282C35]">
                    {probeResult.connection.server_header && (
                      <span>Server: <code className="text-[#F1F3F6] font-mono">{probeResult.connection.server_header}</code></span>
                    )}
                    {probeResult.connection.content_type && (
                      <span>Type: <code className="text-[#F1F3F6] font-mono">{probeResult.connection.content_type.split(";")[0]}</code></span>
                    )}
                    {probeResult.connection.num_redirects > 0 && (
                      <span>Redirects: <code className="text-[#FBBF24] font-mono">{probeResult.connection.num_redirects}</code></span>
                    )}
                  </div>
                </div>

                {/* TreeSHAP Explainability (only for alerts) */}
                {probeResult.top_features && probeResult.top_features.length > 0 && (
                  <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-3">
                    <div className="flex items-center space-x-2">
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                        TreeSHAP Feature Attribution
                      </h3>
                    </div>
                    <p className="text-[11px] text-[#939AA6]">
                      Which real connection metrics most influenced the ML model&apos;s decision:
                    </p>

                    <div className="space-y-2">
                      {probeResult.top_features.map((tf, idx) => (
                        <div key={idx} className="p-2.5 bg-[#181B21] border border-[#282C35] rounded-xs flex items-center justify-between text-xs font-mono">
                          <div>
                            <span className="text-[#F1F3F6] font-medium">{tf.feature}</span>
                            {tf.value !== undefined && tf.value !== null && (
                              <span className="text-[#939AA6] text-[10px] ml-2">value: {String(tf.value)}</span>
                            )}
                          </div>
                          <span className={tf.contribution >= 0 ? "text-[#F87171] font-semibold" : "text-[#34D399]"}>
                            {tf.contribution >= 0 ? `+${tf.contribution.toFixed(4)}` : tf.contribution.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Extracted Features (expandable) */}
                <details className="bg-[#13151A] border border-[#282C35] rounded-sm">
                  <summary className="p-4 cursor-pointer text-xs font-semibold uppercase tracking-wider text-[#F1F3F6] flex items-center space-x-2 select-none">
                    <span>Extracted UNSW-NB15 Features (42 features from real connection)</span>
                  </summary>
                  <div className="px-4 pb-4">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px] font-mono">
                      {Object.entries(probeResult.extracted_features).map(([key, val]) => (
                        <div key={key} className="p-2 bg-[#181B21] border border-[#282C35] rounded-xs">
                          <div className="text-[#939AA6] text-[9px] uppercase">{key}</div>
                          <div className="text-[#F1F3F6] mt-0.5 truncate">{String(val)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </details>

                {/* Action Box & Alert Link */}
                <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-3">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                    Analysis Summary
                  </h3>

                  <p className="text-xs text-[#D1D5DB] leading-relaxed">
                    {probeResult.recommended_action}
                  </p>

                  {probeResult.alert_id && (
                    <div className="pt-2">
                      <Link
                        href={`/alerts/${probeResult.alert_id}`}
                        className="inline-flex items-center space-x-2 px-3 py-1.5 bg-[#181B21] border border-[#3B414E] text-[#38BDF8] hover:text-white hover:border-[#38BDF8] rounded-xs text-xs font-semibold transition-colors"
                      >
                        <span>Open Alert #{probeResult.alert_id.slice(0, 8)} in SOC Triage</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-12 text-center space-y-3">
                <div className="text-sm font-semibold text-[#F1F3F6]">Ready to Probe</div>
                <p className="text-xs text-[#939AA6] max-w-sm mx-auto">
                  Enter any website URL and click &quot;Probe Target Website&quot;. Nexus will make a real HTTP connection,
                  measure actual network metrics (DNS, TCP, TLS, response times), extract flow features, and run
                  them through the LightGBM model to detect anomalous behavior.
                </p>
                <div className="flex flex-wrap justify-center gap-2 pt-3">
                  {["https://google.com", "https://github.com", "https://example.com"].map((url) => (
                    <button
                      key={url}
                      onClick={() => { setTargetUrl(url); }}
                      className="px-2.5 py-1 text-[11px] font-mono bg-[#181B21] border border-[#282C35] text-[#38BDF8] hover:bg-[#1D2027] hover:border-[#38BDF8]/40 rounded-xs transition-colors cursor-pointer"
                    >
                      {url.replace("https://", "")}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

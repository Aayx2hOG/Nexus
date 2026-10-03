"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  fetchSimulationProfiles,
  simulateProbe,
  fetchPresetData,
  submitBatchPredictions,
  fetchModelSummary,
} from "@/lib/api";
import { parseCSVText, chunkFlows, ParsedFlow } from "@/lib/csv";
import { ProbeResult, SimulationProfile } from "@/lib/types";
import {
  UploadCloud,
  FileSpreadsheet,
  Globe,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  ExternalLink,
  ChevronRight,
  Database,
  BarChart2,
  Activity,
  Layers,
  Sparkles,
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

  // Query Simulation Profiles
  const { data: profiles, isLoading: profilesLoading } = useQuery({
    queryKey: ["simulationProfiles"],
    queryFn: fetchSimulationProfiles,
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

  // --- Website Attack Simulator State ---
  const [targetUrl, setTargetUrl] = useState("http://localhost:3000");
  const [selectedProfileId, setSelectedProfileId] = useState("http_exploit");
  const [isProbing, setIsProbing] = useState(false);
  const [probeResult, setProbeResult] = useState<ProbeResult | null>(null);
  const [probeError, setProbeError] = useState<string | null>(null);

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

  // --- Website Probe Handler ---
  const handleRunProbe = async () => {
    if (!targetUrl.trim()) return;
    setIsProbing(true);
    setProbeError(null);
    setProbeResult(null);

    try {
      const res = await simulateProbe({
        target_url: targetUrl.trim(),
        profile_id: selectedProfileId,
      });
      setProbeResult(res);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Website probe simulation failed.";
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
            <span>Traffic Ingestion & Attack Simulator</span>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 bg-[#181B21] border border-[#282C35] text-[#34D399] rounded-sm">
              Live Engine
            </span>
          </h1>
          <p className="text-xs text-[#939AA6] mt-0.5">
            Manually upload flow CSVs for automated model triage or simulate web attacks targeting a specific website.
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
            <Globe className="w-3.5 h-3.5" />
            <span>Website Attack Simulator</span>
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
                    <Layers className="w-4 h-4 text-[#939AA6]" />
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
      {/* TAB 2: Website Attack Simulator & Target Probe                */}
      {/* ============================================================== */}
      {activeTab === "probe" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Target Configuration & Attack Profiles */}
          <div className="lg:col-span-6 space-y-5">
            <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4">
              <div className="flex items-center space-x-2">
                <Globe className="w-4 h-4 text-[#F1F3F6]" />
                <h2 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                  Target Website Configuration
                </h2>
              </div>

              <div>
                <label className="block text-[11px] font-medium text-[#939AA6] mb-1">
                  Target Website URL / Host Endpoint
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={targetUrl}
                    onChange={(e) => setTargetUrl(e.target.value)}
                    placeholder="https://my-target-site.com or http://localhost:3000"
                    className="w-full bg-[#181B21] border border-[#282C35] focus:border-[#38BDF8] text-[#F1F3F6] text-xs font-mono px-3 py-2 rounded-xs outline-hidden"
                  />
                </div>
                <p className="text-[10px] text-[#939AA6] mt-1">
                  Specify any web application or microservice endpoint to simulate real network flow infiltration against.
                </p>
              </div>

              {/* Attack / Traffic Profile Selection */}
              <div>
                <label className="block text-[11px] font-medium text-[#939AA6] mb-2">
                  Select Traffic / Attack Profile
                </label>

                {profilesLoading ? (
                  <div className="py-4 text-center text-xs text-[#939AA6]">Loading profiles...</div>
                ) : (
                  <div className="space-y-2">
                    {profiles?.map((p: SimulationProfile) => {
                      const isSelected = selectedProfileId === p.id;
                      const isAttack = p.expected_verdict === "alert";

                      return (
                        <div
                          key={p.id}
                          onClick={() => setSelectedProfileId(p.id)}
                          className={`p-3 border rounded-sm cursor-pointer transition-colors ${
                            isSelected
                              ? isAttack
                                ? "bg-[#7F1D1D]/15 border-[#F87171] ring-1 ring-[#F87171]/40"
                                : "bg-[#064E3B]/15 border-[#34D399] ring-1 ring-[#34D399]/40"
                              : "bg-[#181B21] border-[#282C35] hover:border-[#3B414E]"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold text-[#F1F3F6]">{p.name}</span>
                            <span
                              className={`text-[9px] uppercase font-mono px-1.5 py-0.5 rounded-xs font-semibold ${
                                isAttack
                                  ? "bg-[#7F1D1D]/40 text-[#F87171] border border-[#F87171]/30"
                                  : "bg-[#064E3B]/40 text-[#34D399] border border-[#34D399]/30"
                              }`}
                            >
                              {p.category}
                            </span>
                          </div>
                          <p className="text-[11px] text-[#939AA6] mt-1">{p.description}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Run Probe Button */}
              <button
                type="button"
                onClick={handleRunProbe}
                disabled={isProbing || !targetUrl.trim()}
                className="w-full py-2.5 px-4 bg-[#F1F3F6] hover:bg-white text-[#0A0B0D] text-xs font-bold uppercase tracking-wider rounded-sm transition-colors cursor-pointer flex items-center justify-center space-x-2 disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{isProbing ? "Sending Traffic & Probing..." : "Send Traffic & Probe Target"}</span>
              </button>

              {probeError && (
                <div className="p-3 bg-[#7F1D1D]/20 border border-[#F87171]/40 rounded-sm text-xs text-[#F87171]">
                  {probeError}
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Live Target Monitor & Did it Get Hit? */}
          <div className="lg:col-span-6 space-y-5">
            {probeResult ? (
              <div className="space-y-4">
                {/* BIG TARGET HIT BANNER */}
                <div
                  className={`p-5 border rounded-sm space-y-3 transition-all ${
                    probeResult.target_hit
                      ? "bg-[#7F1D1D]/20 border-[#F87171] shadow-lg shadow-[#F87171]/5"
                      : "bg-[#064E3B]/20 border-[#34D399] shadow-lg shadow-[#34D399]/5"
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {probeResult.target_hit ? (
                      <div className="w-10 h-10 bg-[#7F1D1D]/40 border border-[#F87171] rounded-full flex items-center justify-center text-[#F87171] animate-pulse">
                        <ShieldAlert className="w-5 h-5" />
                      </div>
                    ) : (
                      <div className="w-10 h-10 bg-[#064E3B]/40 border border-[#34D399] rounded-full flex items-center justify-center text-[#34D399]">
                        <ShieldCheck className="w-5 h-5" />
                      </div>
                    )}
                    <div>
                      <div
                        className={`text-sm font-bold tracking-tight uppercase ${
                          probeResult.target_hit ? "text-[#F87171]" : "text-[#34D399]"
                        }`}
                      >
                        {probeResult.target_hit ? "🚨 TARGET HIT! Malicious Infiltration" : "🟢 TARGET SECURE / ALL CLEAR"}
                      </div>
                      <div className="text-xs text-[#939AA6] mt-0.5">
                        Target Endpoint: <code className="font-mono text-[#F1F3F6]">{probeResult.target_url}</code>
                      </div>
                    </div>
                  </div>

                  {/* Probability Gauge & Threat Dial */}
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
                          probeResult.target_hit ? "text-[#F87171]" : "text-[#34D399]"
                        }`}
                      >
                        {probeResult.threat_level}
                      </div>
                      <div className="text-[9px] text-[#939AA6]">Verdict: {probeResult.decision.toUpperCase()}</div>
                    </div>
                  </div>
                </div>

                {/* Explainability & TreeSHAP Attribution Card */}
                {probeResult.top_features && probeResult.top_features.length > 0 && (
                  <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-3">
                    <div className="flex items-center space-x-2">
                      <Sparkles className="w-4 h-4 text-[#FBBF24]" />
                      <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                        TreeSHAP Explainability (Root-Cause Telemetry)
                      </h3>
                    </div>
                    <p className="text-[11px] text-[#939AA6]">
                      Signed feature attributions indicating exact network telemetry drivers that triggered the detection:
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

                {/* Action Box & Alert Direct Link */}
                <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-3">
                  <div className="flex items-center space-x-2">
                    <Activity className="w-4 h-4 text-[#38BDF8]" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#F1F3F6]">
                      SOC Analyst Action Guidance
                    </h3>
                  </div>

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
                <Globe className="w-10 h-10 text-[#3B414E] mx-auto" />
                <div className="text-sm font-semibold text-[#F1F3F6]">Website Target Monitor Inactive</div>
                <p className="text-xs text-[#939AA6] max-w-sm mx-auto">
                  Configure your target URL on the left, pick an attack profile, and click &quot;Send Traffic &amp; Probe Target&quot; to check in real time whether the website gets hit.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

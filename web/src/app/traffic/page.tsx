"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Cpu,
  Globe2,
  History,
  Layers,
  RefreshCw,
  Upload,
} from "lucide-react";
import {
  Button,
  EmptyState,
  ErrorState,
  InfoHelp,
  PageHeader,
  ScoreScale,
  Status,
  TechnicalValue,
} from "@/components/components";
import { api, isPreviewData } from "@/lib/api";
import { createPreset, parseNexusCsv, predictorFields } from "@/lib/csv";
import type { FlowPrediction, ModelOption, ProbeResult } from "@/lib/types";

const DEFAULT_MODELS: ModelOption[] = [
  {
    id: "uncertainty_band_ae10",
    name: "Uncertainty-band four-mode, AE 10% attack oriented",
    description:
      "LightGBM + Baseline Autoencoder with 10% FPR anomaly boundaries. Focuses on attack recovery in the uncertainty band [0.10, 0.65].",
    architecture: "LightGBM + Baseline AE (4-Mode)",
    decision_threshold: 0.5776925765603604,
    focus: "Attack Oriented (Higher Recall)",
    ae_budget: "10% FPR",
    is_fusion: true,
  },
  {
    id: "mode_confidence_ae05",
    name: "Mode confidence, AE 5%, soc oriented",
    description:
      "LightGBM + Baseline Autoencoder with 5% FPR anomaly boundaries. Focuses on SOC triage and reducing false alarms by 992 cases.",
    architecture: "LightGBM + Baseline AE (Confidence Mode)",
    decision_threshold: 0.5776925765603604,
    focus: "SOC Oriented (Lower False Positives)",
    ae_budget: "5% FPR",
    is_fusion: true,
  },
];

export default function TrafficPage() {
  const router = useRouter();
  const [target, setTarget] = useState("https://example.com");
  const [history, setHistory] = useState<ProbeResult[]>([]);
  const [openResult, setOpenResult] = useState<ProbeResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [csvRows, setCsvRows] = useState<Record<string, string | number>[]>([]);
  const [csvLabels, setCsvLabels] = useState<Array<"Attack" | "Normal" | undefined>>([]);
  const [csvErrors, setCsvErrors] = useState<string[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [results, setResults] = useState<FlowPrediction[]>([]);
  const [resultFilter, setResultFilter] = useState<"all" | "alerts" | "normal">("all");
  const [resultPage, setResultPage] = useState(1);
  const [selectedModel, setSelectedModel] = useState<string>("uncertainty_band_ae10");

  const { data: modelsData } = useQuery({
    queryKey: ["models"],
    queryFn: api.models,
  });

  const modelOptions = modelsData && modelsData.length > 0 ? modelsData : DEFAULT_MODELS;
  const currentModel = modelOptions.find((m) => m.id === selectedModel) || modelOptions[0];

  useEffect(() => {
    const resetLocalTrafficData = () => {
      setFileName("");
      setCsvRows([]);
      setCsvLabels([]);
      setCsvErrors([]);
      setResults([]);
      setOpenResult(null);
      setHistory([]);
    };
    window.addEventListener("nexus-data-mode-change", resetLocalTrafficData);
    return () => window.removeEventListener("nexus-data-mode-change", resetLocalTrafficData);
  }, []);

  const probe = useMutation({
    mutationFn: () => api.probe(target),
    onSuccess: (result) => {
      setOpenResult(result);
      setHistory((existing) => [result, ...existing].slice(0, 20));
    },
  });

  const ingestion = useMutation({
    mutationFn: async () => {
      const data = await api.predict(
        csvRows.map((row, index) =>
          csvLabels[index] ? { ...row, ground_truth: csvLabels[index] as string } : row,
        ),
        selectedModel,
      );
      if (!Array.isArray(data)) throw new Error("Malformed prediction response");
      return data;
    },
    onSuccess: (data) => {
      setResults(data);
      setResultFilter("all");
      setResultPage(1);
    },
  });

  const loadCsvFile = async (file: File) => {
    setFileName(file.name);
    setResults([]);
    const parsed = parseNexusCsv(await file.text());
    setCsvRows(parsed.rows);
    setCsvLabels(parsed.labels);
    setCsvErrors(parsed.errors);
  };

  const loadPreset = (kind: "normal" | "attack" | "mixed") => {
    const preset = createPreset(kind);
    setFileName(kind === "mixed" ? "100 Mixed preset" : `50 ${kind === "normal" ? "Normal" : "Attack"} preset`);
    setCsvRows(preset.map((row) => Object.fromEntries(predictorFields.map((field) => [field, row[field]]))));
    setCsvLabels(preset.map((row) => row.ground_truth as "Attack" | "Normal"));
    setCsvErrors([]);
    setResults([]);
  };

  const filteredResults = results.filter((result) =>
    resultFilter === "all" ? true : resultFilter === "alerts" ? result.decision === "Alert" : result.decision === "Normal",
  );
  const resultPageSize = 10;
  const resultPageCount = Math.max(1, Math.ceil(filteredResults.length / resultPageSize));
  const visibleResults = filteredResults.slice((resultPage - 1) * resultPageSize, resultPage * resultPageSize);
  const hasLabels = results.length > 0 && results.every((result) => result.groundTruth);
  const hasAeMode = results.some((result) => Boolean(result.aeMode));
  const evaluation = hasLabels
    ? results.reduce(
        (counts, result) => {
          const predictedAttack = result.decision === "Alert";
          const actualAttack = result.groundTruth === "Attack";
          if (predictedAttack && actualAttack) counts.tp += 1;
          if (predictedAttack && !actualAttack) counts.fp += 1;
          if (!predictedAttack && !actualAttack) counts.tn += 1;
          if (!predictedAttack && actualAttack) counts.fn += 1;
          return counts;
        },
        { tp: 0, fp: 0, tn: 0, fn: 0 },
      )
    : null;

  return (
    <div className="page">
      <PageHeader
        title="Traffic"
        description="Upload network-flow records or run an experimental website connection probe through NEXUS."
      />
      <section className="workflow-section">
        <div className="workflow-number">01</div>
        <div className="workflow-main">
          <div className="workflow-title">
            <div>
              <h2>CSV flow ingestion</h2>
              <p>Upload network-flow records using the NEXUS 42-feature schema and send them through the prediction pipeline.</p>
            </div>
            <Status tone="info">42 required features</Status>
          </div>

          {/* Model Selector Feature in CSV Ingestion Area */}
          <div className="model-selection-bar">
            <div className="model-select-group">
              <label htmlFor="model-select-dropdown" className="model-select-label">
                <Cpu size={14} />
                <span>DETECTION MODEL</span>
              </label>
              <div className="model-select-wrapper">
                <select
                  id="model-select-dropdown"
                  className="model-select-input"
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setResults([]);
                  }}
                >
                  {modelOptions.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.name}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="model-select-chevron" />
              </div>
            </div>
            {currentModel && (
              <div className="model-pill-badges">
                <span className={`model-pill ${currentModel.is_fusion ? "fusion" : "baseline"}`}>
                  {currentModel.architecture}
                </span>
                <span className="model-pill focus-tag">{currentModel.focus}</span>
                {currentModel.ae_budget && (
                  <span className="model-pill budget-tag">AE Budget: {currentModel.ae_budget}</span>
                )}
              </div>
            )}
          </div>

          <div className="csv-ingestion-grid">
            <label
              className={`upload-zone ${dragActive ? "drag-active" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                const file = event.dataTransfer.files[0];
                if (file) void loadCsvFile(file);
              }}
            >
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void loadCsvFile(file);
                }}
              />
              <Upload size={20} />
              <strong>{fileName || "Select or drop a CSV file"}</strong>
              <span>Headers are checked against the 42-feature NEXUS schema before submission.</span>
            </label>
            <div className="csv-schema">
              <div><span>REQUIRED COLUMNS</span><strong className="mono">42</strong></div>
              <div><span>VALID ROWS</span><strong className="mono">{csvRows.length}</strong></div>
              <div><span>GROUND TRUTH</span><strong>{csvLabels.some(Boolean) ? "Included" : "Not provided"}</strong></div>
            </div>
          </div>
          <div className="preset-row">
            <span>PRESETS</span>
            <Button onClick={() => loadPreset("normal")}>50 Normal</Button>
            <Button onClick={() => loadPreset("attack")}>50 Attack</Button>
            <Button onClick={() => loadPreset("mixed")}>100 Mixed</Button>
          </div>
          {csvErrors.length > 0 && (
            <div className="csv-errors" role="alert">
              <strong>CSV validation failed</strong>
              {csvErrors.slice(0, 8).map((error) => <span key={error}>{error}</span>)}
              {csvErrors.length > 8 && <span>{csvErrors.length - 8} additional errors</span>}
            </div>
          )}
          {csvRows.length > 0 && csvErrors.length === 0 && (
            <div className="csv-preview">
              <div className="csv-preview-head">
                <div>
                  <span>VALIDATED PREVIEW</span>
                  <strong>{fileName}</strong>
                </div>
                <Button variant="primary" disabled={ingestion.isPending} onClick={() => ingestion.mutate()}>
                  {ingestion.isPending ? <RefreshCw className="spin" size={15} /> : <Activity size={15} />}
                  Ingest {csvRows.length} flows with {currentModel.id === "uncertainty_band_ae10" ? "AE 10% model" : currentModel.id === "mode_confidence_ae05" ? "AE 5% model" : "LightGBM"}
                </Button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead><tr>{predictorFields.slice(0, 8).map((field) => <th key={field}>{field}</th>)}</tr></thead>
                  <tbody>
                    {csvRows.slice(0, 5).map((row, index) => (
                      <tr key={index}>{predictorFields.slice(0, 8).map((field) => <td className="mono" key={field}>{row[field]}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <small>Showing the first {Math.min(5, csvRows.length)} rows and 8 of 42 feature columns.</small>
              {ingestion.isError && <ErrorState retry={() => ingestion.mutate()} />}
            </div>
          )}
          {results.length > 0 && (
            <div className="prediction-results">
              <div className="section-heading">
                <div>
                  <h3>Prediction results</h3>
                  <p>Decisions returned by <strong>{currentModel.name}</strong> ({currentModel.architecture}).</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="model-pill fusion">{currentModel.focus}</span>
                  <span className="mono">{results.length} FLOWS</span>
                </div>
              </div>
              <div className="result-summary">
                <div><span>Total</span><strong className="mono">{results.length}</strong></div>
                <div><span>Alerts</span><strong className="mono">{results.filter((result) => result.decision === "Alert").length}</strong></div>
                <div><span>Normal</span><strong className="mono">{results.filter((result) => result.decision === "Normal").length}</strong></div>
                {evaluation && (
                  <>
                    <div><span>Accuracy</span><strong className="mono">{(((evaluation.tp + evaluation.tn) / results.length) * 100).toFixed(1)}%</strong></div>
                    <div><span>Precision</span><strong className="mono">{(evaluation.tp / Math.max(1, evaluation.tp + evaluation.fp) * 100).toFixed(1)}%</strong></div>
                    <div><span>Recall</span><strong className="mono">{(evaluation.tp / Math.max(1, evaluation.tp + evaluation.fn) * 100).toFixed(1)}%</strong></div>
                  </>
                )}
              </div>
              <div className="result-filter">
                {(["all", "alerts", "normal"] as const).map((filter) => (
                  <button key={filter} className={resultFilter === filter ? "active" : ""} onClick={() => { setResultFilter(filter); setResultPage(1); }}>
                    {filter === "all" ? "All" : filter === "alerts" ? "Alerts Only" : "Normal Only"}
                  </button>
                ))}
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Flow ID</th>
                      <th>Decision</th>
                      <th>Score</th>
                      <th>Threshold</th>
                      {hasAeMode && <th>AE Anomaly Mode</th>}
                      <th>Category</th>
                      <th>Alert</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleResults.map((result) => (
                      <tr key={result.flowId}>
                        <td className="mono strong">{result.flowId}</td>
                        <td className={result.decision === "Alert" ? "threat-value" : ""}>
                          <span className={`status ${result.decision === "Alert" ? "status-danger" : "status-success"}`}>
                            {result.decision}
                          </span>
                        </td>
                        <td className="mono">{result.score.toFixed(4)}</td>
                        <td className="mono">{result.threshold.toFixed(4)}</td>
                        {hasAeMode && (
                          <td>
                            <span className={`ae-mode-badge ${result.aeMode || "no_anomaly"}`}>
                              {result.aeMode ? result.aeMode.toUpperCase() : "—"}
                            </span>
                          </td>
                        )}
                        <td>{result.category ?? "Not applicable"}</td>
                        <td>{result.alertId ? <button className="result-link" onClick={() => router.push(`/alerts/${result.alertId}`)}>{result.alertId}</button> : "None"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {filteredResults.length > resultPageSize && (
                <div className="pagination">
                  <span className="mono">PAGE {resultPage} / {resultPageCount}</span>
                  <div>
                    <Button disabled={resultPage === 1} onClick={() => setResultPage((value) => value - 1)}>Previous</Button>
                    <Button disabled={resultPage === resultPageCount} onClick={() => setResultPage((value) => value + 1)}>Next</Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section className="workflow-section">
        <div className="workflow-number">02</div>
        <div className="workflow-main">
          <div className="workflow-title">
            <div>
              <div className="title-with-status">
                <h2>Live website probe</h2>
                <InfoHelp label="live website probe" text="The probe records connection measurements from a real HTTP request and sends the resulting flow features through NEXUS. It is not a vulnerability scan." />
                <Status tone="warning">Experimental</Status>
              </div>
              <p>Measure one website connection and inspect how NEXUS classifies the resulting network flow.</p>
            </div>
            <Globe2 size={24} />
          </div>
          <div className="probe-form">
            <label className="field probe-target">
              <span>TARGET URL</span>
              <div className="input-icon"><Globe2 size={16} /><input value={target} onChange={(e) => setTarget(e.target.value)} /></div>
            </label>
            <Button variant="primary" disabled={!target || probe.isPending} onClick={() => probe.mutate()}>
              {probe.isPending ? <RefreshCw className="spin" size={16} /> : <Activity size={16} />}
              Run probe
            </Button>
          </div>
          <p className="probe-disclaimer">Diagnostic workflow only. This is not a vulnerability scan or penetration test.</p>
          {probe.isError && <ErrorState retry={() => probe.mutate()} />}
          {openResult && (
            <div className="probe-report">
              <div className="report-header">
                <div>
                  <span>DIAGNOSTIC REPORT</span>
                  <h3>{openResult.target}</h3>
                  <TechnicalValue>{new Date(openResult.timestamp).toLocaleString()}</TechnicalValue>
                </div>
                <Status tone={openResult.threatLevel === "Normal" ? "success" : "danger"}>{openResult.verdict}</Status>
              </div>
              <div className="report-grid">
                <div className="report-score">
                  <span>ATTACK SCORE <InfoHelp label="attack score" text="The model's estimate of how suspicious this network flow is. The flow becomes an alert only if this score crosses the decision threshold." /></span>
                  <strong className="mono">{openResult.score.toFixed(4)}</strong>
                  <ScoreScale score={openResult.score} threshold={openResult.threshold} />
                </div>
                <div className="connection-grid">
                  {Object.entries(openResult.measurements).map(([label, value]) => (
                    <div key={label}><span>{label}</span><strong className="mono">{value}</strong></div>
                  ))}
                </div>
              </div>
            </div>
          )}
          <div className="probe-history">
            <div className="section-heading">
              <div><h3>Probe history</h3><p>Results retained in this browser session.</p></div>
              <History size={18} />
            </div>
            {history.length === 0 ? (
              <EmptyState title="No probe history." detail="Run a live probe to retain its result in this session." />
            ) : history.map((item) => (
              <button className="history-row" key={item.id} onClick={() => setOpenResult(item)}>
                <span><Globe2 size={15} /> {item.target}</span>
                <TechnicalValue>{new Date(item.timestamp).toLocaleTimeString()}</TechnicalValue>
                <Status tone={item.threatLevel === "Normal" ? "success" : "danger"}>{item.verdict}</Status>
                <TechnicalValue>{item.score.toFixed(4)}</TechnicalValue>
                <ChevronRight size={16} />
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Check,
  ChevronRight,
  CircleDot,
  Copy,
  Gauge,
  RefreshCw,
  Server,
} from "lucide-react";
import {
  Button,
  ErrorState,
  InfoHelp,
  LoadingState,
  PageHeader,
  Status,
} from "@/components/components";
import { api } from "@/lib/api";

function CopyHash({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="hash-row"
      onClick={() => {
        void navigator.clipboard.writeText(value);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
      }}
    >
      <span className="mono">{value}</span>
      {copied ? <Check size={15} /> : <Copy size={15} />}
    </button>
  );
}

const modelMetricExplanations: Record<string, string> = {
  Precision: "Of the flows classified as attacks, the proportion that were actually attacks in the selection evaluation.",
  Recall: "Of the attacks in the evaluation data, the proportion the model correctly detected.",
  F1: "A combined measure that balances precision and recall.",
  "False positive rate": "The proportion of normal flows incorrectly classified as attacks.",
};

export default function ModelPage() {
  const query = useQuery({ queryKey: ["manifest"], queryFn: api.manifest });
  if (query.isLoading) return <div className="page"><LoadingState label="Loading model manifest" /></div>;
  if (query.isError || !query.data) return <div className="page"><ErrorState retry={() => void query.refetch()} /></div>;
  const manifest = query.data;

  return (
    <div className="page">
      <PageHeader
        eyebrow="PROVENANCE"
        title="Model and manifest"
        description="Technical information about the model running inside NEXUS, including its version, threshold, evaluation results, and artifact integrity."
        actions={<Button onClick={() => void query.refetch()}><RefreshCw size={15} /> Refresh manifest</Button>}
      />
      <section className="manifest-hero">
        <div className="manifest-mark"><Gauge size={28} /></div>
        <div className="manifest-title"><span>PRIMARY SERVING MODEL</span><h2>{manifest.architecture}</h2><p>Supervised network flow classifier</p></div>
        <div className="manifest-kv">
          <span>BUNDLE VERSION <InfoHelp label="Model bundle" text="The packaged model, preprocessor, feature schema, threshold, and validation metadata used by NEXUS." /></span>
          <strong className="mono">{manifest.bundleVersion}</strong>
        </div>
        <div className="manifest-kv">
          <span>DECISION THRESHOLD <InfoHelp label="decision threshold" text="The minimum score required for NEXUS to classify a network flow as an alert." /></span>
          <strong className="mono">{manifest.threshold}</strong>
        </div>
        <Status tone="success">Production path</Status>
      </section>
      <div className="model-grid">
        <section className="content-section selection-panel">
          <span className="section-index">01 / VALIDATED EVALUATION</span>
          <div className="section-heading">
            <div>
              <div className="heading-with-help">
                <h2>Selection results</h2>
                <InfoHelp label="selection results" text="Measurements from the evaluation used to select this model version. They do not represent guaranteed production performance." />
              </div>
              <p>How this model performed during its documented selection evaluation.</p>
            </div>
          </div>
          <div className="selection-grid">
            {Object.entries(manifest.selection).map(([label, value]) => {
              return (
                <div key={label}>
                  <span>
                    {label}
                    {modelMetricExplanations[label] && <InfoHelp label={label} text={modelMetricExplanations[label]} />}
                  </span>
                  <strong className="mono">{String(value)}</strong>
                </div>
              );
            })}
          </div>
        </section>
        <section className="content-section pipeline-panel">
          <span className="section-index">02 / SERVING PATH</span>
          <h2>Production inference</h2>
          <p className="section-explanation">The steps NEXUS follows to turn one network flow into a normal or alert decision.</p>
          <div className="pipeline">
            {["Network flow", "Preprocessing", "LightGBM", "Threshold", "Normal or alert", "TreeSHAP", "SOC review"].map((step, index, all) => (
              <div key={step}><span>{index + 1}</span><strong>{step}</strong>{index < all.length - 1 && <ChevronRight size={15} />}</div>
            ))}
          </div>
        </section>
      </div>
      <section className="content-section">
        <span className="section-index">03 / ARTIFACT INTEGRITY</span>
        <div className="section-heading"><div><h2>Model provenance</h2><p>File fingerprints used to verify that model and source artifacts have not changed.</p></div><Server size={20} /></div>
        <div className="hash-table">
          <div className="hash-group">
            <h3>ARTIFACT HASHES</h3>
            {Object.entries(manifest.artifactHashes).map(([name, hash]) => (
              <div className="hash-item" key={name}><span>{name}</span><CopyHash value={String(hash)} /></div>
            ))}
          </div>
          <div className="hash-group">
            <h3>SOURCE HASHES</h3>
            {Object.entries(manifest.sourceHashes).map(([name, hash]) => (
              <div className="hash-item" key={name}><span>{name}</span><CopyHash value={String(hash)} /></div>
            ))}
          </div>
        </div>
      </section>
      <section className="research-panel">
        <div><CircleDot size={19} /><span>RESEARCH CONTEXT</span></div>
        <div>
          <h2>Selective recovery remains separate from production serving.</h2>
          <p>Research evaluates whether autoencoders, Isolation Forest, latent distance, calibration, and selective fusion can recover attacks missed by the primary LightGBM detector.</p>
          <div className="research-path"><span>LightGBM</span><ChevronRight size={14} /><span>Anomaly detection</span><ChevronRight size={14} /><span>Selective fusion</span><ChevronRight size={14} /><span>Calibration</span><ChevronRight size={14} /><span>Research evaluation</span></div>
        </div>
      </section>
    </div>
  );
}

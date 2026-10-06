"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { fetchShadowPredictions, fetchShadowSummary } from "@/lib/api";

const panel = "bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4";
const pct = (value: number | null) => value === null ? "Unavailable" : `${(value * 100).toFixed(2)}%`;

export default function ShadowPage() {
  const [after, setAfter] = useState(0);
  const summary = useQuery({ queryKey: ["shadow"], queryFn: fetchShadowSummary });
  const rows = useQuery({ queryKey: ["shadowPredictions", after], queryFn: () => fetchShadowPredictions(after) });
  const data = summary.data;
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...data, exported_at: new Date().toISOString() }, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "nexus-shadow-evidence.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <div className="space-y-6 text-sm">
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div>
        <p className="text-xs uppercase tracking-widest text-emerald-400 mb-2">Selective fusion · shadow mode</p>
        <h1 className="text-xl font-semibold">Observe recovery before changing live detection</h1>
        <p className="text-[#939AA6] mt-2">Real model scores from a frozen research pipeline. Operational alerts still come only from live LightGBM.</p>
      </div>
      <button className="underline text-emerald-400" onClick={() => { summary.refetch(); rows.refetch(); }}>Refresh evidence</button>
    </div>
    {(summary.isLoading || rows.isFetching) && <p role="status">Loading shadow evidence…</p>}
    {(summary.error || rows.error) && <p role="alert" className="text-red-400">{(summary.error || rows.error)?.message}</p>}
    {data && <>
      <section className={panel}>
        <p>Shadow status: <strong className={data.status === "ready" ? "text-emerald-400" : "text-amber-300"}>{data.status}</strong></p>
        {data.status !== "ready" ? <p>Configure a parity-verified bundle with NEXUS_SHADOW_BUNDLE_DIR and restart the backend. See docs/SHADOW_FUSION.md. {data.error_code}</p> : <>
          <p className="font-mono text-xs break-all">Candidate: {data.bundle_version} · Live: {data.production_bundle_version}</p>
          <p>Research scenario: {data.held_family} withheld · seed {data.seed} · calibration budget {pct(data.budget)}</p>
          <p className="text-xs break-all text-[#939AA6]">Manifest SHA-256: {data.manifest_sha256}</p>
          <p>Raw-input parity: {data.parity?.status} across {data.parity?.rows.toLocaleString()} archived rows; {data.parity?.decision_mismatches} decision mismatches.</p>
        </>}
        <p className="text-amber-300 text-xs">{data.note}</p>
        <p className="text-xs text-[#939AA6]">Independent confirmation is deferred. Paired metrics below use only scored flows with recorded truth. They are historical replay evidence.</p>
      </section>
      {data.status === "ready" && <>
        <section className={panel}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[["Shadow scored", data.scored], ["Scored with labels", data.labeled_scored], ["Shadow errors", data.errors], ["Not shadowed", data.not_shadowed]].map(([label, count]) =>
              <div key={label}><p className="text-xs text-[#939AA6]">{label}</p><p className="text-2xl mt-1">{count}</p></div>)}
          </div>
          <p>Across scored flows, the candidate would add {data.candidate_additions_vs_live} alerts and omit {data.candidate_removals_vs_live} live alerts. Neither change is applied.</p>
          {data.scored === 0 && <p>No shadow traffic yet. <Link className="underline text-emerald-400" href="/traffic">Run CSV detection</Link> after enabling the bundle.</p>}
          {data.scored > 0 && data.labeled_scored === 0 && <p className="text-amber-300">Scores are available, but accuracy requires separately recorded truth. Run labeled replay before interpreting detection quality.</p>}
        </section>
        <section className={panel}>
          <h2 className="font-semibold">Paired labeled-flow comparison</h2>
          <div className="overflow-x-auto"><table className="w-full text-left">
            <thead><tr><th>Metric</th><th>Live LightGBM</th><th>Research LightGBM</th><th>Research selective fusion</th></tr></thead>
            <tbody>{[
              ["Recall", ...[data.metrics.live, data.metrics.reference, data.metrics.candidate].map(m => pct(m.recall))],
              ["Benign FPR", ...[data.metrics.live, data.metrics.reference, data.metrics.candidate].map(m => pct(m.false_positive_rate))],
              ["Precision", ...[data.metrics.live, data.metrics.reference, data.metrics.candidate].map(m => pct(m.precision))],
              ["Missed attacks", ...[data.metrics.live, data.metrics.reference, data.metrics.candidate].map(m => m.false_negatives)],
            ].map(row => <tr key={row[0]} className="border-t border-[#282C35]">{row.map((cell, i) => <td key={i} className="py-3 pr-3">{cell}</td>)}</tr>)}</tbody>
          </table></div>
          <div className="grid md:grid-cols-2 gap-4">
            {[["Compared with its research baseline", data.vs_reference], ["Compared with the live release", data.vs_live]].map(([title, value]) => {
              if (typeof value === "string") return null;
              return <div key={String(title)} className="bg-[#0A0B0D] p-4 space-y-2">
                <h3>{String(title)}</h3>
                <p>{value.recovered_attacks} attacks recovered · {value.lost_attacks} detections lost</p>
                <p>{value.added_false_positives} false alerts added · {value.removed_false_positives} removed</p>
              </div>;
            })}
          </div>
          <p className="text-xs text-[#939AA6]">Preservation is guaranteed only for the research baseline used to fit this policy. The live release has different weights and a different threshold.</p>
          <button className="text-emerald-400 underline" onClick={download}>Export comparison JSON</button>
        </section>
        <section className={panel}>
          <h2 className="font-semibold">Flow evidence</h2>
          <div className="space-y-3">{rows.data?.items.map(row => <details key={row.sequence} className="border border-[#282C35] p-3">
            <summary className="cursor-pointer break-all"><span className="font-mono text-xs">{row.flow_id}</span> · live {row.live_decision} · shadow {row.decision ?? "error"} · {row.reason?.replaceAll("_", " ") ?? row.error_code}</summary>
            <dl className="grid sm:grid-cols-2 gap-2 mt-3 text-xs">
              <div>Research LightGBM: {row.reference_score?.toFixed(6) ?? "—"} ({row.reference_decision ?? "unavailable"})</div>
              <div>Eligible for recovery: {row.eligible === undefined ? "—" : row.eligible ? "yes" : "no"}</div>
              <div>Reconstruction error: {row.reconstruction_error?.toPrecision(6) ?? "—"}</div>
              <div>Latent distance: {row.latent_distance?.toPrecision(6) ?? "—"}</div>
              <div>Benign anomaly rank: {row.anomaly_rank?.toFixed(6) ?? "—"}</div>
              <div>Fusion score / recovery cutoff: {row.fusion_score?.toFixed(6) ?? "—"} / {row.recovery_threshold?.toFixed(6) ?? "—"}</div>
            </dl>
          </details>)}</div>
          <div className="flex gap-4">
            <button className="underline disabled:opacity-40" disabled={after === 0} onClick={() => setAfter(0)}>First page</button>
            <button className="underline disabled:opacity-40" disabled={!rows.data?.next_after} onClick={() => setAfter(rows.data!.next_after!)}>Next page</button>
          </div>
          <p className="text-xs text-[#939AA6]">Anomaly rank and fusion score are not calibrated attack probabilities. Existing TreeSHAP explanations describe live LightGBM, not the anomaly recovery branch.</p>
        </section>
      </>}
    </>}
  </div>;
}

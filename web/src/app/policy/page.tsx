"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { fetchModelSummary, fetchPolicyReport } from "@/lib/api";

const pct = (value: number | null) => value === null ? "Unavailable" : `${(value * 100).toFixed(2)}%`;
const panel = "bg-[#13151A] border border-[#282C35] rounded-sm p-5 space-y-4";

export default function PolicyPage() {
  const [threshold, setThreshold] = useState<number | null>(null);
  const [prevalence, setPrevalence] = useState(1);
  const model = useQuery({ queryKey: ["modelSummary"], queryFn: fetchModelSummary });
  const cutoff = threshold ?? model.data?.decision_threshold ?? 0.5;
  const [settledCutoff, setSettledCutoff] = useState(cutoff);
  useEffect(() => {
    const timer = setTimeout(() => setSettledCutoff(cutoff), 200);
    return () => clearTimeout(timer);
  }, [cutoff]);
  const report = useQuery({
    queryKey: ["policy", model.data?.bundle_version, settledCutoff],
    queryFn: () => fetchPolicyReport(model.data!.bundle_version, settledCutoff),
    enabled: !!model.data,
  });
  const data = report.data;
  const candidate = data?.candidate;
  const projectedTP = candidate?.recall != null ? candidate.recall * prevalence * 10 : null;
  const projectedFP = candidate?.false_positive_rate != null
    ? candidate.false_positive_rate * (1000 - prevalence * 10) : null;
  const projectedPrecision = projectedTP !== null && projectedFP !== null && projectedTP + projectedFP > 0
    ? projectedTP / (projectedTP + projectedFP) : null;

  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ ...data, exported_at: new Date().toISOString() }, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "nexus-policy-evidence.json"; link.click();
    URL.revokeObjectURL(url);
  }

  return <div className="space-y-6 text-sm">
    <div>
      <p className="text-xs uppercase tracking-widest text-emerald-400 mb-2">Replay policy lab</p>
      <h1 className="text-xl font-semibold">What does another alert actually buy?</h1>
      <p className="text-[#939AA6] mt-2">Compare the recorded detector decisions with a hypothetical score threshold on the same labeled flows.</p>
      <p className="text-amber-300 text-xs mt-2">Exploration only. This does not run fusion, calibrate a release, or change the serving model.</p>
    </div>
    {(model.error || report.error) && <p role="alert" className="text-red-400">{(model.error || report.error)?.message}</p>}
    {model.isLoading && <p role="status">Loading serving model…</p>}
    {model.data && <section className={panel}>
      <div className="flex flex-wrap justify-between gap-3">
        <label htmlFor="cutoff">Hypothetical threshold: <strong>{cutoff.toFixed(4)}</strong></label>
        <button className="text-emerald-400 underline" onClick={() => setThreshold(null)}>Reset to release threshold</button>
      </div>
      <input id="cutoff" type="range" min="0" max="1" step="0.001" value={cutoff}
        className="w-full accent-emerald-400" onChange={e => setThreshold(Number(e.target.value))} />
      <p className="text-xs text-[#939AA6]">Lower thresholds usually catch more attacks and produce more false alerts. Scores are not guarantees of attack probability.</p>
      <p className="font-mono text-xs">Bundle: {model.data.bundle_version} · Release threshold: {model.data.decision_threshold.toFixed(6)}</p>
    </section>}
    {report.isFetching && <p role="status">Computing replay comparison…</p>}
    {data && <>
      <section className={panel}>
        <p className="text-xs text-[#939AA6]">Results at threshold {data.threshold.toFixed(4)}</p>
        <div className="flex flex-wrap justify-between gap-3">
          <p>{data.labeled_predictions} labeled / {data.total_predictions} recorded flows · {data.unlabeled_predictions} excluded without labels</p>
          <div className="flex gap-4">
            <button className="underline" onClick={() => report.refetch()}>Refresh replay</button>
            <button className="text-emerald-400 underline" onClick={download}>Export evidence JSON</button>
          </div>
        </div>
        {data.labeled_predictions === 0 ? <div className="space-y-2 text-[#939AA6]">
          <p>No labeled replay is available for this bundle. CSV detection alone does not attach truth labels.</p>
          <p>Run the replay command from docs/LOCAL_DEMO.md, then refresh. <Link href="/traffic" className="underline">Open traffic ingestion</Link>.</p>
        </div> : <>
          <div className="overflow-x-auto"><table className="w-full text-left">
            <thead><tr className="text-[#939AA6]"><th className="py-2">Metric</th><th>Recorded baseline</th><th>Hypothetical policy</th></tr></thead>
            <tbody>{[
              ["Recall", pct(data.baseline.recall), pct(data.candidate.recall)],
              ["Precision", pct(data.baseline.precision), pct(data.candidate.precision)],
              ["Benign false-positive rate", pct(data.baseline.false_positive_rate), pct(data.candidate.false_positive_rate)],
              ["Alerts on labeled flows", data.baseline.alert_count, data.candidate.alert_count],
              ["Missed attacks", data.baseline.false_negatives, data.candidate.false_negatives],
            ].map(row => <tr key={String(row[0])} className="border-t border-[#282C35]">{row.map((cell, i) => <td key={i} className="py-3">{cell}</td>)}</tr>)}</tbody>
          </table></div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[["Attacks recovered", data.recovered_attacks], ["Detections lost", data.lost_attacks],
              ["Extra false alerts", data.added_false_positives], ["False alerts removed", data.removed_false_positives]].map(([label, count]) =>
              <div key={label} className="bg-[#0A0B0D] p-3"><p className="text-xs text-[#939AA6]">{label}</p><p className="text-2xl mt-1">{count}</p></div>)}
          </div>
        </>}
      </section>
      {data.labeled_predictions > 0 && <section className={panel}>
        <h2 className="font-semibold">Attack-family coverage</h2>
        <p className="text-xs text-[#939AA6]">Dataset labels joined after scoring; these are not predicted families or proof of unseen-attack detection.</p>
        <div className="overflow-x-auto"><table className="w-full text-left">
          <thead><tr><th className="py-2">Family</th><th>Flows</th><th>Baseline recall</th><th>Hypothetical recall</th></tr></thead>
          <tbody>{data.families.map(f => <tr key={f.family} className="border-t border-[#282C35]">
            <td className="py-3">{f.family}</td><td>{f.total}</td><td>{pct(f.baseline_detected / f.total)}</td><td>{pct(f.candidate_detected / f.total)}</td>
          </tr>)}</tbody>
        </table></div>
      </section>}
      <section className={panel}>
        <h2 className="font-semibold">Would an analyst be overwhelmed?</h2>
        <label htmlFor="prevalence">Assumed attack prevalence: {prevalence}% of incoming flows</label>
        <input id="prevalence" type="range" min="0.1" max="20" step="0.1" value={prevalence}
          className="w-full accent-emerald-400" onChange={e => setPrevalence(Number(e.target.value))} />
        <p>Per 1,000 incoming flows: <strong>{projectedTP?.toFixed(1) ?? "—"}</strong> detected attacks and <strong>{projectedFP?.toFixed(1) ?? "—"}</strong> false alerts. Projected precision: <strong>{pct(projectedPrecision)}</strong>.</p>
        <p className="text-xs text-[#939AA6]">Illustration assuming replay recall and benign FPR transfer unchanged. Real deployment and drift can invalidate this assumption; missing denominators remain unavailable.</p>
      </section>
      <p className="text-xs text-[#939AA6]">{data.note} Cohort: all recorded predictions for this bundle; snapshot ends at sequence {data.last_sequence}.</p>
    </>}
  </div>;
}

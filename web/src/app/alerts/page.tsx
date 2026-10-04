"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchAlerts, fetchStats } from "@/lib/api";
import {
  Clock,
  CheckCircle2,
  ArrowRight,
  RefreshCw,
} from "lucide-react";

export default function AlertsPage() {
  const router = useRouter();
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [pageAfter, setPageAfter] = useState<number | undefined>(undefined);
  const [pageHistory, setPageHistory] = useState<number[]>([]);

  // Fetch summary stats
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
    refetchInterval: 10000,
  });

  // Fetch alerts
  const {
    data: alertPage,
    isLoading: alertsLoading,
    isRefetching,
    refetch: refetchAlerts,
  } = useQuery({
    queryKey: ["alerts", severityFilter, pageAfter],
    queryFn: () =>
      fetchAlerts({
        limit: 25,
        after: pageAfter,
        severity: severityFilter !== "all" ? severityFilter : undefined,
      }),
  });

  const alerts = alertPage?.items || [];

  // Filter client-side for source if selected
  const displayedAlerts = alerts.filter((a) => {
    if (sourceFilter !== "all" && a.source !== sourceFilter) return false;
    return true;
  });

  // Keyboard navigation: j / k / Enter
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < displayedAlerts.length - 1 ? prev + 1 : prev
        );
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
      } else if (e.key === "Enter") {
        if (displayedAlerts[selectedIndex]) {
          router.push(`/alerts/${displayedAlerts[selectedIndex].alert_id}`);
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [displayedAlerts, selectedIndex, router]);

  function handleNextPage() {
    if (alertPage?.next_after) {
      setPageHistory((prev) => [...prev, pageAfter || 0]);
      setPageAfter(alertPage.next_after);
      setSelectedIndex(0);
    }
  }

  function handlePrevPage() {
    if (pageHistory.length > 0) {
      const prev = [...pageHistory];
      const last = prev.pop();
      setPageHistory(prev);
      setPageAfter(last === 0 ? undefined : last);
      setSelectedIndex(0);
    }
  }

  return (
    <div className="space-y-5">
      {/* 1. Header & Summary Stats */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-[#F1F3F6]">
            Alert Queue & Triage
          </h1>
          <p className="text-xs text-[#939AA6]">
            Binary intrusion classification queue (threshold shown per model result)
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => {
              refetchStats();
              refetchAlerts();
            }}
            disabled={isRefetching}
            className="flex items-center space-x-1 px-2.5 py-1 text-xs border border-[#282C35] bg-[#181B21] rounded-sm hover:bg-[#1D2027] text-[#F1F3F6] disabled:opacity-50"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isRefetching ? "animate-spin" : ""}`}
            />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Summary Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">Alerts (24h)</div>
          <div className="text-xl font-semibold text-[#F1F3F6] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.alerts_24h.toLocaleString()}
          </div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">Awaiting Review</div>
          <div className="text-xl font-semibold text-[#FBBF24] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.awaiting_review.toLocaleString()}
          </div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">Confirmed Attacks</div>
          <div className="text-xl font-semibold text-[#F87171] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.confirmed_attacks.toLocaleString()}
          </div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">False Positives</div>
          <div className="text-xl font-semibold text-[#34D399] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.false_positives.toLocaleString()}
          </div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">Investigating</div>
          <div className="text-xl font-semibold text-[#939AA6] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.needs_investigation.toLocaleString()}
          </div>
        </div>

        <div className="bg-[#13151A] border border-[#282C35] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#939AA6]">Total Ingested (24h)</div>
          <div className="text-xl font-semibold text-[#F1F3F6] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.total_predictions_24h.toLocaleString()}
          </div>
        </div>
      </div>

      {/* 2. Filters & Table Controls */}
      <div className="bg-[#13151A] border border-[#282C35] rounded-sm p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-1.5">
            <span className="text-[#939AA6]">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => {
                setSeverityFilter(e.target.value);
                setPageAfter(undefined);
                setPageHistory([]);
              }}
              className="bg-[#1D2027] border border-[#282C35] rounded-sm px-2 py-1 text-xs text-[#F1F3F6] focus:outline-hidden"
            >
              <option value="all">All Severities</option>
              <option value="alert">Alert (Single Tier)</option>
              <option value="critical">Critical (Legacy Mock)</option>
              <option value="high">High (Legacy Mock)</option>
              <option value="medium">Medium (Legacy Mock)</option>
              <option value="low">Low (Legacy Mock)</option>
            </select>
          </div>

          <div className="flex items-center space-x-1.5">
            <span className="text-[#939AA6]">Source:</span>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="bg-[#1D2027] border border-[#282C35] rounded-sm px-2 py-1 text-xs text-[#F1F3F6] focus:outline-hidden"
            >
              <option value="all">All Sources</option>
              <option value="model">Model Inference</option>
              <option value="mock">MOCK Only</option>
            </select>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-[#939AA6] text-[11px]">
          <span>Navigate with</span>
          <kbd className="px-1.5 py-0.5 bg-[#1D2027] border border-[#282C35] font-mono rounded-sm text-[#F1F3F6]">
            j
          </kbd>
          <kbd className="px-1.5 py-0.5 bg-[#1D2027] border border-[#282C35] font-mono rounded-sm text-[#F1F3F6]">
            k
          </kbd>
          <span>and</span>
          <kbd className="px-1.5 py-0.5 bg-[#1D2027] border border-[#282C35] font-mono rounded-sm text-[#F1F3F6]">
            Enter
          </kbd>
        </div>
      </div>

      {/* 3. Dense SOC Alert Table */}
      <div className="bg-[#13151A] border border-[#282C35] rounded-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#181B21] border-b border-[#282C35] text-[#939AA6] font-medium">
                <th className="py-2.5 px-3 w-12 text-center">#</th>
                <th className="py-2.5 px-3">Alert ID</th>
                <th className="py-2.5 px-3">Created</th>
                <th className="py-2.5 px-3">Severity</th>
                <th className="py-2.5 px-3">Source</th>
                <th className="py-2.5 px-3">Score / Threshold</th>
                <th className="py-2.5 px-3">Predicted Category</th>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#282C35]">
              {alertsLoading ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#939AA6]">
                    Loading alert queue...
                  </td>
                </tr>
              ) : displayedAlerts.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#939AA6]">
                    No alerts found in queue matching the filters.
                  </td>
                </tr>
              ) : (
                displayedAlerts.map((alert, idx) => {
                  const isSelected = idx === selectedIndex;
                  const score = alert.score ?? alert.probability;
                  const threshold = alert.threshold;

                  return (
                    <tr
                      key={alert.alert_id}
                      onClick={() => setSelectedIndex(idx)}
                      onDoubleClick={() => router.push(`/alerts/${alert.alert_id}`)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-[#1D2027] ring-1 ring-inset ring-[#F1F3F6]/20"
                          : "hover:bg-[#181B21]/60"
                      }`}
                    >
                      <td className="py-2.5 px-3 text-center text-[#939AA6] font-mono text-[11px] tabular-nums">
                        {alert.sequence}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-[11px] text-[#F1F3F6]">
                        {alert.alert_id.slice(0, 8)}...{alert.alert_id.slice(-4)}
                      </td>

                      <td className="py-2.5 px-3 text-[#939AA6] tabular-nums whitespace-nowrap text-[11px]">
                        {alert.created_at.replace("T", " ").replace("Z", "")}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded-xs text-[10px] font-semibold uppercase bg-[#2A1316] text-[#F87171] border border-[#5C1D24]">
                          {alert.severity}
                        </span>
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        {alert.source === "mock" ? (
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded-xs text-[10px] font-semibold uppercase bg-[#2B1F0B] text-[#FBBF24] border border-[#5E4012]">
                            MOCK
                          </span>
                        ) : (
                          <span className="text-[#939AA6] text-[11px] font-mono">
                            model
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap font-mono text-[11px] tabular-nums">
                        {score !== null && score !== undefined ? (
                          <div className="flex items-center space-x-2">
                            <span>{score.toFixed(4)}</span>
                            <span className="text-[#616875]">/</span>
                            <span className="text-[#939AA6]">{threshold?.toFixed(4) ?? "Unavailable"}</span>
                          </div>
                        ) : (
                          "—"
                        )}
                      </td>

                      <td className="py-2.5 px-3 font-medium text-[#F1F3F6]">
                        {alert.predicted_class}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        {alert.feedback_version === 0 ? (
                          <span className="text-[#FBBF24] text-[11px] font-medium flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>Awaiting Review</span>
                          </span>
                        ) : (
                          <span className="text-[#34D399] text-[11px] font-medium flex items-center space-x-1">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>v{alert.feedback_version} Reviewed</span>
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <Link
                          href={`/alerts/${alert.alert_id}`}
                          className="inline-flex items-center space-x-1 text-xs font-medium text-[#F1F3F6] hover:underline"
                        >
                          <span>Review</span>
                          <ArrowRight className="w-3 h-3" />
                        </Link>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination footer */}
        <div className="border-t border-[#282C35] px-3 py-2 flex items-center justify-between text-xs text-[#939AA6]">
          <div>
            Showing {displayedAlerts.length} alerts
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handlePrevPage}
              disabled={pageHistory.length === 0}
              className="px-2.5 py-1 border border-[#282C35] rounded-sm hover:bg-[#1D2027] text-[#F1F3F6] disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={handleNextPage}
              disabled={!alertPage?.next_after}
              className="px-2.5 py-1 border border-[#282C35] rounded-sm hover:bg-[#1D2027] text-[#F1F3F6] disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

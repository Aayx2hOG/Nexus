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
          <h1 className="text-lg font-semibold tracking-tight text-[#1C1C1A]">
            Alert Queue & Triage
          </h1>
          <p className="text-xs text-[#6B6966]">
            Binary intrusion classification queue (threshold: 0.5777)
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => {
              refetchStats();
              refetchAlerts();
            }}
            disabled={isRefetching}
            className="flex items-center space-x-1 px-2.5 py-1 text-xs border border-[#E2E2DD] bg-white rounded-sm hover:bg-[#FAFAF8] text-[#1C1C1A] disabled:opacity-50"
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
        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">Alerts (24h)</div>
          <div className="text-xl font-semibold text-[#1C1C1A] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.alerts_24h.toLocaleString()}
          </div>
        </div>

        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">Awaiting Review</div>
          <div className="text-xl font-semibold text-[#B45309] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.awaiting_review.toLocaleString()}
          </div>
        </div>

        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">Confirmed Attacks</div>
          <div className="text-xl font-semibold text-[#B91C1C] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.confirmed_attacks.toLocaleString()}
          </div>
        </div>

        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">False Positives</div>
          <div className="text-xl font-semibold text-[#047857] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.false_positives.toLocaleString()}
          </div>
        </div>

        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">Investigating</div>
          <div className="text-xl font-semibold text-[#6B6966] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.needs_investigation.toLocaleString()}
          </div>
        </div>

        <div className="bg-white border border-[#E2E2DD] p-3 rounded-sm">
          <div className="text-[11px] font-medium text-[#6B6966]">Total Ingested (24h)</div>
          <div className="text-xl font-semibold text-[#1C1C1A] tabular-nums mt-0.5">
            {statsLoading ? "—" : stats?.total_predictions_24h.toLocaleString()}
          </div>
        </div>
      </div>

      {/* 2. Filters & Table Controls */}
      <div className="bg-white border border-[#E2E2DD] rounded-sm p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-1.5">
            <span className="text-[#6B6966]">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => {
                setSeverityFilter(e.target.value);
                setPageAfter(undefined);
                setPageHistory([]);
              }}
              className="bg-[#FAFAF8] border border-[#E2E2DD] rounded-sm px-2 py-1 text-xs text-[#1C1C1A] focus:outline-hidden"
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
            <span className="text-[#6B6966]">Source:</span>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="bg-[#FAFAF8] border border-[#E2E2DD] rounded-sm px-2 py-1 text-xs text-[#1C1C1A] focus:outline-hidden"
            >
              <option value="all">All Sources</option>
              <option value="model">Model Inference</option>
              <option value="mock">MOCK Only</option>
            </select>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-[#6B6966] text-[11px]">
          <span>Navigate with</span>
          <kbd className="px-1.5 py-0.5 bg-[#FAFAF8] border border-[#E2E2DD] font-mono rounded-sm text-[#1C1C1A]">
            j
          </kbd>
          <kbd className="px-1.5 py-0.5 bg-[#FAFAF8] border border-[#E2E2DD] font-mono rounded-sm text-[#1C1C1A]">
            k
          </kbd>
          <span>and</span>
          <kbd className="px-1.5 py-0.5 bg-[#FAFAF8] border border-[#E2E2DD] font-mono rounded-sm text-[#1C1C1A]">
            Enter
          </kbd>
        </div>
      </div>

      {/* 3. Dense SOC Alert Table */}
      <div className="bg-white border border-[#E2E2DD] rounded-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-[#FAFAF8] border-b border-[#E2E2DD] text-[#6B6966] font-medium">
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
            <tbody className="divide-y divide-[#E2E2DD]">
              {alertsLoading ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#6B6966]">
                    Loading alert queue...
                  </td>
                </tr>
              ) : displayedAlerts.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-12 text-center text-[#6B6966]">
                    No alerts found in queue matching the filters.
                  </td>
                </tr>
              ) : (
                displayedAlerts.map((alert, idx) => {
                  const isSelected = idx === selectedIndex;
                  const score = alert.score ?? alert.probability;
                  const threshold = alert.threshold ?? 0.5777;

                  return (
                    <tr
                      key={alert.alert_id}
                      onClick={() => setSelectedIndex(idx)}
                      onDoubleClick={() => router.push(`/alerts/${alert.alert_id}`)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? "bg-[#FAFAF8] ring-1 ring-inset ring-[#1C1C1A]/20"
                          : "hover:bg-[#FAFAF8]/50"
                      }`}
                    >
                      <td className="py-2.5 px-3 text-center text-[#6B6966] font-mono text-[11px] tabular-nums">
                        {alert.sequence}
                      </td>

                      <td className="py-2.5 px-3 font-mono text-[11px] text-[#1C1C1A]">
                        {alert.alert_id.slice(0, 8)}...{alert.alert_id.slice(-4)}
                      </td>

                      <td className="py-2.5 px-3 text-[#6B6966] tabular-nums whitespace-nowrap text-[11px]">
                        {alert.created_at.replace("T", " ").replace("Z", "")}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span className="inline-flex items-center px-1.5 py-0.5 rounded-xs text-[10px] font-semibold uppercase bg-[#FEF2F2] text-[#B91C1C] border border-[#FCA5A5]">
                          {alert.severity}
                        </span>
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        {alert.source === "mock" ? (
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded-xs text-[10px] font-semibold uppercase bg-[#FFFBEB] text-[#B45309] border border-[#FCD34D]">
                            MOCK
                          </span>
                        ) : (
                          <span className="text-[#6B6966] text-[11px] font-mono">
                            model
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap font-mono text-[11px] tabular-nums">
                        {score !== null && score !== undefined ? (
                          <div className="flex items-center space-x-2">
                            <span>{score.toFixed(4)}</span>
                            <span className="text-[#6B6966]">/</span>
                            <span className="text-[#6B6966]">{threshold.toFixed(4)}</span>
                          </div>
                        ) : (
                          "—"
                        )}
                      </td>

                      <td className="py-2.5 px-3 font-medium text-[#1C1C1A]">
                        {alert.predicted_class}
                      </td>

                      <td className="py-2.5 px-3 whitespace-nowrap">
                        {alert.feedback_version === 0 ? (
                          <span className="text-[#B45309] text-[11px] font-medium flex items-center space-x-1">
                            <Clock className="w-3 h-3" />
                            <span>Awaiting Review</span>
                          </span>
                        ) : (
                          <span className="text-[#047857] text-[11px] font-medium flex items-center space-x-1">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>v{alert.feedback_version} Reviewed</span>
                          </span>
                        )}
                      </td>

                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <Link
                          href={`/alerts/${alert.alert_id}`}
                          className="inline-flex items-center space-x-1 text-xs font-medium text-[#1C1C1A] hover:underline"
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
        <div className="border-t border-[#E2E2DD] px-3 py-2 flex items-center justify-between text-xs text-[#6B6966]">
          <div>
            Showing {displayedAlerts.length} alerts
          </div>

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
              disabled={!alertPage?.next_after}
              className="px-2.5 py-1 border border-[#E2E2DD] rounded-sm hover:bg-[#FAFAF8] disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

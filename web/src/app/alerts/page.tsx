"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Filter, RefreshCw } from "lucide-react";
import {
  Button,
  EmptyState,
  ErrorState,
  InfoHelp,
  LoadingState,
  PageHeader,
} from "@/components/components";
import { api, isPreviewData } from "@/lib/api";
import type { ReviewStatus, Severity } from "@/lib/types";

const statusTone = (status: ReviewStatus) => {
  if (status === "Confirmed Attack") return "danger";
  if (status === "False Positive") return "success";
  if (status === "Investigating") return "warning";
  return "neutral";
};

export default function AlertsPage() {
  const router = useRouter();
  const alertsQuery = useQuery({ queryKey: ["alerts"], queryFn: api.alerts, refetchInterval: 15_000 });
  const statsQuery = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 15_000 });
  const [statusFilter, setStatusFilter] = useState("All statuses");
  const [severityFilter, setSeverityFilter] = useState("All severities");
  const [selected, setSelected] = useState(0);
  const [page, setPage] = useState(1);

  const alerts = useMemo(
    () =>
      (alertsQuery.data ?? []).filter(
        (alert) =>
          (statusFilter === "All statuses" || alert.reviewStatus === statusFilter) &&
          (severityFilter === "All severities" || alert.severity === severityFilter),
      ),
    [alertsQuery.data, statusFilter, severityFilter],
  );

  const rowsPerPage = 10;
  const pageCount = Math.max(1, Math.ceil(alerts.length / rowsPerPage));
  const visibleAlerts = alerts.slice((page - 1) * rowsPerPage, page * rowsPerPage);

  useEffect(() => {
    setPage(1);
    setSelected(0);
  }, [severityFilter, statusFilter, alertsQuery.data]);

  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes((event.target as HTMLElement).tagName)) return;
      if (event.key.toLowerCase() === "j") setSelected((value) => Math.min(value + 1, visibleAlerts.length - 1));
      if (event.key.toLowerCase() === "k") setSelected((value) => Math.max(value - 1, 0));
      if (event.key === "Enter" && visibleAlerts[selected]) router.push(`/alerts/${visibleAlerts[selected].id}`);
    };
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  }, [router, selected, visibleAlerts]);

  const refresh = () => {
    void alertsQuery.refetch();
    void statsQuery.refetch();
  };

  const stats = statsQuery.data;
  const counterItems = stats
    ? [
        {
          label: "Critical",
          value: (alertsQuery.data ?? []).filter((alert) => alert.severity === "Critical").length,
          help: "Alerts assigned the highest available severity level.",
        },
        {
          label: "Investigating",
          value: stats.investigating,
          help: "Alerts still being reviewed with no final decision recorded.",
        },
        {
          label: "Confirmed",
          value: stats.confirmedAttacks,
          help: "Alerts an analyst reviewed and determined represent attacks.",
        },
        {
          label: "False positive",
          value: stats.falsePositives,
          help: "Normal network activity that NEXUS incorrectly flagged as suspicious.",
        },
        {
          label: "Total flows",
          value: stats.totalPredictions,
          help: "All network flows scored by the detector in the reported period.",
        },
      ]
    : [];

  return (
    <div className="page" data-preview={isPreviewData() ? "true" : undefined}>
      <PageHeader
        title="Alerts"
        description="Network activity that NEXUS has flagged as suspicious and sent to an analyst for review."
        actions={
          <>
            {isPreviewData() && <span className="demo-inline">DEMO DATA</span>}
            <Button onClick={refresh}>
              <RefreshCw size={15} /> Refresh
            </Button>
          </>
        }
      />
      <section className="ops-summary" aria-label="Alert statistics">
        {statsQuery.isLoading ? (
          <div className="ops-summary-loading">Reading detection counters</div>
        ) : stats ? (
          <>
            <div className="ops-primary">
              <span>
                Alerts <InfoHelp label="Alert" text="A network flow whose model score crossed the decision threshold and was sent for analyst review." />
              </span>
              <div><strong className="mono">{stats.awaitingReview + stats.investigating}</strong><small>ACTIVE</small></div>
              <em className="mono">{stats.last24Hours} in 24h</em>
            </div>
            <div className="ops-counters">
              {counterItems.map((item) => (
                <div className="ops-counter" key={item.label}>
                  <span>{item.label} <InfoHelp label={item.label} text={item.help} /></span>
                  <strong className="mono">{item.value}</strong>
                </div>
              ))}
            </div>
          </>
        ) : (
          <span className="ops-unavailable">Counters unavailable</span>
        )}
      </section>

      <div className="section-heading">
        <div>
          <h2>Open alerts</h2>
          <p>Alerts that still need attention, newest first.</p>
        </div>
        <div className="shortcut-help">
          <span><kbd>J</kbd><kbd>K</kbd> Navigate</span>
          <span><kbd>Enter</kbd> Open</span>
        </div>
      </div>
      <div className="filter-bar">
        <div className="filter-label"><Filter size={15} /> Filters</div>
        <label>
          <span className="sr-only">Severity</span>
          <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
            <option>All severities</option>
            <option>Critical</option>
            <option>High</option>
            <option>Medium</option>
            <option>Low</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Review status</span>
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option>All statuses</option>
            <option>Pending Review</option>
            <option>Investigating</option>
            <option>Confirmed Attack</option>
            <option>False Positive</option>
          </select>
        </label>
        {(statusFilter !== "All statuses" || severityFilter !== "All severities") && (
          <Button
            onClick={() => {
              setStatusFilter("All statuses");
              setSeverityFilter("All severities");
            }}
          >
            Clear filters
          </Button>
        )}
        <span className="record-count">{alerts.length} records</span>
      </div>

      {alertsQuery.isLoading ? (
        <LoadingState label="Loading alert queue" />
      ) : alertsQuery.isError ? (
        <ErrorState retry={() => void alertsQuery.refetch()} />
      ) : alerts.length === 0 ? (
        <EmptyState title="No alerts require review." detail="The current alert queue is clear for these filters." />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Alert</th>
                <th>Source</th>
                <th>Destination</th>
                <th>Score</th>
                <th>Threshold</th>
                <th>Type</th>
                <th>Status</th>
                <th><span className="sr-only">Action</span></th>
              </tr>
            </thead>
            <tbody>
              {visibleAlerts.map((alert, index) => (
                <tr
                  key={alert.id}
                  className={selected === index ? "selected-row" : ""}
                  onClick={() => setSelected(index)}
                  onDoubleClick={() => router.push(`/alerts/${alert.id}`)}
                >
                  <td className="mono time-cell" title={new Date(alert.createdAt).toLocaleString()}>
                    {new Date(alert.createdAt).toLocaleTimeString([], { hour12: false })}
                  </td>
                  <td className={`mono alert-cell severity-${alert.severity.toLowerCase()}`}>
                    <strong>{alert.id}</strong>
                    <small>{alert.severity.toUpperCase()}</small>
                  </td>
                  <td className="mono">{alert.source}</td>
                  <td className="mono">{alert.destination ?? "Not provided"}</td>
                  <td className="mono score-cell">{alert.score.toFixed(4)}</td>
                  <td className="mono">{alert.threshold.toFixed(4)}</td>
                  <td>{alert.category}</td>
                  <td className={`queue-status queue-status-${statusTone(alert.reviewStatus)}`}>
                    {alert.reviewStatus === "Pending Review" ? "OPEN" : alert.reviewStatus.toUpperCase()}
                  </td>
                  <td>
                    <button className="row-action" onClick={() => router.push(`/alerts/${alert.id}`)} aria-label={`Open ${alert.id}`}>
                      <ArrowRight size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {alerts.length > rowsPerPage && (
        <div className="pagination">
          <span className="mono">PAGE {page} / {pageCount}</span>
          <div>
            <Button disabled={page === 1} onClick={() => { setPage((value) => value - 1); setSelected(0); }}>Previous</Button>
            <Button disabled={page === pageCount} onClick={() => { setPage((value) => value + 1); setSelected(0); }}>Next</Button>
          </div>
        </div>
      )}
    </div>
  );
}

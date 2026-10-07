"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";
import { Button } from "@/components/components";
import { AppearanceSwitcher, DataModeSwitcher, NexusMark } from "@/components/Shell";
import { api, isPreviewData } from "@/lib/api";

export default function HomePage() {
  const router = useRouter();
  const [entering, setEntering] = useState(false);
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 30_000 });

  const enterApplication = () => {
    setEntering(true);
    window.setTimeout(() => router.push("/alerts"), 180);
  };

  return (
    <main className={`home-page ${entering ? "home-entering" : ""}`}>
      <div className="home-toolbar">
        {isPreviewData() && <span className="demo-indicator">DEMO DATA</span>}
        <DataModeSwitcher />
        <AppearanceSwitcher />
      </div>
      <div className="home-intro">
        <div className="home-identity">
          <NexusMark />
        </div>
        <span className="home-kicker">NETWORK INTRUSION DETECTION</span>
        <h1>A security workspace for suspicious network activity.</h1>
        <p>Review alerts, inspect the evidence behind each model decision, and record analyst findings.</p>
        <Button variant="primary" className="home-entry" onClick={enterApplication}>
          Enter NEXUS <ArrowRight size={15} />
        </Button>
      </div>
      <div className="home-status" aria-live="polite">
        {health.isLoading ? (
          <span className="mono">Checking system status</span>
        ) : health.data && health.data.api !== "Unavailable" ? (
          <>
            <div><span>System</span><strong>{health.data.system}</strong></div>
            <div><span>API</span><strong>{health.data.api}</strong></div>
            <div><span>Model</span><strong>{health.data.model}</strong></div>
            {stats.data && <div><span>Awaiting review</span><strong className="mono">{stats.data.awaitingReview}</strong></div>}
          </>
        ) : (
          <div className="home-status-error">
            <span className="home-api-error">NEXUS API unavailable</span>
            <Button
              onClick={() => {
                void health.refetch();
                void stats.refetch();
              }}
            >
              Try again
            </Button>
          </div>
        )}
      </div>
    </main>
  );
}

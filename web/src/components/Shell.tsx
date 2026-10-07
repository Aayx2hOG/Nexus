"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  FileSearch,
  Layers3,
  Menu,
  Network,
  ShieldAlert,
  X,
} from "lucide-react";
import { api, getDataMode, setDataMode, type DataMode } from "@/lib/api";
import { KeyboardHelpModal } from "./KeyboardHelpModal";

type Appearance = "original" | "dark" | "contrast";

function readAppearance(): Appearance {
  if (typeof window === "undefined") return "original";
  const saved = window.localStorage.getItem("nexus-appearance");
  return saved === "dark" || saved === "contrast" ? saved : "original";
}

export function useAppearance() {
  const [appearance, setAppearance] = useState<Appearance>("original");

  useEffect(() => {
    const initial = readAppearance();
    setAppearance(initial);
    document.documentElement.dataset.theme = initial;
  }, []);

  const updateAppearance = (next: Appearance) => {
    setAppearance(next);
    document.documentElement.dataset.theme = next;
    window.localStorage.setItem("nexus-appearance", next);
  };

  return { appearance, setAppearance: updateAppearance };
}

export function AppearanceSwitcher() {
  const { appearance, setAppearance } = useAppearance();
  return (
    <label className="appearance-switcher">
      <span>Appearance</span>
      <select
        aria-label="Appearance mode"
        value={appearance}
        onChange={(event) => setAppearance(event.target.value as Appearance)}
      >
        <option value="original">Original</option>
        <option value="dark">Dark</option>
        <option value="contrast">High Contrast</option>
      </select>
    </label>
  );
}

export function DataModeSwitcher() {
  const queryClient = useQueryClient();
  const [mode, setModeState] = useState<DataMode>("live");

  useEffect(() => {
    setModeState(getDataMode());
  }, []);

  const changeMode = (nextMode: DataMode) => {
    setDataMode(nextMode);
    setModeState(nextMode);
    window.dispatchEvent(new Event("nexus-data-mode-change"));
    void queryClient.resetQueries();
  };

  return (
    <label className="data-mode-switcher">
      <span>Data</span>
      <select
        aria-label="Data source"
        value={mode}
        onChange={(event) => changeMode(event.target.value as DataMode)}
      >
        <option value="live">Live Backend</option>
        <option value="demo">Demo Data</option>
      </select>
    </label>
  );
}

export function NexusMark() {
  return (
    <div className="brand" aria-label="NEXUS">
      <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
        <path d="M5 7L16 16M27 7L16 16M5 25L16 16M27 25L16 16" />
        <circle cx="5" cy="7" r="2" />
        <circle cx="27" cy="7" r="2" />
        <circle cx="5" cy="25" r="2" />
        <circle cx="27" cy="25" r="2" />
        <circle cx="16" cy="16" r="3.2" />
      </svg>
      <span>NEXUS</span>
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });

  const items = [
    { label: "Alerts", path: "/alerts", icon: ShieldAlert },
    { label: "Traffic", path: "/traffic", icon: Network },
    { label: "Missed Attack Review", path: "/review-sample", icon: FileSearch },
    { label: "Model and Manifest", path: "/model", icon: Layers3 },
  ];

  const status = health.data;
  const isHome = pathname === "/";

  // Close sidebar on navigation
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  if (isHome) {
    return <>{children}</>;
  }

  return (
    <div className="app-shell">
      {open && (
        <button
          className="sidebar-scrim"
          onClick={() => setOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="sidebar-top">
          <Link href="/" className="brand-home" aria-label="Open NEXUS home">
            <NexusMark />
          </Link>
          <button
            className="mobile-close"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X size={20} />
          </button>
        </div>
        <nav aria-label="Primary navigation">
          {items.map((item) => {
            const active =
              item.path === "/alerts"
                ? pathname.startsWith("/alerts")
                : pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`nav-item ${active ? "active" : ""}`}
              >
                <Icon size={17} strokeWidth={1.8} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="system-block">
          <div className="system-heading">
            <span>SYSTEM STATUS</span>
            <Activity size={14} />
          </div>
          {health.isLoading ? (
            <div className="health-loading">Checking health endpoint</div>
          ) : (
            [
              ["System", status?.system ?? "Operational"],
              ["API", status?.api === "Operational" ? "Connected" : status?.api ?? "Unavailable"],
              ["Model", status?.model === "Operational" ? "Ready" : status?.model ?? "Unavailable"],
            ].map(([label, value]) => (
              <div className="health-row" key={label}>
                <span>{label}</span>
                <span className={`health-value ${value.toLowerCase()}`}>
                  <i />
                  {value}
                </span>
              </div>
            ))
          )}
        </div>
      </aside>

      <main className="workspace">
        <div className="mobile-bar">
          <button onClick={() => setOpen(true)} aria-label="Open navigation">
            <Menu size={20} />
          </button>
          <NexusMark />
          <span className="mobile-status">
            <i /> Live
          </span>
        </div>
        <div className="global-toolbar">
          <DataModeSwitcher />
          <AppearanceSwitcher />
        </div>
        {children}
        <KeyboardHelpModal />
      </main>
    </div>
  );
}

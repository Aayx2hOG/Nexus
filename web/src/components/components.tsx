"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, Eye, Inbox, RefreshCw } from "lucide-react";

export function Button({
  children,
  variant = "default",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "primary" | "danger";
}) {
  return (
    <button className={`button button-${variant} ${className}`} {...props}>
      {children}
    </button>
  );
}

export function Status({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "danger" | "warning" | "success" | "info" | "neutral";
}) {
  return <span className={`status status-${tone}`}>{children}</span>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </header>
  );
}

export function LoadingState({ label = "Loading operational data" }: { label?: string }) {
  return (
    <div className="state-panel" role="status">
      <RefreshCw size={20} className="spin" />
      <strong>{label}</strong>
      <span>Waiting for the NEXUS API response.</span>
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="state-panel">
      <Inbox size={21} />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function ErrorState({ retry }: { retry?: () => void }) {
  return (
    <div className="state-panel state-error" role="alert">
      <AlertTriangle size={21} />
      <strong>NEXUS API unavailable.</strong>
      <span>Unable to retrieve this data. Check the API connection and try again.</span>
      {retry && <Button onClick={retry}>Retry request</Button>}
    </div>
  );
}

export function ScoreScale({ score, threshold }: { score: number; threshold: number }) {
  return (
    <div className="score-scale" aria-label={`Model score ${score}, threshold ${threshold}`}>
      <div className="scale-labels">
        <span>0.00 Normal region</span>
      </div>
      <div className="scale-track">
        <div className="threat-region" style={{ left: `${threshold * 100}%` }} />
        <div className="threshold-marker" style={{ left: `${threshold * 100}%` }}>
          <span>Threshold {threshold.toFixed(4)}</span>
        </div>
        <div className="score-marker" style={{ left: `${score * 100}%` }}>
          <span>Score {score.toFixed(4)}</span>
        </div>
        <div className="threat-region-tag">
          <span>Threat region 1.00</span>
        </div>
      </div>
    </div>
  );
}

export function TechnicalValue({ children }: { children: ReactNode }) {
  return <span className="mono">{children}</span>;
}

export function InfoHelp({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const root = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const closeOtherHelp = (event: Event) => {
      if ((event as CustomEvent<string>).detail !== id) setOpen(false);
    };
    const closeOnOutsideClick = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("nexus-help-open", closeOtherHelp);
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("nexus-help-open", closeOtherHelp);
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [id]);

  return (
    <span className="info-help" ref={root}>
      <button
        type="button"
        aria-label={`What does ${label} mean?`}
        aria-expanded={open}
        aria-controls={`${id}-explanation`}
        onClick={() => {
          if (!open) window.dispatchEvent(new CustomEvent("nexus-help-open", { detail: id }));
          setOpen((value) => !value);
        }}
      >
        <Eye size={12} />
      </button>
      {open && (
        <span className="info-help-popover" role="dialog" id={`${id}-explanation`}>
          <strong>{label}</strong>
          {text}
        </span>
      )}
    </span>
  );
}

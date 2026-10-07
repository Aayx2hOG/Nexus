"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";

export function KeyboardHelpModal() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "?" && !e.metaKey && !e.ctrlKey) {
        if (
          document.activeElement?.tagName === "INPUT" ||
          document.activeElement?.tagName === "TEXTAREA"
        ) {
          return;
        }
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  if (!open) return null;

  const shortcuts = [
    { key: "j", desc: "Select next row in table" },
    { key: "k", desc: "Select previous row in table" },
    { key: "Enter", desc: "Open selected alert detail view" },
    { key: "?", desc: "Toggle keyboard shortcuts help" },
    { key: "Esc", desc: "Dismiss modal or cancel" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs">
      <div className="bg-[var(--surface)] border border-[var(--border-strong)] rounded-sm shadow-xl max-w-md w-full mx-4 p-5 text-[var(--text)]">
        <div className="flex items-center justify-between border-b border-[var(--border)] pb-3 mb-4">
          <h3 className="text-sm font-semibold tracking-wide">Keyboard Navigation</h3>
          <button
            onClick={() => setOpen(false)}
            className="text-[var(--muted)] hover:text-[var(--text)] p-1 rounded-sm"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-2">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between text-xs py-1">
              <span className="text-[var(--text-secondary)]">{s.desc}</span>
              <kbd className="px-2 py-0.5 font-mono text-[11px] bg-[var(--bg-secondary)] border border-[var(--border-strong)] rounded-xs font-semibold">
                {s.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-5 pt-3 border-t border-[var(--border)] text-right">
          <button
            onClick={() => setOpen(false)}
            className="button"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

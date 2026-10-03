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
    { key: "1", desc: "Quick-select 'Confirmed Attack' in verdict form" },
    { key: "2", desc: "Quick-select 'False Positive' in verdict form" },
    { key: "3", desc: "Quick-select 'Needs Investigation' in verdict form" },
    { key: "?", desc: "Toggle keyboard shortcuts help" },
    { key: "Esc", desc: "Dismiss modal or cancel" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-xs">
      <div className="bg-white border border-[#E2E2DD] rounded-sm shadow-sm max-w-md w-full mx-4 p-5">
        <div className="flex items-center justify-between border-b border-[#E2E2DD] pb-3 mb-4">
          <h3 className="text-sm font-semibold text-[#1C1C1A]">Keyboard Navigation</h3>
          <button
            onClick={() => setOpen(false)}
            className="text-[#6B6966] hover:text-[#1C1C1A] p-1 rounded-sm hover:bg-[#FAFAF8]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-2">
          {shortcuts.map((s) => (
            <div key={s.key} className="flex items-center justify-between text-xs py-1">
              <span className="text-[#6B6966]">{s.desc}</span>
              <kbd className="px-2 py-0.5 font-mono text-[11px] bg-[#FAFAF8] border border-[#E2E2DD] text-[#1C1C1A] rounded-sm font-semibold">
                {s.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="mt-5 pt-3 border-t border-[#E2E2DD] text-[11px] text-[#6B6966] text-right">
          Press <kbd className="px-1.5 py-0.5 font-mono text-[10px] bg-[#FAFAF8] border border-[#E2E2DD] rounded-sm">Esc</kbd> to close
        </div>
      </div>
    </div>
  );
}

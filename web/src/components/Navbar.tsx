"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api";
import { HelpCircle } from "lucide-react";
import { useState } from "react";
import { KeyboardHelpModal } from "./KeyboardHelpModal";

export function Navbar() {
  const pathname = usePathname();
  const [showHelp, setShowHelp] = useState(false);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 15000,
  });

  const isReady = health?.status === "ready";

  const navLinks = [
    { href: "/alerts", label: "Alerts" },
    { href: "/review-sample", label: "Missed Attack Review" },
    { href: "/model", label: "Model & Manifest" },
  ];

  return (
    <>
      <header className="bg-white border-b border-[#E2E2DD] sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <Link href="/alerts" className="flex items-center space-x-2">
              <div className="w-7 h-7 bg-[#1C1C1A] text-white rounded-sm flex items-center justify-center font-bold text-xs">
                N
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="font-semibold text-sm tracking-tight text-[#1C1C1A]">
                  NEXUS
                </span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 bg-[#FAFAF8] border border-[#E2E2DD] text-[#6B6966] rounded-sm">
                  IDS
                </span>
              </div>
            </Link>

            <nav className="flex items-center space-x-1">
              {navLinks.map((link) => {
                const active = pathname.startsWith(link.href);
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`px-3 py-1.5 text-xs font-medium rounded-sm transition-colors ${
                      active
                        ? "bg-[#FAFAF8] text-[#1C1C1A] font-semibold border border-[#E2E2DD]"
                        : "text-[#6B6966] hover:text-[#1C1C1A] hover:bg-[#FAFAF8]"
                    }`}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center space-x-4">
            {/* System Readiness Indicator */}
            <div className="flex items-center space-x-2 text-xs font-mono text-[#6B6966] bg-[#FAFAF8] border border-[#E2E2DD] px-2.5 py-1 rounded-sm">
              <span
                className={`w-2 h-2 rounded-full ${
                  isReady ? "bg-[#047857]" : "bg-[#B45309]"
                }`}
              />
              <span className="text-[11px] tabular-nums">
                {isReady ? `Ready (${health.bundle_version})` : "Not Ready"}
              </span>
            </div>

            {/* Analyst ID */}
            <div className="text-xs text-[#6B6966] hidden md:block">
              Analyst: <span className="font-mono text-[#1C1C1A]">tier1-analyst</span>
            </div>

            {/* Keyboard Shortcuts Trigger */}
            <button
              onClick={() => setShowHelp(true)}
              className="p-1.5 text-[#6B6966] hover:text-[#1C1C1A] border border-[#E2E2DD] rounded-sm hover:bg-[#FAFAF8] transition-colors"
              title="Keyboard Shortcuts (?)"
            >
              <HelpCircle className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      <KeyboardHelpModal open={showHelp} onClose={() => setShowHelp(false)} />
    </>
  );
}

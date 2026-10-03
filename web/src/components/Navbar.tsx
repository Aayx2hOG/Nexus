"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: "/alerts", label: "Alerts" },
    { href: "/traffic", label: "Traffic & Ingestion" },
    { href: "/review-sample", label: "Missed Attack Review" },
    { href: "/model", label: "Model & Manifest" },
  ];

  return (
    <header className="bg-[#13151A] border-b border-[#282C35] sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <Link href="/alerts" className="flex items-center space-x-2">
            <div className="w-7 h-7 bg-[#282C35] text-[#F1F3F6] border border-[#3B414E] rounded-sm flex items-center justify-center font-bold text-xs">
              N
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="font-semibold text-sm tracking-tight text-[#F1F3F6]">
                NEXUS
              </span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 bg-[#1D2027] border border-[#282C35] text-[#939AA6] rounded-sm">
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
                      ? "bg-[#1D2027] text-[#F1F3F6] font-semibold border border-[#282C35]"
                      : "text-[#939AA6] hover:text-[#F1F3F6] hover:bg-[#181B21]"
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}

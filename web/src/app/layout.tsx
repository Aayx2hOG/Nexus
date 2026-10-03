import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "@/components/Providers";
import { Navbar } from "@/components/Navbar";
import { KeyboardHelpModal } from "@/components/KeyboardHelpModal";

export const metadata: Metadata = {
  title: "Nexus IDS Console",
  description: "Network intrusion detection SOC analyst review console",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full bg-[#0A0B0D] text-[#F1F3F6] dark">
      <body className="min-h-full flex flex-col font-sans bg-[#0A0B0D] text-[#F1F3F6]">
        <Providers>
          <Navbar />
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-5">
            {children}
          </main>
          <KeyboardHelpModal />
        </Providers>
      </body>
    </html>
  );
}

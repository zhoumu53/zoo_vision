"use client";

import { useEffect, useState } from "react";
import type { BBox } from "../../lib/types";

const BBOX_COLORS = [
  { border: "border-cyan-400", text: "text-cyan-400", hex: "#22d3ee" },
  { border: "border-amber-400", text: "text-amber-400", hex: "#fbbf24" },
  { border: "border-rose-400", text: "text-rose-400", hex: "#fb7185" },
  { border: "border-emerald-400", text: "text-emerald-400", hex: "#34d399" },
  { border: "border-purple-400", text: "text-purple-400", hex: "#c084fc" },
];

function getColor(idx: number) { return BBOX_COLORS[idx % BBOX_COLORS.length]; }

type PartialElephant = { index: number; bbox: BBox; crop_url: string };

export default function DetectionAnimation({
  previewUrl, progress, stage, partialElephants,
}: {
  previewUrl: string;
  progress: number;
  stage: string;
  partialElephants: PartialElephant[] | null;
}) {
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    if (partialElephants && partialElephants.length > 0 && !revealed) {
      const timer = setTimeout(() => setRevealed(true), 300);
      return () => clearTimeout(timer);
    }
  }, [partialElephants, revealed]);

  const stageLabels: Record<string, string> = {
    "Loading image...": "Reading your photo...",
    "Detecting elephants...": "Scanning for elephants...",
  };
  const displayStage = stageLabels[stage] || stage || "Analyzing...";

  return (
    <div className="w-full">
      <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] p-4 backdrop-blur-2xl">
        <div className="relative overflow-hidden rounded-2xl mb-4">
          <img src={previewUrl} alt="Your photo" className="w-full object-contain rounded-2xl" style={{ maxHeight: 280 }} />

          {!revealed && (
            <>
              <div className="pointer-events-none absolute inset-0 opacity-20" style={{
                background: "linear-gradient(180deg, transparent 0%, transparent 45%, rgba(34,211,238,0.3) 50%, transparent 55%, transparent 100%)",
                backgroundSize: "100% 200%",
                animation: "scanLine 2.5s ease-in-out infinite",
              }} />
              <div className="pointer-events-none absolute" style={{ width: "30%", height: "35%", animation: "viewfinderScan 4s ease-in-out infinite" }}>
                <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-cyan-400/80" style={{ boxShadow: "0 0 8px rgba(34,211,238,0.4)" }} />
                <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-cyan-400/80" style={{ boxShadow: "0 0 8px rgba(34,211,238,0.4)" }} />
                <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-cyan-400/80" style={{ boxShadow: "0 0 8px rgba(34,211,238,0.4)" }} />
                <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-cyan-400/80" style={{ boxShadow: "0 0 8px rgba(34,211,238,0.4)" }} />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                  <div className="w-3 h-[1px] bg-cyan-400/40" />
                  <div className="w-[1px] h-3 bg-cyan-400/40 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
              </div>
              <div className="pointer-events-none absolute inset-0 rounded-2xl" style={{
                background: "radial-gradient(ellipse at center, transparent 40%, rgba(5,5,16,0.3) 100%)",
                animation: "pulse 2s ease-in-out infinite",
              }} />
            </>
          )}

          {revealed && partialElephants && partialElephants.map((el, idx) => {
            const color = getColor(idx);
            const left = (el.bbox.x - el.bbox.w / 2) * 100;
            const top = (el.bbox.y - el.bbox.h / 2) * 100;
            const width = el.bbox.w * 100;
            const height = el.bbox.h * 100;
            return (
              <div key={idx} className={`absolute border-2 ${color.border}`} style={{
                left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%`,
                opacity: 0, transform: "scale(1.1)", animation: `bboxReveal 0.5s ease-out ${idx * 0.2}s forwards`,
              }}>
                <span className={`absolute -top-5 left-0 text-[10px] font-bold ${color.text}`} style={{ textShadow: "0 1px 4px rgba(0,0,0,0.8)" }}>
                  Elephant #{idx + 1}
                </span>
                <div className="absolute inset-0" style={{ boxShadow: `inset 0 0 12px ${color.hex}33, 0 0 12px ${color.hex}22` }} />
              </div>
            );
          })}
        </div>
        <p className="text-center text-sm font-light text-white/60 mb-1">
          {revealed && partialElephants ? `Found ${partialElephants.length} elephant${partialElephants.length !== 1 ? "s" : ""}!` : displayStage}
        </p>
        <p className="text-center text-[10px] text-white/30">{progress}%</p>
      </div>

      <style jsx>{`
        @keyframes viewfinderScan { 0% { top: 8%; left: 10%; } 18% { top: 30%; left: 55%; } 36% { top: 55%; left: 15%; } 54% { top: 15%; left: 60%; } 72% { top: 45%; left: 35%; } 90% { top: 10%; left: 45%; } 100% { top: 8%; left: 10%; } }
        @keyframes scanLine { 0% { background-position: 0% 0%; } 100% { background-position: 0% 100%; } }
        @keyframes bboxReveal { 0% { opacity: 0; transform: scale(1.15); } 60% { opacity: 1; transform: scale(0.97); } 100% { opacity: 1; transform: scale(1); } }
        @keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 0.6; } }
      `}</style>
    </div>
  );
}

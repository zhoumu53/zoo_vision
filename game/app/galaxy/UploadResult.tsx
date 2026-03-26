"use client";

import { useState } from "react";
import type { UploadResult, DetectedElephant } from "../../lib/types";
import { ELEPHANTS } from "../../lib/elephants";

const BBOX_COLORS = [
  { border: "border-cyan-400", bg: "bg-cyan-400", text: "text-cyan-400", ring: "ring-cyan-400/40", hex: "#22d3ee" },
  { border: "border-amber-400", bg: "bg-amber-400", text: "text-amber-400", ring: "ring-amber-400/40", hex: "#fbbf24" },
  { border: "border-rose-400", bg: "bg-rose-400", text: "text-rose-400", ring: "ring-rose-400/40", hex: "#fb7185" },
  { border: "border-emerald-400", bg: "bg-emerald-400", text: "text-emerald-400", ring: "ring-emerald-400/40", hex: "#34d399" },
  { border: "border-purple-400", bg: "bg-purple-400", text: "text-purple-400", ring: "ring-purple-400/40", hex: "#c084fc" },
];

function getColor(idx: number) { return BBOX_COLORS[idx % BBOX_COLORS.length]; }

export default function UploadResultView({
  result, onTryAgain, onSelectElephant,
}: {
  result: UploadResult;
  onTryAgain: () => void;
  onSelectElephant?: (el: DetectedElephant | null) => void;
}) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  const handleElephantClick = (idx: number) => {
    setSelectedIdx(idx);
    if (onSelectElephant && result.detected_elephants[idx]) {
      onSelectElephant(result.detected_elephants[idx]);
    }
  };

  if (result.outcome === "no_elephant_detected") {
    return (
      <div className="space-y-4">
        <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 text-center">
          <div className="mb-4 text-4xl">&#x1F50D;</div>
          <h2 className="mb-2 text-lg font-light text-white/80">No elephant detected</h2>
          <p className="mb-6 text-sm font-light text-white/40">We couldn&apos;t spot an elephant in this photo. Try a clearer image.</p>
          <button onClick={onTryAgain} className="rounded-2xl border border-white/[0.10] bg-white/[0.04] px-6 py-2.5 text-sm font-light text-white/60 transition-all hover:bg-white/[0.08] hover:text-white/80">
            Try Another Photo
          </button>
        </div>
      </div>
    );
  }

  const detected = result.detected_elephants;
  const selected = detected[selectedIdx] || null;
  const topMatch = selected?.nearest_elephants[0] || null;
  const restMatches = selected?.nearest_elephants.slice(1) || [];
  const isNew = selected?.possibly_new ?? false;
  const elephantInfo = ELEPHANTS.find((e) => e.name === topMatch?.elephant_name);

  const simPct = topMatch ? Math.round(topMatch.similarity * 100) : 0;
  const simColor = simPct >= 80 ? "from-cyan-400 to-emerald-400" : simPct >= 60 ? "from-purple-400 to-cyan-400" : "from-amber-400 to-yellow-300";

  return (
    <div className="space-y-4">
      {result.original_url && detected.length > 0 && (
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-3">
          <p className="mb-2 text-xs text-white/40 uppercase tracking-[0.1em]">
            {detected.length} elephant{detected.length !== 1 ? "s" : ""} detected
          </p>
          <div className="relative overflow-hidden rounded-xl">
            <img src={result.original_url} alt="Original upload" className="w-full object-contain" style={{ maxHeight: 240 }} />
            {detected.map((el, idx) => {
              const color = getColor(idx);
              const isSelected = idx === selectedIdx;
              const left = (el.bbox.x - el.bbox.w / 2) * 100;
              const top = (el.bbox.y - el.bbox.h / 2) * 100;
              const width = el.bbox.w * 100;
              const height = el.bbox.h * 100;
              return (
                <button key={idx} onClick={() => handleElephantClick(idx)}
                  className={`absolute border-2 ${color.border} transition-all cursor-pointer ${isSelected ? "ring-2 " + color.ring + " shadow-lg" : "opacity-60 hover:opacity-100"}`}
                  style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
                >
                  <span className={`absolute -top-5 left-0 text-[10px] font-bold ${color.text} ${isSelected ? "" : "opacity-60"}`}>#{idx + 1}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {selected && topMatch && (
        <div className={`rounded-2xl border ${isNew ? "border-amber-400/20" : "border-purple-400/15"} bg-white/[0.03] p-5`}>
          <div className="text-center mb-4">
            <p className="text-[10px] text-white/40 uppercase tracking-[0.15em] mb-1">
              {isNew ? "Closest Match" : "Best Match"}
            </p>
            {isNew ? (
              <h2 className="text-lg font-light text-white">
                <span className="bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent font-normal">Could be a new elephant!</span>
              </h2>
            ) : (
              <h2 className="text-lg font-light text-white">
                Looks like{" "}
                <span className="font-normal" style={{ color: elephantInfo?.color || "#c084fc" }}>{topMatch.elephant_name}</span>
              </h2>
            )}
          </div>

          <div className="flex items-center justify-center gap-5 mb-4">
            <div className="text-center">
              <div className={`h-20 w-20 overflow-hidden rounded-xl border-2 ${getColor(selectedIdx).border}`}>
                <img src={selected.crop_url} alt="Your elephant" className="h-full w-full object-cover" />
              </div>
              <p className="mt-1 text-[10px] text-white/40">Your photo</p>
            </div>
            <div className="text-center">
              <div className={`text-3xl font-light bg-gradient-to-r ${simColor} bg-clip-text text-transparent`}>{simPct}%</div>
              <p className="text-[10px] text-white/40">similarity</p>
            </div>
            {topMatch.sample_crop_path && (
              <div className="text-center">
                <div className="h-20 w-20 overflow-hidden rounded-xl border border-purple-400/20">
                  <img src={topMatch.sample_crop_path} alt={topMatch.elephant_name} className="h-full w-full object-cover" />
                </div>
                <p className="mt-1 text-[10px]" style={{ color: elephantInfo?.color || "#fff" }}>{topMatch.elephant_name}</p>
              </div>
            )}
          </div>
          <p className="text-center text-[10px] text-white/25">{topMatch.image_count} photos in database</p>
        </div>
      )}

      {restMatches.length > 0 && (
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-4">
          <p className="text-[10px] text-white/40 uppercase tracking-[0.1em] mb-3">Other Matches</p>
          <div className="space-y-1.5">
            {restMatches.map((match) => {
              const info = ELEPHANTS.find((e) => e.name === match.elephant_name);
              return (
                <div key={match.elephant_id} className="flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-2">
                  {match.sample_crop_path ? (
                    <div className="h-9 w-9 overflow-hidden rounded-lg border border-white/[0.08]">
                      <img src={match.sample_crop_path} alt={match.elephant_name} className="h-full w-full object-cover" />
                    </div>
                  ) : (
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] bg-white/[0.03] text-sm">🐘</div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-light truncate" style={{ color: info?.color || "#fff" }}>{match.elephant_name}</p>
                    <p className="text-[10px] text-white/30">{match.image_count} photos</p>
                  </div>
                  <div className="text-xs font-light text-white/50">{Math.round(match.similarity * 100)}%</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="text-center">
        <button onClick={onTryAgain} className="rounded-xl border border-white/[0.10] bg-white/[0.04] px-8 py-3 text-sm font-light text-white/60 transition-all hover:bg-white/[0.08] hover:text-white/80">
          Try Another Photo
        </button>
      </div>
    </div>
  );
}

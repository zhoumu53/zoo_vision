"use client";

import { useState } from "react";
import type { UploadResult, DetectedElephant, MatchLevel } from "../../lib/types";
import { ELEPHANTS } from "../../lib/elephants";

const BBOX_COLORS = [
  { border: "border-cyan-400", bg: "bg-cyan-400", text: "text-cyan-400", ring: "ring-cyan-400/40", hex: "#22d3ee" },
  { border: "border-amber-400", bg: "bg-amber-400", text: "text-amber-400", ring: "ring-amber-400/40", hex: "#fbbf24" },
  { border: "border-rose-400", bg: "bg-rose-400", text: "text-rose-400", ring: "ring-rose-400/40", hex: "#fb7185" },
  { border: "border-emerald-400", bg: "bg-emerald-400", text: "text-emerald-400", ring: "ring-emerald-400/40", hex: "#34d399" },
  { border: "border-purple-400", bg: "bg-purple-400", text: "text-purple-400", ring: "ring-purple-400/40", hex: "#c084fc" },
];

function getColor(idx: number) { return BBOX_COLORS[idx % BBOX_COLORS.length]; }

const MATCH_LEVEL_CONFIG: Record<MatchLevel, {
  label: string;
  heading: (name: string, color: string) => JSX.Element;
  subtitle?: string;
  gradient: string;
  borderClass: string;
  profileDim: boolean;
  nameSuffix: string;
}> = {
  same: {
    label: "Match Found",
    heading: (name, color) => (
      <h2 className="text-lg font-light text-white">
        This could be <span className="font-normal" style={{ color }}>{name}</span>
      </h2>
    ),
    gradient: "from-cyan-400 to-emerald-400",
    borderClass: "border-cyan-400/20",
    profileDim: false,
    nameSuffix: "",
  },
  similar: {
    label: "Similar — Not Confirmed",
    heading: (name, color) => (
      <h2 className="text-lg font-light text-white">
        Looks <span className="italic text-white/70">similar</span> to{" "}
        <span className="font-normal" style={{ color }}>{name}</span>
      </h2>
    ),
    subtitle: "The cosine distance is too far for a confident ID",
    gradient: "from-purple-400 to-cyan-400",
    borderClass: "border-purple-400/20",
    profileDim: false,
    nameSuffix: " ?",
  },
  unknown: {
    label: "Unknown Elephant",
    heading: () => (
      <h2 className="text-lg font-light text-white">
        <span className="bg-gradient-to-r from-amber-400 to-rose-400 bg-clip-text text-transparent font-normal">
          We may have never seen this one
        </span>
      </h2>
    ),
    subtitle: "But the most similar elephant we know is:",
    gradient: "from-amber-400 to-rose-400",
    borderClass: "border-amber-400/20",
    profileDim: true,
    nameSuffix: " ?",
  },
};

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
  const elephantInfo = ELEPHANTS.find((e) => e.name === topMatch?.elephant_name);
  const elColor = elephantInfo?.color || "#c084fc";

  const matchLevel: MatchLevel = topMatch?.match_level || "unknown";
  const config = MATCH_LEVEL_CONFIG[matchLevel];
  const simPct = topMatch ? Math.round(topMatch.similarity * 100) : 0;
  const cosDist = topMatch?.cosine_distance ?? 0;
  const profileSrc = topMatch?.profile || topMatch?.sample_crop_path;

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
        <div className={`rounded-2xl border ${config.borderClass} bg-white/[0.03] p-5`}>
          {/* Header */}
          <div className="text-center mb-4">
            <p className={`text-[10px] uppercase tracking-[0.15em] mb-1 ${matchLevel === "same" ? "text-cyan-400/60" : matchLevel === "similar" ? "text-purple-400/60" : "text-amber-400/60"}`}>
              {config.label}
            </p>
            {config.heading(topMatch.elephant_name, elColor)}
            {config.subtitle && (
              <p className="mt-2 text-xs text-white/40">{config.subtitle}</p>
            )}
          </div>

          {/* Comparison */}
          <div className="flex items-center justify-center gap-5 mb-4">
            <div className="text-center">
              <div className={`h-20 w-20 overflow-hidden rounded-xl border-2 ${getColor(selectedIdx).border}`}>
                <img src={selected.crop_url} alt="Your elephant" className="h-full w-full object-cover" />
              </div>
              <p className="mt-1 text-[10px] text-white/40">Your photo</p>
            </div>

            <div className="text-center">
              <div className={`text-3xl font-light bg-gradient-to-r ${config.gradient} bg-clip-text text-transparent`}>
                {simPct}%
              </div>
              <p className="text-[10px] text-white/40">similarity</p>
            </div>

            {profileSrc && (
              <div className="text-center">
                <div className={`h-20 w-20 overflow-hidden rounded-xl border ${config.borderClass} ${config.profileDim ? "opacity-60" : ""}`}>
                  <img src={profileSrc} alt={topMatch.elephant_name} className="h-full w-full object-cover" />
                </div>
                <p className="mt-1 text-[10px]" style={{ color: elColor }}>
                  {topMatch.elephant_name}{config.nameSuffix}
                </p>
              </div>
            )}
          </div>

          {/* Diagnostics */}
          <div className="mt-3 flex justify-center gap-4 text-[9px] text-white/25">
            <span>dist: {cosDist.toFixed(3)}</span>
            {topMatch.margin != null && <span>margin: {(topMatch.margin * 100).toFixed(1)}%</span>}
            {topMatch.vote_ratio != null && <span>vote: {(topMatch.vote_ratio * 100).toFixed(0)}%</span>}
            <span>{topMatch.image_count.toLocaleString()} photos</span>
          </div>
        </div>
      )}

      {restMatches.length > 0 && (
        <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-4">
          <p className="text-[10px] text-white/40 uppercase tracking-[0.1em] mb-3">Other Matches</p>
          <div className="space-y-1.5">
            {restMatches.map((match) => {
              const info = ELEPHANTS.find((e) => e.name === match.elephant_name);
              const levelTag = match.match_level === "same" ? "same" : match.match_level === "similar" ? "similar" : "";
              const tagColor = match.match_level === "same" ? "text-cyan-400/50" : "text-purple-400/50";
              return (
                <div key={match.elephant_id} className="flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-2">
                  {(match.profile || match.sample_crop_path) ? (
                    <div className="h-9 w-9 overflow-hidden rounded-lg border border-white/[0.08]">
                      <img src={match.profile || match.sample_crop_path!} alt={match.elephant_name} className="h-full w-full object-cover" />
                    </div>
                  ) : (
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] bg-white/[0.03] text-sm">&#x1F418;</div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-light truncate" style={{ color: info?.color || "#fff" }}>{match.elephant_name}</p>
                    <p className="text-[10px] text-white/30">{match.image_count} photos</p>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-light text-white/50">{Math.round(match.similarity * 100)}%</div>
                    {levelTag && <div className={`text-[9px] ${tagColor}`}>{levelTag}</div>}
                  </div>
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

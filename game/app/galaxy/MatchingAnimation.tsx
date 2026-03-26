"use client";

import { useEffect, useRef, useState } from "react";
import type { GalaxyElephant, UploadResult } from "../../lib/types";
import { ELEPHANTS } from "../../lib/elephants";

const DECEL_INTERVALS = [100, 120, 150, 200, 280, 380, 520, 700, 950];

export default function MatchingAnimation({
  cropUrl, elephants, progress, stage, finalResult, onComplete,
}: {
  cropUrl: string;
  elephants: GalaxyElephant[];
  progress: number;
  stage: string;
  finalResult: UploadResult | null;
  onComplete: () => void;
}) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [phase, setPhase] = useState<"spinning" | "slowing" | "landed">("spinning");
  const [displaySimilarity, setDisplaySimilarity] = useState<number | null>(null);
  const [landed, setLanded] = useState(false);
  const [reelReady, setReelReady] = useState(false);

  const reelElephants = useRef<GalaxyElephant[]>([]);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const decelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const hasStartedSlowing = useRef(false);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    if (elephants.length === 0) { onCompleteRef.current(); return; }
    const shuffled = [...elephants].sort(() => Math.random() - 0.5);
    reelElephants.current = shuffled;
    setReelReady(true);
  }, [elephants]);

  useEffect(() => {
    if (phase !== "spinning" || reelElephants.current.length === 0) return;
    const spin = () => {
      setCurrentIdx((prev) => (prev + 1) % reelElephants.current.length);
      timerRef.current = setTimeout(spin, 90);
    };
    timerRef.current = setTimeout(spin, 90);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [phase]);

  useEffect(() => {
    if (!finalResult || hasStartedSlowing.current || !reelReady) return;
    hasStartedSlowing.current = true;
    if (timerRef.current) clearTimeout(timerRef.current);
    setPhase("slowing");

    const topMatch = finalResult.detected_elephants[0]?.nearest_elephants[0];
    if (!topMatch) { setPhase("landed"); onCompleteRef.current(); return; }

    const reel = reelElephants.current;
    const nearest = finalResult.detected_elephants[0]?.nearest_elephants || [];

    const finalSequence: GalaxyElephant[] = [];
    for (let i = 0; i < 4; i++) {
      finalSequence.push(reel[Math.floor(Math.random() * reel.length)]);
    }
    const ru2 = nearest[2] ? reel.find((e) => e.elephant_id === nearest[2].elephant_id) : reel[Math.floor(Math.random() * reel.length)];
    const ru1 = nearest[1] ? reel.find((e) => e.elephant_id === nearest[1].elephant_id) : reel[Math.floor(Math.random() * reel.length)];
    const winner = reel.find((e) => e.elephant_id === topMatch.elephant_id);
    if (ru2) finalSequence.push(ru2);
    if (ru1) finalSequence.push(ru1);
    if (winner) finalSequence.push(winner);

    const intervals = DECEL_INTERVALS.slice(0, finalSequence.length);
    let step = 0;
    let cancelled = false;
    const decelerate = () => {
      if (cancelled) return;
      if (step >= finalSequence.length) { setPhase("landed"); setLanded(true); return; }
      const el = finalSequence[step];
      const idx = reel.indexOf(el);
      if (idx >= 0) setCurrentIdx(idx);
      step++;
      decelTimerRef.current = setTimeout(decelerate, intervals[Math.min(step, intervals.length - 1)]);
    };
    decelerate();
    return () => { cancelled = true; if (decelTimerRef.current) clearTimeout(decelTimerRef.current); };
  }, [finalResult, reelReady]);

  useEffect(() => {
    if (!landed || !finalResult) return;
    const topMatch = finalResult.detected_elephants[0]?.nearest_elephants[0];
    if (!topMatch) { onCompleteRef.current(); return; }
    const targetPct = Math.round(topMatch.similarity * 100);
    const delay = setTimeout(() => {
      const startTime = performance.now();
      const duration = 800;
      const animate = (now: number) => {
        const elapsed = now - startTime;
        const t = Math.min(1, elapsed / duration);
        const eased = 1 - Math.pow(1 - t, 3);
        setDisplaySimilarity(Math.round(eased * targetPct));
        if (t < 1) { animFrameRef.current = requestAnimationFrame(animate); }
        else { setTimeout(() => onCompleteRef.current(), 800); }
      };
      animFrameRef.current = requestAnimationFrame(animate);
    }, 400);
    return () => { clearTimeout(delay); if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [landed, finalResult]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (decelTimerRef.current) clearTimeout(decelTimerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  const currentElephant = reelElephants.current[currentIdx] || null;
  const topMatch = finalResult?.detected_elephants[0]?.nearest_elephants[0];
  const isNew = finalResult?.detected_elephants[0]?.possibly_new ?? false;
  const elephantInfo = ELEPHANTS.find((e) => e.name === topMatch?.elephant_name);

  const simPct = displaySimilarity ?? 0;
  const simColor = displaySimilarity !== null
    ? (simPct >= 80 ? "from-cyan-400 to-emerald-400" : simPct >= 60 ? "from-purple-400 to-cyan-400" : "from-amber-400 to-yellow-300")
    : "";

  return (
    <div className="w-full">
      <div className="rounded-3xl border border-purple-400/15 bg-white/[0.03] p-5 backdrop-blur-2xl">
        <div className="text-center mb-4">
          <p className="text-[10px] text-white/40 uppercase tracking-[0.15em] mb-1">Finding Your Match</p>
          <h2 className="text-base font-light text-white/70">
            {landed && topMatch
              ? isNew
                ? <span className="bg-gradient-to-r from-amber-400 to-yellow-300 bg-clip-text text-transparent font-normal">New elephant detected!</span>
                : <>Looks like{" "}<span className="bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent font-normal" style={elephantInfo ? { backgroundImage: `linear-gradient(to right, ${elephantInfo.color}, #00d4aa)` } : undefined}>{topMatch.elephant_name}</span></>
              : "Who could it be?"
            }
          </h2>
        </div>

        <div className="flex items-center justify-center gap-5 mb-4">
          <div className="text-center">
            <div className="h-20 w-20 overflow-hidden rounded-xl border-2 border-cyan-400">
              <img src={cropUrl} alt="Your elephant" className="h-full w-full object-cover" />
            </div>
            <p className="mt-1 text-[10px] text-white/40">Your photo</p>
          </div>

          <div className="text-center min-w-[60px]">
            {displaySimilarity !== null ? (
              <div className={`text-3xl font-light bg-gradient-to-r ${simColor} bg-clip-text text-transparent`}>{simPct}%</div>
            ) : (
              <div className="text-3xl font-light text-rose-400" style={{ animation: landed ? "none" : "questionPulse 1.5s ease-in-out infinite" }}>?</div>
            )}
            <p className="text-[10px] text-white/40">similarity</p>
          </div>

          <div className="text-center">
            <div className="h-20 w-20 overflow-hidden rounded-xl border border-purple-400/20 relative" style={{ filter: phase === "spinning" ? "blur(0.5px)" : "none", transition: "filter 0.3s ease" }}>
              {currentElephant?.sample_crop_path ? (
                <img src={currentElephant.sample_crop_path} alt="?" className="h-full w-full object-cover" style={{
                  opacity: phase === "spinning" ? 0.85 : 1,
                  animation: landed ? "landPulse 0.4s ease-out" : "none",
                }} />
              ) : (
                <div className="flex h-full w-full items-center justify-center bg-white/[0.03] text-2xl">🐘</div>
              )}
              {phase === "spinning" && (
                <div className="absolute inset-0 pointer-events-none" style={{
                  background: "linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%)",
                  animation: "slotShimmer 0.3s linear infinite",
                }} />
              )}
            </div>
            <p className="mt-1 text-[10px] text-white/40">{landed && topMatch ? topMatch.elephant_name : "???"}</p>
          </div>
        </div>

        <p className="text-center text-sm font-light text-white/50 mb-1">
          {landed ? (isNew ? "Possible new elephant!" : "Match found!") : stage || "Matching..."}
        </p>
        <p className="text-center text-[10px] text-white/30">{progress}%</p>
      </div>

      <style jsx>{`
        @keyframes questionPulse { 0%, 100% { opacity: 0.6; transform: scale(1); } 50% { opacity: 1; transform: scale(1.15); } }
        @keyframes landPulse { 0% { transform: scale(1); } 40% { transform: scale(1.08); } 100% { transform: scale(1); } }
        @keyframes slotShimmer { 0% { transform: translateY(-100%); } 100% { transform: translateY(100%); } }
      `}</style>
    </div>
  );
}

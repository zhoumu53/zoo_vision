"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getUploadStatus } from "../../lib/api";
import type { GalaxyElephant, UploadJobStatus, BBox } from "../../lib/types";
import DetectionAnimation from "./DetectionAnimation";
import MatchingAnimation from "./MatchingAnimation";

type PartialElephant = { index: number; bbox: BBox; crop_url: string };

export default function UploadProcessing({
  jobId,
  previewUrl,
  elephants,
  onDone,
}: {
  jobId: string;
  previewUrl: string;
  elephants: GalaxyElephant[];
  onDone: (status: UploadJobStatus) => void;
}) {
  const [status, setStatus] = useState<UploadJobStatus>({
    status: "queued", progress: 0, stage: "Preparing...", result: null, partial: null,
  });
  const [subPhase, setSubPhase] = useState<"detection" | "matching">("detection");
  const [partialElephants, setPartialElephants] = useState<PartialElephant[] | null>(null);
  const [finalResult, setFinalResult] = useState<UploadJobStatus["result"]>(null);
  const finalStatusRef = useRef<UploadJobStatus | null>(null);
  const transitionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const s = await getUploadStatus(jobId);
        if (!active) return;
        setStatus(s);
        if (s.partial?.detected_elephants_partial && s.partial.detected_elephants_partial.length > 0) {
          setPartialElephants((prev) => prev || s.partial!.detected_elephants_partial);
        }
        if (s.status === "done" || s.status === "failed") {
          finalStatusRef.current = s;
          setFinalResult(s.result);
          if (s.status === "failed") { onDone(s); }
          return;
        }
      } catch { /* ignore */ }
      if (active) setTimeout(poll, 2000);
    };
    poll();
    return () => { active = false; };
  }, [jobId, onDone]);

  useEffect(() => {
    if (subPhase !== "detection") return;
    if (partialElephants && partialElephants.length > 0) {
      transitionTimerRef.current = setTimeout(() => setSubPhase("matching"), 2000);
    } else if (finalResult && finalResult.detected_elephants?.length > 0) {
      const fromResult: PartialElephant[] = finalResult.detected_elephants.map((e) => ({
        index: e.index, bbox: e.bbox, crop_url: e.crop_url,
      }));
      setPartialElephants(fromResult);
      transitionTimerRef.current = setTimeout(() => setSubPhase("matching"), 500);
    }
    return () => { if (transitionTimerRef.current) clearTimeout(transitionTimerRef.current); };
  }, [partialElephants, subPhase, finalResult]);

  const handleMatchingComplete = useCallback(() => {
    if (finalStatusRef.current) onDone(finalStatusRef.current);
  }, [onDone]);

  const cropUrl = partialElephants?.[0]?.crop_url || "";

  return (
    <div className="w-full space-y-3">
      {subPhase === "detection" && (
        <DetectionAnimation previewUrl={previewUrl} progress={status.progress} stage={status.stage} partialElephants={partialElephants} />
      )}
      {subPhase === "matching" && (
        <MatchingAnimation cropUrl={cropUrl} elephants={elephants} progress={status.progress} stage={status.stage} finalResult={finalResult} onComplete={handleMatchingComplete} />
      )}
      <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-3">
        <div className="h-1 overflow-hidden rounded-full bg-white/[0.06]">
          <div className="h-full rounded-full bg-gradient-to-r from-purple-400 to-cyan-400 transition-all duration-700 ease-out" style={{ width: `${status.progress}%` }} />
        </div>
      </div>
    </div>
  );
}

"use client";

import { useEffect, useState, useCallback } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { loadGalaxyElephants } from "../../lib/data";
import { uploadElephantPhoto } from "../../lib/api";
import type { GalaxyElephant, UploadResult, UploadJobStatus, Position3D, DetectedElephant } from "../../lib/types";
import GalaxyUpload from "./GalaxyUpload";
import UploadProcessing from "./UploadProcessing";
import UploadResultView from "./UploadResult";

const GalaxyScene = dynamic(() => import("./components/GalaxyScene"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <div className="text-lg font-light tracking-wide text-white/30">Loading galaxy...</div>
    </div>
  ),
});

type UploadPhase = "idle" | "uploading" | "processing" | "result";

export default function GalaxyPage() {
  const [elephants, setElephants] = useState<GalaxyElephant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [uploadedPosition, setUploadedPosition] = useState<Position3D | null>(null);
  const [nearestElephantId, setNearestElephantId] = useState<number | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);

  const [panelOpen, setPanelOpen] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await loadGalaxyElephants();
        setElephants(data);
      } catch (e: any) {
        setError(e?.message || "Failed to load galaxy data");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleUpload = useCallback(async (file: File) => {
    setPhase("uploading");
    setUploadError(null);
    const localPreview = URL.createObjectURL(file);
    setPreviewUrl(localPreview);
    try {
      const resp = await uploadElephantPhoto(file);
      setJobId(resp.job_id);
      setPhase("processing");
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
      setPhase("idle");
      URL.revokeObjectURL(localPreview);
      setPreviewUrl(null);
    }
  }, []);

  const handleProcessingDone = useCallback((status: UploadJobStatus) => {
    if (status.status === "done" && status.result) {
      setUploadResult(status.result);
      setPhase("result");
      const first = status.result.detected_elephants[0];
      if (first) {
        if (first.uploaded_position) setUploadedPosition(first.uploaded_position);
        if (first.nearest_elephants.length > 0) setNearestElephantId(first.nearest_elephants[0].elephant_id);
        if (first.crop_url) setUploadedImageUrl(first.crop_url);
      }
    } else if (status.status === "failed") {
      setUploadError("Processing failed. Please try again.");
      setPhase("idle");
    }
  }, []);

  const handleSelectElephant = useCallback((el: DetectedElephant | null) => {
    if (!el) return;
    setUploadedPosition(el.uploaded_position ?? null);
    setNearestElephantId(el.nearest_elephants[0]?.elephant_id ?? null);
    setUploadedImageUrl(el.crop_url ?? null);
  }, []);

  const handleTryAgain = useCallback(() => {
    setPhase("idle");
    setJobId(null);
    setUploadResult(null);
    setUploadError(null);
    setUploadedPosition(null);
    setNearestElephantId(null);
    setUploadedImageUrl(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  }, [previewUrl]);

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#050510]">
        <div className="text-center">
          <div className="mb-4 text-4xl">&#10022;</div>
          <div className="text-lg font-light tracking-wide text-white/30">
            Mapping the elephant universe...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#050510]">
        <div className="text-center">
          <div className="mb-4 text-2xl text-rose-400/70">Error</div>
          <p className="mb-6 max-w-md text-sm text-white/40">{error}</p>
          <Link href="/" className="rounded-full border border-white/10 px-6 py-2 text-sm text-white/60 hover:border-white/20 hover:text-white">
            Back to Home
          </Link>
        </div>
      </div>
    );
  }

  const nearestElephant = nearestElephantId ? elephants.find((e) => e.elephant_id === nearestElephantId) : null;
  const nearestPosition = nearestElephant?.x != null && nearestElephant?.y != null && nearestElephant?.z != null
    ? { x: nearestElephant.x, y: nearestElephant.y, z: nearestElephant.z }
    : null;

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#050510]">
      <div className="absolute inset-0">
        <GalaxyScene
          elephants={elephants}
          uploadedPosition={uploadedPosition}
          nearestPosition={nearestPosition}
          uploadedImageUrl={uploadedImageUrl}
        />
      </div>

      {/* Top bar */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-center justify-between p-4 sm:p-6">
        <Link href="/" className="pointer-events-auto flex items-center gap-2 rounded-full border border-white/10 bg-black/30 px-4 py-2 text-sm font-light text-white/50 backdrop-blur-md transition-all hover:border-white/20 hover:text-white">
          <span>&larr;</span>
          <span>Home</span>
        </Link>
        <div className="text-right">
          <h1 className="text-lg font-extralight tracking-wide text-white/70 sm:text-xl">Elephant Galaxy</h1>
          <p className="text-xs text-white/30">{elephants.length} elephants &middot; feature space</p>
        </div>
      </div>

      {/* Upload toggle */}
      {!panelOpen && (
        <button onClick={() => setPanelOpen(true)} className="absolute right-4 top-1/2 z-20 -translate-y-1/2 rounded-2xl border border-purple-400/20 bg-black/60 px-4 py-6 text-sm font-light text-purple-300 backdrop-blur-xl transition-all hover:bg-purple-400/10 hover:border-purple-400/30 hover:shadow-[0_0_30px_rgba(123,97,255,0.15)]">
          <div className="flex flex-col items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span className="writing-mode-vertical text-xs tracking-[0.15em] uppercase">Upload Photo</span>
          </div>
        </button>
      )}

      {/* Right panel */}
      <div className={`absolute right-0 top-0 bottom-0 z-20 flex w-full max-w-md flex-col border-l border-white/[0.06] bg-[#050510]/90 backdrop-blur-2xl transition-transform duration-500 ease-out ${panelOpen ? "translate-x-0" : "translate-x-full"}`}>
        <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-4">
          <div>
            <h2 className="text-base font-light text-white/80">Find Your Elephant</h2>
            <p className="text-xs text-white/30">Upload a photo to match</p>
          </div>
          <button onClick={() => setPanelOpen(false)} className="flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.08] text-white/40 transition-colors hover:bg-white/[0.06] hover:text-white/70">
            &times;
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-6">
          {uploadError && (
            <div className="mb-4 rounded-2xl border border-rose-400/20 bg-rose-400/[0.05] px-4 py-2">
              <p className="text-sm text-rose-400/80">{uploadError}</p>
            </div>
          )}
          {phase === "idle" && <GalaxyUpload onUpload={handleUpload} />}
          {phase === "uploading" && (
            <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 text-center">
              <div className="mb-4 animate-pulse text-4xl">&#x1F4E4;</div>
              <p className="text-sm font-light text-white/60">Uploading your photo...</p>
            </div>
          )}
          {phase === "processing" && jobId !== null && previewUrl && (
            <UploadProcessing
              jobId={jobId}
              previewUrl={previewUrl}
              elephants={elephants}
              onDone={handleProcessingDone}
            />
          )}
          {phase === "result" && uploadResult && (
            <UploadResultView result={uploadResult} onTryAgain={handleTryAgain} onSelectElephant={handleSelectElephant} />
          )}
        </div>
      </div>

      {/* Bottom hint */}
      <div className="pointer-events-none absolute bottom-6 left-0 right-0 z-10 text-center">
        <p className="text-xs tracking-widest text-white/15 uppercase">
          Drag to rotate &middot; Scroll to zoom &middot; Click a star to explore
        </p>
      </div>
    </div>
  );
}

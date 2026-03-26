"use client";

import { useCallback, useRef, useState } from "react";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE = 10 * 1024 * 1024;

export default function GalaxyUpload({ onUpload }: { onUpload: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((f: File): string | null => {
    if (!ALLOWED_TYPES.includes(f.type)) return "Please upload a JPG, PNG, or WebP image.";
    if (f.size > MAX_SIZE) return "File is too large (max 10 MB).";
    return null;
  }, []);

  const handleFile = useCallback((f: File) => {
    setError(null);
    const err = validate(f);
    if (err) { setError(err); return; }
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }, [validate]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const reset = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      {!preview ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`flex h-64 cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed transition-all duration-300 ${
            dragOver ? "border-purple-400/50 bg-purple-400/[0.06]" : "border-white/[0.12] bg-white/[0.02] hover:border-white/[0.20] hover:bg-white/[0.04]"
          }`}
        >
          <div className="mb-4 text-4xl text-white/30">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="text-sm font-light text-white/50">Drag & drop an elephant photo here</p>
          <p className="mt-1 text-xs text-white/30">or click to browse</p>
          <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} className="hidden" />
        </div>
      ) : (
        <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] p-6 backdrop-blur-2xl">
          <div className="relative mb-4 overflow-hidden rounded-2xl">
            <img src={preview} alt="Preview" className="w-full max-h-72 object-contain rounded-2xl" />
          </div>
          <p className="mb-4 text-center text-sm text-white/50 truncate">{file?.name}</p>
          <div className="flex gap-3 justify-center">
            <button onClick={reset} className="rounded-2xl border border-white/[0.10] bg-white/[0.04] px-6 py-3 text-sm font-light text-white/60 transition-all duration-300 hover:bg-white/[0.08] hover:text-white/80">
              Cancel
            </button>
            <button onClick={() => file && onUpload(file)} className="rounded-2xl border border-purple-400/20 bg-purple-400/10 px-8 py-3 text-sm font-light text-purple-300 transition-all duration-300 hover:bg-purple-400/20 hover:text-purple-200 hover:shadow-[0_0_30px_rgba(123,97,255,0.15)]">
              Find My Elephant
            </button>
          </div>
        </div>
      )}
      {error && <p className="mt-3 text-center text-sm text-rose-400/80">{error}</p>}
    </div>
  );
}

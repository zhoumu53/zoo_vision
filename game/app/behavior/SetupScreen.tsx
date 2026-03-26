"use client";

import { useState } from "react";
import Link from "next/link";

const PRESETS = [5, 10, 15];

export default function SetupScreen({
  onStart,
  loading,
  error,
}: {
  onStart: (count: number) => void;
  loading: boolean;
  error: string | null;
}) {
  const [selected, setSelected] = useState(10);
  const [custom, setCustom] = useState("");
  const [useCustom, setUseCustom] = useState(false);

  const count = useCustom ? Math.max(1, Math.min(50, Number(custom) || 1)) : selected;

  return (
    <div className="w-full max-w-lg">
      <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl shadow-2xl p-10">
        <div className="text-center mb-10">
          <h1 className="text-5xl font-extralight tracking-widest text-white mb-4">
            Behavior Quiz
          </h1>
          <p className="text-white/60 text-base tracking-wide font-light">
            What is the elephant doing?
          </p>
        </div>

        <div className="mb-6">
          <div className="text-xs text-white/50 uppercase tracking-[0.2em] mb-3">Questions</div>
          <div className="grid grid-cols-3 gap-3">
            {PRESETS.map((n) => (
              <button
                key={n}
                onClick={() => { setSelected(n); setUseCustom(false); }}
                className={`rounded-2xl border px-4 py-4 text-xl font-light transition-all duration-300 ${
                  !useCustom && selected === n
                    ? "border-emerald-400/40 bg-emerald-400/[0.08] text-emerald-300 shadow-[0_0_20px_rgba(16,185,129,0.12)]"
                    : "border-white/[0.08] bg-white/[0.02] text-white/70 hover:border-white/20 hover:text-white/90"
                }`}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-8">
          <button onClick={() => setUseCustom(!useCustom)} className="text-xs text-white/50 hover:text-white/70 transition-colors uppercase tracking-[0.15em]">
            {useCustom ? "← Use preset" : "Custom number →"}
          </button>
          {useCustom && (
            <input type="number" min={1} max={50} value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="1–50"
              className="mt-2 w-full rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-base text-white/90 placeholder-white/30 outline-none focus:border-emerald-400/30 transition-colors font-light"
            />
          )}
        </div>

        <button onClick={() => onStart(count)} disabled={loading}
          className="w-full rounded-2xl border border-emerald-400/25 bg-emerald-400/[0.06] px-6 py-4 text-lg font-light tracking-wide text-emerald-300 transition-all duration-300 hover:bg-emerald-400/[0.12] hover:shadow-[0_0_40px_rgba(16,185,129,0.15)] hover:scale-[1.02] active:scale-[0.97] active:brightness-125 disabled:opacity-30 disabled:pointer-events-none"
        >
          {loading ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-emerald-300/30 border-t-emerald-300" />
              Loading…
            </span>
          ) : "Start Game"}
        </button>

        {error && (
          <div className="mt-5 rounded-xl border border-rose-400/20 bg-rose-400/[0.05] px-4 py-3 text-sm text-rose-300/80 font-light">{error}</div>
        )}

        <div className="mt-6 text-center">
          <Link href="/" className="text-xs text-white/30 hover:text-white/60 transition-colors uppercase tracking-[0.15em]">← Back</Link>
        </div>
      </div>
    </div>
  );
}

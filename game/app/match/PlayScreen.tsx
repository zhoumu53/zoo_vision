"use client";

import { useCallback, useEffect, useState } from "react";
import { MatchPair } from "../../lib/types";

const BROKEN_IMG =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect fill='%23111' width='100' height='100'/%3E%3Ctext x='50' y='55' text-anchor='middle' fill='%23333' font-size='14'%3E?%3C/text%3E%3C/svg%3E";

export default function PlayScreen({
  pair,
  currentIndex,
  total,
  onAnswer,
}: {
  pair: MatchPair;
  currentIndex: number;
  total: number;
  onAnswer: (answer: boolean) => void;
}) {
  const [disabled, setDisabled] = useState(false);
  const [feedback, setFeedback] = useState<"correct" | "wrong" | null>(null);

  const progress = (currentIndex / total) * 100;

  const handleAnswer = useCallback(
    (answer: boolean) => {
      if (disabled) return;
      setDisabled(true);

      const isCorrect = answer === pair.is_same_elephant;
      setFeedback(isCorrect ? "correct" : "wrong");

      setTimeout(() => {
        setFeedback(null);
        setDisabled(false);
        onAnswer(answer);
      }, 600);
    },
    [disabled, pair, onAnswer],
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "y" || e.key === "Y") handleAnswer(true);
      if (e.key === "n" || e.key === "N") handleAnswer(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleAnswer]);

  const feedbackBorder =
    feedback === "correct"
      ? "border-cyan-400/50 shadow-[0_0_40px_rgba(0,212,170,0.2)]"
      : feedback === "wrong"
        ? "border-rose-400/50 shadow-[0_0_40px_rgba(236,72,153,0.2)]"
        : "border-white/[0.06]";

  return (
    <div className="w-full max-w-2xl">
      <div className="mb-2 flex items-center justify-between text-xs text-white/50 uppercase tracking-[0.15em]">
        <span>Question {currentIndex + 1} of {total}</span>
        <span>{Math.round(progress)}%</span>
      </div>
      <div className="mb-8 h-1 w-full overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className={`rounded-3xl border bg-white/[0.03] backdrop-blur-2xl shadow-2xl p-8 transition-all duration-500 ${feedbackBorder}`}>
        <h2 className="text-center text-2xl font-extralight tracking-wide text-white/90 mb-8">
          Are these the same elephant?
        </h2>

        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-6 mb-8">
          <div className="group">
            <div className="aspect-square overflow-hidden rounded-2xl border border-white/[0.08] ring-1 ring-white/[0.04] transition-all duration-300 group-hover:scale-[1.02] group-hover:border-white/20">
              <img
                src={pair.left.image}
                alt="Elephant A"
                className="h-full w-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).src = BROKEN_IMG; }}
              />
            </div>
            <div className="mt-3 text-center text-xs text-white/50 uppercase tracking-[0.2em]">A</div>
          </div>

          <div className="flex flex-col items-center gap-2">
            <div className="h-10 w-px bg-gradient-to-b from-transparent via-white/[0.10] to-transparent" />
            <span className="text-2xl text-white/30 font-extralight">?</span>
            <div className="h-10 w-px bg-gradient-to-b from-transparent via-white/[0.10] to-transparent" />
          </div>

          <div className="group">
            <div className="aspect-square overflow-hidden rounded-2xl border border-white/[0.08] ring-1 ring-white/[0.04] transition-all duration-300 group-hover:scale-[1.02] group-hover:border-white/20">
              <img
                src={pair.right.image}
                alt="Elephant B"
                className="h-full w-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).src = BROKEN_IMG; }}
              />
            </div>
            <div className="mt-3 text-center text-xs text-white/50 uppercase tracking-[0.2em]">B</div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={() => handleAnswer(true)}
            disabled={disabled}
            className="rounded-2xl border border-cyan-400/20 bg-cyan-400/[0.05] px-6 py-4 text-xl font-light text-cyan-300 transition-all duration-300 hover:bg-cyan-400/[0.12] hover:shadow-[0_0_30px_rgba(0,212,170,0.2)] hover:scale-105 active:scale-95 active:brightness-125 disabled:pointer-events-none disabled:opacity-30"
          >
            Yes
            <span className="ml-2 text-xs text-cyan-300/50 uppercase">Y</span>
          </button>
          <button
            onClick={() => handleAnswer(false)}
            disabled={disabled}
            className="rounded-2xl border border-rose-400/20 bg-rose-400/[0.05] px-6 py-4 text-xl font-light text-rose-300 transition-all duration-300 hover:bg-rose-400/[0.12] hover:shadow-[0_0_30px_rgba(236,72,153,0.2)] hover:scale-105 active:scale-95 active:brightness-125 disabled:pointer-events-none disabled:opacity-30"
          >
            No
            <span className="ml-2 text-xs text-rose-300/50 uppercase">N</span>
          </button>
        </div>

        <div className="mt-5 h-7 text-center text-base font-light">
          {feedback === "correct" && <span className="text-cyan-400 animate-pulse">Correct!</span>}
          {feedback === "wrong" && <span className="text-rose-400 animate-pulse">Wrong!</span>}
        </div>
      </div>
    </div>
  );
}

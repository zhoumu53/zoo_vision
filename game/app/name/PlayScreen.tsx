"use client";

import { useCallback, useEffect, useState } from "react";
import { NameQuestion } from "../../lib/types";
import { ELEPHANTS } from "../../lib/elephants";

const BROKEN_IMG =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect fill='%23111' width='100' height='100'/%3E%3Ctext x='50' y='55' text-anchor='middle' fill='%23333' font-size='14'%3E?%3C/text%3E%3C/svg%3E";

export default function PlayScreen({
  question,
  currentIndex,
  total,
  onAnswer,
}: {
  question: NameQuestion;
  currentIndex: number;
  total: number;
  onAnswer: (answer: string) => void;
}) {
  const [disabled, setDisabled] = useState(false);
  const [feedback, setFeedback] = useState<"correct" | "wrong" | null>(null);

  const progress = (currentIndex / total) * 100;

  const handleAnswer = useCallback(
    (name: string) => {
      if (disabled) return;
      setDisabled(true);
      setFeedback(name === question.elephant_name ? "correct" : "wrong");
      setTimeout(() => {
        setFeedback(null);
        setDisabled(false);
        onAnswer(name);
      }, 600);
    },
    [disabled, question, onAnswer],
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const idx = parseInt(e.key) - 1;
      if (idx >= 0 && idx < ELEPHANTS.length) handleAnswer(ELEPHANTS[idx].name);
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
    <div className="w-full max-w-xl">
      <div className="mb-2 flex items-center justify-between text-xs text-white/50 uppercase tracking-[0.15em]">
        <span>Question {currentIndex + 1} of {total}</span>
        <span>{Math.round(progress)}%</span>
      </div>
      <div className="mb-8 h-1 w-full overflow-hidden rounded-full bg-white/[0.06]">
        <div className="h-full rounded-full bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
      </div>

      <div className={`rounded-3xl border bg-white/[0.03] backdrop-blur-2xl shadow-2xl p-8 transition-all duration-500 ${feedbackBorder}`}>
        <h2 className="text-center text-2xl font-extralight tracking-wide text-white/90 mb-6">
          Who is this elephant?
        </h2>

        <div className="mx-auto mb-8 w-64 h-64 overflow-hidden rounded-2xl border border-white/[0.08] ring-1 ring-white/[0.04]">
          <img src={question.image} alt="Elephant" className="h-full w-full object-cover" onError={(e) => { (e.target as HTMLImageElement).src = BROKEN_IMG; }} />
        </div>

        <div className="grid grid-cols-1 gap-3">
          {ELEPHANTS.map((elephant, idx) => (
            <button
              key={elephant.id}
              onClick={() => handleAnswer(elephant.name)}
              disabled={disabled}
              className="rounded-2xl border border-white/[0.08] bg-white/[0.03] px-6 py-3 text-lg font-light transition-all duration-300 hover:bg-white/[0.08] hover:scale-[1.02] active:scale-[0.97] disabled:pointer-events-none disabled:opacity-30 flex items-center gap-3"
              style={{ borderColor: `${elephant.color}20` }}
            >
              <span className="text-xs text-white/40 w-5">{idx + 1}</span>
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: elephant.color }} />
              <span style={{ color: elephant.color }}>{elephant.name}</span>
            </button>
          ))}
        </div>

        <div className="mt-5 h-7 text-center text-base font-light">
          {feedback === "correct" && <span className="text-cyan-400 animate-pulse">Correct!</span>}
          {feedback === "wrong" && <span className="text-rose-400 animate-pulse">Wrong!</span>}
        </div>
      </div>
    </div>
  );
}

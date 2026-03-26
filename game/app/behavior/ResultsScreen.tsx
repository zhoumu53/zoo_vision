"use client";

import Link from "next/link";
import { BehaviorAnswer } from "../../lib/types";
import { BEHAVIORS } from "../../lib/elephants";

const BROKEN_IMG =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect fill='%23111' width='100' height='100'/%3E%3Ctext x='50' y='55' text-anchor='middle' fill='%23333' font-size='14'%3E?%3C/text%3E%3C/svg%3E";

const BEHAVIOR_LABELS: Record<string, string> = {
  Standing: "Standing",
  SleepL: "Sleeping Left",
  SleepR: "Sleeping Right",
};

function getBehaviorColor(name: string): string {
  return BEHAVIORS.find((b) => b.name === name)?.color ?? "#fff";
}

export default function ResultsScreen({
  answers,
  onPlayAgain,
}: {
  answers: BehaviorAnswer[];
  onPlayAgain: () => void;
}) {
  const correct = answers.filter((a) => a.userAnswer === a.question.behavior).length;
  const total = answers.length;
  const pct = total > 0 ? Math.round((correct / total) * 100) : 0;

  const scoreColor =
    pct >= 80 ? "from-cyan-400 to-emerald-400"
    : pct >= 50 ? "from-amber-400 to-yellow-300"
    : "from-rose-400 to-pink-400";

  const message =
    pct >= 90 ? "Amazing! You understand elephant behavior!"
    : pct >= 70 ? "Great job! Solid observation skills."
    : pct >= 50 ? "Not bad! Keep watching."
    : "Keep trying — observation takes practice!";

  return (
    <div className="w-full max-w-2xl">
      <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl shadow-2xl p-10 mb-6">
        <div className="text-center mb-2">
          <div className="text-sm text-white/60 uppercase tracking-[0.2em] mb-4">Final Score</div>
          <div className={`text-7xl font-light tracking-tight bg-gradient-to-r ${scoreColor} bg-clip-text text-transparent`}>
            {correct}/{total}
          </div>
          <div className="mt-3 text-3xl font-light text-white/60">{pct}%</div>
        </div>
        <div className="mx-auto mt-5 mb-5 h-1 max-w-sm overflow-hidden rounded-full bg-white/[0.06]">
          <div className={`h-full rounded-full bg-gradient-to-r ${scoreColor} transition-all duration-1000 ease-out`} style={{ width: `${pct}%` }} />
        </div>
        <p className="text-center text-base font-light text-white/60">{message}</p>
      </div>

      <div className="rounded-3xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl shadow-2xl p-6">
        <div className="text-sm text-white/50 uppercase tracking-[0.15em] mb-5">Review</div>
        <div className="space-y-3">
          {answers.map((a, i) => {
            const isCorrect = a.userAnswer === a.question.behavior;
            return (
              <div key={i} className={`grid grid-cols-[auto_1fr_auto] items-center gap-5 rounded-2xl border p-4 transition-colors ${
                isCorrect ? "border-cyan-400/15 bg-cyan-400/[0.03]" : "border-rose-400/15 bg-rose-400/[0.03]"
              }`}>
                <div className="h-14 w-14 overflow-hidden rounded-xl border border-white/[0.08]">
                  <img src={a.question.image} alt="" className="h-full w-full object-cover" onError={(e) => { (e.target as HTMLImageElement).src = BROKEN_IMG; }} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-normal text-white/70 truncate">
                    {a.question.elephant_name} — <span style={{ color: getBehaviorColor(a.question.behavior) }}>{BEHAVIOR_LABELS[a.question.behavior] || a.question.behavior}</span>
                  </div>
                  {!isCorrect && (
                    <div className="text-xs text-white/50 mt-1">
                      You said: <span style={{ color: getBehaviorColor(a.userAnswer) }}>{BEHAVIOR_LABELS[a.userAnswer] || a.userAnswer}</span>
                    </div>
                  )}
                </div>
                <div className={`flex h-10 w-10 items-center justify-center rounded-full text-base font-medium ${isCorrect ? "bg-cyan-400/15 text-cyan-400" : "bg-rose-400/15 text-rose-400"}`}>
                  {isCorrect ? "✓" : "✗"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-8 flex justify-center gap-4">
        <button onClick={onPlayAgain} className="rounded-2xl border border-white/[0.10] bg-white/[0.04] px-10 py-4 text-base font-light tracking-wide text-white/70 transition-all duration-300 hover:bg-white/[0.08] hover:text-white/90 hover:scale-[1.02] active:scale-[0.97]">
          Play Again
        </button>
        <Link href="/" className="rounded-2xl border border-white/[0.10] bg-white/[0.04] px-10 py-4 text-base font-light tracking-wide text-white/70 transition-all duration-300 hover:bg-white/[0.08] hover:text-white/90 hover:scale-[1.02] active:scale-[0.97]">
          Home
        </Link>
      </div>
    </div>
  );
}

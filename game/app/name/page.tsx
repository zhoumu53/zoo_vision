"use client";

import { useState } from "react";
import { NameQuestion, NameAnswer } from "../../lib/types";
import { loadNameQuestions } from "../../lib/data";
import SetupScreen from "./SetupScreen";
import PlayScreen from "./PlayScreen";
import ResultsScreen from "./ResultsScreen";

type Phase = "setup" | "play" | "results";

export default function NamePage() {
  const [phase, setPhase] = useState<Phase>("setup");
  const [questions, setQuestions] = useState<NameQuestion[]>([]);
  const [answers, setAnswers] = useState<NameAnswer[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStart = async (count: number) => {
    setLoading(true);
    setError(null);
    try {
      const allQuestions = await loadNameQuestions();
      setQuestions(allQuestions.slice(0, count));
      setAnswers([]);
      setCurrentIndex(0);
      setPhase("play");
    } catch (e: any) {
      setError(e?.message || "Failed to load game data. Run generate_game_data.py first.");
    } finally {
      setLoading(false);
    }
  };

  const handleAnswer = (userAnswer: string) => {
    const question = questions[currentIndex];
    setAnswers([...answers, { question, userAnswer }]);
    if (currentIndex + 1 >= questions.length) {
      setPhase("results");
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePlayAgain = () => {
    setQuestions([]);
    setAnswers([]);
    setCurrentIndex(0);
    setError(null);
    setPhase("setup");
  };

  return (
    <div className="fixed inset-0 overflow-auto bg-[#050510]">
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-[radial-gradient(circle,rgba(0,212,170,0.12)_0%,transparent_70%)] animate-aurora-1" />
        <div className="absolute top-[10%] right-[-15%] w-[55%] h-[55%] rounded-full bg-[radial-gradient(circle,rgba(123,97,255,0.10)_0%,transparent_70%)] animate-aurora-2" />
        <div className="absolute bottom-[-10%] left-[20%] w-[50%] h-[50%] rounded-full bg-[radial-gradient(circle,rgba(14,165,233,0.08)_0%,transparent_70%)] animate-aurora-3" />
        <div className="absolute bottom-[20%] right-[10%] w-[40%] h-[40%] rounded-full bg-[radial-gradient(circle,rgba(236,72,153,0.06)_0%,transparent_70%)] animate-aurora-1" />
      </div>

      <div className="relative z-10 flex min-h-screen items-center justify-center px-4 py-12">
        {phase === "setup" && <SetupScreen onStart={handleStart} loading={loading} error={error} />}
        {phase === "play" && questions[currentIndex] && (
          <PlayScreen question={questions[currentIndex]} currentIndex={currentIndex} total={questions.length} onAnswer={handleAnswer} />
        )}
        {phase === "results" && <ResultsScreen answers={answers} onPlayAgain={handlePlayAgain} />}
      </div>
    </div>
  );
}

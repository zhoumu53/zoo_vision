import { MatchPair, NameQuestion, BehaviorQuestion } from "./types";

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export async function loadMatchPairs(): Promise<MatchPair[]> {
  const res = await fetch("/game-data/match-pairs.json");
  if (!res.ok) throw new Error("Failed to load match pairs");
  const data = await res.json();
  return shuffle(data.pairs);
}

export async function loadNameQuestions(): Promise<NameQuestion[]> {
  const res = await fetch("/game-data/name-questions.json");
  if (!res.ok) throw new Error("Failed to load name questions");
  const data = await res.json();
  return shuffle(data.questions);
}

export async function loadBehaviorQuestions(): Promise<BehaviorQuestion[]> {
  const res = await fetch("/game-data/behavior-questions.json");
  if (!res.ok) throw new Error("Failed to load behavior questions");
  const data = await res.json();
  return shuffle(data.questions);
}

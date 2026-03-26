export type ElephantInfo = {
  id: number;
  name: string;
  color: string;
};

export const ELEPHANTS: ElephantInfo[] = [
  { id: 1, name: "Chandra", color: "#F5FFC6" },
  { id: 2, name: "Indi", color: "#B4E1FF" },
  { id: 3, name: "Fahra", color: "#AB87FF" },
  { id: 4, name: "Panang", color: "#EDBBB4" },
  { id: 5, name: "Thai", color: "#C1FF9B" },
];

export const BEHAVIORS = [
  { id: 1, name: "Standing", color: "#7FB069" },
  { id: 2, name: "SleepL", color: "#197278" },
  { id: 3, name: "SleepR", color: "#FF6978" },
];

export function getElephantByName(name: string): ElephantInfo | undefined {
  return ELEPHANTS.find((e) => e.name === name);
}

export function getElephantById(id: number): ElephantInfo | undefined {
  return ELEPHANTS.find((e) => e.id === id);
}

// --------------- Game types ---------------

export type ElephantCrop = {
  image: string;
  elephant_id: number;
  elephant_name: string;
};

export type MatchPair = {
  left: ElephantCrop;
  right: ElephantCrop;
  is_same_elephant: boolean;
};

export type MatchAnswer = {
  pair: MatchPair;
  userAnswer: boolean;
};

export type NameQuestion = {
  image: string;
  elephant_id: number;
  elephant_name: string;
};

export type NameAnswer = {
  question: NameQuestion;
  userAnswer: string;
};

export type BehaviorQuestion = {
  image: string;
  behavior: string;
  elephant_name: string;
};

export type BehaviorAnswer = {
  question: BehaviorQuestion;
  userAnswer: string;
};

// --------------- Galaxy types ---------------

export type Position3D = {
  x: number;
  y: number;
  z: number;
};

export type GalaxyElephant = {
  elephant_id: number;
  elephant_name: string;
  color?: string;
  image_count: number;
  sample_crop_path?: string | null;
  x: number | null;
  y: number | null;
  z: number | null;
};

export type BBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type MatchLevel = "same" | "similar" | "unknown";

export type NearestElephant = {
  elephant_id: number;
  elephant_name: string;
  similarity: number;
  cosine_distance: number;
  match_level: MatchLevel;
  margin: number | null;
  vote_ratio: number | null;
  sample_crop_path: string | null;
  image_count: number;
  position: Position3D | null;
  profile?: string | null;
};

export type DetectedElephant = {
  index: number;
  crop_url: string;
  bbox: BBox;
  nearest_elephants: NearestElephant[];
  uploaded_position: Position3D | null;
  possibly_new?: boolean;
};

export type UploadResult = {
  outcome: "match_found" | "no_elephant_detected";
  original_url: string | null;
  detected_elephants: DetectedElephant[];
};

export type PartialDetection = {
  detected_elephants_partial: Array<{
    index: number;
    bbox: BBox;
    crop_url: string;
  }>;
  original_url: string | null;
};

export type UploadJobStatus = {
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  stage: string;
  result: UploadResult | null;
  partial: PartialDetection | null;
};

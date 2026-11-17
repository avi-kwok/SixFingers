// Mirrors ml/common/api_models.py. Keep in sync by hand.
export type Verdict = "likely_ai" | "uncertain" | "likely_real";

export interface Cue { region: string; cue: string }
export interface JudgeResult {
  model: string;
  confidence: number;
  cues: Cue[];
  explanation: string;
  overrode_detector: boolean;
}
export interface Thresholds { t_ai: number; t_real: number }
export interface DetectorResult {
  score: number;
  band: Verdict;
  backbones: Record<string, number>;
  thresholds: Thresholds;
}
export interface C2PAInfo { present: boolean; ai_generated: boolean | null; issuer: string | null }
export interface Provenance { c2pa: C2PAInfo; exif_hints: string[] }
export interface Timings { validate: number; provenance: number; detector: number; judge: number; total: number }
export interface CheckResponse {
  verdict: Verdict;
  source: "judge" | "detector_only";
  probability_ai: number;
  judge: JudgeResult | null;
  judge_unavailable_reason: "budget" | "timeout" | "error" | "refusal" | "disabled" | null;
  detector: DetectorResult;
  provenance: Provenance;
  model_version: string;
  timings_ms: Timings;
  disclaimer: string;
}

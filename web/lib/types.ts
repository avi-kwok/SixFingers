// Mirrors ml/common/api_models.py. Keep in sync by hand.
export type Verdict = "likely_ai" | "uncertain" | "likely_real";

export interface Cue { region: string; cue: string; box?: [number, number, number, number] | null }  // box: [x0, y0, x1, y1] as 0-1 fractions of the upright image; absent from older API versions
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
  check_id?: string | null;  // for POST /v1/feedback; null when the server could not store it, absent from older API versions
}

export interface FeedbackBody { check_id: string; answer: "right" | "wrong" | "unsure"; actual?: "ai" | "real" | "unknown"; comment?: string }

// Live totals added to GET /v1/scorecard. feedback_right is null until there are at least 20 answers.
export interface Usage { checks_total: number; checks_last_30_days: number; feedback_answers: number; feedback_right: number | null }

// GET /v1/scorecard, written offline by ml/eval/scorecard.py. Rates are shares 0-1.
export interface ScorecardOverall { split: string; n: number; auroc: number | null; tpr_at_5fpr: number | null; ece: number | null }
export interface ScorecardVariant {
  overall: ScorecardOverall | null;
  generators: { generator: string; split: string; n: number; catch_rate: number }[];
  real_sources: { source: string; split: string; n: number; false_alarm_rate: number }[];
}
export interface Scorecard {
  model_version: string;
  date: string;
  t_ai: number;
  detector: { clean: ScorecardVariant; compressed: ScorecardVariant };
  judge: { model: string; split: string; n: number; accuracy: number; uncertain_rate: number; calibrated_ece: number | null; calibration_method: string } | null;
  usage?: Usage | null;
}

import { TOKEN_HEADER, getToken } from "./turnstile";
import type { AudioCheckResponse, CheckResponse, FeedbackBody, Scorecard, VideoCheckResponse } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const MOCK = process.env.NEXT_PUBLIC_MOCK === "1";
export const MAX_BYTES = 10 * 1024 * 1024;
export const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
export const VIDEO_MAX_BYTES = 100 * 1024 * 1024;
export const VIDEO_ACCEPT = ["video/mp4", "video/webm", "video/quicktime"];
export const AUDIO_MAX_BYTES = 25 * 1024 * 1024;
// Browsers name audio types inconsistently (audio/x-wav, audio/x-m4a...), so any audio/* is sent and the server sniffs it.
export const AUDIO_ACCEPT = "audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm";

export class ApiError extends Error {
  constructor(public kind: "rate" | "big" | "type" | "refused" | "link" | "timeout" | "network" | "other", message: string) {
    super(message);
  }
}

export const MOCK_RESULT: CheckResponse = {
  verdict: "likely_ai",
  source: "judge",
  probability_ai: 0.94,
  judge: {
    model: "claude-haiku-5-5",
    confidence: 0.82,
    cues: [
      { region: "hands", cue: "Fingers merge and one hand shows six digits.", box: [0.18, 0.55, 0.42, 0.85] },
      { region: "background text", cue: "Signage letters are warped and unreadable.", box: [0.6, 0.08, 0.92, 0.3] },
      { region: "lighting", cue: "Shadows fall in two directions across the scene.", box: null },
    ],
    explanation: "Detector and visual cues agree: anatomy and text artifacts typical of generated images.",
    overrode_detector: false,
  },
  judge_unavailable_reason: null,
  detector: {
    score: 0.91,
    band: "likely_ai",
    backbones: { clip_vitl14: 0.93, dinov2_vitl14: 0.88, siglip: 0.92 },
    thresholds: { t_ai: 0.7, t_real: 0.25 },
  },
  provenance: { c2pa: { present: true, ai_generated: true, issuer: "Example Generator" }, exif_hints: ["trainedAlgorithmicMedia"] },
  model_version: "v0-mock",
  timings_ms: { validate: 3, provenance: 12, detector: 240, judge: 1800, total: 2100 },
  disclaimer: "Mock data. No real image was analysed.",
};

export const MOCK_VIDEO_RESULT: VideoCheckResponse = {
  verdict: "likely_ai",
  overall_verdict: "likely_ai",
  overall_flagged_by: ["picture"],
  check_id: null,
  source: "judge",
  judge: {
    model: "claude-haiku-5-5",
    confidence: 0.8,
    cues: [
      { frame: 5, time_s: 1.406, region: "hands", cue: "Finger count changes between frames." },
      { frame: 11, time_s: 3.281, region: "background", cue: "Sign lettering smears and reflows." },
    ],
    explanation: "Frames 5 and 11 show anatomy and text artifacts typical of generated video.",
    overrode_detector: false,
  },
  judge_unavailable_reason: null,
  detector: {
    score: 0.86,
    band: "likely_ai",
    frames: [
      { index: 1, time_s: 0.156, score: 0.12, band: "likely_real" },
      { index: 2, time_s: 0.469, score: 0.18, band: "likely_real" },
      { index: 3, time_s: 0.781, score: 0.41, band: "uncertain" },
      { index: 4, time_s: 1.094, score: 0.66, band: "uncertain" },
      { index: 5, time_s: 1.406, score: 0.88, band: "likely_ai" },
      { index: 6, time_s: 1.719, score: 0.79, band: "likely_ai" },
      { index: 7, time_s: 2.031, score: 0.35, band: "uncertain" },
      { index: 8, time_s: 2.344, score: 0.22, band: "likely_real" },
      { index: 9, time_s: 2.656, score: 0.52, band: "uncertain" },
      { index: 10, time_s: 2.969, score: 0.84, band: "likely_ai" },
      { index: 11, time_s: 3.281, score: 0.93, band: "likely_ai" },
      { index: 12, time_s: 3.594, score: 0.71, band: "likely_ai" },
      { index: 13, time_s: 3.906, score: 0.46, band: "uncertain" },
      { index: 14, time_s: 4.219, score: 0.27, band: "uncertain" },
      { index: 15, time_s: 4.531, score: 0.15, band: "likely_real" },
      { index: 16, time_s: 4.844, score: 0.09, band: "likely_real" },
    ],
    thresholds: { t_ai: 0.7, t_real: 0.25 },
  },
  audio: { analysed: false, note: "Audio is not analysed yet." },
  duration_s: 5,
  model_version: "v0-mock",
  timings_ms: { validate: 3, frames: 40, detector: 240, judge: 1500, total: 1900 },
  disclaimer: "Mock data. No real video was analysed.",
};

export const MOCK_AUDIO_RESULT: AudioCheckResponse = {
  verdict: "likely_ai",
  check_id: null,
  source: "detector_only",
  audio: {
    analysed: true,
    p_fake: 0.74,
    band: "likely_ai",
    threshold: 0.18,
    model_version: "a0-mock",
    note: "Prototype. Only the first 4 seconds of speech are scored.",
  },
  duration_s: 6.2,
  timings_ms: { validate: 40, detector: 300, total: 340 },
  disclaimer: "Mock data. No real audio was analysed.",
};

export const MOCK_SCORECARD: Scorecard = {
  model_version: "v0-mock",
  date: "2026-01-01",
  t_ai: 0.5,
  detector: {
    clean: {
      overall: { split: "report_test", n: 600, auroc: 0.98, tpr_at_5fpr: 0.92, ece: 0.02 },
      generators: [{ generator: "flux", split: "report_test", n: 300, catch_rate: 0.96 }],
      real_sources: [
        { source: "raise", split: "report_test", n: 300, false_alarm_rate: 0.18 },
        { source: "coco", split: "val", n: 350, false_alarm_rate: 0.04 },
      ],
    },
    compressed: {
      overall: { split: "report_test", n: 600, auroc: 0.97, tpr_at_5fpr: 0.9, ece: 0.03 },
      generators: [{ generator: "flux", split: "report_test", n: 300, catch_rate: 0.95 }],
      real_sources: [{ source: "raise", split: "report_test", n: 300, false_alarm_rate: 0.15 }],
    },
  },
  judge: { model: "claude-haiku-5-5", split: "report_test", n: 300, accuracy: 0.97, uncertain_rate: 0.02, calibrated_ece: 0.03, calibration_method: "platt" },
  usage: { checks_total: 0, checks_last_30_days: 0, feedback_answers: 0, feedback_right: null },
};

const STATUS: Record<number, [ApiError["kind"], string]> = {
  429: ["rate", "Too many checks. The limit is 10 per hour. Try again later."],
  413: ["big", "That image is too large. The limit is 10 MB."],
  415: ["type", "Unsupported file type. Use JPEG, PNG or WebP."],
  422: ["refused", "We can't analyse this image."],
};

const VIDEO_STATUS: Record<number, [ApiError["kind"], string]> = {
  400: ["other", "Could not read this video."],
  413: ["big", "That video is too large or too long. The limits are 100 MB and 60 seconds."],
  415: ["type", "Unsupported file type. Use MP4, WebM or MOV."],
  422: ["refused", "We can't analyse this video."],
  429: ["rate", "Too many checks. The limit is 10 per hour and a video counts as 3. Try again later."],
  503: ["other", "Video checks are not available right now."],
};

const AUDIO_STATUS: Record<number, [ApiError["kind"], string]> = {
  400: ["other", "Could not read this audio."],
  404: ["other", "Audio checks are not turned on for this server."],
  413: ["big", "That audio is too large or too long. The limits are 25 MB and 60 seconds."],
  415: ["type", "Unsupported file type. Use WAV, MP3, FLAC, OGG, M4A or WebM."],
  422: ["other", "No clear speech found, so the voice was not checked."],
  429: ["rate", "Too many checks. The limit is 10 per hour. Try again later."],
  503: ["other", "Audio checks are not available right now."],
};

const LINK_MSG = "That link can't be used.";
const LINK_EXPIRED = "Couldn't fetch that image. The link may have expired or not be public.";

// XHR rather than fetch so we can report upload progress.
export async function checkImage(file: File | string, onProgress: (frac: number) => void): Promise<CheckResponse> {
  if (MOCK) {
    return new Promise((res) => {
      onProgress(1);
      setTimeout(() => res(MOCK_RESULT), 600);
    });
  }
  const token = await getToken();
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/v1/check`);
    if (token) xhr.setRequestHeader(TOKEN_HEADER, token);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError("network", "Could not reach the server. Check your connection and try again."));
    xhr.ontimeout = xhr.onerror;
    xhr.timeout = 60000;
    xhr.onload = () => {
      if (xhr.status === 200) {
        try { return resolve(JSON.parse(xhr.responseText) as CheckResponse); } catch { /* fall through */ }
      }
      if (typeof file === "string" && xhr.status === 400) {
        let d = "";
        try { d = JSON.parse(xhr.responseText).detail; } catch { /* keep empty */ }
        return reject(new ApiError("link", d === LINK_MSG || d === LINK_EXPIRED ? d : "Something went wrong. Try again."));
      }
      if (xhr.status === 504) return reject(new ApiError("timeout", "The link took too long to respond. Try again."));
      if (xhr.status === 502) return reject(new ApiError("other", "Couldn't fetch that image. Try again or upload the file."));
      const [kind, msg] = STATUS[xhr.status] ?? ["other", "Something went wrong on the server. Try again."];
      reject(new ApiError(kind, msg));
    };
    const form = new FormData();
    if (typeof file === "string") form.append("url", file);
    else form.append("image", file);
    xhr.send(form);
  });
}

export async function checkVideo(file: File, onProgress: (frac: number) => void): Promise<VideoCheckResponse> {
  if (MOCK) {
    return new Promise((res) => {
      onProgress(1);
      setTimeout(() => res(MOCK_VIDEO_RESULT), 600);
    });
  }
  const token = await getToken();
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/v1/check-video`);
    if (token) xhr.setRequestHeader(TOKEN_HEADER, token);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError("network", "Could not reach the server. Check your connection and try again."));
    xhr.ontimeout = () => reject(new ApiError("timeout", "The check took too long. Try a shorter video."));
    // Upload of up to 100 MB plus frame sampling and the explainer can take minutes.
    xhr.timeout = 300000;
    xhr.onload = () => {
      if (xhr.status === 200) {
        try { return resolve(JSON.parse(xhr.responseText) as VideoCheckResponse); } catch { /* fall through */ }
      }
      const [kind, msg] = VIDEO_STATUS[xhr.status] ?? ["other", "Something went wrong on the server. Try again."];
      reject(new ApiError(kind, msg));
    };
    const form = new FormData();
    form.append("video", file);
    xhr.send(form);
  });
}

export function checkAudio(file: File, onProgress: (frac: number) => void): Promise<AudioCheckResponse> {
  if (MOCK) {
    return new Promise((res) => {
      onProgress(1);
      setTimeout(() => res(MOCK_AUDIO_RESULT), 600);
    });
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/v1/check-audio`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError("network", "Could not reach the server. Check your connection and try again."));
    xhr.ontimeout = () => reject(new ApiError("timeout", "The check took too long. Try again."));
    xhr.timeout = 120000; // the first audio check also loads the speech model
    xhr.onload = () => {
      if (xhr.status === 200) {
        try { return resolve(JSON.parse(xhr.responseText) as AudioCheckResponse); } catch { /* fall through */ }
      }
      const [kind, msg] = AUDIO_STATUS[xhr.status] ?? ["other", "Something went wrong on the server. Try again."];
      reject(new ApiError(kind, msg));
    };
    const form = new FormData();
    form.append("audio", file);
    xhr.send(form);
  });
}

/** True when the server has the prototype audio check turned on (it reports audio_enabled in /v1/health). */
export async function audioEnabled(): Promise<boolean> {
  if (MOCK) return true;
  try {
    const r = await fetch(`${API_URL}/v1/health`);
    return (await r.json()).audio_enabled === true;
  } catch {
    return false;
  }
}

const FEEDBACK_STATUS: Record<number, string> = {
  404: "This result has expired, so feedback can no longer be sent.",
  409: "Feedback for this result was already sent.",
  429: "Too much feedback from your connection. Try again later.",
};

// Throws an Error whose message is ready to show.
export async function sendFeedback(body: FeedbackBody): Promise<void> {
  let r: Response;
  try {
    r = await fetch(`${API_URL}/v1/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  if (!r.ok) throw new Error(FEEDBACK_STATUS[r.status] ?? "Feedback could not be sent. Try again later.");
}

export async function getModel(): Promise<unknown> {
  if (MOCK) return { version: "v0-mock", metrics: { report_test: { variants: { clean: { tpr_at_5fpr: 0.9, auroc: 0.96 } } } } };
  const r = await fetch(`${API_URL}/v1/model`);
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

// Resolves null when the server has no scorecard (404) or cannot be reached.
export async function getScorecard(): Promise<Scorecard | null> {
  if (MOCK) return MOCK_SCORECARD;
  try {
    const r = await fetch(`${API_URL}/v1/scorecard`, { cache: "no-store" });
    return r.ok ? ((await r.json()) as Scorecard) : null;
  } catch {
    return null;
  }
}

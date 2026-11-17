import type { CheckResponse } from "./types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const MOCK = process.env.NEXT_PUBLIC_MOCK === "1";
export const MAX_BYTES = 10 * 1024 * 1024;
export const ACCEPT = ["image/jpeg", "image/png", "image/webp"];

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
      { region: "hands", cue: "Fingers merge and one hand shows six digits." },
      { region: "background text", cue: "Signage letters are warped and unreadable." },
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

const STATUS: Record<number, [ApiError["kind"], string]> = {
  429: ["rate", "Too many checks. The limit is 10 per hour. Try again later."],
  413: ["big", "That image is too large. The limit is 10 MB."],
  415: ["type", "Unsupported file type. Use JPEG, PNG or WebP."],
  422: ["refused", "We can't analyse this image."],
};

const LINK_MSG = "That link can't be used.";
const LINK_EXPIRED = "Couldn't fetch that image. The link may have expired or not be public.";

// XHR rather than fetch so we can report upload progress.
export function checkImage(file: File | string, onProgress: (frac: number) => void): Promise<CheckResponse> {
  if (MOCK) {
    return new Promise((res) => {
      onProgress(1);
      setTimeout(() => res(MOCK_RESULT), 600);
    });
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/v1/check`);
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

export async function getModel(): Promise<unknown> {
  if (MOCK) return { version: "v0-mock", metrics: { tpr_at_5fpr_clean: 0.9, auroc: 0.96 } };
  const r = await fetch(`${API_URL}/v1/model`);
  if (!r.ok) throw new Error(String(r.status));
  return r.json();
}

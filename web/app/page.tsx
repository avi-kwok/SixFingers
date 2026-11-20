"use client";
import Link from "next/link";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  ACCEPT, AUDIO_ACCEPT, AUDIO_MAX_BYTES, ApiError, MAX_BYTES, MOCK, VIDEO_ACCEPT, VIDEO_MAX_BYTES,
  audioEnabled, checkAudio, checkImage, checkVideo, getModel, sendFeedback,
} from "@/lib/api";
import { frameThumbs } from "@/lib/frames";
import type { AudioCheckResponse, AudioInfo, CheckResponse, FeedbackBody, Verdict, VideoCheckResponse } from "@/lib/types";

const LABEL: Record<Verdict, string> = {
  likely_ai: "Likely AI-generated",
  uncertain: "Uncertain",
  likely_real: "Likely real",
};

type Mode = "image" | "video" | "audio";

export default function Home() {
  const [mode, setMode] = useState<Mode>("video");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CheckResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoCheckResponse | null>(null);
  const [thumbs, setThumbs] = useState<(string | null)[]>([]);
  const [audioResult, setAudioResult] = useState<AudioCheckResponse | null>(null);
  const [audioOn, setAudioOn] = useState<boolean | null>(null); // null until /v1/health answers
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [link, setLink] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const [facts, setFacts] = useState<string | null>(null);

  useEffect(() => {
    getModel().then((m) => {
      const { version, metrics } = m as { version?: string; metrics?: Record<string, unknown> };
      const clean = (metrics?.report_test as { variants?: { clean?: { tpr_at_5fpr?: unknown; auroc?: unknown } } } | undefined)?.variants?.clean;
      const t = clean?.tpr_at_5fpr;
      const a = clean?.auroc;
      const metric = typeof t === "number" ? `${Math.round(t * 100)}% of an unseen generator's images caught at 5% false alarms` : typeof a === "number" ? `AUROC ${a.toFixed(2)}` : null;
      const parts = [version && `Model ${version}`, metric].filter(Boolean);
      setFacts(parts.length ? parts.join(" · ") : null);
    }).catch(() => setFacts(null));
  }, []);

  const run = useCallback(async (f: File | string) => {
    setResult(null);
    setError(null);
    if (typeof f === "string") {
      setFile(null);
      setPreview(null);
    } else {
      const bad = !ACCEPT.includes(f.type) ? "Unsupported file type. Use JPEG, PNG or WebP." : f.size > MAX_BYTES ? "That image is too large. The limit is 10 MB." : null;
      if (bad) { setFile(null); setPreview(null); return setError(bad); }
      setFile(f);
      setPreview(URL.createObjectURL(f));
    }
    setProgress(0);
    setBusy(true);
    try {
      setResult(await checkImage(f, setProgress));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  const runVideo = useCallback(async (f: File) => {
    setVideoResult(null);
    setThumbs([]);
    setError(null);
    const bad = f.type && !VIDEO_ACCEPT.includes(f.type) ? "Unsupported file type. Use MP4, WebM or MOV." : f.size > VIDEO_MAX_BYTES ? "That video is too large. The limit is 100 MB." : null;
    if (bad) { setFile(null); setPreview(null); return setError(bad); }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setProgress(0);
    setBusy(true);
    try {
      const r = await checkVideo(f, setProgress);
      setVideoResult(r);
      setBusy(false); // thumbnails can take a while; do not keep the UI locked
      setThumbs(await frameThumbs(f, r.detector.frames.map((x) => x.time_s)));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  const runAudio = useCallback(async (f: File) => {
    setAudioResult(null);
    setError(null);
    if (f.size > AUDIO_MAX_BYTES) { setFile(null); setPreview(null); return setError("That audio file is too large. The limit is 25 MB."); }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setProgress(0);
    setBusy(true);
    try {
      setAudioResult(await checkAudio(f, setProgress));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }, []);

  const pick = (f: File) => busy ? undefined : (mode === "video" ? runVideo(f) : mode === "audio" ? runAudio(f) : run(f));

  const switchMode = useCallback((m: Mode) => {
    setMode(m);
    setFile(null);
    setPreview(null);
    setResult(null);
    setVideoResult(null);
    setThumbs([]);
    setAudioResult(null);
    setError(null);
  }, []);

  useEffect(() => {
    // audioEnabled() returns false on fetch errors too, so retry a few times
    let alive = true;
    (async () => {
      for (let i = 0; i < 4 && alive; i++) {
        if (i) await new Promise((r) => setTimeout(r, 2000));
        if (await audioEnabled()) { if (alive) setAudioOn(true); return; }
      }
      if (alive) setAudioOn(false);
    })();
    return () => { alive = false; };
  }, []);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (busy) return;
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) { e.preventDefault(); if (mode !== "image") switchMode("image"); run(f); return; }
      const t = e.clipboardData?.getData("text")?.trim() ?? "";
      if (/^https:\/\/\S+$/.test(t) && !(e.target instanceof HTMLInputElement)) {
        e.preventDefault();
        if (mode !== "image") switchMode("image");
        setLink(t);
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [run, mode, busy, switchMode]);

  const isVideo = mode === "video";
  const isAudio = mode === "audio";

  return (
    <>
      <section className="hero" aria-label="Introduction">
        <h1>{isVideo ? "Is this video AI-generated? We check the picture and the voice." : isAudio ? "Does this voice sound AI-generated?" : "Does this image look AI-generated?"}</h1>
        <p className="muted">
          {isVideo
            ? audioOn
              ? "SixFingers checks a video two ways: it looks at the picture (frames scored by the detector and read by a visual judge) and listens to the voice (a voice deepfake check), then gives one overall verdict."
              : audioOn === null
                ? "SixFingers checks the picture of a video (frames scored by the detector and read by a visual judge), and the voice when the voice check is on, then gives one overall verdict."
                : "SixFingers looks at the picture of a video: frames scored by the detector and read by a visual judge, then one verdict. The voice check is off on this server, so the voice is not checked."
            : isAudio
              ? "Prototype. Checks whether speech sounds machine-made, from the first 4 seconds only."
              : "Whole images made by a generator only. Edited, inpainted or upscaled photos are out of scope."}
        </p>
        {isVideo && (
          <ol className="steps" aria-label="How it works">
            <li><b>Upload a video</b></li>
            <li>{audioOn ? "We check the frames and the voice" : "We check the frames"}</li>
            <li><b>One verdict{audioOn ? ", with which part flagged it" : ""}</b></li>
          </ol>
        )}
        {mode === "image" && (
          <>
            <ol className="steps" aria-label="How it works">
              <li><b>Upload</b></li>
              <li>Provenance check</li>
              <li>3 detector probes (CLIP, DINOv2, SigLIP)</li>
              <li>Claude Haiku visual check</li>
              <li><b>Verdict</b></li>
            </ol>
            <p className="facts">{facts && `${facts} · `}<Link href="/scorecard">Full scorecard</Link></p>
          </>
        )}
      </section>
      {MOCK && <p role="status" className="tag">Mock mode: no server call</p>}

      <div className="tabs" role="tablist" aria-label="What to check">
        {(audioOn ? (["video", "image", "audio"] as const) : (["video", "image"] as const)).map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            id={`tab-${m}`}
            aria-selected={mode === m}
            aria-controls="check-panel"
            disabled={busy}
            onClick={() => switchMode(m)}
          >
            {m === "image" ? "Image" : m === "video" ? "Video" : "Audio"}
          </button>
        ))}
      </div>

      <div role="tabpanel" id="check-panel" aria-labelledby={`tab-${mode}`}>
        <label
          className={`drop${over ? " over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault(); setOver(false);
            const f = e.dataTransfer.files[0];
            if (f) pick(f);
          }}
        >
          <input
            ref={input}
            type="file"
            accept={isVideo ? [...VIDEO_ACCEPT, ".mp4", ".webm", ".mov"].join(",") : isAudio ? AUDIO_ACCEPT : ACCEPT.join(",")}
            disabled={busy}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) pick(f); e.target.value = ""; }}
          />
          <strong>
            {isVideo ? "Drop a video or choose a file" : isAudio ? "Drop an audio file or choose a file" : "Drop an image, choose a file, or paste with Ctrl/Cmd+V"}
          </strong>
          <br />
          <span className="muted">
            {isVideo
              ? "MP4, WebM or MOV, up to 100 MB and 60 seconds. A video counts as 3 checks."
              : isAudio
                ? "WAV, MP3, FLAC, OGG, M4A or WebM, up to 25 MB and 60 seconds"
                : "JPEG, PNG or WebP, up to 10 MB"}
          </span>
        </label>

        {mode === "image" && (
          <form
            onSubmit={(e) => { e.preventDefault(); if (link.trim()) run(link.trim()); }}
            style={{ display: "flex", gap: 8, margin: "12px 0" }}
          >
            <input
              type="url"
              aria-label="Image link"
              placeholder="or paste an image link (https://…)"
              value={link}
              disabled={busy}
              onChange={(e) => setLink(e.target.value)}
              style={{ flex: 1, minWidth: 0 }}
            />
            <button type="submit" disabled={busy || !link.trim()}>Check</button>
          </form>
        )}
        {isVideo && (
          <p className="muted"><small>{audioOn ? "The voice check is a prototype and only the first 4 seconds of speech are scored. " : ""}Results can be wrong.</small></p>
        )}
        <p className="muted">
          <small>By uploading{isVideo ? " a video" : isAudio ? " audio" : " or linking an image"} you agree to the <Link href="/terms">Terms</Link>.</small>
        </p>

        {preview && !(mode === "image" && result) && (
          <div className="card">
            {isVideo ? (
              <video className="preview" src={preview} controls muted playsInline />
            ) : isAudio ? (
              <audio src={preview} controls style={{ width: "100%" }} aria-label={file ? `Preview of ${file.name}` : "Preview"} />
            ) : (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img className="preview" src={preview} alt={file ? `Preview of ${file.name}` : "Preview"} />
            )}
          </div>
        )}

        {busy && (
          <div className="card" role="status">
            <label htmlFor="prog">{progress < 1 ? "Uploading…" : "Analysing…"}</label>
            <progress id="prog" value={progress < 1 ? progress : undefined} max={1} />
          </div>
        )}

      </div>

      <div aria-live="polite" className="report">
        {error && <div className="card err" role="alert">{error}</div>}
        {result && <ResultCard r={result} preview={preview} name={file?.name} />}
        {result?.check_id && !MOCK && <FeedbackCard key={result.check_id} checkId={result.check_id} />}
        {videoResult && <VideoResultCard r={videoResult} thumbs={thumbs} />}
        {videoResult?.check_id && !MOCK && <FeedbackCard key={videoResult.check_id} checkId={videoResult.check_id} />}
        {audioResult && <AudioResultCard r={audioResult} />}
        {audioResult?.check_id && !MOCK && <FeedbackCard key={audioResult.check_id} checkId={audioResult.check_id} />}
      </div>
    </>
  );
}

function ResultCard({ r, preview, name }: { r: CheckResponse; preview: string | null; name?: string }) {
  const c2pa = r.provenance.c2pa;
  const cues = r.judge?.cues ?? [];
  const boxed = !!preview && cues.some((c) => c.box);
  const [active, setActive] = useState<number | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [stamp, setStamp] = useState<string | null>(null);
  const hl = (i: number) => (boxed ? { onMouseEnter: () => setActive(i), onMouseLeave: () => setActive(null), onFocus: () => setActive(i), onBlur: () => setActive(null) } : {});

  const download = () => {
    flushSync(() => setStamp(new Date().toISOString().slice(0, 16).replace("T", " ") + " UTC"));
    window.print();  // the report is this card with print CSS; nothing leaves the browser
  };

  return (
    <section className="card" aria-label="Result">
      <p className="print-only"><strong>SixFingers report</strong>{stamp && ` · ${stamp}`}</p>
      {c2pa.present && (
        <p className="banner" role="note">
          Signed by {c2pa.issuer ?? "an unknown issuer"}{c2pa.ai_generated ? ": marked AI-generated" : ""}
        </p>
      )}
      <p>
        <span className={`badge ${r.verdict}`}>{LABEL[r.verdict]}</span>{" "}
        {r.source === "detector_only" && <span className="tag">Detector only</span>}
      </p>
      <p style={{ fontSize: "2rem", fontWeight: 700, margin: "4px 0", color: "var(--ink)" }}>{pct(r.probability_ai)} likely AI</p>
      <p className="muted no-print" style={{ margin: 0 }}><small><Link href="/scorecard#calibration">What this number means</Link></small></p>
      <p className="muted"><small>Detector score: {pct(r.detector.score)}</small></p>
      {r.verdict === "uncertain" && (
        <p className="note">
          <strong>We would rather say we don&apos;t know than guess.</strong> The evidence points both ways, so we are not calling this image AI or real.
          Our calibrated estimate is {pct(r.probability_ai)} likely AI: of images that get this number, about {pct(r.probability_ai)} turn out to be AI.
        </p>
      )}

      {preview && (
        <figure className={`evidence${showBoxes ? "" : " boxes-off"}`}>
          <div className="frame">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="preview" src={preview} alt={`${name ? `Your image ${name}` : "Your image"}${boxed ? ", with numbered boxes matching the cues listed below" : ""}`} />
            {cues.map((c, i) => c.box && (
              <span
                key={i}
                aria-hidden="true"
                className={`box${active === i ? " on" : ""}`}
                style={{ left: `${c.box[0] * 100}%`, top: `${c.box[1] * 100}%`, width: `${(c.box[2] - c.box[0]) * 100}%`, height: `${(c.box[3] - c.box[1]) * 100}%` }}
                onMouseEnter={() => setActive(i)}
                onMouseLeave={() => setActive(null)}
              ><b>{i + 1}</b></span>
            ))}
          </div>
          {boxed && (
            <figcaption className="no-print">
              <label><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Show evidence boxes</label>
            </figcaption>
          )}
        </figure>
      )}

      {r.judge ? (
        <>
          <p>{r.judge.explanation}</p>
          {cues.length > 0 && (
            <ol className="cues">{cues.map((c, i) => (
              <li key={i} className={active === i ? "on" : undefined} tabIndex={boxed && c.box ? 0 : undefined} {...(c.box ? hl(i) : {})}>
                <strong>{c.region}:</strong> {c.cue}{boxed && !c.box && <span className="muted"> (whole image)</span>}
              </li>
            ))}</ol>
          )}
          {r.judge.overrode_detector && (
            <p><strong>Note:</strong> {disagreement(r)}</p>
          )}
        </>
      ) : (
        <p className="muted">{unavailable(r.judge_unavailable_reason)}</p>
      )}

      <h2>Detector</h2>
      <p>Score {pct(r.detector.score)} AI-like. Band: {LABEL[r.detector.band]}.</p>

      <h2>Provenance</h2>
      <p>
        {c2pa.present ? (
          <span className="tag">
            C2PA manifest found{c2pa.ai_generated ? ": marked AI-generated" : ""}{c2pa.issuer ? ` by ${c2pa.issuer}` : ""}
          </span>
        ) : (
          <span className="muted">No C2PA manifest. That proves nothing either way.</span>
        )}
        {r.provenance.exif_hints.map((h) => <span className="tag" key={h}>{h}</span>)}
      </p>

      <p className="muted"><small>Scope: fully AI-generated images only. Edited or upscaled images are not covered. Model {r.model_version}.</small></p>
      <p className="muted"><small>{r.disclaimer}</small></p>
      <p className="no-print">
        <button type="button" onClick={download}>Download report</button>{" "}
        <small className="muted">Opens the print dialog; choose &quot;Save as PDF&quot;. Made in your browser, nothing is uploaded or stored.</small>
      </p>
    </section>
  );
}

type Answer = FeedbackBody["answer"];
type Actual = NonNullable<FeedbackBody["actual"]>;
const ANSWERS: [Answer, string][] = [["right", "Right"], ["wrong", "Wrong"], ["unsure", "Not sure"]];
const ACTUALS: [Actual, string][] = [["ai", "AI-made"], ["real", "Real"], ["unknown", "Don't know"]];
const MAX_COMMENT = 500;

function Choices<T extends string>({ label, options, value, onPick }: { label: string; options: [T, string][]; value: T | null; onPick: (v: T) => void }) {
  return (
    <fieldset className="choices">
      <legend>{label}</legend>
      {options.map(([v, text]) => (
        <button key={v} type="button" className="choice" aria-pressed={value === v} onClick={() => onPick(v)}>{text}</button>
      ))}
    </fieldset>
  );
}

function FeedbackCard({ checkId }: { checkId: string }) {
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [actual, setActual] = useState<Actual | null>(null);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (sent) {
    return <section className="card no-print" aria-label="Feedback"><p role="status" style={{ margin: 0 }}>Thank you. Your answer helps us see how often SixFingers gets it right.</p></section>;
  }
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!answer) return;
    setSending(true);
    setError(null);
    try {
      await sendFeedback({ check_id: checkId, answer, ...(actual && { actual }), ...(comment.trim() && { comment: comment.trim() }) });
      setSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback could not be sent. Try again later.");
    } finally {
      setSending(false);
    }
  };
  return (
    <section className="card no-print feedback" aria-label="Feedback">
      <form onSubmit={submit}>
        <Choices label="Was this right?" options={ANSWERS} value={answer} onPick={setAnswer} />
        {answer && (
          <>
            <Choices label="What was it really? (optional)" options={ACTUALS} value={actual} onPick={setActual} />
            <label htmlFor="fb-comment">Comment (optional)</label>
            <textarea id="fb-comment" rows={3} maxLength={MAX_COMMENT} value={comment} aria-describedby="fb-hint" onChange={(e) => setComment(e.target.value)} />
            <p id="fb-hint" className="muted"><small>{comment.length}/{MAX_COMMENT} characters. Please do not type names, contact details or other personal information. <Link href="/privacy">What we store</Link></small></p>
            <button type="submit" disabled={sending}>{sending ? "Sending…" : "Send"}</button>
          </>
        )}
        {error && <p className="fb-err" role="alert">{error}</p>}
      </form>
    </section>
  );
}

const FLAGGED_BY = { picture: "the picture check", voice: "the voice check" } as const;

function overallNote(r: VideoCheckResponse) {
  const voice = r.audio.analysed ? r.audio.band : null;
  if (r.overall_verdict === "likely_ai") return `Flagged by ${r.overall_flagged_by.map((p) => FLAGGED_BY[p]).join(" and ")}. Either part is enough, so some real videos get flagged.`;
  if (r.overall_verdict === "likely_real") return voice ? "The picture and the voice both look real." : "The picture looks real. There is no voice result.";
  return voice ? "The picture and the voice do not both look real, and neither says AI." : "The picture check is unsure.";
}

function VideoResultCard({ r, thumbs }: { r: VideoCheckResponse; thumbs: (string | null)[] }) {
  const cited = new Set(r.judge?.cues.map((c) => c.frame) ?? []);
  return (
    <section className="card" aria-label="Result">
      <h2>Overall</h2>
      <p><span className={`badge ${r.overall_verdict}`}>{LABEL[r.overall_verdict]}</span></p>
      <p>{overallNote(r)}</p>

      <div className="part">
      <h2>Picture</h2>
      <p>
        <span className={`badge ${r.verdict}`}>{LABEL[r.verdict]}</span>{" "}
        {r.source === "detector_only" && <span className="tag">Detector only</span>}
      </p>
      {r.judge ? (
        <>
          <p>{r.judge.explanation}</p>
          {r.judge.cues.length > 0 && (
            <ul>
              {r.judge.cues.map((c, i) => (
                <li key={i}><strong>Frame #{c.frame} at {c.time_s.toFixed(1)} s, {c.region}:</strong> {c.cue}</li>
              ))}
            </ul>
          )}
          {r.judge.overrode_detector && (
            <p><strong>Note:</strong> the explainer disagreed with the statistical detector (which said &quot;{LABEL[r.detector.band]}&quot;) and overrode it.</p>
          )}
        </>
      ) : (
        <p className="muted">{unavailable(r.judge_unavailable_reason)}</p>
      )}

      <ol className="strip" aria-label="Sampled frames">
        {r.detector.frames.map((f, i) => {
          const isCited = cited.has(f.index);
          const thumb = thumbs[i];
          return (
            <li key={f.index} className={"shot" + (isCited ? " cited" : "")}>
              {thumb && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={thumb} alt={`Frame ${f.index} at ${f.time_s.toFixed(1)} seconds`} />
              )}
              <div>#{f.index} · {f.time_s.toFixed(1)} s</div>
              <div><span className={f.band}>{Math.round(f.score * 100)}% AI-like</span></div>
              {isCited && <span className="tag">Cited</span>}
            </li>
          );
        })}
      </ol>
      <p>Video score {(r.detector.score * 100).toFixed(0)}% AI-like (average of the 4 most AI-like frames). Band: {LABEL[r.detector.band]}.</p>

      </div>

      <div className="part">
        <h2>Voice</h2>
        {r.audio.analysed ? (
          <>
            <p><span className={`badge ${r.audio.band}`}>{LABEL[r.audio.band]}</span></p>
            <AudioScore a={r.audio} />
          </>
        ) : (
          <p className="muted">{/no audio track/i.test(r.audio.note) ? "No voice found in this video." : r.audio.note}</p>
        )}
      </div>

      <p className="muted"><small>Scope: fully AI-generated videos, judged from still frames. Model {r.model_version}.</small></p>
      <p className="muted"><small>{r.disclaimer}</small></p>
    </section>
  );
}

function AudioResultCard({ r }: { r: AudioCheckResponse }) {
  return (
    <section className="card" aria-label="Result">
      <p>
        <span className={`badge ${r.verdict}`}>{LABEL[r.verdict]}</span>{" "}
        <span className="tag">Detector only</span>
      </p>
      <AudioScore a={r.audio} />
      <p className="muted">There is no written explanation for audio: the explainer cannot hear sound.</p>
      <p className="muted"><small>{r.disclaimer}</small></p>
    </section>
  );
}

function AudioScore({ a }: { a: AudioInfo }) {
  if (!a.analysed) return <p className="muted">{a.note}</p>;
  return (
    <>
      <p>
        <span className="tag">Prototype</span> Chance the voice is AI-generated: {(a.p_fake * 100).toFixed(0)}%. Band: {LABEL[a.band]}.
      </p>
      <p className="muted"><small>Limits: only the first 4 seconds are scored, and the model was trained on read English speech. Model {a.model_version}.</small></p>
    </>
  );
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

function disagreement(r: CheckResponse) {
  const lean = { likely_ai: "AI", likely_real: "real", uncertain: "unsure" } as const;
  const found = r.verdict === "likely_ai" ? "AI-generated" : "real";
  const lead = r.detector.band === "uncertain" ? "The detector was unsure" : `The detector leaned ${lean[r.detector.band]}`;
  return `${lead}, but Claude's visual check found it ${found}.`;
}

function unavailable(reason: CheckResponse["judge_unavailable_reason"]) {
  if (reason === "verification") return "We couldn't confirm you're human, so only the detector result is shown.";
  if (reason === "budget") return "The daily limit for written explanations has been reached. Showing the detector result only.";
  return "The explanation step was unavailable. Showing the detector result only.";
}

"use client";
import Link from "next/link";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { ACCEPT, ApiError, MAX_BYTES, MOCK, checkImage, getModel, sendFeedback } from "@/lib/api";
import type { CheckResponse, FeedbackBody, Verdict } from "@/lib/types";

const LABEL: Record<Verdict, string> = {
  likely_ai: "Likely AI-generated",
  uncertain: "Uncertain",
  likely_real: "Likely real",
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CheckResponse | null>(null);
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
      if (!ACCEPT.includes(f.type)) return setError("Unsupported file type. Use JPEG, PNG or WebP.");
      if (f.size > MAX_BYTES) return setError("That image is too large. The limit is 10 MB.");
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

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) { e.preventDefault(); run(f); return; }
      const t = e.clipboardData?.getData("text")?.trim() ?? "";
      if (/^https:\/\/\S+$/.test(t) && !(e.target instanceof HTMLInputElement)) { e.preventDefault(); setLink(t); }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [run]);

  return (
    <>
      <section className="hero" aria-label="Introduction">
        <h1>Does this image look AI-generated?</h1>
        <p className="muted">Whole images made by a generator only. Edited, inpainted or upscaled photos are out of scope.</p>
        <ol className="steps" aria-label="How it works">
          <li><b>Upload</b></li>
          <li>Provenance check</li>
          <li>3 detector probes (CLIP, DINOv2, SigLIP)</li>
          <li>Claude Haiku visual check</li>
          <li><b>Verdict</b></li>
        </ol>
        <p className="facts">{facts && `${facts} · `}<Link href="/scorecard">Full scorecard</Link></p>
      </section>
      {MOCK && <p role="status" className="tag">Mock mode: no server call</p>}

      <label
        className={`drop${over ? " over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault(); setOver(false);
          const f = e.dataTransfer.files[0];
          if (f) run(f);
        }}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT.join(",")}
          disabled={busy}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }}
        />
        <strong>Drop an image, choose a file, or paste with Ctrl/Cmd+V</strong>
        <br /><span className="muted">JPEG, PNG or WebP, up to 10 MB</span>
      </label>
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
      <p className="muted"><small>By uploading or linking an image you agree to the <Link href="/terms">Terms</Link>.</small></p>

      {preview && !result && (
        <div className="card">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="preview" src={preview} alt={file ? `Preview of ${file.name}` : "Preview"} />
        </div>
      )}

      {busy && (
        <div className="card" role="status">
          <label htmlFor="prog">{progress < 1 ? "Uploading…" : "Analysing…"}</label>
          <progress id="prog" value={progress < 1 ? progress : undefined} max={1} />
        </div>
      )}

      <div aria-live="polite" className="report">
        {error && <div className="card err" role="alert">{error}</div>}
        {result && <ResultCard r={result} preview={preview} name={file?.name} />}
        {result?.check_id && !MOCK && <FeedbackCard key={result.check_id} checkId={result.check_id} />}
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

const pct = (p: number) => `${Math.round(p * 100)}%`;

function disagreement(r: CheckResponse) {
  const lean = { likely_ai: "AI", likely_real: "real", uncertain: "unsure" } as const;
  const found = r.verdict === "likely_ai" ? "AI-generated" : "real";
  const lead = r.detector.band === "uncertain" ? "The detector was unsure" : `The detector leaned ${lean[r.detector.band]}`;
  return `${lead}, but Claude's visual check found it ${found}.`;
}

function unavailable(reason: CheckResponse["judge_unavailable_reason"]) {
  if (reason === "budget") return "The daily limit for written explanations has been reached. Showing the detector result only.";
  return "The explanation step was unavailable. Showing the detector result only.";
}

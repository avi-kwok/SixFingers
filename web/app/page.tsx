"use client";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { ACCEPT, ApiError, MAX_BYTES, MOCK, checkImage } from "@/lib/api";
import type { CheckResponse, Verdict } from "@/lib/types";

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
      <h1>Does this image look AI-generated?</h1>
      <p className="muted">Whole images made by a generator only. Edited, inpainted or upscaled photos are out of scope.</p>
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

      {preview && (
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

      <div aria-live="polite">
        {error && <div className="card err" role="alert">{error}</div>}
        {result && <ResultCard r={result} />}
      </div>
    </>
  );
}

function ResultCard({ r }: { r: CheckResponse }) {
  const c2pa = r.provenance.c2pa;
  return (
    <section className="card" aria-label="Result">
      <p>
        <span className={`badge ${r.verdict}`}>{LABEL[r.verdict]}</span>{" "}
        {r.source === "detector_only" && <span className="tag">Detector only</span>}
      </p>

      {r.judge ? (
        <>
          <p>{r.judge.explanation}</p>
          {r.judge.cues.length > 0 && (
            <ul>{r.judge.cues.map((c, i) => <li key={i}><strong>{c.region}:</strong> {c.cue}</li>)}</ul>
          )}
          {r.judge.overrode_detector && (
            <p><strong>Note:</strong> the explainer disagreed with the statistical detector (which said &quot;{LABEL[r.detector.band]}&quot;) and overrode it.</p>
          )}
        </>
      ) : (
        <p className="muted">{unavailable(r.judge_unavailable_reason)}</p>
      )}

      <h2>Detector</h2>
      <p>Score {(r.detector.score * 100).toFixed(0)}% AI-like. Band: {LABEL[r.detector.band]}.</p>

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
    </section>
  );
}

function unavailable(reason: CheckResponse["judge_unavailable_reason"]) {
  if (reason === "budget") return "The daily limit for written explanations has been reached. Showing the detector result only.";
  return "The explanation step was unavailable. Showing the detector result only.";
}

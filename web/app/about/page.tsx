import { getModel } from "@/lib/api";

export const metadata = { title: "About - SixFingers" };
export const dynamic = "force-dynamic";

function flatten(o: unknown, prefix = ""): [string, string][] {
  if (o === null || typeof o !== "object") return [[prefix, String(o)]];
  return Object.entries(o as Record<string, unknown>).flatMap(([k, v]) =>
    flatten(v, prefix ? `${prefix}.${k}` : k),
  );
}

export default async function About() {
  let rows: [string, string][] | null = null;
  try {
    const m = (await getModel()) as { metrics?: unknown; version?: string };
    rows = [...(m.version ? [["version", m.version] as [string, string]] : []), ...flatten(m.metrics ?? {}, "")]
      .filter(([k]) => k);
  } catch { /* API down: show the text without numbers */ }

  return (
    <>
      <h1>How it works</h1>
      <h2>How a video is checked</h2>
      <p>A video gets two checks and one overall verdict. <strong>Picture:</strong> we pull frames from the video, the detector scores each one, and a visual judge (Claude Haiku 5.5) reads them and cites the frames that gave it away. <strong>Voice:</strong> when the voice check is on, a prototype deepfake detector scores the soundtrack, using only the first 4 seconds of speech. A video with no soundtrack, or with only music or background noise and no clear speech, has no voice result, and the overall verdict then uses the picture alone.</p>
      <p>The overall verdict is a simple rule: if the picture or the voice says likely AI, the overall verdict is likely AI and we say which part flagged it. If both say likely real, it is likely real. Anything else is uncertain. Flagging on either part raises false alarms, and the voice check is a prototype, so results can be wrong.</p>
      <p>The rest of this page describes the image check.</p>
      <h2>1. Checks the file&apos;s paper trail</h2>
      <p>Some generators sign their images (C2PA) or write tags in the metadata. If found, we show it. Missing metadata means nothing, since it is easily stripped.</p>
      <h2>2. A statistical detector</h2>
      <p>Several large vision models turn the image into numbers. A small classifier on each learned, from real and AI examples, what generated images tend to look like. Their opinions are averaged into one score, then placed in a band (likely AI, uncertain, likely real).</p>
      <h2>3. A language model explains</h2>
      <p>Claude looks at the image on its own, without the detector scores, and writes a short explanation with visual cues. It may disagree with the detector; if so, we say so. If it is unavailable we show the detector alone.</p>
      <h2>The two percentages</h2>
      <p><strong>X% likely AI</strong> is Claude&apos;s confidence that the image is AI-generated, adjusted on 300 validation images and checked on separate test images, so that, for example, answers of about 80% are right about 80% of the time. If Claude is unavailable, this number is the detector score instead.</p>
      <p><strong>Detector score</strong> is the statistical detector&apos;s own probability, calibrated the same way. When the two disagree we say so, and the first number is the one to read.</p>
      <p>The <a href="/scorecard">scorecard</a> shows what each number means and how often we are right, per image generator and per photo source.</p>
      <h2>Reading a result</h2>
      <ul>
        <li>If the file is signed (C2PA), the signature is shown first, above our verdict, because it says who made the image.</li>
        <li>Numbered boxes on your image mark where the visual check saw each cue. Hover or focus a cue to highlight its box. Links show the cue list only.</li>
        <li>&quot;Uncertain&quot; means the evidence points both ways. We would rather say we don&apos;t know than guess.</li>
        <li>&quot;Download report&quot; builds a printable page in your browser; save it as a PDF from the print dialog. Nothing is uploaded or stored.</li>
      </ul>
      <h2>Limits</h2>
      <ul>
        <li>Only fully AI-generated still images. Edited, inpainted or upscaled images are out of scope.</li>
        <li>Videos: up to 60 seconds and 100 MB, and a video counts as 3 checks. The picture check looks at up to 20 still frames, not at motion. The voice check is a prototype, scores only the first 4 seconds of speech, and is skipped when it finds no clear speech.</li>
        <li>It gives a probability-style judgment, not proof. New generators can fool it.</li>
        <li>Heavy compression and resizing lower accuracy.</li>
      </ul>
      <h2>Measured results</h2>
      {rows && rows.length ? (
        <table>
          <tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}</tbody>
        </table>
      ) : (
        <p className="muted">Metrics are unavailable right now.</p>
      )}
    </>
  );
}

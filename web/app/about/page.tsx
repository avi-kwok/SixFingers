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

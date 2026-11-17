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
      <p>Claude looks at the image together with the detector scores and writes a short explanation with visual cues. It may disagree with the detector; if so, we say so. If it is unavailable we show the detector alone.</p>
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

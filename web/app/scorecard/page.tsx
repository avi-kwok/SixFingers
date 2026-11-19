import { getScorecard } from "@/lib/api";
import type { ScorecardVariant } from "@/lib/types";

export const metadata = { title: "Scorecard - SixFingers" };
export const dynamic = "force-dynamic";

const NAMES: Record<string, string> = {
  flux: "FLUX",
  raise: "RAISE (raw camera photos)",
  coco: "COCO",
  openimages: "Open Images",
  openimages_original: "Open Images originals (full-size camera files)",
};
const name = (k: string) => NAMES[k] ?? k;
const pct = (p: number | null | undefined) => (typeof p === "number" ? `${Math.round(p * 100)}%` : "–");

// Rate for the same row in the compressed run, if it was measured there.
const caught = (v: ScorecardVariant, g: string) => v.generators.find((x) => x.generator === g)?.catch_rate;
const flagged = (v: ScorecardVariant, src: string) => v.real_sources.find((x) => x.source === src)?.false_alarm_rate;

export default async function ScorecardPage() {
  const s = await getScorecard();
  if (!s) {
    return (
      <>
        <h1>Scorecard</h1>
        <p className="muted">The scorecard is unavailable right now. Try again later.</p>
      </>
    );
  }
  const { clean, compressed: comp } = s.detector;
  const j = s.judge;
  const valRows = clean.real_sources.some((r) => r.split === "val");
  return (
    <>
      <h1>Scorecard</h1>
      <p className="muted">Model {s.model_version}, measured {s.date}. All numbers come from images the detector never trained on.</p>

      <h2>What we catch</h2>
      <p>Share of AI images we flag as AI, per image generator. These generators were held back from training, so this is how we do on a generator we have never seen.</p>
      <table>
        <thead><tr><th scope="col">Generator</th><th scope="col">Images</th><th scope="col">Caught</th><th scope="col">Caught after compression</th></tr></thead>
        <tbody>{clean.generators.map((g) => (
          <tr key={g.generator}><th scope="row">{name(g.generator)}</th><td>{g.n}</td><td>{pct(g.catch_rate)}</td><td>{pct(caught(comp, g.generator))}</td></tr>
        ))}</tbody>
      </table>

      <h2>How often real photos are wrongly flagged</h2>
      <p>Share of real photos we wrongly call AI, per photo source. Lower is better.</p>
      <table>
        <thead><tr><th scope="col">Source</th><th scope="col">Photos</th><th scope="col">Wrongly flagged</th><th scope="col">After compression</th></tr></thead>
        <tbody>{clean.real_sources.map((r) => (
          <tr key={r.source}><th scope="row">{name(r.source)}{r.split === "val" && "*"}</th><td>{r.n}</td><td>{pct(r.false_alarm_rate)}</td><td>{pct(flagged(comp, r.source))}</td></tr>
        ))}</tbody>
      </table>
      {valRows && <p className="muted"><small>* Validation photos. The detector never trained on them, but its cut-off was tuned so that about 5% of them are flagged, so these rows sit near 5% by design. &quot;–&quot; means not measured.</small></p>}

      <h2>Clean vs compressed</h2>
      <p>Images shared online are usually re-compressed. We test each image as stored and after compression.</p>
      <table>
        <thead><tr><th scope="col">All test images</th><th scope="col">Clean</th><th scope="col">Compressed</th></tr></thead>
        <tbody>
          <tr><th scope="row">AI images caught when 5% of real photos are flagged</th><td>{pct(clean.overall?.tpr_at_5fpr)}</td><td>{pct(comp.overall?.tpr_at_5fpr)}</td></tr>
          <tr><th scope="row">Ranking score (AUROC: 100% is perfect, 50% is a coin flip)</th><td>{pct(clean.overall?.auroc)}</td><td>{pct(comp.overall?.auroc)}</td></tr>
          <tr><th scope="row">Images tested</th><td>{clean.overall?.n ?? "–"}</td><td>{comp.overall?.n ?? "–"}</td></tr>
        </tbody>
      </table>

      <h2>The visual check (Claude Haiku)</h2>
      {j ? (
        <table>
          <tbody>
            <tr><th scope="row">Right answer (an &quot;uncertain&quot; counts as wrong)</th><td>{pct(j.accuracy)}</td></tr>
            <tr><th scope="row">Answered &quot;uncertain&quot;</th><td>{pct(j.uncertain_rate)}</td></tr>
            <tr><th scope="row">Test images</th><td>{j.n}</td></tr>
          </tbody>
        </table>
      ) : <p className="muted">Not measured for this model.</p>}

      <h2 id="calibration">What the percentage means</h2>
      <p>
        When we say 80% AI, about 80% of such images are AI.
        {j?.calibrated_ece != null && ` On test images the gap between what we say and how often it is true averages ${(j.calibrated_ece * 100).toFixed(1)} percentage points for the visual check`}
        {j?.calibrated_ece != null && clean.overall?.ece != null && ` and ${(clean.overall.ece * 100).toFixed(1)} for the detector`}
        {j?.calibrated_ece != null && "."}
        {" "}A number near 50% means the evidence is mixed.
      </p>

      <h2>Honest limits</h2>
      <ul>
        <li>Only fully AI-generated still images. Edited, inpainted or upscaled photos are out of scope.</li>
        <li>Few test generators. A new generator can fool the detector until we train on it.</li>
        <li>Some kinds of real photos are wrongly flagged far more often than others (see the table above).</li>
        <li>A result is a probability, not proof. Do not use it alone to accuse anyone.</li>
      </ul>
    </>
  );
}

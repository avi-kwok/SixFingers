export const metadata = { title: "Privacy - SixFingers" };

export default function Privacy() {
  return (
    <>
      <h1>Privacy</h1>
      <ul>
        <li>Images are processed in memory and are not stored. A video is written to a temporary folder only while it is checked, and is deleted right after. Nothing is stored. There is no account and no login.</li>
        <li>For the explanation step, the image, or for a video one picture made of up to 20 still frames, is sent to the Anthropic API (Claude) to be analysed. The video file and its audio are never sent. Anthropic&apos;s own terms apply to that step.</li>
        <li>Before the explanation step runs, Cloudflare Turnstile may run a bot check in your browser. Cloudflare receives browser signals for that check; see <a href="https://www.cloudflare.com/turnstile-privacy-policy/">Cloudflare&apos;s Turnstile privacy addendum</a>.</li>
        <li>We log only: time, verdict, score, image or video size and format, video length, latency, and a salted hash of your IP that rotates daily (for rate limiting). We never log image or video bytes, frames or hashes.</li>
        <li>If you paste an image link, our server fetches that image, analyses it in memory and does not store it. We log only the link&apos;s host name, never the full link.</li>
      </ul>

      <h2>Usage counts and feedback</h2>
      <ul>
        <li>For every finished check we add 1 to a daily total, kept per UTC date, input type (image file, image link, video or audio) and verdict. The totals appear on the <a href="/scorecard">scorecard</a>.</li>
        <li>Each result gets a random check ID. For 24 hours the server keeps that ID with the date, input type, verdict, the percentage shown, the model version and whether the explanation step answered, so feedback can be matched to the result it is about. After 24 hours the ID can no longer receive feedback, and it is deleted at the next check. If you send feedback, the ID is kept with your feedback so the same result cannot be answered twice; it is random and not linked to you.</li>
        <li>If you use the &quot;Was this right?&quot; box, we keep your answer (right, wrong or not sure), what you say the image, video or audio really was (if you choose), your comment (if you write one), and the result details listed above. Sending feedback is optional.</li>
        <li>Comments are free text and are stored as written. Please do not type names, contact details or other personal information.</li>
        <li>The counts and feedback never include your IP address or a hash of it, your browser or device details, or the file or link you checked. Feedback is rate-limited the same way as checks: a salted, daily-rotating hash of your IP is kept in memory only and is never saved with the counts or the feedback.</li>
      </ul>
    </>
  );
}

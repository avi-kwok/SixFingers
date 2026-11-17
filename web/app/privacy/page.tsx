export const metadata = { title: "Privacy - SixFingers" };

export default function Privacy() {
  return (
    <>
      <h1>Privacy</h1>
      <ul>
        <li>Your image is processed in memory and is not stored. There is no account and no login.</li>
        <li>For the explanation step, the image is sent to the Anthropic API (Claude) to be analysed. Anthropic&apos;s own terms apply to that step.</li>
        <li>We log only: time, verdict, score, image size and format, latency, and a salted hash of your IP that rotates daily (for rate limiting). We never log image bytes or image hashes.</li>
        <li>If you paste an image link, our server fetches that image, analyses it in memory and does not store it. We log only the link&apos;s host name, never the full link.</li>
      </ul>
    </>
  );
}

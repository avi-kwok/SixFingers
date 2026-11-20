// Thumbnails are cut in the browser from the file the user picked. Nothing is uploaded.

function waitVideo(v: HTMLVideoElement, ev: "loadeddata" | "seeked", ms: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (err?: string) => {
      clearTimeout(timer);
      v.removeEventListener(ev, onOk);
      v.removeEventListener("error", onErr);
      if (err) reject(new Error(err));
      else resolve();
    };
    const onOk = () => finish();
    const onErr = () => finish("decode");
    const timer = setTimeout(() => finish("timeout"), ms);
    v.addEventListener(ev, onOk);
    v.addEventListener("error", onErr);
  });
}

export async function frameThumbs(file: File, times: number[], width = 160): Promise<(string | null)[]> {
  const none = times.map((): string | null => null);
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  try {
    v.src = url;
    await waitVideo(v, "loadeddata", 10000);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx || !v.videoWidth || !v.videoHeight) return none;
    canvas.width = width;
    canvas.height = Math.round((width * v.videoHeight) / v.videoWidth);
    const out: (string | null)[] = [];
    for (const t of times) {
      try {
        // Some containers report no duration; then seek straight to t.
        const end = Number.isFinite(v.duration) ? Math.max(0, v.duration - 0.01) : t;
        const target = Math.min(t, end);
        if (Math.abs(v.currentTime - target) > 0.001) {
          v.currentTime = target;
          await waitVideo(v, "seeked", 5000);
        }
        ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
        out.push(canvas.toDataURL("image/jpeg", 0.7));
      } catch {
        out.push(null);
      }
    }
    return out;
  } catch {
    return none;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Cloudflare Turnstile, explicit rendering. Off (no script, no widget) unless NEXT_PUBLIC_TURNSTILE_SITE_KEY is set.
export const SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY ?? "";
export const TOKEN_HEADER = "cf-turnstile-response";

type Api = {
  render: (el: HTMLElement, o: Record<string, unknown>) => string;
  execute: (id: string) => void;
  reset: (id: string) => void;
};
declare global { interface Window { turnstile?: Api } }

let ready: Promise<Api> | null = null;
let widget: { id: string; settle: (t: string | null) => void } | null = null;

function load(): Promise<Api> {
  ready ??= new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    s.async = true;
    s.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error("turnstile")));
    s.onerror = () => { ready = null; reject(new Error("turnstile")); };
    document.head.appendChild(s);
  });
  return ready;
}

/** A fresh single-use token for one check, or null when Turnstile is off or fails (the server then shows the detector only). */
export async function getToken(): Promise<string | null> {
  if (!SITE_KEY) return null;
  try {
    const api = await load();
    return await new Promise<string | null>((resolve) => {
      let timer: ReturnType<typeof setTimeout>;
      const settle = (t: string | null) => { clearTimeout(timer); resolve(t); };
      timer = setTimeout(() => settle(null), 30000); // leaves time for a visitor to tick a challenge
      if (widget) {
        widget.settle = settle;
        api.reset(widget.id); // tokens are single use, so every check starts a new challenge
        api.execute(widget.id);
        return;
      }
      const box = document.createElement("div");
      box.style.cssText = "position:fixed;bottom:16px;left:50%;transform:translateX(-50%);z-index:50";
      document.body.appendChild(box);
      const id = api.render(box, {
        sitekey: SITE_KEY,
        execution: "execute", // the challenge runs only when a check starts
        appearance: "interaction-only", // hidden unless the visitor has to do something
        callback: (t: string) => widget?.settle(t),
        "error-callback": () => widget?.settle(null),
        "timeout-callback": () => widget?.settle(null),
      });
      widget = { id, settle };
      api.execute(id);
    });
  } catch {
    return null;
  }
}

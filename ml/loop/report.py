"""Nightly report (JSON + Markdown) and Discord post. The webhook URL never reaches logs."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "<redacted>") if secret else text


def to_markdown(r: dict) -> str:
    g = r.get("gate") or {}
    lines = [
        f"# Nightly loop {r['date']}",
        f"- Result: **{r['outcome']}**" + (f" (candidate {r['candidate']})" if r.get("candidate") else ""),
        f"- Live model: {r.get('current_before')} -> {r.get('current_after')}",
        f"- Images added: {r.get('images_added', 'n/a')}",
        f"- Paid spend: ${r.get('spend_usd', 0.0):.2f}",
    ]
    if r.get("error"):
        lines.append(f"- Error: {r['error']}")
    for k in ("candidate_metrics", "current_metrics"):
        m = g.get(k)
        if m:
            lines.append(f"- {k.split('_')[0].title()} on gate: TPR@5%FPR {m['tpr_at_5fpr']:.3f}, real FPR {m['real_fpr']:.3f}")
    for why in g.get("reasons", []):
        lines.append(f"- Rejected because: {why}")
    return "\n".join(lines) + "\n"


def write_report(r: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / f"{r['date']}.json", out_dir / f"{r['date']}.md"
    j.write_text(json.dumps(r, indent=2))
    m.write_text(to_markdown(r))
    return j, m


def post_discord(markdown: str, url: str | None = None) -> bool:
    """Best-effort. Returns False on any failure; the error text is scrubbed of the URL."""
    url = url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return False
    req = urllib.request.Request(
        url,
        data=json.dumps({"content": markdown[:1900]}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "sixfingers-loop"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except Exception as e:  # noqa: BLE001
        print("discord post failed:", redact(f"{type(e).__name__}: {e}", url))
        return False

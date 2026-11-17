"""Step 2: look for signed C2PA manifests and metadata hints. Absence proves nothing."""
import io
import json
import logging
import re

from PIL import Image

from ml.common.api_models import C2PAInfo, Provenance

log = logging.getLogger("sixfingers")

try:
    import c2pa
except Exception:  # shortcut: no C2PA on platforms without the wheel, add the wheel when one exists
    c2pa = None

AI_SOURCE_TYPES = {"trainedalgorithmicmedia", "compositesynthetic"}  # exact IPTC tokens; "composite with..." is out of scope
SOFTWARE_HINTS = ("midjourney", "dall-e", "dall·e", "stable diffusion", "firefly", "comfyui",
                  "automatic1111", "novelai", "flux", "imagen", "ideogram", "leonardo")


def _c2pa(data: bytes, mime: str) -> C2PAInfo:
    if c2pa is None:
        return C2PAInfo()
    try:
        with c2pa.Reader(mime, io.BytesIO(data)) as reader:
            store = json.loads(reader.json())
            state = str(reader.get_validation_state())
    except Exception:  # no manifest raises too; that is the normal case
        return C2PAInfo()
    return _info(store, state)


def _info(store: dict, state: str) -> C2PAInfo:
    valid = state.lower().endswith(("valid", "trusted")) and "invalid" not in state.lower()
    m = (store.get("manifests") or {}).get(store.get("active_manifest"), {})
    ai = False
    for a in m.get("assertions", []):
        for act in (a.get("data") or {}).get("actions", []) if isinstance(a.get("data"), dict) else []:
            if str(act.get("digitalSourceType", "")).lower().rsplit("/", 1)[-1] in AI_SOURCE_TYPES:
                ai = True
    # an unverified manifest is attacker-controlled: no AI badge, no issuer text for the judge prompt
    issuer = str((m.get("signature_info") or {}).get("issuer") or "")[:80] or None
    return C2PAInfo(present=True, valid=valid, ai_generated=(ai or None) if valid else None,
                    issuer=issuer if valid else None)


def _hints(data: bytes, img: Image.Image) -> list[str]:
    hints: list[str] = []
    low = data[:2_000_000].lower()  # XMP/IPTC live near the start; no XML parser needed for a substring check
    for t in re.findall(rb"digitalsourcetype/([a-z]+)", low):
        if t.decode() in AI_SOURCE_TYPES:
            hints.append(f"IPTC digitalSourceType: {t.decode()}")
    texts = [str(v) for v in img.info.values() if isinstance(v, str)]  # PNG text chunks
    try:
        texts.append(str(img.getexif().get(0x0131, "")))  # EXIF Software
    except Exception:
        pass
    for s in texts:
        for h in SOFTWARE_HINTS:
            if h in s.lower():
                hints.append(f"Metadata mentions {h}")
    if b"parameters" in data[:4096] and b"Steps:" in data[:200_000]:
        hints.append("PNG has Stable Diffusion generation parameters")
    return sorted(set(hints))


def read_provenance(data: bytes, mime: str, img: Image.Image) -> Provenance:
    return Provenance(c2pa=_c2pa(data, mime), exif_hints=_hints(data, img))

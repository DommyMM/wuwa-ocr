"""Download the 38 Phantom echo skins as EXTRA templates for the base echo's id.

    py bench/fetch_phantom_icons.py

Phantoms are cosmetic: same cost, same legal sonata sets, same stat pools. So the
phantom FLAG is worthless to an optimizer and we do not try to detect it -- a
Phantom tile matching its base id is the CORRECT answer.

The reason to hold the art anyway is defensive. A Phantom is a RECOLOR, so its hue
is shifted away from the base template. In the four Nightmare families whose sets are
identical to their base (Crownless, Feilian Beringal, Inferno Rider, Thundering Mephis)
the sonata badge is mute, so identity rests on gradient and hue alone -- and for
Feilian Beringal gradient is blind (base-vs-Nightmare template NCC = 0.937), leaving
HUE AS THE ONLY SIGNAL. Feilian Beringal has a phantom skin and its Nightmare does not.
Comparing that shifted hue against non-phantom templates is exactly how a Phantom base
gets flipped to a Nightmare.

Registering the phantom art as a second template under the SAME id removes the trap:
identify.py scores every variant and keeps the best, so the phantom matches phantom art
and still reports the base id.

Saved as Data/EchoPhantoms/<echo_id>.png (id-native, matching Data/Echoes/<id>.webp).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

# Wuthery 403s the default urllib agent (same fix as fetch_stat_icons.py).
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
CDN_BASE = "https://files.wuthery.com"
ENCORE_BASE = "https://api.encore.moe/resource/Data"

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "Data" / "EchoPhantoms"
# The frontend's CDN echo table is the only source carrying phantomIcon; backend
# Data/Echoes.json is the trimmed (id/name/cost/setIds) mirror.
FRONTEND_PUBLIC = BACKEND.parent / "wuwabuilds" / "public"
CDN_ECHOES = FRONTEND_PUBLIC / "Data" / "Echoes.json"


def to_source(raw: str) -> str | Path:
    """Port of wuwabuilds/lib/echo.ts::toImageUrl -- the paths are NOT uniform.

    Since the image mirror (wuwabuilds scripts/mirror_images_to_public.py) the normal
    case is a site-relative /assets/ path, whose file already sits in the frontend's
    public/ dir — read it from disk. The CDN URL forms survive for pre-mirror
    snapshots: newly-shipped echoes are not on Wuthery yet and carry an absolute
    encore URL, so blindly prefixing CDN_BASE yields
    'https://files.wuthery.comhttps://api.encore...'.
    """
    if raw.startswith("/assets/"):
        return FRONTEND_PUBLIC / raw.lstrip("/")
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("/d/"):
        return f"{CDN_BASE}{raw}"
    if raw.startswith("/Game/"):
        return f"{ENCORE_BASE}{raw}"
    return raw


def main() -> int:
    rows = json.loads(CDN_ECHOES.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else list(rows.values())

    urls = {
        str(r["id"]): to_source(r["phantomIcon"])
        for r in rows
        if r.get("phantomIcon")
    }
    OUT.mkdir(parents=True, exist_ok=True)

    def get(item: tuple[str, str | Path]) -> str:
        eid, src = item
        # Keep the source extension: the /assets/ mirror and encore serve .webp,
        # Wuthery .png. cv2.imdecode reads both, and identify.py globs on the id,
        # not the suffix.
        suffix = src.suffix if isinstance(src, Path) else Path(src).suffix
        dest = OUT / f"{eid}{suffix or '.png'}"
        # Suffix-agnostic: a skin fetched as .png pre-mirror must not be
        # re-fetched as .webp — identify.py globs on the id, so both would load.
        if any((OUT / f"{eid}{s}").exists() for s in (".png", ".webp")):
            return f"  have  {eid}"
        try:
            if isinstance(src, Path):
                dest.write_bytes(src.read_bytes())
            else:
                with urlopen(Request(src, headers={"User-Agent": UA}), timeout=30) as r:
                    dest.write_bytes(r.read())
            return f"  got   {eid}{dest.suffix}  ({dest.stat().st_size:,} B)"
        except Exception as exc:
            return f"  FAIL  {eid}  {src}  {exc}"

    print(f"{len(urls)} phantom skins -> {OUT}")
    fails = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for line in ex.map(get, sorted(urls.items())):
            fails += line.startswith("  FAIL")
            print(line)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

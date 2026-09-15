"""Download Phantom echo skins as extra templates under their base echo's id

    py bench/fetch_phantom_icons.py

Phantoms are cosmetic, so a Phantom tile matching its base id is correct and the flag is never detected
But a Phantom is a recolor, and Feilian Beringal splits from its Nightmare by hue alone, so its Phantom can flip
identify.py keeps each id's best-scoring variant, so the phantom matches its own art and still reports the base id
Saved as Data/EchoPhantoms/<echo_id> with the source's extension, id-native like Data/Echoes
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

# Wuthery 403s the default urllib agent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
CDN_BASE = "https://files.wuthery.com"
ENCORE_BASE = "https://api.encore.moe/resource/Data"

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "Data" / "EchoPhantoms"
# Only the frontend's echo table carries phantomIcon, since backend Data/Echoes.json is a trimmed mirror
FRONTEND_PUBLIC = BACKEND.parent / "wuwabuilds" / "public"
CDN_ECHOES = FRONTEND_PUBLIC / "Data" / "Echoes.json"


def to_source(raw: str) -> str | Path:
    """Port of the frontend's toImageUrl in echo.ts, keep the path forms in step

    Site-relative /assets/ paths are already in the frontend's public/ dir, so they read from disk
    Absolute URLs pass through, since prefixing CDN_BASE onto an encore URL breaks it
    /d/ and /Game/ forms remain for pre-mirror snapshots
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
        # Keep the source extension, since identify.py globs on the id and cv2.imdecode reads .webp and .png alike
        suffix = src.suffix if isinstance(src, Path) else Path(src).suffix
        dest = OUT / f"{eid}{suffix or '.png'}"
        # Check both suffixes, since a .png fetched pre-mirror plus a new .webp would both load as variants
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

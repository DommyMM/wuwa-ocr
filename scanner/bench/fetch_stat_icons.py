"""Download the 17 unique stat icons referenced by Data/Stats.json

20 stats share 17 icons since flat and percent HP, ATK and DEF share one each, and their disjoint legal sets split them
So an icon match names the stat in all 9 WuWa languages without OCR
Saved under Data/Stats with the icon URL's file name
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

# Wuthery 403s the default urllib agent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "Data" / "Stats"


def main() -> None:
    stats = json.loads((BACKEND / "Data" / "Stats.json").read_text(encoding="utf-8"))
    urls = {v["icon"].rsplit("/", 1)[-1]: v["icon"] for v in stats.values() if v.get("icon")}
    OUT.mkdir(parents=True, exist_ok=True)

    def get(item: tuple[str, str]) -> str:
        fname, url = item
        dest = OUT / fname
        if dest.exists():
            return f"  have  {fname}"
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                dest.write_bytes(r.read())
            return f"  got   {fname}  ({dest.stat().st_size:,} B)"
        except Exception as exc:
            return f"  FAIL  {fname}  {exc}"

    print(f"{len(urls)} unique icons -> {OUT}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        for line in ex.map(get, sorted(urls.items())):
            print(line)


if __name__ == "__main__":
    sys.exit(main())

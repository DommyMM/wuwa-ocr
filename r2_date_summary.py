"""Summarize local r2-backup images by modified time.

Run `sync_r2.py --run` first if you want mtimes aligned to R2 LastModified.

Examples:
    py backend\r2_date_summary.py
    py backend\r2_date_summary.py --since 2026-06-07T19:00:00-07:00
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_R2_DIR = ROOT / "r2-backup"


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.glob(suffix))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", type=Path, default=DEFAULT_R2_DIR)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()

    since_ts = parse_time(args.since)
    until_ts = parse_time(args.until)

    rows = []
    for path in image_paths(args.r2_dir):
        ts = path.stat().st_mtime
        if since_ts is not None and ts < since_ts:
            continue
        if until_ts is not None and ts >= until_ts:
            continue
        dt = datetime.fromtimestamp(ts).astimezone()
        rows.append((dt, path))

    rows.sort()
    by_day = Counter(dt.date().isoformat() for dt, _ in rows)
    by_hour = Counter(dt.strftime("%Y-%m-%d %H:00 %z") for dt, _ in rows)

    print(f"r2_dir={args.r2_dir}")
    print(f"selected={len(rows)} since={args.since or '*'} until={args.until or '*'}")

    if rows:
        print(f"first={rows[0][0].isoformat()} {rows[0][1].name}")
        print(f"last ={rows[-1][0].isoformat()} {rows[-1][1].name}")

    print("\nBY DAY")
    for key, count in sorted(by_day.items()):
        print(f"{key} {count}")

    print("\nBY HOUR")
    for key, count in sorted(by_hour.items()):
        print(f"{key} {count}")

    if args.show:
        print(f"\nFIRST {args.show}")
        for dt, path in rows[: args.show]:
            print(f"{dt.isoformat()} {path.name}")
        print(f"\nLAST {args.show}")
        for dt, path in rows[-args.show :]:
            print(f"{dt.isoformat()} {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

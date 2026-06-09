"""Stage a date-filtered subset of r2-backup for frontend /bulk-import.

The script hardlinks by default, so staging is fast and does not duplicate image
bytes. It falls back to copying if hardlinks are unavailable.

Run `sync_r2.py --run` first so local mtimes reflect R2 LastModified.

Examples:
    py backend\stage_r2_backfill.py --since 2026-06-07T19:00:00-07:00
    py backend\stage_r2_backfill.py --since 2026-06-07T19:00:00-07:00 --limit 100
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_R2_DIR = ROOT / "r2-backup"
DEFAULT_OUT_DIR = ROOT / "r2-backfill-3.4"


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


def stage_file(src: Path, dest: Path, copy: bool) -> str:
    if dest.exists():
        return "exists"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dest)
        return "copied"
    try:
        os.link(src, dest)
        return "linked"
    except OSError:
        shutil.copy2(src, dest)
        return "copied"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", type=Path, default=DEFAULT_R2_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking")
    parser.add_argument("--clean", action="store_true", help="Clear the output directory first")
    args = parser.parse_args()

    since_ts = parse_time(args.since)
    until_ts = parse_time(args.until)

    selected = []
    for path in image_paths(args.r2_dir):
        ts = path.stat().st_mtime
        if since_ts is not None and ts < since_ts:
            continue
        if until_ts is not None and ts >= until_ts:
            continue
        selected.append(path)

    if args.offset:
        selected = selected[args.offset:]
    if args.limit:
        selected = selected[: args.limit]

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"linked": 0, "copied": 0, "exists": 0}
    for src in selected:
        status = stage_file(src, args.out_dir / src.name, args.copy)
        counts[status] += 1

    print(f"selected={len(selected)} out_dir={args.out_dir}")
    print(f"linked={counts['linked']} copied={counts['copied']} exists={counts['exists']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
batch_ocr.py — full pipeline OCR on every image in r2-backup/.

Sends all 10 regions per image to the running server concurrently,
merges results into one AnalysisData object per image, saves to:
  ../ocr_results.json   (repo root — accessible by both backend and frontend)

Usage:
  py batch_ocr.py
"""
import asyncio
import aiohttp
import argparse
import base64
import cv2
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_URL    = "http://localhost:5000"
R2_BACKUP   = Path(__file__).parent.parent / "r2-backup"
OUT_FILE    = Path(__file__).parent.parent / "ocr_results.json"
DEFAULT_CONCURRENCY = 40  # 10 regions x 4 images in-flight at once

REGIONS = {
    "character": (0.0328, 0.0074, 0.3021, 0.0833),
    "watermark":  (0.0073, 0.0741, 0.1304, 0.1370),
    "forte":      (0.4057, 0.0222, 0.7422, 0.5917),
    "sequences":  (0.0703, 0.4787, 0.3318, 0.5843),
    "weapon":     (0.7542, 0.3843, 0.9828, 0.5843),
    "echo1":      (0.0125, 0.6019, 0.2042, 0.9843),
    "echo2":      (0.2057, 0.6019, 0.3974, 0.9843),
    "echo3":      (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4":      (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5":      (0.7911, 0.6019, 0.9833, 0.9843),
}


def crop_b64(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    crop = img[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
    _, buf = cv2.imencode(".png", crop)
    return base64.b64encode(buf).decode()


async def post_region(session, sem, img_name, region, b64):
    headers = {"X-OCR-Region": region, "Content-Type": "application/json"}
    async with sem:
        try:
            async with session.post(
                f"{BASE_URL}/api/ocr",
                json={"image": b64},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                return img_name, region, await resp.json()
        except Exception as e:
            return img_name, region, {"success": False, "error": str(e)}


async def run_all(images, concurrency: int):
    tasks = []
    sem   = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        for img_path, img in images:
            for region, coords in REGIONS.items():
                b64 = crop_b64(img, *coords)
                tasks.append(post_region(session, sem, img_path.name, region, b64))

        print(f"  Firing {len(tasks)} requests ({concurrency} concurrent) ...")
        return await asyncio.gather(*tasks)


def merge_results(raw_results):
    """Group by image name, merge all region results into one AnalysisData dict."""
    by_image = {}
    for img_name, region, body in raw_results:
        if img_name not in by_image:
            by_image[img_name] = {"image": img_name}
        if not body.get("success"):
            continue
        analysis = body.get("analysis")
        if analysis:
            by_image[img_name][region] = analysis

    return list(by_image.values())


def is_valid(entry):
    """Skip entries that look like invalid/corrupt images."""
    # Must have character and at least one echo
    has_char  = isinstance(entry.get("character"), dict) and entry["character"].get("id")
    has_echo  = any(isinstance(entry.get(f"echo{i}"), dict) for i in range(1, 6))
    has_water = isinstance(entry.get("watermark"), dict) and entry["watermark"].get("uid")
    return has_char and has_echo and has_water


def parse_since(value: str | None) -> float | None:
    if not value:
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(
            "Invalid --since. Use ISO format, e.g. "
            "2026-06-07T19:00:00-07:00"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def select_images(args) -> list[Path]:
    all_images = sorted(
        list(args.r2_dir.glob("*.jpg")) +
        list(args.r2_dir.glob("*.jpeg")) +
        list(args.r2_dir.glob("*.png")) +
        list(args.r2_dir.glob("*.webp"))
    )

    since_ts = parse_since(args.since)
    if since_ts is not None:
        all_images = [path for path in all_images if path.stat().st_mtime >= since_ts]

    if args.offset:
        all_images = all_images[args.offset:]
    if args.limit:
        all_images = all_images[:args.limit]

    return all_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", type=Path, default=R2_BACKUP)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    parser.add_argument("--since", help="Only process files with mtime >= ISO timestamp")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    all_images = select_images(args)
    if not all_images:
        print(f"ERROR: no images selected from {args.r2_dir}")
        sys.exit(1)

    if args.list_only:
        print(f"Selected {len(all_images)} images")
        for path in all_images:
            print(path)
        return

    print(f"[1/4] Loading {len(all_images)} images ...")
    loaded = [(p, cv2.imread(str(p))) for p in all_images]
    loaded = [(p, img) for p, img in loaded if img is not None]
    print(f"      {len(loaded)} loaded successfully")

    print(f"[2/4] Running full OCR via server ...")
    raw = asyncio.run(run_all(loaded, args.concurrency))

    print(f"[3/4] Merging results ...")
    merged  = merge_results(raw)
    valid   = [e for e in merged if is_valid(e)]
    skipped = len(merged) - len(valid)
    print(f"      {len(valid)} valid  |  {skipped} skipped (no character/watermark)")

    print(f"[4/4] Saving to {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*52}")
    print(f"  Images processed : {len(loaded)}")
    print(f"  Valid entries    : {len(valid)}")
    print(f"  Output           : {args.out}")
    print()


if __name__ == "__main__":
    main()

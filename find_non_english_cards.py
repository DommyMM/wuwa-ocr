"""Find likely non-English build-card screenshots in a local r2-backup folder.

This is a fast scout, not the full import parser. By default it OCRs only the
stat-name strips from the five echo panels, then writes live JSONL checkpoints
plus final candidate manifests for targeted parser validation.

Usage:
  py find_non_english_cards.py ../r2-backup
  py find_non_english_cards.py ../r2-backup --workers 8 --limit 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import pytesseract
from rapidfuzz import fuzz


BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
DEFAULT_R2_DIR = ROOT / "r2-backup"
DEFAULT_OUT = BACKEND_DIR / "forensics" / "non_english_ocr_scan"
DEFAULT_LANG = "eng+fra+jpn+chi_tra+chi_sim"

ECHO_BAND = {"x1": 0.0125, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843}
ECHO_REGIONS = [
    {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
]
ECHO_SUBSTAT_NAMES = {"x1": 36 / 368, "x2": 290 / 368, "y1": 228 / 413, "y2": 400 / 413}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LANG_KEYS = ("fr", "ja", "zh-Hans", "zh-Hant")


def image_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def parse_since(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit("Invalid --since. Use ISO format like 2026-06-07T19:00:00-07:00") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def select_images(args: argparse.Namespace) -> list[Path]:
    paths = image_paths(args.r2_dir)
    since_ts = parse_since(args.since)
    if since_ts is not None:
        paths = [path for path in paths if path.stat().st_mtime >= since_ts]
    if args.offset:
        paths = paths[args.offset:]
    if args.limit:
        paths = paths[:args.limit]
    return paths


def normalize_alias(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("％", "%")
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    text = text.casefold()
    return re.sub(r"[\s:：・.'’`´\-_/()+]+", "", text)


def load_aliases(stats_path: Path) -> list[dict[str, str]]:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    aliases: list[dict[str, str]] = []
    for canonical, translations in stats.items():
        en_norm = normalize_alias(str(translations.get("en", "")))
        for lang in LANG_KEYS:
            value = str(translations.get(lang, "") or "")
            normalized = normalize_alias(value)
            if not value or not normalized or normalized == en_norm:
                continue
            aliases.append({
                "canonical": canonical,
                "lang": lang,
                "label": value,
                "normalized": normalized,
            })
    return aliases


def crop_echo_band(img) -> Any:
    h, w = img.shape[:2]
    return img[
        round(ECHO_BAND["y1"] * h): round(ECHO_BAND["y2"] * h),
        round(ECHO_BAND["x1"] * w): round(ECHO_BAND["x2"] * w),
    ]


def _crop_norm(img, region: dict[str, float]):
    h, w = img.shape[:2]
    return img[
        round(region["y1"] * h): round(region["y2"] * h),
        round(region["x1"] * w): round(region["x2"] * w),
    ]


def crop_echo_name_strips(img) -> Any:
    strips = []
    for region in ECHO_REGIONS:
        echo = _crop_norm(img, region)
        strip = _crop_norm(echo, ECHO_SUBSTAT_NAMES)
        if strip.size:
            strips.append(strip)
    if not strips:
        return crop_echo_band(img)

    target_width = max(strip.shape[1] for strip in strips)
    padded = []
    for strip in strips:
        if strip.shape[1] == target_width:
            padded.append(strip)
            continue
        pad = target_width - strip.shape[1]
        padded.append(cv2.copyMakeBorder(strip, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return cv2.vconcat(padded)


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(scaled, (0, 0), 3)
    sharp = cv2.addWeighted(scaled, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 135, 255, cv2.THRESH_BINARY)
    return thresh


def script_counts(text: str) -> dict[str, int]:
    return {
        "japanese": len(re.findall(r"[\u3040-\u30ff]", text)),
        "cjk": len(re.findall(r"[\u3400-\u9fff]", text)),
        "hangul": len(re.findall(r"[\uac00-\ud7af]", text)),
        "latin_accent": len(re.findall(r"[À-ÖØ-öø-ÿ]", text)),
    }


def alias_hits(text: str, aliases: list[dict[str, str]], cutoff: int) -> list[dict[str, Any]]:
    normalized_text = normalize_alias(text)
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for alias in aliases:
        score = 100 if alias["normalized"] in normalized_text else fuzz.partial_ratio(alias["normalized"], normalized_text)
        if score < cutoff:
            continue
        key = (alias["canonical"], alias["lang"], alias["label"])
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            "canonical": alias["canonical"],
            "lang": alias["lang"],
            "label": alias["label"],
            "score": round(float(score), 1),
        })
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:12]


def classify(text: str, hits: list[dict[str, Any]], scripts: dict[str, int]) -> tuple[bool, str, int]:
    if hits:
        langs = Counter("zh" if hit["lang"].startswith("zh-") else hit["lang"] for hit in hits)
        lang, count = langs.most_common(1)[0]
        if count >= 3:
            return True, lang, min(98, 65 + count * 8)
    if scripts["cjk"] >= 3 and scripts["cjk"] > scripts["japanese"] * 2:
        return True, "zh", 85 + min(10, scripts["cjk"])
    if scripts["japanese"] > 0:
        return True, "ja", 90 + min(10, scripts["japanese"])
    if scripts["cjk"] > 0:
        return True, "zh", 85 + min(10, scripts["cjk"])
    if scripts["hangul"] > 0:
        return True, "ko", 85 + min(10, scripts["hangul"])
    if scripts["latin_accent"] > 0:
        return True, "latin-accent", 70 + min(10, scripts["latin_accent"])
    if hits:
        langs = Counter("zh" if hit["lang"].startswith("zh-") else hit["lang"] for hit in hits)
        lang, count = langs.most_common(1)[0]
        return count >= 2, lang, min(95, 55 + count * 10)
    return False, "unknown", 0


def scan_one(
    path: Path,
    aliases: list[dict[str, str]],
    lang: str,
    cutoff: int,
    crop_mode: str,
    ocr_timeout: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "decode_ok": False,
        "candidate": False,
        "language": "unknown",
        "confidence": 0,
        "scripts": {},
        "hits": [],
        "text": "",
        "error": "",
        "elapsed_ms": 0,
    }
    started = time.perf_counter()
    img = cv2.imread(str(path))
    if img is None:
        base["error"] = "decode_failed"
        base["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return base

    base["decode_ok"] = True
    base["width"] = int(img.shape[1])
    base["height"] = int(img.shape[0])
    try:
        crop = crop_echo_name_strips(img) if crop_mode == "names" else crop_echo_band(img)
        ocr_image = crop if crop_mode == "names" else preprocess_for_ocr(crop)
        text = pytesseract.image_to_string(ocr_image, lang=lang, config="--psm 6", timeout=ocr_timeout)
    except Exception as exc:
        base["error"] = str(exc)
        base["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return base

    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    scripts = script_counts(text)
    hits = alias_hits(text, aliases, cutoff)
    candidate, language, confidence = classify(text, hits, scripts)
    base.update({
        "candidate": candidate,
        "language": language,
        "confidence": confidence,
        "scripts": scripts,
        "hits": hits,
        "text": text[:1200],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    })
    return base


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["file", "language", "confidence", "elapsed_ms", "bytes", "width", "height", "hit_count", "top_hits", "text"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "file": row["file"],
                "language": row["language"],
                "confidence": row["confidence"],
                "elapsed_ms": row.get("elapsed_ms", 0),
                "bytes": row["bytes"],
                "width": row.get("width", 0),
                "height": row.get("height", 0),
                "hit_count": len(row["hits"]),
                "top_hits": "; ".join(f"{hit['lang']}:{hit['canonical']}={hit['label']}({hit['score']})" for hit in row["hits"][:5]),
                "text": row["text"].replace("\n", " / "),
            })


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find likely non-English build-card screenshots")
    parser.add_argument("r2_dir", type=Path, nargs="?", default=DEFAULT_R2_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=BACKEND_DIR / "Data" / "Stats.json")
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--since")
    parser.add_argument("--cutoff", type=int, default=82)
    parser.add_argument("--crop-mode", choices=("names", "band"), default="names")
    parser.add_argument("--ocr-timeout", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Skip files already present in results.jsonl")
    args = parser.parse_args()

    if not args.stats.exists():
        raise SystemExit(f"Stats translations not found: {args.stats}. Run sync_backend.py first.")

    paths = select_images(args)
    if not paths:
        raise SystemExit(f"No images selected from {args.r2_dir}")

    aliases = load_aliases(args.stats)
    args.out.mkdir(parents=True, exist_ok=True)
    results_jsonl = args.out / "results.jsonl"
    candidates_jsonl = args.out / "candidates.jsonl"
    progress_json = args.out / "progress.json"

    completed: set[str] = set()
    if args.resume and results_jsonl.exists():
        for line in results_jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("file"):
                completed.add(str(row["file"]))
        paths = [path for path in paths if path.name not in completed]

    if not args.resume:
        results_jsonl.write_text("", encoding="utf-8")
        candidates_jsonl.write_text("", encoding="utf-8")

    total = len(paths)
    print(
        f"Scanning {total} image(s) with {args.workers} workers; "
        f"aliases={len(aliases)} crop={args.crop_mode} resume_skipped={len(completed)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(scan_one, path, aliases, args.lang, args.cutoff, args.crop_mode, args.ocr_timeout)
            for path in paths
        ]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            append_jsonl(results_jsonl, row)
            if row["candidate"]:
                append_jsonl(candidates_jsonl, row)
                counts[row["language"]] += 1
            elif row["error"]:
                counts["error"] += 1
            else:
                counts["ok"] += 1

            if i == 1 or i % max(1, args.progress_every) == 0 or i == total:
                elapsed = max(0.001, time.perf_counter() - started)
                rate = i / elapsed
                remaining = (total - i) / rate if rate > 0 else 0
                candidate_count = sum(v for k, v in counts.items() if k not in {"ok", "error"})
                progress = {
                    "processed": i,
                    "total": total,
                    "percent": round(i * 100 / max(1, total), 2),
                    "rate_per_sec": round(rate, 3),
                    "eta_seconds": round(remaining),
                    "counts": dict(counts),
                    "last_file": row["file"],
                }
                write_progress(progress_json, progress)
                print(
                    f"processed {i}/{total} ({progress['percent']}%) "
                    f"rate={progress['rate_per_sec']}/s eta={progress['eta_seconds']}s "
                    f"candidates={candidate_count} last={row['file']}",
                    flush=True,
                )

    candidates = sorted(
        [row for row in rows if row["candidate"]],
        key=lambda row: (row["confidence"], len(row["hits"]), row["file"]),
        reverse=True,
    )
    summary = {
        "total": len(rows),
        "decoded": sum(1 for row in rows if row["decode_ok"]),
        "candidates": len(candidates),
        "languages": dict(Counter(row["language"] for row in candidates)),
        "known_samples": {
            name: next((row for row in rows if row["file"].lower() == name.lower()), None)
            for name in ("f32421ba8b1f3dc0.jpg", "cce1a0f29186891b.jpg", "5e17036118784d4b.jpg")
        },
    }

    (args.out / "candidates.json").write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.out / "candidates.csv", candidates)

    print(f"\nScanned: {summary['total']}  decoded: {summary['decoded']}  candidates: {summary['candidates']}")
    print(f"Languages: {summary['languages']}")
    print(f"Wrote: {args.out / 'candidates.json'}")
    print(f"Wrote: {args.out / 'candidates.csv'}")
    print(f"Wrote: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

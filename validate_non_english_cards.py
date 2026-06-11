"""Validate localized build-card candidates through the backend echo parser.

This consumes candidate rows produced by ``find_non_english_cards.py`` and
runs the same frontend echo crops through ``card.process_card``. Results are
written as JSONL plus a compact summary and failure CSV under forensics.

Usage:
  py validate_non_english_cards.py
  py validate_non_english_cards.py --workers 8 --limit 20
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

from card import process_card


BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
DEFAULT_INPUT = BACKEND_DIR / "forensics" / "non_english_validation" / "build_card_signal_candidates.json"
DEFAULT_OUT = BACKEND_DIR / "forensics" / "non_english_validation"

ECHO_REGIONS = {
    "echo1": {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2": {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3": {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4": {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5": {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}


def crop_region(img, region: dict[str, float]):
    h, w = img.shape[:2]
    return img[
        round(region["y1"] * h): round(region["y2"] * h),
        round(region["x1"] * w): round(region["x2"] * w),
    ]


def load_candidates(path: Path, *, only_build_signal: bool) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if only_build_signal:
        rows = [row for row in rows if row.get("build_card_signal", True)]
    return rows


def echo_ok(result: dict[str, Any], min_substats: int) -> bool:
    if not result.get("success"):
        return False
    analysis = result.get("analysis") or {}
    main = analysis.get("main") or {}
    substats = analysis.get("substats") or []
    return bool(main.get("name")) and len(substats) >= min_substats


def scan_image(row: dict[str, Any], r2_dir: str, min_substats: int) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(r2_dir) / row["file"]
    result: dict[str, Any] = {
        "file": row["file"],
        "language": row.get("language", "unknown"),
        "confidence": row.get("confidence", 0),
        "hit_count": row.get("hit_count", len(row.get("hits", []))),
        "line_count": row.get("line_count", 0),
        "decode_ok": False,
        "echoes": {},
        "ok_echoes": 0,
        "failed_echoes": 0,
        "elapsed_ms": 0,
    }

    img = cv2.imread(str(path))
    if img is None:
        result["error"] = "decode_failed"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return result

    result["decode_ok"] = True
    for region, coords in ECHO_REGIONS.items():
        crop = crop_region(img, coords)
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = process_card(crop, region)
        ok = echo_ok(parsed, min_substats)
        result["echoes"][region] = {
            "ok": ok,
            "main": (parsed.get("analysis") or {}).get("main", {}),
            "substats": (parsed.get("analysis") or {}).get("substats", []),
            "name": (parsed.get("analysis") or {}).get("name", {}),
            "element": (parsed.get("analysis") or {}).get("element", ""),
            "error": parsed.get("error", ""),
        }
        result["ok_echoes" if ok else "failed_echoes"] += 1

    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return result


def flatten_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("decode_ok"):
            failures.append({
                "file": row["file"],
                "language": row["language"],
                "region": "",
                "main": "",
                "substat_count": 0,
                "error": row.get("error", "decode_failed"),
            })
            continue
        for region, echo in row["echoes"].items():
            if echo["ok"]:
                continue
            failures.append({
                "file": row["file"],
                "language": row["language"],
                "region": region,
                "main": echo.get("main", {}).get("name", ""),
                "substat_count": len(echo.get("substats", [])),
                "error": echo.get("error", ""),
            })
    return failures


def write_failures_csv(path: Path, failures: list[dict[str, Any]]) -> None:
    fields = ["file", "language", "region", "main", "substat_count", "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def summarize(rows: list[dict[str, Any]], failures: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    by_language: dict[str, dict[str, int]] = defaultdict(lambda: {
        "images": 0,
        "echoes": 0,
        "ok_echoes": 0,
        "failed_echoes": 0,
    })
    for row in rows:
        lang = row["language"]
        by_language[lang]["images"] += 1
        by_language[lang]["echoes"] += len(row.get("echoes", {}))
        by_language[lang]["ok_echoes"] += row.get("ok_echoes", 0)
        by_language[lang]["failed_echoes"] += row.get("failed_echoes", 0)

    return {
        "counts": {
            "images": len(rows),
            "decode_failed": sum(1 for row in rows if not row.get("decode_ok")),
            "echoes": sum(len(row.get("echoes", {})) for row in rows),
            "ok_echoes": sum(row.get("ok_echoes", 0) for row in rows),
            "failed_echoes": sum(row.get("failed_echoes", 0) for row in rows),
            "failed_images": len({failure["file"] for failure in failures}),
        },
        "by_language": dict(sorted(by_language.items())),
        "failure_reasons": dict(Counter(failure["error"] or "parse_miss" for failure in failures)),
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate localized build-card candidates")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--r2-dir", type=Path, default=ROOT / "r2-backup")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--min-substats", type=int, default=3)
    parser.add_argument("--include-low-signal", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    candidates = load_candidates(args.input, only_build_signal=not args.include_low_signal)
    if args.offset:
        candidates = candidates[args.offset:]
    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        raise SystemExit(f"No candidates selected from {args.input}")

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "build_card_parser_results.jsonl"
    summary_path = args.out / "build_card_parser_summary.json"
    failures_path = args.out / "build_card_parser_failures.csv"
    results_path.write_text("", encoding="utf-8")

    print(
        f"Validating {len(candidates)} image(s), workers={args.workers}, "
        f"min_substats={args.min_substats}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(scan_image, row, str(args.r2_dir), args.min_substats)
            for row in candidates
        ]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

            if i == 1 or i % max(1, args.progress_every) == 0 or i == len(candidates):
                elapsed = max(0.001, time.perf_counter() - started)
                rate = i / elapsed
                remaining = (len(candidates) - i) / rate if rate > 0 else 0
                ok_echoes = sum(item.get("ok_echoes", 0) for item in rows)
                failed_echoes = sum(item.get("failed_echoes", 0) for item in rows)
                print(
                    f"processed {i}/{len(candidates)} "
                    f"rate={rate:.3f}/s eta={remaining:.0f}s "
                    f"ok_echoes={ok_echoes} failed_echoes={failed_echoes} "
                    f"last={row['file']}",
                    flush=True,
                )

    failures = flatten_failures(rows)
    summary = summarize(rows, failures, time.perf_counter() - started)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_failures_csv(failures_path, failures)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote: {results_path}", flush=True)
    print(f"Wrote: {summary_path}", flush=True)
    print(f"Wrote: {failures_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark current card.py against a historical card.py.

The benchmark exports an old ``card.py`` from git history, then runs the same
frontend-style regions for small image sets through old and current parser
versions. It measures subprocess wall time, so Python/module startup is
included, and each child reports its internal processing time.

Usage:
  py benchmark_card_versions.py
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
DEFAULT_OUT = BACKEND_DIR / "forensics" / "card_version_benchmark"
DEFAULT_OLD_REF = "91f9d2b"

DEFAULT_ENGLISH = [
    "00055f05eb843ecf.jpg",
    "0009c98f56d51b47.jpg",
    "0015bcd42f0b33b3.jpg",
]
DEFAULT_LOCALIZED = [
    "f32421ba8b1f3dc0.jpg",
    "cce1a0f29186891b.jpg",
    "5e17036118784d4b.jpg",
]

IMPORT_REGIONS = {
    "character": {"x1": 0.0328, "x2": 0.3021, "y1": 0.0074, "y2": 0.0833},
    "watermark": {"x1": 0.0073, "x2": 0.1304, "y1": 0.0741, "y2": 0.1370},
    "forte": {"x1": 0.4057, "x2": 0.7422, "y1": 0.0222, "y2": 0.5917},
    "sequences": {"x1": 0.0703, "x2": 0.3318, "y1": 0.4787, "y2": 0.5843},
    "weapon": {"x1": 0.7542, "x2": 0.9828, "y1": 0.3843, "y2": 0.5843},
    "echo1": {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2": {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3": {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4": {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5": {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}

CARD_MODULE = None


def crop_region(img: np.ndarray, region: dict[str, float]) -> np.ndarray:
    h, w = img.shape[:2]
    return img[
        round(region["y1"] * h): round(region["y2"] * h),
        round(region["x1"] * w): round(region["x2"] * w),
    ]


def encode_tasks(image_paths: list[Path]) -> list[tuple[str, str, bytes]]:
    tasks: list[tuple[str, str, bytes]] = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to decode image: {image_path}")
        for region, coords in IMPORT_REGIONS.items():
            crop = crop_region(img, coords)
            ok, buf = cv2.imencode(".png", crop)
            if not ok:
                raise RuntimeError(f"Failed to encode crop: {image_path.name} {region}")
            tasks.append((image_path.name, region, buf.tobytes()))
    return tasks


def init_worker(card_path: str) -> None:
    global CARD_MODULE
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    spec = importlib.util.spec_from_file_location("bench_card_module", card_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load card module: {card_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CARD_MODULE = module


def run_region_task(task: tuple[str, str, bytes]) -> dict[str, Any]:
    if CARD_MODULE is None:
        raise RuntimeError("Worker not initialized")

    image_name, region, crop_bytes = task
    started = time.perf_counter()
    arr = np.frombuffer(crop_bytes, dtype=np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    result = CARD_MODULE.process_card(crop, region)
    analysis = result.get("analysis") or {}
    echo_substats = len(analysis.get("substats") or []) if region.startswith("echo") else None
    return {
        "image": image_name,
        "region": region,
        "success": bool(result.get("success")),
        "echo_substats": echo_substats,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def summarize_region_results(results: list[dict[str, Any]], internal_seconds: float) -> dict[str, Any]:
    by_region: dict[str, list[float]] = {}
    for row in results:
        by_region.setdefault(row["region"], []).append(float(row["elapsed_ms"]))

    echoes = [row for row in results if row["region"].startswith("echo")]
    return {
        "internal_seconds": round(internal_seconds, 3),
        "region_count": len(results),
        "success_count": sum(1 for row in results if row["success"]),
        "echo_count": len(echoes),
        "echoes_with_3plus_substats": sum(1 for row in echoes if (row.get("echo_substats") or 0) >= 3),
        "region_ms_avg": {
            region: round(statistics.mean(values), 1)
            for region, values in sorted(by_region.items())
        },
        "slowest_regions": sorted(results, key=lambda row: row["elapsed_ms"], reverse=True)[:8],
    }


def child_main(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    image_paths = [Path(path) for path in args.images]
    tasks = encode_tasks(image_paths)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.region_workers,
        initializer=init_worker,
        initargs=(str(args.card_path),),
    ) as executor:
        results = list(executor.map(run_region_task, tasks))

    payload = summarize_region_results(results, time.perf_counter() - started)
    payload["images"] = [path.name for path in image_paths]
    payload["card_path"] = str(args.card_path)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


def export_old_card(ref: str, out_dir: Path) -> Path:
    old_dir = out_dir / "old"
    old_dir.mkdir(parents=True, exist_ok=True)
    old_card = old_dir / f"card_{ref}.py"
    command = ["git", "-c", "safe.directory=*", "-C", str(BACKEND_DIR), "show", f"{ref}:card.py"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    old_card.write_text(completed.stdout, encoding="utf-8")
    return old_card


def run_case(
    *,
    label: str,
    card_label: str,
    card_path: Path,
    images: list[Path],
    region_workers: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--card-path",
        str(card_path),
        "--region-workers",
        str(region_workers),
        "--images",
        *[str(path) for path in images],
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    payload.update({
        "label": label,
        "card_version": card_label,
        "subprocess_wall_seconds": round(wall_seconds, 3),
        "stdout_prefix": lines[:-1],
    })
    return payload


def write_markdown(path: Path, results: list[dict[str, Any]], old_ref: str) -> None:
    lines = [
        "# Card Parser Version Benchmark",
        "",
        f"Old parser ref: `{old_ref}`",
        "",
        "| Set | Parser | Subprocess wall | Internal time | Regions | Success | Echoes >=3 substats |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {label} | {card_version} | {wall:.3f}s | {internal:.3f}s | {regions} | {success} | {echo_ok}/{echoes} |".format(
                label=row["label"],
                card_version=row["card_version"],
                wall=row["subprocess_wall_seconds"],
                internal=row["internal_seconds"],
                regions=row["region_count"],
                success=row["success_count"],
                echo_ok=row["echoes_with_3plus_substats"],
                echoes=row["echo_count"],
            )
        )

    lines.extend([
        "",
        "## Images",
        "",
    ])
    for row in results:
        lines.append(f"- {row['label']} / {row['card_version']}: {', '.join(row['images'])}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Subprocess wall time includes Python startup, data/template loading, crop encoding, worker startup, and parser execution.",
        "- Internal time is measured inside the child process for the same work.",
        "- Region workers process the frontend import regions in parallel processes to avoid stdout-redirection races in `process_card`.",
        "- This benchmark is a local apples-to-apples comparison, not an exact production HTTP latency trace.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark old vs current card.py")
    parser.add_argument("--r2-dir", type=Path, default=ROOT / "r2-backup")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-ref", default=DEFAULT_OLD_REF)
    parser.add_argument("--region-workers", type=int, default=10)
    parser.add_argument("--english", nargs="*", default=DEFAULT_ENGLISH)
    parser.add_argument("--localized", nargs="*", default=DEFAULT_LOCALIZED)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--card-path", type=Path)
    parser.add_argument("--images", nargs="*", default=[])
    args = parser.parse_args()

    if args.child:
        if not args.card_path or not args.images:
            raise SystemExit("--child requires --card-path and --images")
        return child_main(args)

    args.out.mkdir(parents=True, exist_ok=True)
    old_card = export_old_card(args.old_ref, args.out)
    current_card = BACKEND_DIR / "card.py"
    english_images = [args.r2_dir / name for name in args.english]
    localized_images = [args.r2_dir / name for name in args.localized]

    results = []
    for label, images in (("english", english_images), ("localized", localized_images)):
        for card_label, card_path in (("old", old_card), ("current", current_card)):
            print(f"Running {label} / {card_label}...", flush=True)
            results.append(
                run_case(
                    label=label,
                    card_label=card_label,
                    card_path=card_path,
                    images=images,
                    region_workers=args.region_workers,
                )
            )

    (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out / "summary.md", results, args.old_ref)
    print((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

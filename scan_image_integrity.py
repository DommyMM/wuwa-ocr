"""
scan_image_integrity.py — fast batch review scanner for build-card images.

This combines the practical checks from the exploratory scripts:
- decode/shape/format inventory
- JPG size hints
- position-aware echo row darkness/run outliers

It writes a CSV plus JSON review queue. It intentionally does not delete files.

Usage:
  py scan_image_integrity.py ../r2-backup --out ../forensics/integrity_scan
  py scan_image_integrity.py <image> --out ../forensics/one_image
  py scan_image_integrity.py suspicious.jpg --baseline ../r2-backup
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


REGIONS = {
    "echo1": (0.0125, 0.6019, 0.2042, 0.9843),
    "echo2": (0.2057, 0.6019, 0.3974, 0.9843),
    "echo3": (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4": (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5": (0.7911, 0.6019, 0.9833, 0.9843),
}

ROWS = [
    (0.520, 0.585),
    (0.600, 0.665),
    (0.682, 0.747),
    (0.765, 0.830),
    (0.850, 0.915),
]


def image_paths(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if root.is_file():
        return [root] if root.suffix.lower() in exts else []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def crop(img: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region
    return img[round(y1 * h): round(y2 * h), round(x1 * w): round(x2 * w)]


def longest_dark_run_ratio(mask: np.ndarray, width: int) -> float:
    longest = 0
    for row in mask:
        run = 0
        for value in row:
            run = run + 1 if value else 0
            longest = max(longest, run)
    return float(longest / max(1, width))


def panel_row_features(panel: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dark_ratios = []
    run_ratios = []
    luma_means = []

    for y1, y2 in ROWS:
        roi = gray[round(y1 * h): round(y2 * h), round(0.08 * w): round(0.95 * w)]
        dark = roi < 16
        dark_ratios.append(float(np.mean(dark)))
        run_ratios.append(longest_dark_run_ratio(dark, w))
        luma_means.append(float(np.mean(roi)))

    return {
        "dark_avg": float(np.mean(dark_ratios)),
        "dark_max": float(np.max(dark_ratios)),
        "run_avg": float(np.mean(run_ratios)),
        "run_max": float(np.max(run_ratios)),
        "mean_luma": float(np.mean(luma_means)),
    }


def percentile(value: float, values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    return float(100.0 * np.mean(arr < value))


def scan_file(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    base = {
        "file": path.name,
        "extension": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "decode_ok": False,
        "width": 0,
        "height": 0,
        "aspect": 0.0,
    }
    img = cv2.imread(str(path))
    if img is None:
        return base, []

    h, w = img.shape[:2]
    base.update({
        "decode_ok": True,
        "width": w,
        "height": h,
        "aspect": float(w / h),
    })

    panels = []
    for panel, region in REGIONS.items():
        panels.append({"file": path.name, "panel": panel, **panel_row_features(crop(img, region))})
    return base, panels


def verdict_for_image(
    info: dict[str, object],
    panels: dict[str, dict[str, object]],
    dark_baselines: dict[str, list[float]],
    run_baselines: dict[str, list[float]],
    pctl_threshold: float,
) -> tuple[str, list[str], dict[str, object]]:
    reasons = []

    if not info["decode_ok"]:
        return "reject", ["decode_failed"], {}

    aspect = float(info["aspect"])
    width = int(info["width"])
    height = int(info["height"])
    ext = str(info["extension"])
    size = int(info["bytes"])

    if not (1.74 <= aspect <= 1.81):
        reasons.append("non_kurobot_aspect")
    if width < 1200 or height < 650:
        reasons.append("too_small_for_ocr")
    if ext in {".jpg", ".jpeg"} and size > 650_000:
        reasons.append("large_jpg_size")
    if ext not in {".jpg", ".jpeg"}:
        reasons.append("non_jpg_format")

    panel_evidence = {}
    row_hits = 0
    for panel in ("echo3", "echo4", "echo5"):
        p = panels.get(panel)
        if not p:
            continue
        dark = float(p["dark_avg"])
        run = float(p["run_avg"])
        dark_pct = percentile(dark, dark_baselines[panel])
        run_pct = percentile(run, run_baselines[panel])
        hit = dark_pct >= pctl_threshold and run_pct >= pctl_threshold
        if hit:
            row_hits += 1
        panel_evidence[panel] = {
            "darkAvg": dark,
            "darkPercentile": dark_pct,
            "runAvg": run,
            "runPercentile": run_pct,
            "hit": hit,
        }

    if row_hits >= 2:
        reasons.append("echo_row_darkness_outlier")

    if "non_kurobot_aspect" in reasons or "too_small_for_ocr" in reasons:
        verdict = "reject"
    elif "echo_row_darkness_outlier" in reasons:
        verdict = "suspect"
    elif reasons:
        verdict = "review"
    else:
        verdict = "ok"

    return verdict, reasons, panel_evidence


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scan_paths(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image_infos = []
    panel_rows = []

    for i, path in enumerate(paths, 1):
        info, panels = scan_file(path)
        image_infos.append(info)
        panel_rows.extend(panels)
        if i % 500 == 0:
            print(f"processed {i}/{len(paths)}")

    return image_infos, panel_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="image file or directory to scan")
    parser.add_argument("--baseline", type=Path, help="directory to use for panel-position baselines; defaults to scanned directory or file parent")
    parser.add_argument("--out", type=Path, default=Path("../forensics/integrity_scan"))
    parser.add_argument("--pctl", type=float, default=99.5)
    args = parser.parse_args()

    target_paths = image_paths(args.path)
    if not target_paths:
        raise SystemExit(f"no images found at {args.path}")

    if args.baseline:
        baseline_root = args.baseline
    elif args.path.is_file():
        baseline_root = args.path.parent
    else:
        baseline_root = args.path

    baseline_paths = image_paths(baseline_root)
    if not baseline_paths:
        raise SystemExit(f"no baseline images found at {baseline_root}")

    same_scan = [p.resolve() for p in baseline_paths] == [p.resolve() for p in target_paths]

    print(f"Building baseline from {len(baseline_paths)} images...")
    baseline_infos, baseline_panel_rows = scan_paths(baseline_paths)

    if same_scan:
        image_infos = baseline_infos
        target_panel_rows = baseline_panel_rows
    else:
        print(f"Scanning {len(target_paths)} target image(s)...")
        image_infos, target_panel_rows = scan_paths(target_paths)

    dark_baselines: dict[str, list[float]] = defaultdict(list)
    run_baselines: dict[str, list[float]] = defaultdict(list)
    for row in baseline_panel_rows:
        panel = str(row["panel"])
        dark_baselines[panel].append(float(row["dark_avg"]))
        run_baselines[panel].append(float(row["run_avg"]))

    panels_by_file: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in target_panel_rows:
        panel = str(row["panel"])
        panels_by_file[str(row["file"])][panel] = row

    review_items = []
    summary_rows = []
    counts: dict[str, int] = defaultdict(int)

    for info in image_infos:
        panels = panels_by_file[str(info["file"])]
        verdict, reasons, evidence = verdict_for_image(info, panels, dark_baselines, run_baselines, args.pctl)
        counts[verdict] += 1
        row = {
            **info,
            "verdict": verdict,
            "reasons": ";".join(reasons),
            "echo3_dark_pct": evidence.get("echo3", {}).get("darkPercentile", ""),
            "echo4_dark_pct": evidence.get("echo4", {}).get("darkPercentile", ""),
            "echo5_dark_pct": evidence.get("echo5", {}).get("darkPercentile", ""),
            "echo3_run_pct": evidence.get("echo3", {}).get("runPercentile", ""),
            "echo4_run_pct": evidence.get("echo4", {}).get("runPercentile", ""),
            "echo5_run_pct": evidence.get("echo5", {}).get("runPercentile", ""),
        }
        summary_rows.append(row)
        if verdict != "ok":
            review_items.append({
                "key": info["file"],
                "verdict": verdict,
                "reasons": reasons,
                "image": info,
                "panels": evidence,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "integrity_scan.csv", summary_rows)
    write_csv(args.out / "panel_metrics.csv", target_panel_rows)
    with (args.out / "review_queue.json").open("w", encoding="utf-8") as f:
        json.dump(review_items, f, indent=2)

    print(f"\nScanned {len(target_paths)} target image(s)")
    for verdict in ("ok", "review", "suspect", "reject"):
        print(f"{verdict:7s}: {counts[verdict]}")
    print(f"\nWrote: {args.out / 'integrity_scan.csv'}")
    print(f"Wrote: {args.out / 'panel_metrics.csv'}")
    print(f"Wrote: {args.out / 'review_queue.json'}")

    print("\nReview queue preview:")
    for item in review_items[:30]:
        print(f"{item['key']:24s} {item['verdict']:7s} {','.join(item['reasons'])}")


if __name__ == "__main__":
    main()

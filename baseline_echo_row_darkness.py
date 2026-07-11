"""
baseline_echo_row_darkness.py — scan local build-card images for dark row runs.

This is a targeted follow-up to forensics_echo_integrity.py. It measures fixed
substat-row regions in each echo panel and reports images where panels 3-5 have
unusually large near-black rectangular runs. In the confirmed edited sample,
this was the strongest low-false-positive signal.

Usage:
  py baseline_echo_row_darkness.py ../r2-backup --suspect <image>.jpg
"""

from __future__ import annotations

import argparse
import csv
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


def crop(img, region):
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


def panel_features(panel) -> dict[str, float]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    dark_ratios = []
    run_ratios = []
    luma_means = []
    row_edges = []

    for y1, y2 in ROWS:
        roi = gray[round(y1 * h): round(y2 * h), round(0.08 * w): round(0.95 * w)]
        dark = roi < 16
        dark_ratios.append(float(np.mean(dark)))
        run_ratios.append(longest_dark_run_ratio(dark, w))
        luma_means.append(float(np.mean(roi)))
        row_profile = roi.mean(axis=1)
        row_edges.append(float(np.max(np.abs(np.diff(row_profile)))) if len(row_profile) > 1 else 0.0)

    return {
        "dark_avg": float(np.mean(dark_ratios)),
        "dark_max": float(np.max(dark_ratios)),
        "run_avg": float(np.mean(run_ratios)),
        "run_max": float(np.max(run_ratios)),
        "mean_luma": float(np.mean(luma_means)),
        "row_edge": float(np.mean(row_edges)),
    }


def image_rows(path: Path) -> list[dict[str, object]]:
    img = cv2.imread(str(path))
    if img is None or img.shape[0] < 500:
        return []
    rows = []
    for panel, region in REGIONS.items():
        features = panel_features(crop(img, region))
        rows.append({"file": path.name, "panel": panel, **features})
    return rows


def robust_z(value: float, values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.std(arr) or 1.0)
    return float((value - median) / scale)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--suspect", type=Path)
    parser.add_argument("--out", type=Path, default=Path("../forensics/baseline_row_darkness.csv"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    files = sorted(
        list(args.image_dir.glob("*.jpg"))
        + list(args.image_dir.glob("*.jpeg"))
        + list(args.image_dir.glob("*.png"))
    )

    rows: list[dict[str, object]] = []
    for i, path in enumerate(files, 1):
        rows.extend(image_rows(path))
        if i % 500 == 0:
            print(f"processed {i}/{len(files)}")

    if not rows:
        raise SystemExit("no readable images found")

    write_csv(args.out, rows)
    print(f"wrote {args.out} ({len(rows)} panel rows)")

    by_panel: dict[str, list[float]] = defaultdict(list)
    by_file: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        panel = str(row["panel"])
        by_panel[panel].append(float(row["dark_avg"]))
        by_file[str(row["file"])][panel] = row

    thresholds = {panel: float(np.percentile(values, 99.5)) for panel, values in by_panel.items()}
    print("p99.5 dark_avg thresholds:")
    for panel in sorted(thresholds):
        print(f"  {panel}: {thresholds[panel]:.6f}")

    scored = []
    for file_name, panels in by_file.items():
        count = 0
        score = 0.0
        details = []
        for panel in ("echo3", "echo4", "echo5"):
            value = float(panels[panel]["dark_avg"])
            run = float(panels[panel]["run_avg"])
            z = robust_z(value, by_panel[panel])
            if value > thresholds[panel]:
                count += 1
            score += max(0.0, z)
            details.append(f"{panel}:dark={value:.3f},run={run:.3f},z={z:.1f}")
        if count:
            scored.append((count, score, file_name, details))

    print(f"\ntop {args.top} images by panels 3-5 dark-row score:")
    for count, score, file_name, details in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:args.top]:
        print(f"{file_name:24s} count={count} score={score:.1f}  " + " | ".join(details))

    if args.suspect:
        suspect_name = args.suspect.name
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        for idx, (_, score, file_name, details) in enumerate(ranked, 1):
            if file_name == suspect_name:
                print(f"\nsuspect rank by score: {idx} of {len(ranked)}")
                print(f"{suspect_name}: score={score:.1f}  " + " | ".join(details))
                break


if __name__ == "__main__":
    main()

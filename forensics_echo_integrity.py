"""Exploratory echo-panel integrity checks that make no moderation decisions

Crops the five echo panels, computes per-panel consistency metrics and writes debug images that make edits easy to spot

Usage:
  py forensics_echo_integrity.py suspect.jpg --reference clean.jpeg
  py forensics_echo_integrity.py suspect.jpg --out ../forensics/suspect
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


IMPORT_REGIONS = {
    "echo1": {"x1": 0.0125, "x2": 0.2042, "y1": 0.6019, "y2": 0.9843},
    "echo2": {"x1": 0.2057, "x2": 0.3974, "y1": 0.6019, "y2": 0.9843},
    "echo3": {"x1": 0.4016, "x2": 0.5938, "y1": 0.6019, "y2": 0.9843},
    "echo4": {"x1": 0.5969, "x2": 0.7891, "y1": 0.6019, "y2": 0.9843},
    "echo5": {"x1": 0.7911, "x2": 0.9833, "y1": 0.6019, "y2": 0.9843},
}

SUB_ROWS = [
    (0.520, 0.585),
    (0.600, 0.665),
    (0.682, 0.747),
    (0.765, 0.830),
    (0.850, 0.915),
]


@dataclass
class PanelMetrics:
    image: str
    panel: str
    gold_ratio: float
    gold_sat_mean: float
    gold_val_mean: float
    gold_lab_l_std: float
    gold_edge_density: float
    border_gold_ratio: float
    right_text_gold_ratio: float
    row_gold_ratio: float
    row_luma_mean: float
    row_luma_std: float
    row_luma_range_mean: float
    row_laplacian_var: float
    row_edge_density: float
    row_band_contrast_std: float
    ela_mean: float
    ela_p95: float
    ela_row_mean: float
    ela_row_p95: float
    jpeg_blockiness: float


def crop_region(img: np.ndarray, region: dict[str, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = round(region["x1"] * w)
    x2 = round(region["x2"] * w)
    y1 = round(region["y1"] * h)
    y2 = round(region["y2"] * h)
    return img[y1:y2, x1:x2]


def rel_crop(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[round(y1 * h): round(y2 * h), round(x1 * w): round(x2 * w)]


def mask_gold(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Gold UI spans muted beige to saturated yellow, so the S floor admits compressed text but not white echo art
    return cv2.inRange(hsv, np.array([10, 25, 55]), np.array([45, 255, 255]))


def mask_rows(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = round(0.08 * w), round(0.94 * w)
    for y1, y2 in SUB_ROWS:
        mask[round(y1 * h): round(y2 * h), x1:x2] = 255
    return mask


def masked_values(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = gray[mask > 0]
    if vals.size == 0:
        return np.array([0], dtype=np.float32)
    return vals.astype(np.float32)


def edge_density(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    edges = cv2.Canny(gray, 60, 140)
    if mask is not None:
        denom = max(1, cv2.countNonZero(mask))
        return float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / denom)
    return float(np.count_nonzero(edges) / edges.size)


def jpeg_ela_bgr(img: np.ndarray, quality: int = 90) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB"))
    recompressed_bgr = cv2.cvtColor(recompressed, cv2.COLOR_RGB2BGR)
    return cv2.absdiff(img, recompressed_bgr)


def jpeg_blockiness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if g.shape[0] < 16 or g.shape[1] < 16:
        return 0.0
    vertical_boundaries = np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean()
    horizontal_boundaries = np.abs(g[8::8, :] - g[7:-1:8, :]).mean()
    vertical_inside = np.abs(g[:, 1:] - g[:, :-1]).mean()
    horizontal_inside = np.abs(g[1:, :] - g[:-1, :]).mean()
    inside = max(1e-6, (vertical_inside + horizontal_inside) / 2)
    return float((vertical_boundaries + horizontal_boundaries) / (2 * inside))


def band_contrast_std(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    vals = []
    x1, x2 = round(0.08 * w), round(0.94 * w)
    for y1, y2 in SUB_ROWS:
        band = gray[round(y1 * h): round(y2 * h), x1:x2]
        vals.append(float(np.mean(band)))
    return float(np.std(vals))


def panel_metrics(image_name: str, panel_name: str, panel: np.ndarray) -> PanelMetrics:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(panel, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    gold = mask_gold(panel)
    rows = mask_rows(panel)
    gold_vals_l = masked_values(lab[:, :, 0], gold)
    row_vals = masked_values(gray, rows)

    border = np.zeros(gray.shape, dtype=np.uint8)
    h, w = gray.shape
    b = max(2, round(w * 0.025))
    border[:, :b] = 255
    border[:, w - b:] = 255
    border[h - b:, :] = 255

    right_text = np.zeros(gray.shape, dtype=np.uint8)
    right_text[round(0.08 * h): round(0.42 * h), round(0.63 * w): round(0.97 * w)] = 255

    ela = jpeg_ela_bgr(panel)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_row_vals = masked_values(ela_gray, rows)

    row_crop = rel_crop(panel, 0.08, 0.50, 0.96, 0.94)
    row_gray = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)

    return PanelMetrics(
        image=image_name,
        panel=panel_name,
        gold_ratio=float(cv2.countNonZero(gold) / gold.size),
        gold_sat_mean=float(np.mean(masked_values(hsv[:, :, 1], gold))),
        gold_val_mean=float(np.mean(masked_values(hsv[:, :, 2], gold))),
        gold_lab_l_std=float(np.std(gold_vals_l)),
        gold_edge_density=edge_density(gray, gold),
        border_gold_ratio=float(cv2.countNonZero(cv2.bitwise_and(gold, gold, mask=border)) / max(1, cv2.countNonZero(border))),
        right_text_gold_ratio=float(cv2.countNonZero(cv2.bitwise_and(gold, gold, mask=right_text)) / max(1, cv2.countNonZero(right_text))),
        row_gold_ratio=float(cv2.countNonZero(cv2.bitwise_and(gold, gold, mask=rows)) / max(1, cv2.countNonZero(rows))),
        row_luma_mean=float(np.mean(row_vals)),
        row_luma_std=float(np.std(row_vals)),
        row_luma_range_mean=float(np.mean(np.ptp(cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY), axis=1))),
        row_laplacian_var=float(cv2.Laplacian(row_gray, cv2.CV_64F).var()),
        row_edge_density=edge_density(gray, rows),
        row_band_contrast_std=band_contrast_std(gray),
        ela_mean=float(np.mean(ela_gray)),
        ela_p95=float(np.percentile(ela_gray, 95)),
        ela_row_mean=float(np.mean(ela_row_vals)),
        ela_row_p95=float(np.percentile(ela_row_vals, 95)),
        jpeg_blockiness=jpeg_blockiness(gray),
    )


def robust_zscores(rows: list[PanelMetrics]) -> dict[str, dict[str, float]]:
    metric_names = [k for k in asdict(rows[0]).keys() if k not in {"image", "panel"}]
    out: dict[str, dict[str, float]] = {r.panel: {} for r in rows}
    for metric in metric_names:
        vals = np.array([float(getattr(r, metric)) for r in rows], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        scale = 1.4826 * mad if mad > 1e-9 else float(np.std(vals) or 1.0)
        for r, v in zip(rows, vals):
            out[r.panel][metric] = float((v - med) / scale)
    return out


def save_heatmaps(out_dir: Path, image_name: str, panels: dict[str, np.ndarray]) -> None:
    for panel_name, panel in panels.items():
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        gold = mask_gold(panel)
        rows = mask_rows(panel)
        ela = cv2.cvtColor(jpeg_ela_bgr(panel), cv2.COLOR_BGR2GRAY)
        ela_heat = cv2.applyColorMap(cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_MAGMA)

        overlay = panel.copy()
        overlay[gold > 0] = (0.35 * overlay[gold > 0] + 0.65 * np.array([0, 215, 255])).astype(np.uint8)
        overlay[rows > 0] = (0.80 * overlay[rows > 0] + 0.20 * np.array([255, 0, 0])).astype(np.uint8)

        edges = cv2.Canny(gray, 60, 140)
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(out_dir / f"{image_name}_{panel_name}.png"), panel)
        cv2.imwrite(str(out_dir / f"{image_name}_{panel_name}_gold_rows_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{image_name}_{panel_name}_ela.png"), ela_heat)
        cv2.imwrite(str(out_dir / f"{image_name}_{panel_name}_edges.png"), edge_vis)


def write_csv(path: Path, rows: Iterable[PanelMetrics]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_outliers(rows: list[PanelMetrics]) -> list[dict[str, object]]:
    z = robust_zscores(rows)
    summary = []
    for row in rows:
        strongest = sorted(z[row.panel].items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        score = math.sqrt(sum(v * v for _, v in strongest))
        summary.append({
            "panel": row.panel,
            "score": score,
            "strongest": [{"metric": k, "z": v} for k, v in strongest],
        })
    return sorted(summary, key=lambda item: item["score"], reverse=True)


def analyze_image(path: Path, out_dir: Path) -> tuple[list[PanelMetrics], dict[str, np.ndarray]]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"failed to read image: {path}")
    panels = {name: crop_region(img, region) for name, region in IMPORT_REGIONS.items()}
    metrics = [panel_metrics(path.stem, name, panel) for name, panel in panels.items()]
    save_heatmaps(out_dir, path.stem, panels)
    return metrics, panels


def compare_panel_geometry(out_dir: Path, suspect_panels: dict[str, np.ndarray], reference_panels: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    results = {}
    for name, panel in suspect_panels.items():
        ref = reference_panels[name]
        ref = cv2.resize(ref, (panel.shape[1], panel.shape[0]))
        # Heavy blur suppresses text and echo identity, leaving the color and lighting structure to compare
        a = cv2.GaussianBlur(panel, (0, 0), 9)
        b = cv2.GaussianBlur(ref, (0, 0), 9)
        diff = cv2.absdiff(a, b)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        rows = mask_rows(panel)
        gold = mask_gold(panel)
        results[name] = {
            "blurred_diff_mean": float(np.mean(gray)),
            "blurred_diff_row_mean": float(np.mean(masked_values(gray, rows))),
            "gold_mask_iou": float(cv2.countNonZero(cv2.bitwise_and(gold, mask_gold(ref))) / max(1, cv2.countNonZero(cv2.bitwise_or(gold, mask_gold(ref))))),
        }
        heat = cv2.applyColorMap(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out_dir / f"compare_{name}_blurred_diff.png"), heat)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    out_dir = args.out or (args.image.parent / f"{args.image.stem}_forensics")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[PanelMetrics] = []
    suspect_metrics, suspect_panels = analyze_image(args.image, out_dir)
    all_metrics.extend(suspect_metrics)

    report = {
        "image": str(args.image),
        "outDir": str(out_dir),
        "suspectOutliers": summarize_outliers(suspect_metrics),
    }

    if args.reference:
        reference_metrics, reference_panels = analyze_image(args.reference, out_dir)
        all_metrics.extend(reference_metrics)
        report["reference"] = str(args.reference)
        report["referenceOutliers"] = summarize_outliers(reference_metrics)
        report["referenceComparison"] = compare_panel_geometry(out_dir, suspect_panels, reference_panels)

    write_csv(out_dir / "metrics.csv", all_metrics)
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote metrics and debug images to: {out_dir}")


if __name__ == "__main__":
    main()

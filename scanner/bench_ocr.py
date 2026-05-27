"""bench_ocr.py — OCR engine comparison for echo-bag panel crops.

Engines: RapidOCR (ONNX), Tesseract (pytesseract).
Regions: echo_stats (raw), echo_name_cost (raw + #efe4a4 color mask).

Reports cold-start, warm per-call latency, and accuracy vs hardcoded ground
truth. Full raw OCR output is written to bench_results.txt.

Usage:  py bench_ocr.py [echo_bag_dir]
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wuwa_scanner.layout import REGIONS, proportional_crop  # noqa: E402


GROUND_TRUTH = {
    "Screenshot 2026-05-27 140417": {
        "name": "Phantom: Sigillum",
        "cost": 4,
        "level": 25,
        "stats": [
            ("Crit. DMG", "44.0%"),
            ("ATK", "150"),
            ("DEF", "10.0%"),
            ("ATK", "11.6%"),
            ("Crit. Rate", "8.7%"),
            ("Crit. DMG", "18.6%"),
            ("Basic Attack DMG Bonus", "6.4%"),
        ],
    },
    "Screenshot 2026-05-27 140447": {
        "name": "Nightmare: Hecate",
        "cost": 4,
        "level": 25,
        "stats": [
            ("Crit. DMG", "44.0%"),
            ("ATK", "150"),
            ("Resonance Liberation DMG Bonus", "9.4%"),
            ("Resonance Skill DMG Bonus", "7.1%"),
            ("DEF", "60"),
            ("ATK", "7.9%"),
            ("Crit. Rate", "7.5%"),
        ],
    },
    "Screenshot 2026-05-27 140456": {
        "name": "Reminiscence: Denia",
        "cost": 4,
        "level": 25,
        "stats": [
            ("Crit. DMG", "44.0%"),
            ("ATK", "150"),
            ("DEF", "11.8%"),
            ("ATK", "50"),
            ("ATK", "11.6%"),
            ("Crit. Rate", "6.9%"),
            ("Crit. DMG", "15.0%"),
        ],
    },
}


def name_color_mask(img_bgr: np.ndarray, target_hex: str = "efe4a4", tol: float = 70.0) -> np.ndarray:
    """Keep only pixels within `tol` Euclidean RGB distance of `target_hex`.

    Returns a 1-channel image: pixels matching the target become black (text),
    everything else white. This shape is what Tesseract prefers; RapidOCR
    accepts it too via opencv's auto-3ch promotion.
    """
    r = int(target_hex[0:2], 16)
    g = int(target_hex[2:4], 16)
    b = int(target_hex[4:6], 16)
    target = np.array([b, g, r], dtype=np.int16)  # cv2 is BGR
    diff = img_bgr.astype(np.int16) - target
    dist = np.sqrt((diff * diff).sum(axis=2))
    mask = dist <= tol
    out = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)
    out[mask] = 0
    return out


@dataclass
class OcrResult:
    text: str
    lines: list[str]
    elapsed_ms: float


class RapidEngine:
    name = "RapidOCR"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()

    def run(self, img) -> OcrResult:
        t = time.perf_counter()
        result, _ = self.ocr(img)
        elapsed = (time.perf_counter() - t) * 1000
        if not result:
            return OcrResult("", [], elapsed)
        lines = [str(r[1]) for r in result]
        return OcrResult("\n".join(lines), lines, elapsed)


class TesseractEngine:
    name = "Tesseract"

    def __init__(self):
        import pytesseract
        self.pt = pytesseract
        _ = pytesseract.image_to_string(np.zeros((20, 50, 3), dtype=np.uint8))

    def run(self, img) -> OcrResult:
        t = time.perf_counter()
        text = self.pt.image_to_string(img)
        elapsed = (time.perf_counter() - t) * 1000
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return OcrResult(text, lines, elapsed)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_name_cost(lines: list[str], truth: dict) -> dict:
    blob = _norm(" ".join(lines))
    blob_compact = blob.replace(" ", "")
    name_norm = _norm(truth["name"])
    # Allow prefix-stripped fallback: "Phantom: Sigillum" may render as "Sigillum"
    base_name = name_norm.split(":")[-1].strip() if ":" in name_norm else name_norm
    name_hit = name_norm in blob or base_name in blob
    cost_hit = bool(re.search(rf"cost\s*{truth['cost']}\b", blob))
    level_hit = f"+{truth['level']}" in blob_compact or f"{truth['level']}" in blob_compact
    return {"name": name_hit, "cost": cost_hit, "level": level_hit}


def score_stats(lines: list[str], truth: dict) -> dict:
    matched = 0
    missing = []
    blob = _norm(" ".join(lines))
    blob_compact = blob.replace(" ", "")
    line_blobs = [_norm(ln) for ln in lines]
    for stat_name, value in truth["stats"]:
        sn = _norm(stat_name)
        vn = _norm(value).replace(" ", "")
        # Fuzzy name match against any line, and value present anywhere.
        name_hit = any(fuzz.partial_ratio(sn, lb) >= 85 for lb in line_blobs) or sn in blob
        value_hit = vn in blob_compact
        if name_hit and value_hit:
            matched += 1
        else:
            missing.append((stat_name, value, name_hit, value_hit))
    return {"matched": matched, "total": len(truth["stats"]), "missing": missing}


def main() -> None:
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "echo_bag"
    images = sorted(p for p in src.glob("*.png") if p.stem in GROUND_TRUTH)
    if not images:
        raise SystemExit(f"no scored PNGs in {src}")

    print("Cold start:")
    t = time.perf_counter()
    rapid = RapidEngine()
    print(f"  RapidOCR   {(time.perf_counter() - t) * 1000:7.0f} ms")
    t = time.perf_counter()
    tess = TesseractEngine()
    print(f"  Tesseract  {(time.perf_counter() - t) * 1000:7.0f} ms")
    engines = [rapid, tess]

    print("\nWarmup pass (not timed in totals)...")
    warm = cv2.imread(str(images[0]))
    for eng in engines:
        _ = eng.run(proportional_crop(warm, REGIONS["echo_stats"]))

    print("\n" + "=" * 96)
    print(f"{'Echo':22s} {'Engine':10s} {'Region':18s} {'ms':>7s}  Result")
    print("-" * 96)

    log = []
    totals = {(e.name, r): [] for e in engines for r in ("echo_stats", "name_cost raw", "name_cost mask")}

    for img_path in images:
        truth = GROUND_TRUTH[img_path.stem]
        img = cv2.imread(str(img_path))
        stats_crop = proportional_crop(img, REGIONS["echo_stats"])
        name_crop = proportional_crop(img, REGIONS["echo_name_cost"])
        name_masked = name_color_mask(name_crop)
        echo_label = truth["name"].split(":")[-1].strip()

        cases = [
            ("echo_stats", stats_crop),
            ("name_cost raw", name_crop),
            ("name_cost mask", name_masked),
        ]

        for eng in engines:
            for region_label, crop in cases:
                r = eng.run(crop)
                totals[(eng.name, region_label)].append(r.elapsed_ms)
                if region_label == "echo_stats":
                    sc = score_stats(r.lines, truth)
                    summary = f"{sc['matched']}/{sc['total']} stats"
                else:
                    sc = score_name_cost(r.lines, truth)
                    flags = ("N" if sc["name"] else "n") + ("C" if sc["cost"] else "c") + ("L" if sc["level"] else "l")
                    summary = f"[{flags}]"
                print(f"{echo_label:22s} {eng.name:10s} {region_label:18s} {r.elapsed_ms:7.1f}  {summary}")
                log.append(f"\n=== {img_path.stem} | {eng.name} | {region_label} ({r.elapsed_ms:.1f} ms) ===")
                log.extend(r.lines)
                log.append(f"  score: {sc}")

    print("-" * 96)
    print("\nWarm-call averages:")
    for (eng_name, region), times in totals.items():
        if times:
            print(f"  {eng_name:10s} {region:18s} {sum(times) / len(times):7.1f} ms (n={len(times)})")

    log_path = here / "bench_results.txt"
    log_path.write_text("\n".join(log), encoding="utf-8")
    print(f"\nFull OCR output -> {log_path}")


if __name__ == "__main__":
    main()

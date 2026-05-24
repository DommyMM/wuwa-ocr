"""
optimize_crops.py - sweep crop geometry for fixed-layout import recognition.

This is an offline Phase 2 tool. It does not change runtime coordinates. It
evaluates candidate crop boxes against an explicit gold-label JSON file and
writes a ranked report under backend/benchmarks/crop_sweeps/.

Usage:
  py optimize_crops.py --labels labels.json --task weapon_sift
  py optimize_crops.py --labels labels.json --task character_sift --image-root ..
  py optimize_crops.py --labels labels.json --task echo_icon --regions echo1 echo2
  py optimize_crops.py --labels labels.json --task watermark_uid --save-debug
  py optimize_crops.py --labels labels.json --task echo_main --regions echo1
  py optimize_crops.py --labels labels.json --task echo_substats --regions echo1 echo2

Gold-label JSON shape:
{
  "images": [
    {
      "file": "r2-backup/example.jpg",
      "character": {"id": "1108"},
      "weapon": {"id": "21020086"},
      "watermark": {"uid": "500006092"},
      "echoes": {
        "echo1": {
          "id": "60001995",
          "main": {"name": "ATK%"},
          "substats": [{"name": "ATK%", "value": "10.1%"}]
        }
      }
    }
  ]
}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pytesseract

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))


# Full-card normalized regions. These mirror the frontend import layout plus
# the Phase 1 character/weapon recognition crops.
FULL_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "character": (0.0328, 0.0074, 0.3021, 0.0833),
    "character_splash": (0.0200, 0.1000, 0.2700, 0.4500),
    "watermark": (0.0073, 0.0741, 0.1304, 0.1370),
    "watermark_uid": (0.0100, 0.1060, 0.1150, 0.1245),
    "weapon": (0.7542, 0.3843, 0.9828, 0.5843),
    "weapon_icon": (0.7590, 0.4120, 0.8310, 0.5380),
    "echo1": (0.0125, 0.6019, 0.2042, 0.9843),
    "echo2": (0.2057, 0.6019, 0.3974, 0.9843),
    "echo3": (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4": (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5": (0.7911, 0.6019, 0.9833, 0.9843),
}

# Absolute subregions inside the current 368/369 x 413 echo crop.
ECHO_SUBREGIONS: dict[str, tuple[int, int, int, int]] = {
    "icon": (0, 0, 188, 182),
    "main": (195, 66, 366, 148),
}

SUBSTAT_ROWS: list[tuple[int, int, int, int]] = [
    (36, 228, 359, 262),
    (36, 262, 359, 296),
    (36, 296, 359, 330),
    (36, 330, 359, 364),
    (36, 364, 359, 400),
]

FORTE_LEVEL_BOXES: list[tuple[int, int, int, int]] = [
    (270, 144, 389, 204),
    (48, 302, 158, 356),
    (467, 296, 596, 357),
    (122, 545, 247, 602),
    (386, 544, 518, 601),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Candidate:
    dx: int
    dy: int
    pad: int

    @property
    def key(self) -> str:
        return f"dx{self.dx:+d}_dy{self.dy:+d}_pad{self.pad:+d}"


@dataclass
class Prediction:
    value: str
    confidence: float = 0.0
    margin: float = 0.0
    raw: str = ""


class SiftMatcher:
    def __init__(self, template_dir: Path, max_side: int = 420):
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        self.templates: dict[str, tuple[tuple[cv2.KeyPoint, ...], np.ndarray]] = {}

        for path in sorted(template_dir.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = resize_long_side(img, max_side)
            kp, des = self.sift.detectAndCompute(img, None)
            if des is not None and len(des) >= 2:
                self.templates[path.stem] = (tuple(kp), des)

    def match(self, image: np.ndarray) -> Prediction:
        if not self.templates:
            return Prediction("", raw="no_templates")

        kp1, des1 = self.sift.detectAndCompute(image, None)
        if des1 is None or len(des1) < 2:
            return Prediction("", raw="no_features")

        scores: list[tuple[str, float]] = []
        for template_id, (kp2, des2) in self.templates.items():
            try:
                matches = self.flann.knnMatch(des1, des2, k=2)
            except cv2.error:
                continue
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            denom = max(len(kp1), len(kp2))
            score = len(good) / denom if denom else 0.0
            scores.append((template_id, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores:
            return Prediction("", raw="no_scores")

        best_id, best = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        top = ";".join(f"{tid}:{score:.4f}" for tid, score in scores[:5])
        return Prediction(best_id, confidence=float(best), margin=float(best - second), raw=top)


def resize_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def norm_to_px(region: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = region
    return clamp_box(round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h), w, h)


def candidate_box(base: tuple[int, int, int, int], cand: Candidate, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = base
    return clamp_box(
        x1 + cand.dx - cand.pad,
        y1 + cand.dy - cand.pad,
        x2 + cand.dx + cand.pad,
        y2 + cand.dy + cand.pad,
        w,
        h,
    )


def crop_px(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def crop_full_region(img: np.ndarray, key: str, cand: Candidate) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    base = norm_to_px(FULL_REGIONS[key], w, h)
    box = candidate_box(base, cand, w, h)
    return crop_px(img, box), box


def crop_echo_subregion(
    img: np.ndarray,
    echo_key: str,
    sub_box: tuple[int, int, int, int],
    cand: Candidate,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    ex1, ey1, ex2, ey2 = norm_to_px(FULL_REGIONS[echo_key], w, h)
    sx1, sy1, sx2, sy2 = sub_box
    base = (ex1 + sx1, ey1 + sy1, ex1 + sx2, ey1 + sy2)
    box = candidate_box(base, cand, w, h)
    return crop_px(img, box), box


def preprocess_for_tess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(bilateral, (0, 0), 3)
    sharp = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
    return thresh


def normalize_digits(text: str) -> str:
    return re.sub(r"\D+", "", text)


def normalize_value(text: str) -> str:
    return re.sub(r"\s+", "", text.strip()).replace("％", "%")


def load_card_helpers() -> dict[str, Any]:
    from card import clean_stat_name, validate_stat, validate_substat_name, validate_value
    from data import MAIN_STAT_NAMES

    return {
        "clean_stat_name": clean_stat_name,
        "validate_stat": validate_stat,
        "validate_substat_name": validate_substat_name,
        "validate_value": validate_value,
        "MAIN_STAT_NAMES": MAIN_STAT_NAMES,
    }


def recognize_uid(img: np.ndarray) -> Prediction:
    processed = preprocess_for_tess(img)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(processed, config=config)
    return Prediction(normalize_digits(raw), raw=raw.strip())


def recognize_forte_level(img: np.ndarray) -> Prediction:
    processed = preprocess_for_tess(img)
    config = "--psm 7 -c tessedit_char_whitelist=LVlv.0123456789/"
    raw = pytesseract.image_to_string(processed, config=config).strip()
    match = re.search(r"(?i)(?:lv\.?)?(\d{1,2})(?:/10)?", raw)
    return Prediction(match.group(1) if match else "", raw=raw)


def recognize_main_name(img: np.ndarray, helpers: dict[str, Any]) -> Prediction:
    processed = preprocess_for_tess(img)
    raw = pytesseract.image_to_string(processed).strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    text = " ".join(lines)
    if not text:
        return Prediction("", raw=raw)
    parts = text.rsplit(" ", 1)
    name = parts[0] if len(parts) == 2 else text
    value = parts[1] if len(parts) == 2 else ""
    cleaned = helpers["clean_stat_name"](name, value)
    matched = helpers["validate_stat"](cleaned, helpers["MAIN_STAT_NAMES"])
    if matched in {"HP", "ATK", "DEF"}:
        matched = f"{matched}%" if "%" in value or "%" in text else matched
    return Prediction(str(matched), raw=raw)


def recognize_substat_row(img: np.ndarray, helpers: dict[str, Any]) -> Prediction:
    processed = preprocess_for_tess(img)
    config = "--psm 7"
    raw = pytesseract.image_to_string(processed, config=config).strip()
    text = re.sub(r"\s+", " ", raw)
    parts = text.rsplit(" ", 1)
    if len(parts) != 2:
        return Prediction("", raw=raw)
    name_raw, value_raw = parts
    name = helpers["validate_substat_name"](name_raw, value_raw)
    value = helpers["validate_value"](value_raw, name)
    return Prediction(f"{name}|{normalize_value(value)}", raw=raw)


def image_entries(labels_path: Path, image_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    entries = payload.get("images", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        raise ValueError("labels must be an object with images[] or a list")

    resolved: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or "file" not in entry:
            continue
        path = Path(str(entry["file"]))
        if not path.is_absolute():
            path = image_root / path
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        next_entry = dict(entry)
        next_entry["_path"] = path
        resolved.append(next_entry)
    return resolved


def candidate_grid(delta: int, pad: int, step: int) -> list[Candidate]:
    offsets = list(range(-delta, delta + 1, max(1, step)))
    pads = list(range(-pad, pad + 1, max(1, step)))
    if 0 not in offsets:
        offsets.append(0)
    if 0 not in pads:
        pads.append(0)
    return [
        Candidate(dx, dy, p)
        for dx in sorted(set(offsets))
        for dy in sorted(set(offsets))
        for p in sorted(set(pads))
    ]


def expected_for_task(entry: dict[str, Any], task: str, region: str | None, row_index: int | None) -> str:
    if task == "character_sift":
        return str(entry.get("character", {}).get("id", ""))
    if task == "weapon_sift":
        return str(entry.get("weapon", {}).get("id", ""))
    if task == "watermark_uid":
        return str(entry.get("watermark", {}).get("uid", ""))
    if task == "forte_digit":
        levels = entry.get("forte", [])
        if isinstance(levels, list) and row_index is not None and row_index < len(levels):
            return str(levels[row_index])
        return ""

    echoes = entry.get("echoes", {})
    echo = echoes.get(region or "", {}) if isinstance(echoes, dict) else {}
    if task == "echo_icon":
        return str(echo.get("id", ""))
    if task == "echo_main":
        return str(echo.get("main", {}).get("name", ""))
    if task == "echo_substats":
        rows = echo.get("substats", [])
        if isinstance(rows, list) and row_index is not None and row_index < len(rows):
            name = str(rows[row_index].get("name", ""))
            value = normalize_value(str(rows[row_index].get("value", "")))
            return f"{name}|{value}"
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_debug_crop(out_dir: Path, image_name: str, task: str, label: str, crop: np.ndarray) -> None:
    safe_label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", label)
    path = out_dir / "debug_crops" / task / f"{Path(image_name).stem}_{safe_label}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)


def build_predictor(task: str) -> tuple[Callable[..., Prediction], dict[str, Any]]:
    state: dict[str, Any] = {}
    if task == "character_sift":
        matcher = SiftMatcher(BACKEND_DIR / "Data" / "Characters")
        return lambda crop, **_: matcher.match(crop), state
    if task == "weapon_sift":
        matcher = SiftMatcher(BACKEND_DIR / "Data" / "Weapons")
        return lambda crop, **_: matcher.match(crop), state
    if task == "echo_icon":
        matcher = SiftMatcher(BACKEND_DIR / "Data" / "Echoes")
        return lambda crop, **_: matcher.match(crop), state
    if task == "watermark_uid":
        return lambda crop, **_: recognize_uid(crop), state
    if task == "forte_digit":
        return lambda crop, **_: recognize_forte_level(crop), state
    if task in {"echo_main", "echo_substats"}:
        helpers = load_card_helpers()
        state["helpers"] = helpers
        if task == "echo_main":
            return lambda crop, **_: recognize_main_name(crop, helpers), state
        return lambda crop, **_: recognize_substat_row(crop, helpers), state
    raise ValueError(f"unsupported task: {task}")


def iter_eval_targets(
    entry: dict[str, Any],
    task: str,
    regions: list[str],
) -> Iterable[tuple[str | None, int | None, str]]:
    if task in {"character_sift", "weapon_sift", "watermark_uid"}:
        expected = expected_for_task(entry, task, None, None)
        if expected:
            yield None, None, expected
        return

    if task == "forte_digit":
        for i in range(5):
            expected = expected_for_task(entry, task, None, i)
            if expected:
                yield None, i, expected
        return

    for region in regions:
        if task in {"echo_icon", "echo_main"}:
            expected = expected_for_task(entry, task, region, None)
            if expected:
                yield region, None, expected
        elif task == "echo_substats":
            for i in range(5):
                expected = expected_for_task(entry, task, region, i)
                if expected:
                    yield region, i, expected


def crop_for_task(
    img: np.ndarray,
    task: str,
    candidate: Candidate,
    region: str | None,
    row_index: int | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if task == "character_sift":
        return crop_full_region(img, "character_splash", candidate)
    if task == "weapon_sift":
        return crop_full_region(img, "weapon_icon", candidate)
    if task == "watermark_uid":
        return crop_full_region(img, "watermark_uid", candidate)
    if task == "echo_icon":
        if not region:
            raise ValueError("echo_icon requires region")
        return crop_echo_subregion(img, region, ECHO_SUBREGIONS["icon"], candidate)
    if task == "echo_main":
        if not region:
            raise ValueError("echo_main requires region")
        return crop_echo_subregion(img, region, ECHO_SUBREGIONS["main"], candidate)
    if task == "echo_substats":
        if not region or row_index is None:
            raise ValueError("echo_substats requires region and row")
        return crop_echo_subregion(img, region, SUBSTAT_ROWS[row_index], candidate)
    if task == "forte_digit":
        if row_index is None:
            raise ValueError("forte_digit requires row")
        # Forte boxes are absolute inside the forte outer crop.
        h, w = img.shape[:2]
        fx1, fy1, _, _ = norm_to_px(FULL_REGIONS["forte"], w, h)
        sx1, sy1, sx2, sy2 = FORTE_LEVEL_BOXES[row_index]
        base = (fx1 + sx1, fy1 + sy1, fx1 + sx2, fy1 + sy2)
        box = candidate_box(base, candidate, w, h)
        return crop_px(img, box), box
    raise ValueError(f"unsupported task: {task}")


def run_sweep(args: argparse.Namespace) -> int:
    entries = image_entries(args.labels, args.image_root)
    if args.max_images:
        entries = entries[: args.max_images]
    if not entries:
        print(f"ERROR: no labeled images found in {args.labels}")
        return 1

    predictor, _ = build_predictor(args.task)
    candidates = candidate_grid(args.delta, args.pad, args.step)
    out_dir = args.out or BACKEND_DIR / "benchmarks" / "crop_sweeps" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    images: dict[Path, np.ndarray] = {}
    for entry in entries:
        path = entry["_path"]
        img = cv2.imread(str(path))
        if img is None:
            print(f"WARNING: failed to read {path}")
            continue
        images[path] = img

    if not images:
        print("ERROR: no readable labeled images")
        return 1

    summary_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []

    print(f"Task: {args.task}")
    print(f"Images: {len(images)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Output: {out_dir}")

    for idx, cand in enumerate(candidates, 1):
        total = correct = low_conf = 0
        confs: list[float] = []
        margins: list[float] = []
        first_box: tuple[int, int, int, int] | None = None

        for entry in entries:
            path = entry["_path"]
            img = images.get(path)
            if img is None:
                continue

            for region, row_index, expected in iter_eval_targets(entry, args.task, args.regions):
                crop, box = crop_for_task(img, args.task, cand, region, row_index)
                pred = predictor(crop, region=region, row_index=row_index)
                if first_box is None:
                    first_box = box

                total += 1
                is_correct = pred.value == expected
                correct += 1 if is_correct else 0
                confs.append(pred.confidence)
                margins.append(pred.margin)
                if pred.confidence < args.low_confidence:
                    low_conf += 1

                if not is_correct:
                    target = region or "global"
                    if row_index is not None:
                        target = f"{target}:row{row_index + 1}"
                    mismatch_rows.append({
                        "candidate": cand.key,
                        "image": path.name,
                        "target": target,
                        "expected": expected,
                        "predicted": pred.value,
                        "confidence": f"{pred.confidence:.6f}",
                        "margin": f"{pred.margin:.6f}",
                        "raw": pred.raw,
                        "box": ",".join(str(v) for v in box),
                    })
                    if args.save_debug:
                        save_debug_crop(out_dir, path.name, args.task, f"{cand.key}_{target}", crop)

        accuracy = correct / total if total else 0.0
        summary_rows.append({
            "candidate": cand.key,
            "dx": cand.dx,
            "dy": cand.dy,
            "pad": cand.pad,
            "total": total,
            "correct": correct,
            "accuracy": f"{accuracy:.6f}",
            "avg_confidence": f"{float(np.mean(confs)) if confs else 0.0:.6f}",
            "avg_margin": f"{float(np.mean(margins)) if margins else 0.0:.6f}",
            "low_confidence": low_conf,
            "example_box": ",".join(str(v) for v in first_box) if first_box else "",
        })

        if idx % 25 == 0 or idx == len(candidates):
            best = max(summary_rows, key=lambda row: (float(row["accuracy"]), float(row["avg_margin"])))
            print(
                f"  {idx}/{len(candidates)} best={best['candidate']} "
                f"acc={best['accuracy']} margin={best['avg_margin']}"
            )

    summary_rows.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["avg_margin"]),
            float(row["avg_confidence"]),
            -int(row["low_confidence"]),
        ),
        reverse=True,
    )

    report = {
        "task": args.task,
        "labels": str(args.labels),
        "image_root": str(args.image_root),
        "regions": args.regions,
        "delta": args.delta,
        "pad": args.pad,
        "step": args.step,
        "low_confidence": args.low_confidence,
        "best": summary_rows[0] if summary_rows else None,
        "summary": summary_rows,
    }

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "mismatches.csv", mismatch_rows)

    print("\nBest candidates:")
    for row in summary_rows[:10]:
        print(
            f"  {row['candidate']:20s} acc={row['accuracy']} "
            f"avg_margin={row['avg_margin']} low_conf={row['low_confidence']}"
        )
    print(f"\nWrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'summary.csv'}")
    if mismatch_rows:
        print(f"Wrote: {out_dir / 'mismatches.csv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize fixed OCR/recognition crops against gold labels")
    parser.add_argument("--labels", type=Path, required=True, help="Gold-label JSON")
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root for relative image paths")
    parser.add_argument("--task", required=True, choices=[
        "character_sift",
        "weapon_sift",
        "echo_icon",
        "watermark_uid",
        "forte_digit",
        "echo_main",
        "echo_substats",
    ])
    parser.add_argument("--regions", nargs="*", default=["echo1", "echo2", "echo3", "echo4", "echo5"],
                        help="Echo regions to evaluate for echo_* tasks")
    parser.add_argument("--delta", type=int, default=8, help="Max x/y translation in pixels")
    parser.add_argument("--pad", type=int, default=6, help="Max uniform crop padding in pixels")
    parser.add_argument("--step", type=int, default=4, help="Sweep step in pixels")
    parser.add_argument("--max-images", type=int, default=0, help="Limit labeled images for quick runs")
    parser.add_argument("--low-confidence", type=float, default=0.05, help="Low-confidence threshold for SIFT tasks")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--save-debug", action="store_true", help="Save mismatch crops")
    args = parser.parse_args()

    args.image_root = args.image_root.resolve()
    args.labels = args.labels.resolve()
    if args.out:
        args.out = args.out.resolve()
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())

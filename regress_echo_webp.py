"""Compare current echo templates against an all-WebP echo template set.

This exercises the same full-flow echo icon path used by the backend:
card.match_icon() with cost, badge, and family validation. It does not run OCR.

Examples:
    py backend\regress_echo_webp.py --limit 500
    py backend\regress_echo_webp.py --limit 1000 --stride 3 --show-matches
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    raise SystemExit(
        "Missing cv2. Run this inside the backend Python environment, or install "
        "backend/requirements.txt first."
    )

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import card  # noqa: E402
import data  # noqa: E402


REGIONS = {
    "echo1": (0.0125, 0.6019, 0.2042, 0.9843),
    "echo2": (0.2057, 0.6019, 0.3974, 0.9843),
    "echo3": (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4": (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5": (0.7911, 0.6019, 0.9833, 0.9843),
}

_BASE_TEMPLATES = None
_BASE_FEATURES = None
_WEBP_TEMPLATES = None
_WEBP_FEATURES = None


def image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(root.glob(suffix))
    return sorted(paths)


def crop_region(img, region: tuple[float, float, float, float]):
    x1, y1, x2, y2 = region
    h, w = img.shape[:2]
    px1, py1 = round(x1 * w), round(y1 * h)
    px2, py2 = round(x2 * w), round(y2 * h)
    return img[py1:py2, px1:px2]


def build_webp_templates(source_dir: Path, temp_dir: Path, quality: int) -> None:
    for source in sorted(source_dir.glob("*")):
        if source.suffix.lower() not in {".png", ".webp"}:
            continue
        target = temp_dir / f"{source.stem}.webp"
        if source.suffix.lower() == ".webp":
            shutil.copyfile(source, target)
            continue
        img = data._read_template_image(source)
        if img is None:
            raise RuntimeError(f"Could not decode template: {source}")
        ok = cv2.imwrite(str(target), img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"Could not write WebP template: {target}")


def load_echo_features(template_dir: Path):
    sift = cv2.SIFT_create()
    templates = {}
    features = {}
    for path in sorted(template_dir.glob("*.webp")):
        img = data._read_template_image(path)
        if img is None:
            raise RuntimeError(f"Could not decode WebP template: {path}")
        img = cv2.resize(img, (188, 188))
        templates[path.stem] = img
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            features[path.stem] = (kp, des)
    return templates, features


def set_echo_templates(templates, features) -> None:
    data.ICON_TEMPLATES.clear()
    data.ICON_TEMPLATES.update(templates)
    data.TEMPLATE_FEATURES.clear()
    data.TEMPLATE_FEATURES.update(features)
    card._ECHO_FAMILY_INDEX = None


def identify(crop):
    with contextlib.redirect_stdout(io.StringIO()):
        echo_id, confidence, element = card.match_icon(crop)
    return echo_id, round(float(confidence), 4), element


def init_worker(webp_dir: str) -> None:
    global _BASE_TEMPLATES, _BASE_FEATURES, _WEBP_TEMPLATES, _WEBP_FEATURES
    _BASE_TEMPLATES = dict(data.ICON_TEMPLATES)
    _BASE_FEATURES = dict(data.TEMPLATE_FEATURES)
    _WEBP_TEMPLATES, _WEBP_FEATURES = load_echo_features(Path(webp_dir))


def compare_image(path_str: str):
    path = Path(path_str)
    img = cv2.imread(str(path))
    if img is None:
        return {
            "error": (str(path), "could not read image"),
            "mismatches": [],
            "same": 0,
            "conf_delta_total": 0.0,
            "max_conf_delta": 0.0,
        }

    mismatches = []
    same = 0
    conf_delta_total = 0.0
    max_conf_delta = 0.0

    for region_name, region in REGIONS.items():
        crop = crop_region(img, region)

        set_echo_templates(_BASE_TEMPLATES, _BASE_FEATURES)
        base = identify(crop)

        set_echo_templates(_WEBP_TEMPLATES, _WEBP_FEATURES)
        webp = identify(crop)

        if base[0] != webp[0] or base[2] != webp[2]:
            mismatches.append((path.name, region_name, base, webp))
        else:
            same += 1
            delta = abs(base[1] - webp[1])
            conf_delta_total += delta
            max_conf_delta = max(max_conf_delta, delta)

    return {
        "error": None,
        "mismatches": mismatches,
        "same": same,
        "conf_delta_total": conf_delta_total,
        "max_conf_delta": max_conf_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-dir", type=Path, default=BACKEND_DIR.parent / "r2-backup")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--show-matches", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if not args.r2_dir.exists():
        raise SystemExit(f"r2 dir not found: {args.r2_dir}")

    all_paths = image_paths(args.r2_dir)
    selected = all_paths[args.offset :: max(args.stride, 1)][: args.limit]
    if not selected:
        raise SystemExit("No regression images selected")

    base_templates = dict(data.ICON_TEMPLATES)
    base_features = dict(data.TEMPLATE_FEATURES)
    template_dir = BACKEND_DIR / "Data" / "Echoes"

    temp_root = Path(tempfile.mkdtemp(prefix="wuwa_echo_webp_"))
    start = time.time()
    try:
        build_webp_templates(template_dir, temp_root, args.quality)
        webp_templates, webp_features = load_echo_features(temp_root)
        print(
            f"selected={len(selected)} images panels={len(selected) * len(REGIONS)} "
            f"workers={args.workers} quality={args.quality}",
            flush=True,
        )

        mismatches = []
        same = 0
        conf_delta_total = 0.0
        max_conf_delta = 0.0
        errors = []

        if args.workers <= 1:
            init_worker(str(temp_root))
            for idx, path in enumerate(selected, 1):
                result = compare_image(str(path))
                if result["error"]:
                    errors.append(result["error"])
                mismatches.extend(result["mismatches"])
                same += result["same"]
                conf_delta_total += result["conf_delta_total"]
                max_conf_delta = max(max_conf_delta, result["max_conf_delta"])
                if args.show_matches and not result["mismatches"]:
                    print(f"MATCH {path.name}")
                if idx % args.progress_every == 0:
                    print(f"processed {idx}/{len(selected)} images", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(str(temp_root),),
            ) as executor:
                futures = [executor.submit(compare_image, str(path)) for path in selected]
                for idx, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    if result["error"]:
                        errors.append(result["error"])
                    mismatches.extend(result["mismatches"])
                    same += result["same"]
                    conf_delta_total += result["conf_delta_total"]
                    max_conf_delta = max(max_conf_delta, result["max_conf_delta"])
                    if idx % args.progress_every == 0:
                        print(f"processed {idx}/{len(selected)} images", flush=True)

        set_echo_templates(base_templates, base_features)

        elapsed = time.time() - start
        total_panels = len(selected) * len(REGIONS)
        avg_delta = conf_delta_total / same if same else 0.0

        print(
            f"current_templates={len(base_features)} webp_templates={len(webp_features)} "
            f"images={len(selected)} panels={total_panels} errors={len(errors)} "
            f"seconds={elapsed:.1f}"
        )
        print(
            f"mismatches={len(mismatches)} same={same} "
            f"same_match_avg_conf_delta={avg_delta:.5f} max_delta={max_conf_delta:.4f}"
        )

        if errors:
            print("\nERRORS")
            for path, error in errors[:20]:
                print(f"{path}: {error}")

        if mismatches:
            print("\nMISMATCHES")
            for image_name, region_name, base, webp in mismatches:
                print(f"{image_name} {region_name} current={base} webp={webp}")

        return 1 if mismatches or errors else 0
    finally:
        set_echo_templates(base_templates, base_features)
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

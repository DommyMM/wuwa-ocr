"""Detect substat rows re-rendered in an image editor on build cards

A re-typed row is brighter than its neighbours with a lower error level, since it skipped the original JPEG quantization
ela_delta ignores background level, so the wrapped labels that break echo_bed_score's gradient don't move it
Only combined flags, since highlighted substat rows push genuine cards to gap 41 with near-zero ela_delta
Corpus validation: docs/card-forgery-detection.md

Usage:
  py forensics_card_render.py ../r2-backup --out ../forensics/card_render
  py forensics_card_render.py suspect.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# Panel x-ratios mirror forensics_echo_integrity.py's IMPORT_REGIONS
# Row geometry is absolute pixels, since every genuine card is exactly 1920x1080
PANEL_X = (0.0125, 0.2057, 0.4016, 0.5969, 0.7911)
ROW_Y = (886, 920, 954, 988, 1022)
ROW_H = 16
LABEL_X0, LABEL_X1 = 34, 265
TEXT_LEVEL = 110          # a label pixel, below this is panel background
MIN_TEXT_PX = 80          # fewer than this means the slot is empty, not dim
TOP_N = 40                # fixed pixel count -> independent of glyph count
MIN_GRP = 3               # smallest believable number of edited rows

# GAP_TRIGGER only picks which cards pay the re-encode, and trips on 2/2625 genuine cards
# COMBINED_FLAG sits between the genuine peak of 2.00 and the two known forgeries at 40.8 and 64.4
GAP_TRIGGER = 5.0
COMBINED_FLAG = 5.0

EXPECTED = (1080, 1920)


def encoder_signature(raw: bytes) -> dict[str, Any]:
    """Container and JPEG encoder fingerprint from the raw bytes, without decoding

    Escalation signal only, never grounds for rejection
    Corpus mixes frontend canvas re-encodes with original upload bytes, so a foreign signature is often a stale client
    """
    if not raw.startswith(b"\xff\xd8"):
        return {"fmt": "PNG" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "other"}

    i, samp, dqt, app0 = 2, None, None, False
    while i < len(raw) - 1:
        if raw[i] != 0xFF:
            i += 1
            continue
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = int.from_bytes(raw[i + 2:i + 4], "big")
        if marker == 0xE0:
            app0 = True
        elif marker == 0xDB and dqt is None:
            dqt = raw[i + 5:i + 5 + 64]
        elif marker in (0xC0, 0xC1, 0xC2):
            count = raw[i + 9]
            samp = tuple(raw[i + 10 + 3 * k + 1] for k in range(count))
            break
        i += 2 + length
    return {
        "fmt": "JPEG",
        "subsampling": "4:4:4" if samp == (17, 17, 17) else "4:2:0",
        "app0": app0,
        "dqt_mean": round(float(np.mean(list(dqt))), 2) if dqt else None,
        "bytes": len(raw),
    }


def _label_cells(gray: np.ndarray) -> list[tuple[int, int]]:
    """(y, x0) of each substat label cell that holds text"""
    cells = []
    for ratio in PANEL_X:
        px = int(ratio * EXPECTED[1])
        for y in ROW_Y:
            x0 = px + LABEL_X0
            patch = gray[y:y + ROW_H, x0:px + LABEL_X1]
            if (patch > TEXT_LEVEL).sum() >= MIN_TEXT_PX:
                cells.append((y, x0))
    return cells


def render_consistency(image: np.ndarray, ela: np.ndarray | None = None) -> dict[str, Any]:
    """Score how uniformly the substat rows were rendered

    Without `ela` the caller skips the re-encode, and `ela_delta` and `combined` come back None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cells = _label_cells(gray)
    if len(cells) < 2 * MIN_GRP:
        return {"cells": len(cells), "gap": None, "tone_sd": None,
                "ela_delta": None, "combined": None}

    scored = []
    for y, x0 in cells:
        patch = gray[y:y + ROW_H, x0:x0 + (LABEL_X1 - LABEL_X0)]
        tone = float(np.sort(patch.ravel())[::-1][:TOP_N].mean())
        err = (float(np.percentile(ela[y:y + ROW_H, x0:x0 + (LABEL_X1 - LABEL_X0)], 99))
               if ela is not None else 0.0)
        scored.append((tone, err))
    scored.sort(key=lambda t: t[0])

    tones = np.array([s[0] for s in scored])
    errs = np.array([s[1] for s in scored])

    # Largest tone gap leaving at least MIN_GRP rows each side
    # A partly re-rendered card splits there into original and redrawn rows, while a single-pass card has no clean split
    gap, delta = 0.0, 0.0
    for k in range(MIN_GRP, len(scored) - MIN_GRP + 1):
        candidate = float(tones[k] - tones[k - 1])
        if candidate > gap:
            gap = candidate
            # Dim minus bright group error, large on a real edit since redrawn text is bright with low error
            delta = float(errs[:k].mean() - errs[k:].mean())

    out = {"cells": len(scored), "gap": round(gap, 3),
           "tone_sd": round(float(tones.std()), 3)}
    if ela is None:
        out["ela_delta"] = None
        out["combined"] = None
    else:
        out["ela_delta"] = round(delta, 3)
        out["combined"] = round(gap * max(delta, 0.0), 3)
    return out


def error_level(image: np.ndarray, path: str | Path) -> np.ndarray:
    """Per-pixel error after re-encoding at the genuine pipeline's JPEG settings, ~20 ms"""
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=80, subsampling=2)
    again = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR).astype(np.float32)
    return np.abs(image.astype(np.float32) - again).max(axis=2)


def analyze(path: Path) -> dict[str, Any] | None:
    """Full two-stage pass on one file, None when it isn't a readable card"""
    try:
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != EXPECTED:
            return None

        cheap = render_consistency(image)
        if cheap["gap"] is None:
            return None

        # Raw bytes are read only after the card check, since corpus directories also hold multi-hundred-MB db dumps
        row: dict[str, Any] = {"file": path.name}
        row.update(encoder_signature(path.read_bytes()))
        row.update(cheap)

        if cheap["gap"] >= GAP_TRIGGER:
            row.update(render_consistency(image, error_level(image, path)))
        row["flagged"] = bool(row.get("combined") is not None
                              and row["combined"] >= COMBINED_FLAG)
        return row
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="image file or directory to scan")
    parser.add_argument("--out", type=Path, default=Path("../forensics/card_render"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    files = (sorted(p for p in args.path.rglob("*") if p.is_file())
             if args.path.is_dir() else [args.path])
    print(f"scanning {len(files)} file(s) with {args.workers} workers ...")

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(analyze, files, chunksize=16), 1):
            if result:
                rows.append(result)
            if i % 2000 == 0:
                print(f"  ...{i}/{len(files)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    flagged = [r for r in rows if r["flagged"]]
    triggered = [r for r in rows if r.get("ela_delta") is not None]

    if rows:
        with (args.out / "card_render.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "review_queue.json").write_text(
        json.dumps(sorted(flagged, key=lambda r: -r["combined"]), indent=2),
        encoding="utf-8",
    )

    print(f"\nscored     {len(rows)} cards")
    print(f"triggered  {len(triggered)} paid the re-encode (gap >= {GAP_TRIGGER})")
    print(f"FLAGGED    {len(flagged)} (combined >= {COMBINED_FLAG})")
    for r in sorted(flagged, key=lambda r: -r["combined"])[:20]:
        print(f"  {r['combined']:8.2f}  gap={r['gap']:6.2f} ela_delta={r['ela_delta']:6.2f}  {r['file']}")
    print(f"\nwrote {args.out}/card_render.csv and review_queue.json")
    print("Review before acting. Do not auto-delete: see docs/image-integrity.md.")


if __name__ == "__main__":
    main()

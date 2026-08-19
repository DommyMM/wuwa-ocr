"""
forensics_card_render.py — detect re-rendered substat rows on build cards.

Answers a question the existing phases cannot: "were all 25 substat rows drawn
in the same render pass?" A row that was re-typed in an image editor is brighter
than its neighbours AND has a lower error level, because it never went through
the original JPEG quantization. Genuine cards show no such pairing however you
split them.

This is the confirmation `echo_bed_score` has been waiting for. Phase B is
observe-only because wrapped substat labels ("Resonance Liberation DMG Bonus")
break its background-gradient assumption — and those wraps were exactly what the
2026-08 forgeries used. `ela_delta` ignores background level entirely, so a
wrapped label does not perturb it.

Validated 2026-08-19 on 2625 cards stratified >=300 per upload month across the
whole corpus (500/month for 2026-05..08):

  statistic   genuine max   genuine p99.9   the two known forgeries
  tone_sd          18.48            5.95    10.45 / 11.17
  gap              41.27            4.67     8.10 / 12.18
  ela_delta         0.40            0.09     5.04 / 5.29
  combined          2.00            0.14     40.8 / 64.4

Use `combined` only. `gap` and `tone_sd` are NOT safe alone at corpus scale:
cards whose substat rows carry highlight/selection bands reach gap=41 while
being perfectly genuine, and `ela_delta` is what tells "different background"
apart from "different layer".

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


# Panel x-ratios mirror IMPORT_REGIONS in forensics_echo_integrity.py. The row
# geometry is absolute because every genuine card is exactly 1920x1080; a
# different size is already rejected upstream by validate_image_integrity.
PANEL_X = (0.0125, 0.2057, 0.4016, 0.5969, 0.7911)
ROW_Y = (886, 920, 954, 988, 1022)
ROW_H = 16
LABEL_X0, LABEL_X1 = 34, 265
TEXT_LEVEL = 110          # a label pixel; below this is panel background
MIN_TEXT_PX = 80          # fewer than this means the slot is empty, not dim
TOP_N = 40                # fixed pixel count -> independent of glyph count
MIN_GRP = 3               # smallest believable number of edited rows

# A card must clear BOTH to be worth reporting. GAP_TRIGGER alone fires on
# 0.08% of genuine cards (2/2625) and exists only to skip the re-encode.
GAP_TRIGGER = 5.0
COMBINED_FLAG = 5.0

EXPECTED = (1080, 1920)


def encoder_signature(raw: bytes) -> dict[str, Any]:
    """Walk the JPEG markers without decoding. ~1.3 microseconds per card.

    Escalation signal ONLY, never grounds for rejection: the corpus contains two
    encoder eras (frontend canvas-recompression before 2026-07, original input
    bytes after), so a "foreign" signature is often just a stale client.
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
    """Yield (y, x0) for each substat label cell that actually holds text."""

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
    """Score how uniformly the substat rows were rendered.

    `ela` is optional so callers that only want the cheap trigger can skip the
    re-encode; without it `ela_delta` and `combined` are None.
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

    # Largest clean split leaving at least MIN_GRP rows either side. A card whose
    # rows were all drawn together has no such split; a partly re-rendered one
    # separates into "original" and "redrawn" groups.
    gap, delta = 0.0, 0.0
    for k in range(MIN_GRP, len(scored) - MIN_GRP + 1):
        candidate = float(tones[k] - tones[k - 1])
        if candidate > gap:
            gap = candidate
            # Dim group minus bright group: re-rendered text is bright and has
            # LOW error level, so a real edit makes this large and positive.
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
    """Requantize at the genuine pipeline's own settings and diff. ~20 ms."""

    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=80, subsampling=2)
    again = cv2.cvtColor(np.array(Image.open(buf)), cv2.COLOR_RGB2BGR).astype(np.float32)
    return np.abs(image.astype(np.float32) - again).max(axis=2)


def analyze(path: Path) -> dict[str, Any] | None:
    """Full two-stage pass on one file. Returns None if it is not a card."""

    try:
        raw = path.read_bytes()
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != EXPECTED:
            return None

        cheap = render_consistency(image)
        if cheap["gap"] is None:
            return None

        row: dict[str, Any] = {"file": path.name}
        row.update(encoder_signature(raw))
        row.update(cheap)

        # Only pay for the re-encode when the cheap trigger fires.
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

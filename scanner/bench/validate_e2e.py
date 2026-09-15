"""End-to-end stat icon and substat value check on the three labelled 4K echoes

    py bench/validate_e2e.py

Panel X is hand-measured and the Y band runs past any substat wrap, while icon column, rows and value cells self-locate
Echo 2 holds a two-line wrap with a flat DEF, and echo 3 two consecutive wraps with an ATK% substat
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wuwa_scanner.stats import FAMILY, find_rows, resolve_stat, value_cells  # noqa: E402

REF_W, REF_H = 3840, 2160
PANEL_X = (2620 / REF_W, 3720 / REF_W)   # hand-measured
STATS_Y = (0.400, 0.790)                 # extended well past any wrap
VALUE_FRAC = 0.74
NUM_RX = re.compile(r"\d+(?:[.,]\d+)?")

# (stat, value), rows 0 and 1 are main and innate, derived from cost in the pipeline and listed only to score icons
GOLD = {
    "bag_4k_01.jpg": [
        ("Crit DMG", 44.0), ("ATK", 150.0),
        ("Heavy Attack DMG Bonus", 7.9), ("HP%", 10.1), ("Energy Regen", 9.2),
        ("Crit DMG", 21.0), ("ATK", 40.0),
    ],
    "bag_4k_02.jpg": [
        ("Healing Bonus", 26.4), ("ATK", 150.0),
        ("Heavy Attack DMG Bonus", 7.9), ("DEF", 60.0), ("Crit DMG", 12.6),
        ("Resonance Liberation DMG Bonus", 7.1), ("Crit Rate", 6.9),
    ],
    # Frostbite Coleoid, cost 3: two consecutive wraps, ATK% 7.9 against flat ATK's 30-60, and innate ATK 100 not 150
    "bag_4k_03_cost3.jpg": [
        ("Glacio DMG", 30.0), ("ATK", 100.0),
        ("Resonance Skill DMG Bonus", 10.9), ("Resonance Liberation DMG Bonus", 10.1),
        ("Crit DMG", 21.0), ("Crit Rate", 8.7), ("ATK%", 7.9),
    ],
}


def read_values(cells: list[np.ndarray]) -> list[float | None]:
    """One Tesseract process with a result per cell, never one stitched image, which lets a dropped line shift rows"""
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, im in enumerate(cells):
            up = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            g = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            p = f"{td}/c{i:02d}.png"
            cv2.imwrite(p, th)
            paths.append(p)
        Path(f"{td}/l.txt").write_text("\n".join(paths))
        out = subprocess.run(
            ["tesseract", f"{td}/l.txt", "stdout", "--psm", "7",
             "-c", "tessedit_char_whitelist=0123456789.%"],
            capture_output=True, text=True,
        ).stdout
    vals: list[float | None] = []
    for page in out.split("\f")[: len(cells)]:
        m = NUM_RX.search(page.replace(" ", "").replace("\n", "").replace("%", ""))
        try:
            vals.append(float(m.group().replace(",", ".")) if m else None)
        except ValueError:
            vals.append(None)
    return vals


def main() -> None:
    import data

    total_icons = total_vals = ok_icons = ok_vals = 0

    for name, gold in GOLD.items():
        path = Path("samples") / name
        if not path.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        st = np.ascontiguousarray(
            img[int(STATS_Y[0] * h):int(STATS_Y[1] * h),
                int(PANEL_X[0] * w):int(PANEL_X[1] * w)]
        )
        sw = st.shape[1]

        rows = find_rows(st)
        cells = value_cells(st, rows, VALUE_FRAC)
        blank = np.zeros((8, 8, 3), np.uint8)
        nums = read_values([c if c is not None else blank for c in cells])

        print(f"\n{'=' * 78}\n{name}   rows found: {len(rows)}  (gold {len(gold)})\n{'=' * 78}")
        print(f"{'#':>2} {'STAT (icon)':32s} {'IoU':>5s} {'marg':>5s} {'read':>7s} "
              f"{'snap':>7s}  verdict")
        print("-" * 78)

        for i, (r, num) in enumerate(zip(rows, nums)):
            g_stat, g_val = gold[i] if i < len(gold) else ("?", 0.0)
            # % is never read, so percent-ness is a first guess by magnitude that the legal sets below settle
            stat = resolve_stat(r["icon"], "%" if num is not None and num < 100 else "")
            # Flat and percent legal sets are disjoint, so the nearest legal value within 2.0 picks the member
            members = [m for m in (stat, stat.rstrip("%") if stat else "") if m]
            chosen, snapped = stat, num
            if num is not None:
                best = None
                for m in {stat, (stat or "").rstrip("%"), (stat or "").rstrip("%") + "%"}:
                    legal = data.SUB_STATS.get(m)
                    if not legal:
                        continue
                    c = min(legal, key=lambda v: abs(float(v) - num))
                    d = abs(float(c) - num)
                    if d <= 2.0 and (best is None or d < best[0]):
                        best = (d, m, float(c))
                if best:
                    _, chosen, snapped = best

            icon_ok = (chosen or "").rstrip("%") == g_stat.rstrip("%")
            val_ok = snapped is not None and abs(snapped - g_val) < 0.05
            is_sub = i >= 2
            total_icons += 1
            ok_icons += icon_ok
            if is_sub:
                total_vals += 1
                ok_vals += val_ok

            tag = "OK" if icon_ok and (val_ok or not is_sub) else \
                  f"MISS exp {g_stat} {g_val:g}"
            rd = "-" if num is None else f"{num:g}"
            sn = "-" if snapped is None else f"{snapped:g}"
            note = "" if is_sub else "  (derived, not OCR'd)"
            print(f"{i:>2} {str(chosen):32s} {r['iou']:5.2f} {r['margin']:5.2f} "
                  f"{rd:>7s} {sn:>7s}  {tag}{note}")

    print(f"\n{'=' * 78}")
    print(f"icons: {ok_icons}/{total_icons}    substat values: {ok_vals}/{total_vals}")


if __name__ == "__main__":
    main()

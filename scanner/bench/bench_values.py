"""THE decisive bench: which engine reads one icon-anchored value cell, fast?

    .bench-venv/Scripts/python.exe bench/bench_values.py samples/bag_4k_01.jpg

Why this is the only OCR question left
--------------------------------------
Stat NAMES come from the 17 icon templates: 0.3 ms/row, no OCR, and correct in
all 9 WuWa languages. Rows are anchored on icon blobs, so name/value alignment is
guaranteed by construction and the wrap problem disappears.

That leaves VALUES. They must be read per-row, not as a batched column: batching
lets the engine drop a line and shift every row below it, which is precisely the
drift card.py's reconcile_echo_substat_rows exists to survive. So we pay one OCR
call per row (~7/echo) and the only thing that matters is per-call cost on a tiny
crop, in-process.

pytesseract is disqualified on its face: ~154 ms/call, essentially all of it
subprocess spawn. At 7 rows that is >1 s/echo of pure overhead.
"""
from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.engines import ALL_ENGINES  # noqa: E402
from wuwa_scanner.stats import find_rows  # noqa: E402

STATS_BOX = (0.690, 0.415, 0.975, 0.715)
ICON_FRAC = 0.075
VALUE_FRAC = 0.74

# Row 0 is the MAIN stat and row 1 the INNATE base stat. Neither needs OCR: the
# icon gives the stat, and the value is fully determined by (cost, stat, level)
# via Data/EchoStats.json. The roadmap already mandates this ("echo main stat
# value | derive from cost and stat name | Do not OCR").
#
# So only the SUBSTAT rows are an OCR problem: 5 cells per echo, not 7.
SKIP_ROWS = 2

# (stat, true numeric value). The icon supplies the stat; OCR supplies only digits.
TRUTH = [
    ("Heavy Attack DMG Bonus", 7.9), ("HP%", 10.1), ("Energy Regen", 9.2),
    ("Crit DMG", 21.0), ("ATK", 40.0),
]
NUM_RX = re.compile(r"\d+(?:[.,]\d+)?")

RUNS = 5


def parse_num(lines: list[str]) -> float | None:
    """Digits only. The '%' is NEVER read.

    The stat family (from the icon) already determines whether the value is a
    percent, and for the three flat/percent families the legal sets are disjoint
    (HP% 6.4-11.6 vs HP 320-580; ATK% 6.4-11.6 vs ATK 30-60; DEF% 8.1-14.7 vs
    DEF 40-70), so the NUMBER alone resolves it. Engines that drop the '%' are
    therefore not wrong in any way the pipeline cares about.
    """
    blob = "".join(lines).replace(" ", "").replace("%", "")
    m = NUM_RX.search(blob)
    if not m:
        return None
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return None


def snap(stat: str, num: float | None, legal_by_stat: dict) -> float | None:
    """Snap a read number to the stat's closed legal set. Arbitration only:
    the reader must be discriminative, the legal set never *guesses* the value.
    """
    if num is None:
        return None
    legal = legal_by_stat.get(stat)
    if not legal:
        return num
    best = min(legal, key=lambda v: abs(float(v) - num))
    return float(best) if abs(float(best) - num) <= 2.0 else None


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import data

    legal = dict(data.SUB_STATS)
    # Main-stat rows are not substats; allow them through unsnapped.
    truth_stats = [t[0] for t in TRUTH]
    truth_nums = [t[1] for t in TRUTH]

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/bag_4k_01.jpg")
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = STATS_BOX
    stats = np.ascontiguousarray(img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)])
    sh, sw = stats.shape[:2]

    rows = find_rows(stats)[SKIP_ROWS:]  # substat rows only
    cells = [
        np.ascontiguousarray(stats[max(0, r["band"][0] - 6):min(sh, r["band"][1] + 6), int(sw * VALUE_FRAC):])
        for r in rows
    ]
    print(f"{path.name} {w}x{h} -> {len(cells)} SUBSTAT value cells, "
          f"each ~{cells[0].shape[1]}x{cells[0].shape[0]}")
    print("(main + innate rows are derived from cost via EchoStats.json, never OCR'd)")
    print("scoring: digits only, then snapped to the icon-identified stat's legal set\n")

    print(f"{'engine':28s} {'load':>7s} {'ms/cell':>8s} {'ms/echo':>8s} {'acc':>6s}   snapped reads")
    print("-" * 108)

    for eng in ALL_ENGINES:
        ok, reason = eng.available()
        if not ok:
            print(f"{eng.name:28s} {'-':>7s} {'-':>8s} {'-':>8s} {'SKIP':>6s}   {reason[:42]}")
            continue

        t0 = time.perf_counter()
        eng.load()
        load_ms = (time.perf_counter() - t0) * 1000

        try:
            eng.read_batch(cells)  # warm before timing
            samples: list[float] = []
            per_row: list[list[str]] = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                per_row = eng.read_batch(cells)
                samples.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:
            print(f"{eng.name:28s} {'-':>7s} {'-':>8s} {'-':>8s} {'ERR':>6s}   "
                  f"{type(exc).__name__}: {str(exc)[:50]}")
            continue

        got = [snap(s, parse_num(lines), legal) for s, lines in zip(truth_stats, per_row)]
        hits = sum(g is not None and abs(g - t) < 0.05 for g, t in zip(got, truth_nums))
        echo_ms = statistics.median(samples)
        shown = [("-" if g is None else f"{g:g}") for g in got]
        print(f"{eng.name:28s} {load_ms:6.0f}m {echo_ms / len(cells):7.1f}m {echo_ms:7.1f}m "
              f"{hits}/{len(TRUTH):<4d}   {shown}")

    print("-" * 108)
    print(f"truth: {[f'{n:g}' for n in truth_nums]}")
    print("\nms/echo = ms/cell x 7 rows. The click+settle wall is ~150-250 ms, so any")
    print("engine under ~150 ms/echo disappears entirely behind navigation.")


if __name__ == "__main__":
    main()

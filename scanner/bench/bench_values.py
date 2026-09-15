"""Which engine reads one icon-anchored substat value cell accurately and fast

    .bench-venv/Scripts/python.exe bench/bench_values.py samples/bag_4k_01.jpg

Stat names come from the 17 icon templates (0.3 ms/row, no OCR), and icon-anchored rows keep names and values aligned
Values are read one cell per row, since a batched column lets an engine drop a line and shift every row below
So what matters is the cost of reading 5 tiny substat cells per echo
Per-call pytesseract is ~154 ms of process spawn, so the Tesseract engine batches cells into one process
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

# Rows 0 and 1 are the main and innate stats, whose values follow from cost, stat and level via Data/EchoStats.json
SKIP_ROWS = 2

# (stat, true value), the icon supplies the stat and OCR only the digits
TRUTH = [
    ("Heavy Attack DMG Bonus", 7.9), ("HP%", 10.1), ("Energy Regen", 9.2),
    ("Crit DMG", 21.0), ("ATK", 40.0),
]
NUM_RX = re.compile(r"\d+(?:[.,]\d+)?")

RUNS = 5


def parse_num(lines: list[str]) -> float | None:
    """First number in the read, ignoring '%'

    Flat and percent legal sets are disjoint (ATK 30-60, ATK% 6.4-11.6), so the number alone picks the member
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
    """Snap a read number to the stat's legal set within 2.0, or pass it through when the stat has none

    Arbitration only, so the reader must discriminate and the legal set never guesses the value
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

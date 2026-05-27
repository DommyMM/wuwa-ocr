"""Bench: SIFT echo identification + OCR substats on bag-view samples.

Measures latency and correctness against ground truth, with:
- Cost prefilter for SIFT (match cost badge first, narrow template pool)
- Warmed-up OCR engines (cold-start cost reported separately)
- Substat row parsing + fuzzy match against data.SUB_STATS
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/user/wuwa-ocr-api")

from extract_regions import REGIONS, proportional_crop  # noqa: E402

import data  # noqa: E402
from rapidocr_onnxruntime import RapidOCR  # noqa: E402
import pytesseract  # noqa: E402
from rapidfuzz import process, fuzz  # noqa: E402

SAMPLES = [
    {
        "path": "/home/user/wuwa-ocr-api/scanner_proto/samples/bag_view_01.png",
        "expected": {
            "echo": "Devotee's Flesh",
            "echo_id": "60001105",
            "cost": 1,
            "equipped_by": "Cartethyia",
            "stats": [
                ("HP", "22.8%"),
                ("HP", "2280"),
                ("Crit. Rate", "6.9%"),
                ("Crit. DMG", "13.8%"),
                ("HP", "470"),
                ("Resonance Liberation DMG Bonus", "10.1%"),
                ("HP", "10.1%"),
            ],
        },
    },
    {
        "path": "/home/user/wuwa-ocr-api/scanner_proto/samples/bag_view_02.jpg",
        "expected": {
            "echo": "Dreamless",
            "echo_id": "60000535",
            "cost": 4,
            "equipped_by": "Camellya",
            "stats": [
                ("Crit. Rate", "22.0%"),
                ("ATK", "150"),
                ("Basic Attack DMG Bonus", "7.9%"),
                ("Resonance Liberation DMG Bonus", "8.6%"),
                ("DEF", "60"),
                ("HP", "390"),
                ("ATK", "6.4%"),
            ],
        },
    },
]


def time_call(fn, *a, **kw):
    t0 = time.perf_counter()
    result = fn(*a, **kw)
    return result, (time.perf_counter() - t0) * 1000.0


# ---------- COST badge detection ----------

def identify_cost(badge_bgr):
    """Match the badge against cost1/3/4 templates. Returns (cost, score)."""
    badge_gray = cv2.cvtColor(badge_bgr, cv2.COLOR_BGR2GRAY)
    best, best_score = None, -1.0
    for cost, tpl in data.COST_TEMPLATES.items():
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim == 3 else tpl
        h, w = tpl_gray.shape
        if badge_gray.shape[0] < h or badge_gray.shape[1] < w:
            badge_resized = cv2.resize(badge_gray, (max(w, badge_gray.shape[1]), max(h, badge_gray.shape[0])))
        else:
            badge_resized = badge_gray
        res = cv2.matchTemplate(badge_resized, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best, best_score = cost, score
    return best, best_score


# ---------- SIFT echo identification with cost prefilter ----------
# Mirrors card.py:406-424 (match_icon) — same FLANN config, same conf formula.
# Difference: card.py crops the top-left 188x188 of an export card; we get the
# icon from the bag detail-panel preview, so we resize.

# Pre-bucket templates by cost so cost-filtered SIFT skips iteration entirely.
_TEMPLATES_BY_COST: dict = {}


def build_cost_buckets():
    if _TEMPLATES_BY_COST:
        return
    for echo_id, kp_des in data.TEMPLATE_FEATURES.items():
        cost = data.ECHO_COSTS.get(echo_id)
        _TEMPLATES_BY_COST.setdefault(cost, {})[echo_id] = kp_des


def sift_identify_echo(icon_bgr, cost_filter=None, resize_to=(188, 188), ratio=0.75):
    """Match icon against echo templates using SIFT + FLANN.

    Returns: (best_name, best_id, confidence, full_ranked_list, candidate_count)
    """
    build_cost_buckets()
    sift = cv2.SIFT_create()
    icon_resized = cv2.resize(icon_bgr, resize_to, interpolation=cv2.INTER_CUBIC) if resize_to else icon_bgr
    icon_kp, icon_des = sift.detectAndCompute(icon_resized, None)
    if icon_des is None or len(icon_des) < 2:
        return None, None, 0.0, [], 0

    candidates = _TEMPLATES_BY_COST.get(cost_filter, {}) if cost_filter else data.TEMPLATE_FEATURES
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    results = []
    for echo_id, (tpl_kp, tpl_des) in candidates.items():
        if tpl_des is None or len(tpl_des) < 2:
            continue
        matches = flann.knnMatch(icon_des, tpl_des, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
        denom = max(len(icon_kp), len(tpl_kp), 1)
        conf = len(good) / denom
        name = data.ECHO_NAME_MAP.get(echo_id, echo_id)
        results.append((name, echo_id, conf))
    results.sort(key=lambda r: -r[2])
    if not results:
        return None, None, 0.0, [], len(candidates)
    return results[0][0], results[0][1], results[0][2], results, len(candidates)


# ---------- OCR ----------

def ocr_tesseract(img_bgr, psm=6):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(gray, config=f"--psm {psm}")


_rapid = None


def get_rapid():
    global _rapid
    if _rapid is None:
        _rapid = RapidOCR()
    return _rapid


def ocr_rapid(img_bgr):
    result, _ = get_rapid()(img_bgr)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


# ---------- Stat row parsing ----------

VALUE_RX = re.compile(r"^([0-9]+(?:[.,][0-9])?)(%?)$")
JUNK_RX = re.compile(r"^[\W_]{1,2}$")  # single/double-char punctuation/symbols


def _best_match(raw, choices, threshold=60):
    res = process.extractOne(raw, choices, scorer=fuzz.WRatio)
    if not res:
        return None, 0.0
    best, score, _ = res
    return (best, score) if score >= threshold else (None, score)


def parse_stat_rows(raw_text, valid_stat_names):
    """Pair lines into (stat_name, value) tuples.

    RapidOCR emits lines top-to-bottom but stat names may wrap to a second
    line below the value (e.g. 'Resonance Liberation' / '10.1%' / 'DMG Bonus').
    Strategy:
      1. Strip junk tokens (single symbols from icon glyphs).
      2. Walk lines; when we hit a value, pair it with the accumulated name.
      3. If the pair doesn't fuzzy-match, try joining a subsequent line as
         a wrap continuation.
    """
    raw_tokens = [t.strip() for t in raw_text.split("\n") if t.strip()]
    tokens = [t for t in raw_tokens if not JUNK_RX.match(t)]

    parsed = []
    pending = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if VALUE_RX.match(tok.replace(",", "")) and pending:
            name = " ".join(pending).replace(".", ". ").replace("  ", " ").strip()
            matched, score = _best_match(name, valid_stat_names)

            # If poor match and next line is a non-value, try appending it
            # (handles 'Resonance Liberation' / value / 'DMG Bonus' wrap).
            if (not matched or score < 85) and i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if not VALUE_RX.match(nxt.replace(",", "")):
                    extended = f"{name} {nxt}"
                    matched2, score2 = _best_match(extended, valid_stat_names)
                    if score2 > score:
                        name = extended
                        matched, score = matched2, score2
                        i += 1  # consume the continuation

            parsed.append({"raw_name": name, "matched": matched, "score": score, "value": tok})
            pending = []
        else:
            pending.append(tok)
        i += 1
    return parsed


# ---------- Main bench ----------

def warmup_ocr():
    dummy = (255 * cv2.imread("/home/user/wuwa-ocr-api/scanner_proto/crops/stats_block.png")).clip(0, 255).astype("uint8")
    if dummy is None:
        return
    ocr_tesseract(dummy)
    ocr_rapid(dummy)


def main():
    print(f"Loaded {len(data.TEMPLATE_FEATURES)} echo templates, {len(data.COST_TEMPLATES)} cost templates")

    # OCR cold-start
    t0 = time.perf_counter()
    get_rapid()  # init
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"RapidOCR init: {init_ms:.1f}ms")

    # Warmup
    stub = cv2.imread("/home/user/wuwa-ocr-api/scanner_proto/crops/stats_block.png")
    ocr_tesseract(stub)
    ocr_rapid(stub)
    print("OCR warmup complete\n")

    valid_stats = list(data.SUB_STATS.keys()) + list(data.MAIN_STATS.get("4cost", {}).keys() if hasattr(data, "MAIN_STATS") else [])
    valid_stats = sorted(set(valid_stats))

    for s in SAMPLES:
        print("=" * 76)
        print(f"Sample: {Path(s['path']).name}  expected: {s['expected']['echo']}  cost {s['expected']['cost']}")
        print("=" * 76)
        img = cv2.imread(s["path"])
        h, w = img.shape[:2]
        print(f"  dims: {w}x{h}")

        icon = proportional_crop(img, REGIONS["icon_preview"])
        stats_crop = proportional_crop(img, REGIONS["stats_block"])
        name_crop = proportional_crop(img, REGIONS["echo_name"])
        equipped = proportional_crop(img, REGIONS["equipped_by"])
        cost_crop = proportional_crop(img, REGIONS["cost_badge"])

        # Cost
        (cost, cost_score), t_cost = time_call(identify_cost, cost_crop)
        cost_ok = cost == s["expected"]["cost"]
        print(f"\n  cost      [{t_cost:6.1f}ms]  -> {cost} (score {cost_score:.3f}) {'OK' if cost_ok else 'WRONG, expected ' + str(s['expected']['cost'])}")

        # SIFT (unfiltered)
        (n, eid, conf, ranked, cc), t_sift_full = time_call(sift_identify_echo, icon, None)
        print(f"\n  SIFT full [{t_sift_full:6.1f}ms over {cc} templates] -> {n!r} conf={conf:.3f}")
        for nm, eid_, c in ranked[:5]:
            mark = "*" if nm == s["expected"]["echo"] else " "
            print(f"    {mark} {c:.3f}  {nm}")
        rank_full = next((i for i, r in enumerate(ranked) if r[0] == s["expected"]["echo"]), -1)
        print(f"    expected rank in full ranking: #{rank_full + 1} of {len(ranked)}")

        # SIFT (cost-filtered) — uses detected cost, so a wrong cost would propagate
        (n2, eid2, conf2, ranked2, cc2), t_sift_filt = time_call(sift_identify_echo, icon, cost)
        sift_ok = n2 == s["expected"]["echo"]
        print(f"\n  SIFT cost={cost} [{t_sift_filt:6.1f}ms over {cc2} templates] -> {n2!r} conf={conf2:.3f}  {'OK' if sift_ok else 'WRONG'}")
        for nm, eid_, c in ranked2[:5]:
            mark = "*" if nm == s["expected"]["echo"] else " "
            print(f"    {mark} {c:.3f}  {nm}")
        rank_filt = next((i for i, r in enumerate(ranked2) if r[0] == s["expected"]["echo"]), -1)
        if rank_filt > 4:
            for i in range(max(0, rank_filt - 1), min(len(ranked2), rank_filt + 2)):
                nm, eid_, c = ranked2[i]
                print(f"    @ #{i+1} {c:.3f}  {nm}")
        print(f"    expected rank in cost-filtered ranking: #{rank_filt + 1} of {len(ranked2)}")

        # SIFT with larger upscale — does interpolation help low-res icons?
        (n3, eid3, conf3, ranked3, _), t_sift_big = time_call(sift_identify_echo, icon, cost, (384, 384))
        sift_big_ok = n3 == s["expected"]["echo"]
        rank_big = next((i for i, r in enumerate(ranked3) if r[0] == s["expected"]["echo"]), -1)
        print(f"\n  SIFT cost={cost} upscale=384 [{t_sift_big:6.1f}ms] -> {n3!r} conf={conf3:.3f}  {'OK' if sift_big_ok else 'WRONG, expected at #' + str(rank_big + 1)}")

        # Stats OCR
        tess_stats, t_tess_s = time_call(ocr_tesseract, stats_crop)
        rapid_stats, t_rapid_s = time_call(ocr_rapid, stats_crop)
        print(f"\n  stats_block tesseract [{t_tess_s:6.1f}ms]:")
        for ln in tess_stats.strip().splitlines():
            print(f"    > {ln}")
        print(f"  stats_block rapidocr  [{t_rapid_s:6.1f}ms]:")
        for ln in rapid_stats.strip().splitlines():
            print(f"    > {ln}")

        parsed = parse_stat_rows(rapid_stats, valid_stats)
        print(f"\n  parsed rows ({len(parsed)} of {len(s['expected']['stats'])} expected):")
        expected = s["expected"]["stats"]
        for i, row in enumerate(parsed):
            exp = expected[i] if i < len(expected) else ("?", "?")
            ok_name = row["matched"] == exp[0] if row["matched"] else False
            ok_val = row["value"].replace(" ", "") == exp[1].replace(" ", "")
            mark = "OK" if ok_name and ok_val else "??"
            print(f"    {mark} parsed=({row['matched']!r}, {row['value']!r})  expected=({exp[0]!r}, {exp[1]!r})  score={row['score']}")

        # Name OCR (sanity check; SIFT is the real signal)
        n_tess, t_n_tess = time_call(ocr_tesseract, name_crop, 7)
        n_rapid, t_n_rapid = time_call(ocr_rapid, name_crop)
        echo_names = list(data.ECHO_NAME_MAP.values())
        best_tess = process.extractOne(n_tess.strip(), echo_names) if n_tess.strip() else None
        best_rapid = process.extractOne(n_rapid.strip(), echo_names) if n_rapid.strip() else None
        print(f"\n  echo_name tess  [{t_n_tess:6.1f}ms] raw={n_tess.strip()!r}  fuzzy={best_tess}")
        print(f"  echo_name rapid [{t_n_rapid:6.1f}ms] raw={n_rapid.strip()!r}  fuzzy={best_rapid}")

        # Equipped-by
        eq_rapid, t_eq = time_call(ocr_rapid, equipped)
        eq_clean = re.sub(r"(?i)equipped\s*by", "", eq_rapid).strip()
        chars = data.CHARACTER_NAMES
        best_eq = process.extractOne(eq_clean, chars) if eq_clean else None
        eq_ok = best_eq and best_eq[0] == s["expected"]["equipped_by"]
        print(f"\n  equipped_by [{t_eq:6.1f}ms] raw={eq_rapid.strip()!r}  fuzzy={best_eq}  {'OK' if eq_ok else 'WRONG'}")

        print()


if __name__ == "__main__":
    main()

import cv2
import pytesseract
import re
from data import CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_ID_MAP, MAIN_STAT_NAMES, MAIN_STATS, SUB_STATS, ECHO_SET_IDS, SET_NAME_BY_ID, ECHO_COSTS, ECHO_NAME_MAP, ROVER_GENDER_BY_ID, ROVER_ELEMENT_BY_ID, ICON_TEMPLATES, TEMPLATE_FEATURES, COST_TEMPLATES, determine_element
import numpy as np
from rapidfuzz import fuzz, process
from typing import Collection, Tuple
from cv2 import FlannBasedMatcher
from pathlib import Path
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading


class _ThreadLocalStdout:
    """Stdout shim routing writes to a per-thread buffer while one is pushed, else to the real stream"""

    def __init__(self, real):
        self._real = real
        self._local = threading.local()

    def push(self) -> None:
        self._local.buffer = io.StringIO()

    def pop(self) -> str:
        buf = getattr(self._local, "buffer", None)
        self._local.buffer = None
        return buf.getvalue() if buf is not None else ""

    def write(self, s):
        buf = getattr(self._local, "buffer", None)
        return buf.write(s) if buf is not None else self._real.write(s)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name):
        # Delegate everything else (reconfigure, encoding, isatty, ...) to the real stream
        return getattr(self._real, name)


_STDOUT = _ThreadLocalStdout(sys.stdout)
sys.stdout = _STDOUT

# Minimum fuzz.ratio for a weapon-name read, since real reads score ~92-100 and garbled text under ~40
WEAPON_NAME_MIN_SCORE = 75

# OCR-noise spellings per element, extend when a new Rover element ships
ROVER_ELEMENT_ALIASES = {
    "Aero": ("aero", "acro"),
    "Spectro": ("spectro", "speetro"),
    "Havoc": ("havoc", "lavoc"),
    "Electro": ("electro", "clectro"),
}
# Derived from Characters.json and data.py's ROVER_GENDER_BY_ID, so a new Rover element only needs its two ids there
ROVER_IDS_BY_GENDER_ELEMENT = {
    (gender, ROVER_ELEMENT_BY_ID[cid]): cid
    for cid, gender in ROVER_GENDER_BY_ID.items()
    if cid in ROVER_ELEMENT_BY_ID
}
ROVER_KNOWN_ELEMENTS = {element for _gender, element in ROVER_IDS_BY_GENDER_ELEMENT}
# Measured badge hue medians, valid for Rover since its badge keeps element color on the purple header
ROVER_BADGE_HUE_ANCHORS = {
    "Spectro": 26,
    "Aero": 77,
    "Electro": 135,
    "Havoc": 161,
}


FORTE_REGIONS = {
    "normal": {"x1": 270, "y1": 144, "x2": 389, "y2": 204},
    "skill": {"x1": 48, "y1": 302, "x2": 158, "y2": 356},
    "circuit": {"x1": 467, "y1": 296, "x2": 596, "y2": 357},
    "intro": {"x1": 122, "y1": 545, "x2": 247, "y2": 602},
    "lib": {"x1": 386, "y1": 544, "x2": 518, "y2": 601}
}

SEQUENCE_REGIONS = {
    "S1": {"center": (55, 58), "width": 30, "height": 26},
    "S2": {"center": (130, 58), "width": 30, "height": 26},
    "S3": {"center": (210, 58), "width": 30, "height": 26},
    "S4": {"center": (290, 58), "width": 30, "height": 26},
    "S5": {"center": (369, 58), "width": 30, "height": 26},
    "S6": {"center": (449, 58), "width": 30, "height": 26}
}

# Active nodes have a large pale center, lossy JPEGs drop it to ~0.71 but inactive nodes stay under 0.58
SEQUENCE_ACTIVE_GRAY_RATIO = 0.65

ECHO_REGIONS = {
    "main": {"x1": 195, "y1": 66, "x2": 366, "y2": 148},
    "subs_names": {"x1": 36, "y1": 228, "x2": 290, "y2": 400},
    "subs_values": {"x1": 290, "y1": 228, "x2": 359, "y2": 400}
}


def preprocess_region(image):
    """Grayscale, denoise, sharpen and binarize at a fixed 140 threshold for Tesseract"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    blur = cv2.GaussianBlur(bilateral, (0,0), 3)
    sharp = cv2.addWeighted(bilateral, 1.5, blur, -0.5, 0)
    _, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
    return thresh

def clean_stat_name(name: str, value: str) -> str:
    name = re.sub(r'\s+', ' ', name.strip()).replace("Crit.", "Crit").rstrip('.')
    if name.upper() in ["ATK", "HP", "DEF"] and "%" in value:
        return f"{name.upper()}%"
    return name.upper() if name.upper() in ["ATK", "HP", "DEF"] else name

def validate_stat(name: str, valid_names: Collection[str]) -> str:
    if not valid_names:
        return name
    match = process.extractOne(name, list(valid_names))
    return match[0] if match else name

# A band edge can clip the 'o' of a wrapped Resonance name ('Rescnance Skill DMG'), which then fuzzy-matches Crit DMG
# 'skill' and 'liberation' occur in no other stat, so they decide the name outright
_RESONANCE_BY_FRAGMENT = (
    ("skill", "Resonance Skill DMG Bonus"),
    ("liberation", "Resonance Liberation DMG Bonus"),
)

# Plain fuzz.ratio a name read needs against its resolved stat, below which the row resolves to nothing
# WRatio picks the stat but can't gate it, since it scores fragments like 'ne' at 90 against 'Energy Regen'
# Over the 6000-card gates, good reads score >= 77 and garbage <= 58
SUBSTAT_NAME_MATCH_FLOOR = 65


def _substat_name_confidence(raw: str, value: str) -> float:
    """Plain-ratio similarity of a raw name read to its resolved stat, 0 if none"""
    if not raw:
        return 0.0
    stat = validate_substat_name(raw, value or "1%")
    if not stat:
        return 0.0
    cleaned = clean_stat_name(raw, value or "1%")
    if any(key in _canonical_stat_fragment(cleaned) for key, _ in _RESONANCE_BY_FRAGMENT):
        return 100.0
    return float(fuzz.ratio(cleaned.lower(), stat.lower()))


def validate_substat_name(name: str, value: str) -> str:
    cleaned = clean_stat_name(name, value)
    fragment = _canonical_stat_fragment(cleaned)
    for key, stat in _RESONANCE_BY_FRAGMENT:
        if key in fragment and stat in SUB_STATS:
            return stat
    matched = validate_stat(cleaned, SUB_STATS.keys())
    if not matched or fuzz.ratio(cleaned.lower(), matched.lower()) < SUBSTAT_NAME_MATCH_FLOOR:
        return ""
    base = matched.replace("%", "")
    if base in {"HP", "ATK", "DEF"}:
        return f"{base}%" if "%" in value else base
    return matched

# Percent-stat snap tolerance absorbs display rounding (DEF 11.9% for the 11.8 roll) since legal rolls sit >= 0.7 apart
# Flat stats must match exactly
SUBSTAT_SNAP_TOLERANCE = 0.15
FLAT_SUBSTATS = ("HP", "ATK", "DEF")


def validate_value(value: str, stat_name: str) -> str:
    if not SUB_STATS or stat_name not in SUB_STATS:
        return value
    had_percent = "%" in value
    try:
        numeric = float(value.replace('%', '').strip())
    except ValueError:
        return value
    legal = SUB_STATS[stat_name]
    closest = min(legal, key=lambda x: abs(numeric - x))
    tolerance = 0.0 if stat_name in FLAT_SUBSTATS else SUBSTAT_SNAP_TOLERANCE
    if abs(numeric - closest) <= tolerance:
        snapped = format_stat_value(closest)
        return f"{snapped}%" if had_percent else snapped
    return value

def is_legal_substat_value(value: str, stat_name: str) -> bool:
    if not SUB_STATS or stat_name not in SUB_STATS:
        return False

    try:
        numeric = float(value.replace('%', '').strip())
    except (TypeError, ValueError):
        return False

    return any(abs(numeric - float(valid)) <= 0.05 for valid in SUB_STATS[stat_name])

def format_stat_value(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"

def _crop_region(image: np.ndarray, box: dict) -> np.ndarray:
    return image[box["y1"]:box["y2"], box["x1"]:box["x2"]]


# pytesseract spawns a process per call and each reloads the model: ~78ms on Railway for under 5ms of OCR
# Tesseract's list-file mode runs N images through one process, split by form feeds, identical to N separate calls
# One config per batch, with temp PNGs in /dev/shm when present so they never touch disk
_TESS_BATCH_DIR = os.environ.get("TESS_BATCH_DIR") or ("/dev/shm" if os.path.isdir("/dev/shm") else None)


def tess_batch(images: list[np.ndarray], config: str = "") -> list[str]:
    """OCR several images in one Tesseract process, returning one text per image in order"""
    if not images:
        return []
    work = tempfile.mkdtemp(prefix="tb_", dir=_TESS_BATCH_DIR)
    try:
        paths = []
        for i, im in enumerate(images):
            path = os.path.join(work, f"{i}.png")
            cv2.imwrite(path, im)
            paths.append(path)
        listing = os.path.join(work, "list.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            fh.write("\n".join(paths) + "\n")
        cmd = [pytesseract.pytesseract.tesseract_cmd, listing, "stdout", *shlex.split(config)]
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=120).stdout
        pages = out.split("\f")
        if len(pages) < len(images):
            raise RuntimeError(f"tesseract batch returned {len(pages)} pages for {len(images)} images")
        return [page.strip() for page in pages[:len(images)]]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _main_strip_lines(main_img: np.ndarray) -> list[str]:
    """Read the echo main strip on plain grayscale at 2x, skipping preprocess_region

    Fixed 140 threshold shreds the dim name row, so psm 3 reads nothing and resolve_echo_main falls to HP%
    Grayscale held 99.99% across a softness ladder where the threshold fell from 99.60% to 7.79%
    Sweep and rejected options: docs/echo-main-strip.md
    """
    upscaled = cv2.resize(
        cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY), None,
        fx=2, fy=2, interpolation=cv2.INTER_CUBIC,
    )
    return [
        l.strip()
        for l in pytesseract.image_to_string(upscaled, config="--psm 6").splitlines()
        if l.strip()
    ]


def _legal_main_values(cost: int) -> dict[str, str]:
    """Name to canonical Lv.25 value ('22.8%') for each variable main stat of an echo cost

    Variable mains are always percent stats, while the flat innate HP/ATK sits in the substat block
    """
    return {n: f"{format_stat_value(v[-1])}%" for n, v in MAIN_STATS.get(f"{cost}cost", {}).items()}


def _parse_main_line(line: str) -> tuple[str, str]:
    """Split an echo main OCR line 'Crit DMG 44%' into ('Crit DMG', '44%')"""
    parts = line.rsplit(' ', 1)
    if len(parts) == 2 and re.search(r'\d', parts[1]):
        return parts[0], parts[1]
    return line, ""


def _clean_main_name(raw_name: str, raw_value: str) -> str:
    name = clean_stat_name(raw_name, raw_value)
    return f"{name}%" if name in ("HP", "ATK", "DEF") else name


def _tiebreak_main_name(candidates: list[str], tess_name: str | None) -> str | None:
    """Break a main-stat tie by the Tesseract name read"""
    if not tess_name:
        return None
    probe = _clean_main_name(*_parse_main_line(tess_name))
    match = process.extractOne(probe, candidates, scorer=fuzz.WRatio, score_cutoff=60)
    return match[0] if match else None


def resolve_echo_main(cost: int, raw_name: str, raw_value: str) -> dict:
    """Resolve an echo's main stat against the legal mains for its cost

    A soft strip can fuzzy-match a name illegal for the cost (Crit DMG on a 1-cost), which would reject the card
    A legal name is trusted and its value snapped to the Lv.25 canonical
    An illegal name is a misread, so the value decides and the name read breaks ties (the 3-cost 30.0% cluster)
    """
    legal = _legal_main_values(cost)
    if not legal:
        # Unidentified echo (cost 0) has no legal set, so validate against every main stat name
        validated = validate_stat(clean_stat_name(raw_name, raw_value), MAIN_STAT_NAMES)
        if validated in ("HP", "ATK", "DEF"):
            validated = f"{validated}%"
        return {"name": validated, "value": raw_value}

    name = _clean_main_name(raw_name, raw_value)
    match = process.extractOne(name, list(legal), scorer=fuzz.WRatio, score_cutoff=82)
    if match:
        chosen = match[0]
        if legal[chosen] != raw_value:
            print(f"Main stat snapped: {raw_name!r} {raw_value!r} -> {chosen} {legal[chosen]} (cost {cost})")
        return {"name": chosen, "value": legal[chosen]}

    # Illegal-for-cost name is a misread, so recover from the value
    target = None
    if m := re.search(r'\d+(?:\.\d+)?', raw_value or ""):
        target = float(m.group())

    if target is not None:
        dist = lambda n: abs(float(legal[n].rstrip('%')) - target)
        ranked = sorted(legal, key=dist)
        near = [n for n in ranked if dist(n) <= 1.0]
        if len(near) == 1:
            chosen = near[0]                                          # value alone resolves it
        elif near:                                                   # tie among nearby mains
            chosen = _tiebreak_main_name(near, raw_name) or near[0]
        else:                                                        # value matches nothing: trust the name
            chosen = _tiebreak_main_name(ranked, raw_name) or ranked[0]
    else:                                                            # unreadable value: name read, else primary main
        chosen = _tiebreak_main_name(list(legal), raw_name) or next(iter(legal))

    print(f"Main stat recovered: {raw_name!r} {raw_value!r} -> {chosen} {legal[chosen]} (cost {cost}, illegal-for-cost name)")
    return {"name": chosen, "value": legal[chosen]}


# Each substat row is read on its own fixed band with psm 7, so page segmentation can never drop a row
# Grid is pixel-exact (first row at 15.5 px, 34 px pitch) and wrapped names spill into the gap without shifting rows
# Names use the band's first line, since the closed vocabulary resolves a wrapped name from its first half
# Measurements and rejected options: docs/echo-substats.md
SUBSTAT_ROW_FIRST = 15.5
SUBSTAT_ROW_PITCH = 34
SUBSTAT_ROW_HALF = 14
# A wrapped stat's spill reaches the top 4 px of the next row's band, where psm 7 reads it as '[1]' and resolves HP
# Only the row after a wrapping stat gets these tighter tops, since on every row they clip the 'o' of "Resonance"
SUBSTAT_NAME_TOPS_AFTER_WRAP = (12, 10, 9, 8, 7)
WRAPPING_SUBSTATS = ("Resonance Liberation DMG Bonus", "Resonance Skill DMG Bonus")
SUBSTAT_ROWS = 5
SUBSTAT_NAME_CONFIG = "--psm 7"
SUBSTAT_VALUE_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789.%"
SUBSTAT_VALUE_RETRY_UPSCALE = 3


def _substat_band(crop: np.ndarray, row: int, top: int = SUBSTAT_ROW_HALF) -> np.ndarray:
    centre = SUBSTAT_ROW_FIRST + SUBSTAT_ROW_PITCH * row
    return crop[max(0, int(centre - top)):int(centre + SUBSTAT_ROW_HALF), :]


def _upscale2(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _upscale_retry(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image, None, fx=SUBSTAT_VALUE_RETRY_UPSCALE, fy=SUBSTAT_VALUE_RETRY_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )


def _resolve_substat(raw_name: str, raw_value: str) -> dict | None:
    """One (name, value) row, or None when it is not a legal roll for that stat

    Flat vs percent HP/ATK/DEF is decided by magnitude (flat ATK 30-60, ATK% 6.4-11.6), so a dropped % is harmless
    """
    if not raw_name or not raw_value:
        return None
    try:
        # A stray trailing '.' or '%' on the read ('10.17.') must not void the row
        numeric = float(raw_value.replace('%', '').strip().rstrip('.'))
    except ValueError:
        return None
    name = validate_substat_name(raw_name, raw_value)
    base = name.rstrip('%')
    if base in FLAT_SUBSTATS:
        name = base if numeric >= 20 else base + "%"
    value = validate_value(format_stat_value(numeric) + ("" if name in FLAT_SUBSTATS else "%"), name)
    if not is_legal_substat_value(value, name):
        return None
    return {"name": name, "value": value}


def _read_value_bands(names: list[str], value_bands: list[np.ndarray], render) -> list[str]:
    """Values at 2x, then one 3x retry for any row whose 2x read is not a legal roll

    Each scale misreads bands the other reads right (2x gives '3390' for 390)
    A legal 2x read is never replaced, and a 3x read is taken only when legal
    `render(band, upscale)` is the image Tesseract sees, shared by both passes
    """
    values = tess_batch([render(b, _upscale2) for b in value_bands], SUBSTAT_VALUE_CONFIG)
    bad = [i for i, (n, v) in enumerate(zip(names, values)) if n and _resolve_substat(n, v) is None]
    if bad:
        retry = tess_batch([render(value_bands[i], _upscale_retry) for i in bad], SUBSTAT_VALUE_CONFIG)
        for i, v in zip(bad, retry):
            if _resolve_substat(names[i], v) is not None:
                values[i] = v
    return values


def _substat_bands(names: list[str], values: list[str]) -> list[dict | None]:
    """One entry per band, None where the band isn't a legal row, so two passes can merge band by band"""
    return [_resolve_substat(n, v) for n, v in zip(names, values)]


def _reread_after_wrap(names: list[str], values: list[str], names_img: np.ndarray, render) -> list[str]:
    """Re-read the row after each wrapping stat with tighter tops and keep the more confident read

    Full band can catch the spilled 'DMG Bonus' ('[1]') while a tight band can clip a high row ('[3' for DEF)
    Fragments score ~0 against any stat and real names ~100, so higher confidence wins and a tie keeps the primary
    """
    after = [k for k in range(1, SUBSTAT_ROWS)
             if names[k - 1] and validate_substat_name(names[k - 1], values[k - 1] or "1%") in WRAPPING_SUBSTATS]
    if not after:
        return names
    tops = SUBSTAT_NAME_TOPS_AFTER_WRAP
    tight = tess_batch(
        [render(_substat_band(names_img, k, t)) for k in after for t in tops],
        SUBSTAT_NAME_CONFIG,
    )
    names = list(names)
    for i, k in enumerate(after):
        for n in tight[i * len(tops):(i + 1) * len(tops)]:
            if _substat_name_confidence(n, values[k]) > _substat_name_confidence(names[k], values[k]):
                names[k] = n
    return names


def read_substat_rows(names_img: np.ndarray, values_img: np.ndarray) -> tuple[list[dict], list[str], list[str]]:
    """Substats by fixed row band, returning (rows, raw_names, raw_values)

    Thresholded pass ('B') runs first, and plain grayscale ('G') only when it resolves fewer than five rows
    Grayscale recovers dim cards the threshold shreds but loses ~1.2% on bright ones, so it is only a fallback
    """
    name_bands = [_substat_band(names_img, k) for k in range(SUBSTAT_ROWS)]
    value_bands = [_substat_band(values_img, k) for k in range(SUBSTAT_ROWS)]

    names = tess_batch([preprocess_region(b) for b in name_bands], SUBSTAT_NAME_CONFIG)
    values = _read_value_bands(names, value_bands, lambda b, up: preprocess_region(up(b)))
    names = _reread_after_wrap(names, values, names_img, preprocess_region)
    bands = _substat_bands(names, values)
    if sum(b is not None for b in bands) >= SUBSTAT_ROWS:
        return [b for b in bands if b], names, values

    gray = lambda b: cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    g_names = tess_batch([_upscale2(gray(b)) for b in name_bands], SUBSTAT_NAME_CONFIG)
    g_values = _read_value_bands(g_names, value_bands, lambda b, up: up(gray(b)))
    g_names = _reread_after_wrap(g_names, g_values, names_img, lambda b: _upscale2(gray(b)))
    g_bands = _substat_bands(g_names, g_values)

    # Pass with more resolved rows wins (thresholded on a tie) and the other fills only its empty bands
    # Letting the loser replace resolved rows swapped in legal but wrong values
    if sum(g is not None for g in g_bands) > sum(b is not None for b in bands):
        lost = "B"
        win_bands, win_names, win_values = g_bands, g_names, g_values
        fill_bands, fill_names, fill_values = bands, names, values
    else:
        lost = "G"
        win_bands, win_names, win_values = bands, names, values
        fill_bands, fill_names, fill_values = g_bands, g_names, g_values

    merged_names, merged_values, filled = list(win_names), list(win_values), 0
    for k, (w, l) in enumerate(zip(win_bands, fill_bands)):
        if w is None and l is not None:
            win_bands[k] = l
            merged_names[k], merged_values[k] = fill_names[k], fill_values[k]
            filled += 1
    rows = [b for b in win_bands if b]
    if filled:
        print(f"Substats: {lost} filled {filled} empty row(s) -> {len(rows)} rows")
    return rows, merged_names, merged_values


def validate_character_name(raw_name: str) -> str:
    if not CHARACTER_NAMES:
        return raw_name
    match = process.extractOne(raw_name, CHARACTER_NAMES)
    return match[0] if match else raw_name

def parse_character_title(text: str) -> dict:
    level = 1
    if match := re.search(r'\bLV\.?\s*(\d+)\b', text, re.IGNORECASE):
        level = int(match.group(1))

    name_text = re.sub(r'\bLV\.?\s*\d+\b', ' ', text, flags=re.IGNORECASE)
    name_text = re.sub(r'\s+', ' ', name_text).strip()
    compact_name = re.sub(r'[^a-z]', '', name_text.lower())
    rover_element = next((
        element
        for element, aliases in ROVER_ELEMENT_ALIASES.items()
        if any(alias in compact_name for alias in aliases)
    ), None)

    if "rover" in compact_name and rover_element:
        char_name = f"Rover: {rover_element}"
        return {
            "name": char_name,
            "id": CHARACTER_ID_MAP.get(char_name, ""),
            "level": level,
            "element": rover_element,
        }

    char_name = validate_character_name(name_text)
    return {"name": char_name, "id": CHARACTER_ID_MAP.get(char_name, ""), "level": level}

def parse_watermark_username(lines: list[str], uid: int) -> str:
    """Strip the 'Player ID:' label off the name line, and any UID that bled into it"""
    if not lines:
        return ""
    first_line = lines[0].strip()
    if ':' in first_line:
        username = first_line.split(':', 1)[-1].strip()
    else:
        username = re.sub(r'^(?:Player\s*ID|Name|Neme)[.:;]?\s*', '', first_line, flags=re.IGNORECASE).strip()
    if uid > 0 and str(uid) in username:
        username = username.replace(str(uid), "").strip().rstrip(':').strip()
    return username


# Watermark's two lines are read separately so the UID line can take a digits-only whitelist and a 2x upscale
# Upscale fixes Tesseract's 9-to-3 misread (UID errors 1.7% to 0.17% over 600 cards)
# Scoped to this strip since pale text on dark reads well upscaled, but the same upscale wrecks forte's bright artwork
WATERMARK_UID_LINE_TOP = 0.46
WATERMARK_UPSCALE = 2
WATERMARK_UID_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"


# A wrong 9-digit UID is permanent because uid is sticky on dedup conflict, so a second read corroborates it
# Primary read is the UID line with digits only, second is the whole strip free-form at 2x
# Reads that aren't 9 digits are dropped, and two valid reads that disagree write no UID (a refusal costs one re-upload)
WATERMARK_UID_GUARD_UPSCALE = 2
UID_DIGITS = 9                       # lb enforces ^\d{9}$


def _uid_in_text(text: str) -> int:
    for line in text.split("\n"):
        if match := re.search(r'\d{6,12}', line):
            return int(match.group(0))
    return 0


def read_watermark(image: np.ndarray) -> dict:
    """Username from the free-form strip, UID from an upscaled digits-only line read"""
    height = image.shape[0]
    uid_line = image[int(height * WATERMARK_UID_LINE_TOP):, :]
    upscaled = cv2.resize(
        uid_line, None, fx=WATERMARK_UPSCALE, fy=WATERMARK_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    uid = 0
    if match := re.search(
        r'\d{6,12}',
        pytesseract.image_to_string(preprocess_region(upscaled), config=WATERMARK_UID_CONFIG),
    ):
        uid = int(match.group(0))

    # 2x corroborating read and 1x username read share a config, so they share one Tesseract process
    guard_text, name_text = tess_batch([
        preprocess_region(cv2.resize(
            image, None, fx=WATERMARK_UID_GUARD_UPSCALE, fy=WATERMARK_UID_GUARD_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )),
        preprocess_region(image),
    ])
    corroborating = _uid_in_text(guard_text)
    candidates = {c for c in (uid, corroborating) if len(str(c)) == UID_DIGITS}
    if len(candidates) == 1:
        resolved = candidates.pop()
        if resolved != uid:
            print(f"UID guard: primary {uid} malformed, using corroborating read {resolved}")
        uid = resolved
    else:
        if candidates:
            print(f"UID guard: reads disagree ({uid} vs {corroborating}); writing no UID")
        uid = 0

    name_lines = [line.strip() for line in name_text.splitlines() if line.strip()]
    return {"username": parse_watermark_username(name_lines, uid), "uid": uid}


def get_element_region(image):
    """Set badge crop of an echo panel"""
    h, w = image.shape[:2]
    return image[int(h*0.027):int(h*0.148), int(w*0.654):int(w*0.797)]


def get_echo_cost(image: np.ndarray) -> int:
    """Echo cost by template-matching the cost badge, 0 when unsure"""
    cost_img = image[9:61, 302:345]

    if not COST_TEMPLATES:
        return 0

    gray = cv2.cvtColor(cost_img, cv2.COLOR_BGR2GRAY)
    scores = []
    for cost, template in COST_TEMPLATES.items():
        tmpl = template
        if tmpl.shape != gray.shape:
            tmpl = cv2.resize(tmpl, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        score = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED).max()
        scores.append((cost, score))

    best_cost, best_score = max(scores, key=lambda item: item[1])
    return best_cost if best_score >= 0.2 else 0

# SIFT confidence needed before the badge may promote a base echo to its Nightmare variant
# Wrong base-to-Nightmare flips sat at conf <= 0.28 and clean echoes at 0.3+, while demotions need no floor
NIGHTMARE_PROMOTE_FLOOR = 0.30

def echo_family_key(template_id: str) -> str:
    """Group an echo with its Nightmare variant by the shared body name

    Nightmare recolors share the base silhouette, so SIFT can't separate them and the set badge does
    Phantom echoes share the base echo's id, so they need no handling here
    """
    name = ECHO_NAME_MAP.get(template_id, template_id)
    return re.sub(r'^Nightmare:\s*', '', name).strip().lower()

_ECHO_FAMILY_INDEX: dict[str, list[str]] | None = None

def _echo_family_index() -> dict[str, list[str]]:
    global _ECHO_FAMILY_INDEX
    if _ECHO_FAMILY_INDEX is None:
        index: dict[str, list[str]] = {}
        for template_id in ECHO_NAME_MAP:
            if template_id not in TEMPLATE_FEATURES:
                continue
            index.setdefault(echo_family_key(template_id), []).append(template_id)
        _ECHO_FAMILY_INDEX = index
    return _ECHO_FAMILY_INDEX


def validate_echo_family_by_element(
    best_match: str,
    best_conf: float,
    sorted_matches: list[tuple[str, float]],
    element_region: np.ndarray,
    detected_element: int | None,
) -> tuple[str, float, int | None]:
    """Resolve same-body variant confusion using the visible set badge"""
    variants = _echo_family_index().get(echo_family_key(best_match), [])
    if len(variants) < 2:
        return best_match, best_conf, detected_element

    family_set_ids = sorted({
        sid for variant in variants for sid in ECHO_SET_IDS.get(variant, [])
    })
    if len(family_set_ids) < 2:
        return best_match, best_conf, detected_element

    # Combined-family badge only selects the variant, and the caller recomputes the set from that variant's own sets
    badge = determine_element(element_region, family_set_ids)
    candidates = [
        variant
        for variant in variants
        if badge in ECHO_SET_IDS.get(variant, [])
    ]
    if badge is None or not candidates:
        # No badge keeps SIFT's pick, and a badge matching no variant can't happen since it comes from family_set_ids
        return best_match, best_conf, detected_element

    conf_of = dict(sorted_matches)
    chosen = max(candidates, key=lambda variant: conf_of.get(variant, 0.0))
    if chosen == best_match:
        return best_match, best_conf, detected_element

    # Promoting to Nightmare is the risky direction since a low-confidence wrong-body match can force a bogus set
    # Demoting back to base needs no floor, since base is the common case
    promoting_to_nightmare = (
        "Nightmare" in ECHO_NAME_MAP.get(chosen, "")
        and "Nightmare" not in ECHO_NAME_MAP.get(best_match, "")
    )
    if promoting_to_nightmare and best_conf < NIGHTMARE_PROMOTE_FLOOR:
        return best_match, best_conf, detected_element

    print(f"Family badge validation: {best_match} -> {chosen} (badge {SET_NAME_BY_ID.get(badge, badge)})")
    # Reset element so the caller recomputes it from the new identity's own sets
    return chosen, conf_of.get(chosen, best_conf), None


# Same-silhouette recolors (the six Kernel Puppets) near-tie under grayscale SIFT, but body hue separates them
# The S>=80/V>=60 mask drops the shared silver/gold trim and dark background
# Recolors score 0.8+ on their own template and <0.35 on siblings, so these floors leave different-body ties to SIFT
HUE_ARBITRATION_MIN_SCORE = 0.5
HUE_ARBITRATION_MIN_MARGIN = 0.2

def _icon_hue_hist(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    cv2.normalize(hist, hist)
    return hist

def arbitrate_by_icon_hue(icon_img: np.ndarray, candidates: list[tuple[str, float]]):
    """Pick among near-tied SIFT candidates by icon hue-histogram similarity

    Returns (echo_name, sift_conf, hue_score) when hue is decisive, else None
    """
    query = _icon_hue_hist(icon_img)
    scored = []
    for name, conf in candidates:
        tmpl = ICON_TEMPLATES.get(name)
        if tmpl is None:
            continue
        score = cv2.compareHist(query, _icon_hue_hist(tmpl), cv2.HISTCMP_CORREL)
        scored.append((score, name, conf))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best, second = scored[0], scored[1]
    if best[0] >= HUE_ARBITRATION_MIN_SCORE and (best[0] - second[0]) >= HUE_ARBITRATION_MIN_MARGIN:
        return best[1], best[2], best[0]
    return None


def _identify_icon_core(image: np.ndarray):
    """SIFT match plus close-match disambiguation, before family validation

    Returns (echo_name, confidence, element, sorted_matches, element_region)
    """
    icon_img = image[0:182, 0:188]
    sift = cv2.SIFT.create()
    kp1, des1 = sift.detectAndCompute(icon_img, None)
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    detected_element = None  # set only when close matches read the badge here

    # Cost badge narrows the sweep to ~44 same-cost templates, ~3.6x fewer FLANN matches
    # An unsure cost (0) sweeps everything, so a missed cost never drops the true echo
    actual_cost = get_echo_cost(image)
    if actual_cost in (1, 3, 4):
        candidate_names = [n for n in TEMPLATE_FEATURES if ECHO_COSTS.get(n, 0) == actual_cost]
        if not candidate_names:
            candidate_names = list(TEMPLATE_FEATURES)
    else:
        candidate_names = list(TEMPLATE_FEATURES)

    matches = []
    for name in candidate_names:
        kp2, des2 = TEMPLATE_FEATURES[name]
        matches_list = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
        confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        matches.append((name, confidence))

    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    best_match, best_conf = sorted_matches[0]
    element_region = get_element_region(image)

    # On a near tie the badge separates look-alike bodies (Chirpuff vs Gulpuff), while Nightmare variants settle later
    if len(sorted_matches) > 1 and (best_conf - sorted_matches[1][1]) < 0.1:
        # Soft images deflate every confidence, so on a collapsed scale the floor drops toward 30% of best
        # Noise-level candidates stay out, so a misread badge can't hand them the win
        close_floor = min(0.1, max(best_conf - 0.1, best_conf * 0.3))
        close_matches = [(name, conf) for name, conf in sorted_matches if conf > close_floor]
        if len(close_matches) >= 2:
            print(f"Close matches detected: {[(n, f'{c:.4f}') for n, c in close_matches]}")

            candidate_set_ids = set()
            for name, _ in close_matches:
                candidate_set_ids.update(ECHO_SET_IDS.get(name, []))
            detected_element = determine_element(element_region, list(candidate_set_ids))

            element_matches = [
                (name, conf)
                for name, conf in close_matches
                if detected_element in ECHO_SET_IDS.get(name, [])
            ]
            # A badge matching exactly one candidate decides alone
            # Otherwise icon hue arbitrates the badge survivors, or every close match when the badge decided nothing
            if detected_element is not None and len(element_matches) == 1:
                best_match, best_conf = element_matches[0]
                print(f"-> badge {SET_NAME_BY_ID.get(detected_element, detected_element)} -> '{best_match}'")
            else:
                pool = element_matches if len(element_matches) >= 2 else close_matches
                arbitrated = arbitrate_by_icon_hue(icon_img, pool)
                if arbitrated is not None:
                    best_match, best_conf, hue_score = arbitrated
                    print(f"-> icon hue {hue_score:.3f} -> '{best_match}'")

    return best_match, best_conf, detected_element, sorted_matches, element_region

def match_icon(image: np.ndarray) -> Tuple[str, float, int | None]:
    """Best echo match with its SIFT confidence and set badge id

    Set badge arbitrates same-body variants such as base vs Nightmare
    """
    best_match, best_conf, detected_element, sorted_matches, element_region = _identify_icon_core(image)
    best_match, best_conf, detected_element = validate_echo_family_by_element(
        best_match,
        best_conf,
        sorted_matches,
        element_region,
        detected_element,
    )
    if detected_element is None:
        detected_element = determine_element(element_region, best_match)
    return (best_match, best_conf, detected_element)

def parse_sequence_region(image) -> int:
    """Count active sequence nodes using HSV gray detection"""
    GRAY_HSV = {
        'lower': np.array([0, 0, 160]),
        'upper': np.array([40, 180, 255])
    }
    active_count = 0
    
    for region in SEQUENCE_REGIONS.values():
        center_x, center_y = region["center"]
        half_w = region["width"] // 2
        half_h = region["height"] // 2
        
        x1 = max(0, center_x - half_w)
        x2 = min(image.shape[1], center_x + half_w)
        y1 = max(0, center_y - half_h)
        y2 = min(image.shape[0], center_y + half_h)
        
        sequence_img = image[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(sequence_img, cv2.COLOR_BGR2HSV)
        gray_mask = cv2.inRange(hsv, GRAY_HSV['lower'], GRAY_HSV['upper'])
        gray_ratio = np.count_nonzero(gray_mask) / gray_mask.size
        
        if gray_ratio > SEQUENCE_ACTIVE_GRAY_RATIO:
            active_count += 1
    
    return active_count

def _canonical_stat_fragment(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.lower())

# Character and weapon sub-boxes are fractions of server.py's IMPORT_REGIONS crops and must change with them
# SIFT abstains below its confidence and margin floors (Rover, look-alike weapons, junk) and falls back to OCR
DATA_DIR = Path(__file__).resolve().parent / "Data"

# Splash has no fixed frame, so this box is tuned, not measured (SIFT accept 97.8% to 99.2%, no identity changes)
# x fractions are scaled by 0.32/0.38 so the pixels stayed put when the server region widened
CHAR_SPLASH_SUBBOX = (0.0842, 0.16, 0.8421, 0.9455)        # splash within character region
CHAR_NAME_SUBBOX = (0.0863, 0.0135, 0.7949, 0.1515)    # name strip within character region
CHAR_ELEMENT_SUBBOX = (0.0152, 0.025, 0.0884, 0.14)      # element badge left of name strip
CHAR_SIFT_MAX_SIDE = 150
CHAR_CONF_FLOOR = 0.10
CHAR_MARGIN_FLOOR = 0.04

# Measured by per-pixel variance over 400 stacked panels, since the frame never changes and only the icon varies
# Square box matches the square weapon art (SIFT accept 77.9% to 88.3%, no identity changes)
WEAP_ICON_SUBBOX = (0.0456, 0.1806, 0.3098, 0.7176)   # square icon within weapon panel
WEAP_SIFT_MAX_SIDE = 120
WEAP_CONF_FLOOR = 0.08
WEAP_MARGIN_FLOOR = 0.03

CHARACTER_ID_NAME = {cid: name for name, cid in CHARACTER_ID_MAP.items()}
WEAPON_ID_NAME = {wid: name for name, wid in WEAPON_ID_MAP.items()}

_CHARACTER_FEATURES = None
_WEAPON_FEATURES = None


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _subcrop(img: np.ndarray, box: tuple) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)])


def _load_asset_features(folder: str, max_side: int) -> dict:
    sift = cv2.SIFT.create()
    feats = {}
    for path in sorted((DATA_DIR / folder).glob("*.webp")):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        kp, des = sift.detectAndCompute(_resize_max_side(img, max_side), None)
        if des is not None:
            feats[path.stem] = (kp, des)
    return feats


def _match_asset(region: np.ndarray, feats: dict) -> tuple:
    """Top template by SIFT good-match ratio, returning (id, confidence, margin)"""
    sift = cv2.SIFT.create()
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    kp1, des1 = sift.detectAndCompute(region, None)
    if des1 is None or len(kp1) < 2:
        return None, 0.0, 0.0
    scores = []
    for name, (kp2, des2) in feats.items():
        ml = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in (pr for pr in ml if len(pr) == 2) if m.distance < 0.7 * n.distance]
        conf = len(good) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        scores.append((name, conf))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_id, best_conf = scores[0]
    margin = best_conf - scores[1][1] if len(scores) > 1 else best_conf
    return best_id, best_conf, margin


def _detect_rover_badge_element(region_img: np.ndarray) -> str | None:
    badge = _subcrop(region_img, CHAR_ELEMENT_SUBBOX)
    hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    hues = hsv[:, :, 0][mask > 0]
    if hues.size < 10:
        return None
    median_hue = float(np.median(hues))

    def hue_distance(anchor: int) -> float:
        delta = abs(median_hue - anchor)
        return min(delta, 180 - delta)

    anchors = {
        element: anchor
        for element, anchor in ROVER_BADGE_HUE_ANCHORS.items()
        if element in ROVER_KNOWN_ELEMENTS
    }
    if not anchors:
        return None
    element, distance = min(
        ((element, hue_distance(anchor)) for element, anchor in anchors.items()),
        key=lambda item: item[1],
    )
    return element if distance <= 18 else None


def _rover_analysis(cid: str | None, element: str | None, level: int = 90) -> dict | None:
    if not cid:
        return None
    gender = ROVER_GENDER_BY_ID.get(cid)
    if not gender or not element:
        return None
    resolved_id = ROVER_IDS_BY_GENDER_ELEMENT.get((gender, element))
    if not resolved_id:
        return None
    name = f"Rover: {element}"
    return {"name": name, "id": resolved_id, "level": level, "element": element}


# Level is read from the gold LV. pill alone, since psm 7 commits to one polarity per line and misses dark-on-gold text
# Pill floats with name length, so it is located by hue, cropped and inverted to light-on-dark
# Its right 22% is decorative stripes that read as a trailing 7 or 1, so it is cropped off
# 3x reads before 2x, which returned empty on 8/600 pills, and both renders share one batched spawn
CHAR_LEVEL_PILL_HSV = (np.array([12, 90, 90]), np.array([38, 255, 255]))
CHAR_LEVEL_PILL_STRIPE_FRACTION = 0.22
CHAR_LEVEL_CONFIG = "--psm 7 -c tessedit_char_whitelist=LV.0123456789"


def _level_from_text(text: str) -> int:
    match = re.search(r'(?i)lv\.?\s*(\d{1,2})', text) or re.search(r'\d{1,2}', text)
    if not match:
        return 0
    level = int(match.group(1) if match.lastindex else match.group(0))
    return level if 1 <= level <= 90 else 0


def read_character_level(region_img: np.ndarray) -> int:
    """Level from the LV. pill, 0 when the pill is absent or unreadable"""
    height, width = region_img.shape[:2]
    _, top, _, bottom = CHAR_NAME_SUBBOX
    strip = region_img[int(height * top):int(height * bottom), :]
    if strip.size == 0:
        return 0
    mask = cv2.inRange(cv2.cvtColor(strip, cv2.COLOR_BGR2HSV), *CHAR_LEVEL_PILL_HSV)
    mask[:, :int(strip.shape[1] * 0.25)] = 0            # element icon lives at far left
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w < 20 or h < 8:
        return 0
    pad = 3
    pill = cv2.cvtColor(strip[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad], cv2.COLOR_BGR2GRAY)
    pill = pill[:, :int(pill.shape[1] * (1 - CHAR_LEVEL_PILL_STRIPE_FRACTION))]

    def render(factor: int) -> np.ndarray:
        up = cv2.resize(pill, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(cv2.bitwise_not(up), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    for text in tess_batch([render(3), render(2)], CHAR_LEVEL_CONFIG):
        if level := _level_from_text(text):
            return level
    return 0


def _character_title_text(strip: np.ndarray) -> str:
    """Letters-only Tesseract read of the name strip, separate so tests can inject a title"""
    return pytesseract.image_to_string(
        preprocess_region(strip),
        config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
    )


def _read_character_title(region_img: np.ndarray) -> dict:
    """Name from a letters-only Tesseract read of the strip, level from the pill"""
    text = _character_title_text(_subcrop(region_img, CHAR_NAME_SUBBOX))
    parsed = parse_character_title("\n".join(line.strip() for line in text.splitlines() if line.strip()))
    if level := read_character_level(region_img):
        parsed["level"] = level
    return parsed


def recognize_character_asset(region_img: np.ndarray) -> dict:
    """SIFT the character splash, or OCR the name strip on abstain (Rover, junk)

    Level comes from the pill, and a non-Rover SIFT accept falls back to 90 when it is unreadable
    """
    global _CHARACTER_FEATURES
    if _CHARACTER_FEATURES is None:
        _CHARACTER_FEATURES = _load_asset_features("Characters", CHAR_SIFT_MAX_SIDE)
    splash = _resize_max_side(_subcrop(region_img, CHAR_SPLASH_SUBBOX), CHAR_SIFT_MAX_SIDE)
    cid, conf, margin = _match_asset(splash, _CHARACTER_FEATURES)
    parsed = None
    if cid in ROVER_GENDER_BY_ID and conf >= CHAR_CONF_FLOOR:
        # Newer Rover titles carry the element while older Spectro/Aero cards say only "Rover"
        # An explicit title wins and the badge covers an unsuffixed or unreadable title
        parsed = _read_character_title(region_img)
        title_element = parsed.get("element")
        badge_element = _detect_rover_badge_element(region_img)
        if title_element and badge_element and title_element != badge_element:
            print(
                "Rover title/badge disagreement: "
                f"title={title_element} badge={badge_element}; using title"
            )
        rover = _rover_analysis(
            cid,
            title_element or badge_element,
            parsed.get("level", 90),
        )
        if rover is not None:
            return rover
    if cid and cid not in ROVER_GENDER_BY_ID and conf >= CHAR_CONF_FLOOR and margin >= CHAR_MARGIN_FLOOR:
        return {
            "name": CHARACTER_ID_NAME.get(cid, ""),
            "id": cid,
            "level": read_character_level(region_img) or 90,
        }
    if parsed is None:
        parsed = _read_character_title(region_img)
    if "rover" in re.sub(r'[^a-z]', '', parsed.get("name", "").lower()):
        rover = _rover_analysis(cid, parsed.get("element") or _detect_rover_badge_element(region_img), parsed.get("level", 90))
        if rover is not None:
            return rover
    return parsed


# Weapon level is read from its own box on both paths, since SIFT can't see level and ~6% of weapons aren't level 90
# 2x, psm 7 and a digit whitelist matched the RapidOCR reference on 800/800 cards
WEAPON_LEVEL_BOX = {"x1": 191, "y1": 79, "x2": 269, "y2": 133}
WEAPON_LEVEL_UPSCALE = 2
WEAPON_LEVEL_CONFIG = "--psm 7 -c tessedit_char_whitelist=LV.0123456789"


def read_weapon_level(region_img: np.ndarray) -> int:
    """Weapon level from its own box, 0 when unreadable so callers can fall back"""
    box = WEAPON_LEVEL_BOX
    raw = region_img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
    if raw.size == 0:
        return 0
    upscaled = cv2.resize(
        raw, None, fx=WEAPON_LEVEL_UPSCALE, fy=WEAPON_LEVEL_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    text = pytesseract.image_to_string(
        preprocess_region(upscaled), config=WEAPON_LEVEL_CONFIG
    )
    match = re.search(r'(?i)lv\.?\s*(\d+)', text)
    return int(match.group(1)) if match else 0


# Name strip for the ~12% of panels where icon SIFT abstains, fuzzy-matched against WEAPON_NAMES
# Read twice in one spawn: thresholded first, then plain grayscale for dark uploads whose gold text never reaches 140
# A blank-art panel resolves to no name, which the frontend's signature-weapon fallback keys on
WEAPON_NAME_BOX = {"x1": 152, "y1": 25, "x2": 437, "y2": 79}
WEAPON_NAME_UPSCALE = 2
WEAPON_NAME_CONFIG = "--psm 7"


def read_weapon_name(region_img: np.ndarray) -> str | None:
    """Weapon name from the strip resolved against the known list, None when unreadable"""
    box = WEAPON_NAME_BOX
    raw = region_img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
    if raw.size == 0 or not WEAPON_NAMES:
        return None
    upscaled = cv2.resize(
        raw, None, fx=WEAPON_NAME_UPSCALE, fy=WEAPON_NAME_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    renders = [preprocess_region(upscaled), cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)]
    for text in tess_batch(renders, WEAPON_NAME_CONFIG):
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first:
            continue
        match = process.extractOne(first, WEAPON_NAMES, scorer=fuzz.ratio, score_cutoff=WEAPON_NAME_MIN_SCORE)
        if match:
            return match[0]
    return None


def recognize_weapon_asset(region_img: np.ndarray) -> dict:
    """SIFT the weapon icon, or OCR the name strip on abstain (look-alike icons)

    A blank panel stays empty (name and id "") so the frontend's signature-weapon fallback applies
    Level comes from its own box on both paths, falling back to 90, or 1 on a blank panel
    """
    global _WEAPON_FEATURES
    if _WEAPON_FEATURES is None:
        _WEAPON_FEATURES = _load_asset_features("Weapons", WEAP_SIFT_MAX_SIDE)
    icon = _resize_max_side(_subcrop(region_img, WEAP_ICON_SUBBOX), WEAP_SIFT_MAX_SIDE)
    wid, conf, margin = _match_asset(icon, _WEAPON_FEATURES)
    if wid and conf >= WEAP_CONF_FLOOR and margin >= WEAP_MARGIN_FLOOR:
        return {
            "name": WEAPON_ID_NAME.get(wid, ""),
            "id": wid,
            "level": read_weapon_level(region_img) or 90,
        }
    name = read_weapon_name(region_img)
    # The level pill renders even on a blank-art panel, so level is read on this path too
    level = read_weapon_level(region_img)
    if not name:
        return {"name": "", "id": "", "level": level or 1}
    return {"name": name, "id": WEAPON_ID_MAP.get(name, ""), "level": level or 90}


# Localized cards OCR their substat values fine but their names match the English vocabulary poorly
# server.py flags a card non-English when enough values are present but few names match
_SUBSTAT_VOCAB = list(SUB_STATS.keys())
_NUMERIC_VALUE_RE = re.compile(r"^\d{1,4}(\.\d)?%?$")

def echo_language_signal(cleaned_names: list[str], values: list[str]) -> dict:
    """Per-echo English-confidence: matched/total substat names + count of real values"""
    names = [n for n in cleaned_names if len(n) >= 2]
    name_good = sum(
        1 for n in names
        if (process.extractOne(n, _SUBSTAT_VOCAB, scorer=fuzz.WRatio) or (None, 0))[1] >= 80
    )
    num_values = sum(1 for v in values if _NUMERIC_VALUE_RE.match(v.strip()))
    return {"nameGood": name_good, "nameTotal": len(names), "numValues": num_values}

def process_card(image, region: str):
    """Recognize one region and return its result plus the lines it logged

    Stdout during the call is captured per thread and returned under "logs" for server.py to print in region order
    """
    if image is None:
        return {"success": False, "error": "No image data provided", "logs": []}

    _STDOUT.push()
    try:
        result = _process_card_inner(image, region)
    except Exception as e:
        result = {"success": False, "error": str(e)}
    finally:
        captured = _STDOUT.pop()

    result["logs"] = [s for s in (line.rstrip() for line in captured.splitlines()) if s]
    return result


def _process_card_inner(image, region: str):
    if region == "sequences":
        sequence = parse_sequence_region(image)
        return {
            "success": True,
            "analysis": {"sequence": sequence}
        }
    elif region == "forte":
        forte_data = {"levels": [0] * 5}
        processed = preprocess_region(image)

        # One Tesseract process for all five nodes
        node_crops = [processed[c["y1"]:c["y2"], c["x1"]:c["x2"]] for c in FORTE_REGIONS.values()]
        for i, text in enumerate(tess_batch(node_crops)):
            match = re.search(r'(?i)lv\.(\d+)(?:/10)?', text)
            if match:
                forte_data["levels"][i] = int(match.group(1))

        return {
            "success": True,
            "analysis": forte_data
        }
    elif region == "character":
        return {"success": True, "analysis": recognize_character_asset(image)}
    elif region == "weapon":
        return {"success": True, "analysis": recognize_weapon_asset(image)}
    elif region == "watermark":
        return {"success": True, "analysis": read_watermark(image)}
    elif region.startswith("echo"):
        # Main stat is read raw here and resolved once the echo's cost is known
        main_img = _crop_region(image, ECHO_REGIONS["main"])
        main_lines = _main_strip_lines(main_img)
        main_line = " ".join(main_lines[:2]) if len(main_lines) >= 2 else (main_lines[0] if main_lines else "")
        raw_main_name, raw_main_value = _parse_main_line(main_line)

        names_img = _crop_region(image, ECHO_REGIONS["subs_names"])
        values_img = _crop_region(image, ECHO_REGIONS["subs_values"])
        substats, raw_names, raw_values = read_substat_rows(names_img, values_img)
        for i, row in enumerate(substats, 1):
            print(f"Substat {i}: '{row['name']} {row['value']}'")
        lang_signal = echo_language_signal(raw_names, raw_values)

        echo_id, confidence, set_id = match_icon(image)
        if set_id is not None and set_id not in ECHO_SET_IDS.get(echo_id, []):
            set_id = None
        element_name = SET_NAME_BY_ID.get(set_id) if set_id is not None else None
        print(f"Echo identified: {echo_id} (confidence: {confidence:.2%})")
        main = resolve_echo_main(ECHO_COSTS.get(echo_id, 0), raw_main_name, raw_main_value)
        print(f"Echo '{echo_id}' -> Set: {element_name} (id {set_id})")
        print(f"Final echo result: main={main}, substats={substats}")

        return {
            "success": True,
            "analysis": {
                "name": {"name": ECHO_NAME_MAP.get(echo_id, echo_id), "id": echo_id, "confidence": float(confidence)},
                "main": main,
                "substats": substats,
                "element": element_name,
                "setId": set_id,
                "langSignal": lang_signal,
            }
        }
    else:
        return {"success": False, "error": f"Unsupported region: {region}"}

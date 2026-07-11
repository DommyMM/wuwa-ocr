import cv2
import pytesseract
import re
from data import CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_ID_MAP, MAIN_STAT_NAMES, MAIN_STATS, SUB_STATS, ECHO_SET_IDS, SET_NAME_BY_ID, ECHO_COSTS, ECHO_NAME_MAP, ROVER_GENDER_BY_ID, ROVER_ELEMENT_BY_ID, ICON_TEMPLATES, TEMPLATE_FEATURES, COST_TEMPLATES, Rapid, determine_element
import numpy as np
from rapidfuzz import fuzz, process
from typing import Tuple
from cv2 import SIFT_create, FlannBasedMatcher
from pathlib import Path
import io
import sys
import threading


class _ThreadLocalStdout:
    """Process-wide stdout shim that routes writes to a per-thread buffer when one is active, otherwise to the real stream.
    """

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
        # Delegate everything else (reconfigure, encoding, isatty, ...) to the real stream.
        return getattr(self._real, name)


_STDOUT = _ThreadLocalStdout(sys.stdout)
sys.stdout = _STDOUT

# Minimum fuzz.ratio score for a weapon-name OCR read to be trusted. Real reads
# (even with OCR noise) score ~92-100; unreadable/garbled text stays under ~40.
# Below this, the weapon is reported as missing rather than guessed.
WEAPON_NAME_MIN_SCORE = 75

# OCR-noise spellings per element; extend when a new Rover element ships.
ROVER_ELEMENT_ALIASES = {
    "Aero": ("aero", "acro"),
    "Spectro": ("spectro", "speetro"),
    "Havoc": ("havoc", "lavoc"),
}
# Everything below derives from Characters.json + the hand-kept gender ids in
# data.py — a new Rover element only needs its two ids added to
# ROVER_GENDER_BY_ID (plus alias/hue entries above/below if wanted).
ROVER_IDS_BY_GENDER_ELEMENT = {
    (gender, ROVER_ELEMENT_BY_ID[cid]): cid
    for cid, gender in ROVER_GENDER_BY_ID.items()
    if cid in ROVER_ELEMENT_BY_ID
}
ROVER_KNOWN_ELEMENTS = {element for _gender, element in ROVER_IDS_BY_GENDER_ELEMENT}
# The base elemental sonata sets are named exactly by element, so inverting
# SET_NAME_BY_ID over the known Rover elements yields the badge set ids (4/5/6).
ROVER_SET_ID_TO_ELEMENT = {
    set_id: name
    for set_id, name in SET_NAME_BY_ID.items()
    if name in ROVER_KNOWN_ELEMENTS
}
ROVER_BADGE_SET_IDS = list(ROVER_SET_ID_TO_ELEMENT)
# Empirical badge-crop hue medians for the color fallback when badge SIFT abstains.
ROVER_BADGE_HUE_ANCHORS = {
    "Spectro": 26,
    "Aero": 77,
    "Havoc": 161,
}


WEAPON_REGIONS = {
    "name": {"x1": 152, "y1": 25, "x2": 437, "y2": 79},
    "level": {"x1": 191, "y1": 79, "x2": 269, "y2": 133}
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

ECHO_REGIONS = {
    "main": {"x1": 195, "y1": 66, "x2": 366, "y2": 148},
    "subs_names": {"x1": 36, "y1": 228, "x2": 290, "y2": 400},
    "subs_values": {"x1": 290, "y1": 228, "x2": 359, "y2": 400}
}


def process_ocr(name: str, image: np.ndarray) -> str:
    """Process image with appropriate OCR engine"""
    if name == "character":
        # Parallel hybrid: Tesseract for name accuracy + Rapid for level detection
        from concurrent.futures import ThreadPoolExecutor

        def run_tesseract():
            processed_image = preprocess_region(image)
            return pytesseract.image_to_string(processed_image, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')

        def run_rapid():
            result, _ = Rapid(image)
            return "\n".join(text for _, text, _ in result) if result else ""

        with ThreadPoolExecutor(max_workers=2) as executor:
            tess_future = executor.submit(run_tesseract)
            rapid_future = executor.submit(run_rapid)
            name_text = tess_future.result()
            rapid_text = rapid_future.result()

        return f"{name_text.strip()}\n{rapid_text.strip()}"
    elif name == "weapon":
        # Keep Rapid OCR for weapons
        result, _ = Rapid(image)
        if result:
            return "\n".join(text for _, text, _ in result)
        return ""
    else:
        # Default tesseract with preprocessing for other regions
        image = preprocess_region(image)
        return pytesseract.image_to_string(image)

def preprocess_region(image):
    """Lighter preprocessing to preserve text clarity"""
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

def validate_stat(name: str, valid_names: set) -> str:
    if not valid_names:
        return name
    match = process.extractOne(name, list(valid_names))
    return match[0] if match else name

def validate_substat_name(name: str, value: str) -> str:
    cleaned = clean_stat_name(name, value)
    matched = validate_stat(cleaned, SUB_STATS.keys())
    base = matched.replace("%", "")
    if base in {"HP", "ATK", "DEF"}:
        return f"{base}%" if "%" in value else base
    return matched

def validate_value(value: str, stat_name: str) -> str:
    if not SUB_STATS or stat_name not in SUB_STATS:
        return value
        
    had_percent = "%" in value
    clean_value = value.replace('%', '')
    
    try:
        valid_values = [str(v) for v in SUB_STATS[stat_name]]
        match = process.extractOne(clean_value, valid_values)
        if match:
            float_value = float(clean_value)
            matched_value = float(match[0])
            if abs(float_value - matched_value) > 2.0:
                closest = min(SUB_STATS[stat_name], key=lambda x: abs(float_value - x))
                if abs(float_value - closest) <= 1.0:
                    return f"{closest}%" if had_percent else str(closest)
            else:
                return f"{match[0]}%" if had_percent else match[0]
                
    except (ValueError, KeyError):
        pass
    return value

def is_legal_substat_value(value: str, stat_name: str) -> bool:
    if not SUB_STATS or stat_name not in SUB_STATS:
        return False

    try:
        numeric = float(value.replace('%', '').strip())
    except (TypeError, ValueError):
        return False

    return any(abs(numeric - float(valid)) <= 0.05 for valid in SUB_STATS[stat_name])

def choose_substat_value(stat_name: str, tess_value: str, rapid_value: str | None) -> str:
    name_from_tess = validate_substat_name(stat_name, tess_value)
    if is_legal_substat_value(tess_value, name_from_tess):
        return tess_value

    if not rapid_value:
        return tess_value

    name_from_rapid = validate_substat_name(stat_name, rapid_value)
    if is_legal_substat_value(rapid_value, name_from_rapid):
        print(f"Value OCR fallback: '{stat_name} {tess_value}' -> '{stat_name} {rapid_value}'")
        return rapid_value

    return tess_value

def rapid_text_lines(image) -> list[str]:
    result, _ = Rapid(image)
    return [text for _, text, _ in result] if result else []

def substat_pair_score(names: list[str], values: list[str]) -> tuple[int, int]:
    """Score paired OCR rows by valid rows first, then unique stat types."""
    legal = 0
    unique: set[str] = set()
    for name, value in zip(names[:5], values[:5]):
        stat_name = validate_substat_name(name, value)
        if is_legal_substat_value(value, stat_name):
            legal += 1
            unique.add(stat_name)
    return legal, len(unique)

def has_invalid_substat_pair(names: list[str], values: list[str]) -> bool:
    return any(
        not is_legal_substat_value(value, validate_substat_name(name, value))
        for name, value in zip(names, values)
    )

def reconcile_echo_substat_rows(
    names_img,
    values_img,
    names_lines: list[str],
    tess_values: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Align substat name/value rows without assuming maxed echoes have 5 rows."""
    names = merge_wrapped_substat_names(names_lines)
    values = tess_values
    rapid_values: list[str] = []

    if len(names) != len(values):
        candidate_names = merge_wrapped_substat_names(rapid_text_lines(names_img))
        candidate_values = rapid_text_lines(values_img)
        rapid_values = candidate_values
        target_count = max(len(names), len(values))

        if len(candidate_names) > len(names) and len(candidate_names) >= target_count:
            names = candidate_names
        if len(candidate_values) > len(values) and len(candidate_values) >= len(names):
            values = candidate_values

        # When Tesseract drops a numeric row and invents a wrapped-name tail,
        # the count guard above rejects the better Rapid pair because there are
        # "too many" Tesseract names. Prefer the pair with more legal substat
        # rows (and then more unique stat types) so flat HP/ATK/DEF rows do not
        # shift following percent values onto duplicate names.
        if substat_pair_score(candidate_names, candidate_values) > substat_pair_score(names, values):
            names = candidate_names
            values = candidate_values

    if not rapid_values and has_invalid_substat_pair(names, values):
        rapid_values = rapid_text_lines(values_img)

    return names, values, rapid_values

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


def _tess_lines(image: np.ndarray) -> list[str]:
    return [l.strip() for l in pytesseract.image_to_string(image).splitlines() if l.strip()]


def _rapid_main_line(main_img: np.ndarray) -> str:
    """Rapid OCR of the echo main strip, collapsed to one 'Name Value' line."""
    lines = rapid_text_lines(main_img)
    return " ".join(lines[:2]) if len(lines) >= 2 else (lines[0] if lines else "")


def _legal_main_values(cost: int) -> dict[str, str]:
    """name -> canonical Lv.25 value ('22.8%') for an echo cost's variable main stats.

    The variable main stat (what shows in ECHO_REGIONS['main']) is always a percent
    stat from MAIN_STATS; the flat innate HP/ATK (DEFAULT_MAIN_STATS) lives in the
    substat block, not here.
    """
    return {n: f"{format_stat_value(v[-1])}%" for n, v in MAIN_STATS.get(f"{cost}cost", {}).items()}


def _parse_main_line(line: str) -> tuple[str, str]:
    """Split an echo main OCR line 'Crit DMG 44%' into ('Crit DMG', '44%')."""
    parts = line.rsplit(' ', 1)
    if len(parts) == 2 and re.search(r'\d', parts[1]):
        return parts[0], parts[1]
    return line, ""


def _clean_main_name(raw_name: str, raw_value: str) -> str:
    name = clean_stat_name(raw_name, raw_value)
    return f"{name}%" if name in ("HP", "ATK", "DEF") else name


def _name_in(candidates: list[str], read: str | None) -> str | None:
    if not read:
        return None
    probe = _clean_main_name(*_parse_main_line(read))
    match = process.extractOne(probe, candidates, scorer=fuzz.WRatio, score_cutoff=60)
    return match[0] if match else None


def _tiebreak_main_name(candidates: list[str], tess_name: str | None, rapid_provider=None) -> str | None:
    """Break a main-stat tie by name: cheap Tesseract read first, then a lazy Rapid read.

    Rapid is the expensive engine, so it is only invoked when the Tesseract name
    can't resolve the tie (and only when a provider is passed at all).
    """
    if chosen := _name_in(candidates, tess_name):
        return chosen
    if rapid_provider is not None:
        rapid_line = rapid_provider() if callable(rapid_provider) else rapid_provider
        if chosen := _name_in(candidates, rapid_line):
            return chosen
    return None


def resolve_echo_main(cost: int, raw_name: str, raw_value: str, rapid_main=None) -> dict:
    """Resolve an echo's main stat against what its cost actually allows.

    The name is read by bare Tesseract off a small, often-soft strip, so on
    re-encoded/low-detail uploads it fuzzy-matches to a stat that's illegal for the
    cost (e.g. Crit DMG on a 1-cost, whose only legal mains are HP%/ATK%/DEF%) and
    the card gets rejected. Every cost has a fixed legal set with known Lv.25 values:
      - a legal-for-cost name is trusted, its value snapped to the +25 canonical;
      - an illegal name is a confirmed misread, recovered from the *value* (the
        reliable anchor), with the Rapid main read only breaking value ties (e.g. the
        3-cost 30.0% cluster where the value alone can't separate the mains).
    `rapid_main` is an optional zero-arg callable returning the Rapid main line; it is
    invoked lazily, only when a tie actually needs breaking.
    """
    legal = _legal_main_values(cost)
    if not legal:
        # Unidentified echo (cost 0): keep the cost-blind validation so identified
        # echoes improve without regressing the unknown-cost path.
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

    # Illegal-for-cost name => confirmed misread. Recover from the value.
    target = None
    if m := re.search(r'\d+(?:\.\d+)?', raw_value or ""):
        target = float(m.group())

    if target is not None:
        dist = lambda n: abs(float(legal[n].rstrip('%')) - target)
        ranked = sorted(legal, key=dist)
        near = [n for n in ranked if dist(n) <= 1.0]
        if len(near) == 1:
            chosen = near[0]                                          # value alone resolves it (no Rapid)
        elif near:                                                   # genuine tie among present mains
            chosen = _tiebreak_main_name(near, raw_name, rapid_main) or near[0]
        else:                                                        # value matches nothing: trust the name, no Rapid
            chosen = _tiebreak_main_name(ranked, raw_name) or ranked[0]
    else:                                                            # unreadable strip: default to primary main, no Rapid
        chosen = _tiebreak_main_name(list(legal), raw_name) or next(iter(legal))

    print(f"Main stat recovered: {raw_name!r} {raw_value!r} -> {chosen} {legal[chosen]} (cost {cost}, illegal-for-cost name)")
    return {"name": chosen, "value": legal[chosen]}


def parse_echo_substats(lines: list[str]) -> list[dict]:
    """Validate paired 'Name Value' substat lines into legal {name, value} dicts."""
    substats = []
    for i, line in enumerate(lines, 1):
        print(f"Substat {i}: '{line}'")
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
        stat_name, stat_value = parts
        name = validate_substat_name(stat_name, stat_value)
        value = validate_value(stat_value, name)
        if not is_legal_substat_value(value, name):
            print(f"Skipping illegal substat value: '{line}' -> {name} {value}")
            continue
        substats.append({"name": name, "value": value})
    return substats

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

def parse_region_text(name, text):
    match name:
        case "character":
            return parse_character_title(text)
            
        case "watermark":
            lines = text.split('\n')
            username = ""
            uid = 0
            for line in lines:
                if uid_match := re.search(r'\d{6,12}', line):
                    uid = int(uid_match.group(0))
                    break
            if lines:
                first_line = lines[0].strip()
                if ':' in first_line:
                    username = first_line.split(':', 1)[-1].strip()
                else:
                    username = re.sub(r'^(?:Player\s*ID|Name|Neme)[.:;]?\s*', '', first_line, flags=re.IGNORECASE).strip()
                if uid > 0 and str(uid) in username:
                    username = username.replace(str(uid), "").strip().rstrip(':').strip()
            return {
                "username": username,
                "uid": uid
            }



        case "weapon":
            # Match against the known weapon list with a length-sensitive scorer
            # and a cutoff so unreadable text resolves to "missing" instead of
            # snapping to the nearest (often long) name like Legend of Drunken
            # Hero. Real reads score ~92-100 even with OCR noise; garbage stays
            # well under the cutoff. Empty/below-cutoff -> "" so the frontend can
            # apply its signature-weapon fallback or flag the weapon as missing.
            def validate_weapon_name(raw_name: str):
                if not WEAPON_NAMES or not raw_name:
                    return None
                match = process.extractOne(
                    raw_name, WEAPON_NAMES,
                    scorer=fuzz.ratio, score_cutoff=WEAPON_NAME_MIN_SCORE,
                )
                return match[0] if match else None
            lines = text.split('\n')
            raw_name = lines[0].strip() if lines else ""
            weapon_name = validate_weapon_name(raw_name)
            # Scan every line for the level: when the name doesn't render, OCR
            # returns only "LV.xx" and it lands on line 0 (the name slot), so
            # restricting to lines[1:] would miss it and default to 1.
            level = 1
            for line in lines:
                if "LV." in line:
                    match = re.search(r'LV\.(\d+)', line)
                    if match:
                        level = int(match.group(1))
                        break
            return {
                "name": weapon_name or "",
                "id": WEAPON_ID_MAP.get(weapon_name, "") if weapon_name else "",
                "level": level
            }
        case _:
            return text

def get_element_region(image):
    """Extract element region from individual echo image"""
    h, w = image.shape[:2]
    return image[int(h*0.027):int(h*0.148), int(w*0.654):int(w*0.797)]


def get_echo_cost(image: np.ndarray) -> int:
    """Get echo cost from image region"""
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

# Minimum SIFT confidence required before the badge may promote a base echo to
# its rarer Nightmare variant. Calibrated against data: every wrong
# base->nightmare flip there sat at conf <= 0.28 (wrong-body matches), while clean
# echoes score 0.3+. Demotions (nightmare->base) are unconditional.
NIGHTMARE_PROMOTE_FLOOR = 0.30

def echo_family_key(template_id: str) -> str:
    """Group an echo with its Nightmare variant by the shared body name.

    Nightmare echoes are recolors that carry a "Nightmare: " name prefix but the
    same silhouette as their base, so SIFT can't separate them — the badge does
    (see validate_echo_family_by_element). Phantom echoes are not handled here:
    they share the base echo's canonical id, so there is no separate template.
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
    """Resolve same-body variant confusion using the visible set badge."""
    variants = _echo_family_index().get(echo_family_key(best_match), [])
    if len(variants) < 2:
        return best_match, best_conf, detected_element

    family_set_ids = sorted({
        sid for variant in variants for sid in ECHO_SET_IDS.get(variant, [])
    })
    if len(family_set_ids) < 2:
        return best_match, best_conf, detected_element

    # The badge across the family's combined sets is used only to *select the
    # variant*. The element shown is recomputed by the caller from the chosen
    # variant's own legal sets, so a non-flip echo behaves exactly as before.
    badge = determine_element(element_region, family_set_ids)
    candidates = [
        variant
        for variant in variants
        if badge in ECHO_SET_IDS.get(variant, [])
    ]
    if not candidates:
        # determine_element only returns a member of family_elements, so every
        # badge belongs to some variant; this branch is defensive.
        return best_match, best_conf, detected_element

    conf_of = dict(sorted_matches)
    chosen = max(candidates, key=lambda variant: conf_of.get(variant, 0.0))
    if chosen == best_match:
        return best_match, best_conf, detected_element

    # Promoting SIFT's pick *to* a Nightmare variant is the risky direction:
    # Nightmare echoes are rare, and a low-confidence (wrong-body) SIFT match can
    # land on a family whose colors don't even include the true badge, forcing the
    # badge onto a bogus Nightmare sonata. Only promote when SIFT identified the
    # body confidently enough to trust it. Demoting a Nightmare guess back to base
    # needs no floor — base is the overwhelmingly common reality.
    promoting_to_nightmare = (
        "Nightmare" in ECHO_NAME_MAP.get(chosen, "")
        and "Nightmare" not in ECHO_NAME_MAP.get(best_match, "")
    )
    if promoting_to_nightmare and best_conf < NIGHTMARE_PROMOTE_FLOOR:
        return best_match, best_conf, detected_element

    print(f"Family badge validation: {best_match} -> {chosen} (badge {SET_NAME_BY_ID.get(badge, badge)})")
    # Reset element so the caller recomputes it from the new identity's own sets.
    return chosen, conf_of.get(chosen, best_conf), None


# Same-silhouette recolor families (e.g. the six Kernel Puppets) near-tie under
# SIFT because its descriptors are grayscale gradients, and the badge can't pick
# a winner when the recolors share a set. Body hue separates them decisively:
# S>=80/V>=60 drops the shared washed-out silver/gold trim and dark background
# that dilute the histogram. True recolors score 0.8+ against their own template
# and <0.35 against siblings, so the floors below only fire on recolor-style
# ties and leave different-body ties (e.g. Chirpuff vs Gulpuff) to SIFT.
HUE_ARBITRATION_MIN_SCORE = 0.5
HUE_ARBITRATION_MIN_MARGIN = 0.2

def _icon_hue_hist(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 80, 60]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    cv2.normalize(hist, hist)
    return hist

def arbitrate_by_icon_hue(icon_img: np.ndarray, candidates: list[tuple[str, float]]):
    """Pick among near-tied SIFT candidates by icon hue-histogram similarity.

    Returns (echo_name, sift_conf, hue_score) when hue is decisive, else None.
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
    """SIFT match plus close-match/cost disambiguation before family validation.

    Returns:
        Tuple of (echo_name, confidence, element, sorted_matches, element_region)
    """
    icon_img = image[0:182, 0:188]
    sift = SIFT_create()
    kp1, des1 = sift.detectAndCompute(icon_img, None)
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    detected_element = None  # Initialize to avoid duplicate element detection

    # Cost prefilter: the visible cost badge is cheap to read (template match)
    # and partitions the ~163 echoes into cost 1/3/4 buckets, so the SIFT sweep
    # only scores the ~44 templates of the detected cost instead of all of them
    # (~3.6x fewer FLANN matches, the dominant per-echo cost). get_echo_cost
    # returns 0 when unsure (match score < 0.2); an unknown cost falls back to
    # the full sweep, so a missed cost can never drop the true echo. This
    # replaces the old post-hoc cost tiebreaker: a cost-homogeneous sweep has
    # nothing left for it to fix.
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

    # When the top SIFT candidates are near-tied, the badge element disambiguates
    # look-alikes across different bodies (e.g. Chirpuff vs Gulpuff). Same-body
    # base/Nightmare confusion is resolved later by validate_echo_family_by_element.
    if len(sorted_matches) > 1 and (best_conf - sorted_matches[1][1]) < 0.1:
        # Soft images (WebP re-encodes, heavy blur) deflate every confidence, so
        # a fixed conf > 0.1 floor can leave the pool with a single entry and
        # skip disambiguation exactly when SIFT is least trustworthy. Keep the
        # historical floor for healthy images, but on a collapsed scale widen
        # down to 30% of best: still "similar to best", while keeping noise-level
        # junk out of the pool — a misread badge must never hand the win to a
        # candidate SIFT scored at noise level.
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
            # A badge pointing to exactly one candidate is decisive on its own.
            # Otherwise arbitrate the survivors (or all close matches when the
            # badge decided nothing) by icon hue — decisive for recolor ties.
            if len(element_matches) == 1:
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
    """SIFT-based icon matching - returns best match with confidence check and element.

    The visible set badge arbitrates same-body variants such as base vs Nightmare
    echoes, where icon SIFT can prefer the wrong regional variant.
    """
    best_match, best_conf, detected_element, sorted_matches, element_region = _identify_icon_core(image)
    best_match, best_conf, detected_element = validate_echo_family_by_element(
        best_match,
        best_conf,
        sorted_matches,
        element_region,
        detected_element,
    )
    # Only detect element if we haven't already
    if detected_element is None:
        detected_element = determine_element(element_region, best_match)
    return (best_match, best_conf, detected_element)

def parse_sequence_region(image) -> int:
    """Count active sequence nodes using HSV gray detection"""
    GRAY_HSV = {
        'lower': np.array([0, 0, 160]),
        'upper': np.array([40, 180, 255])
    }
    GRAY_THRESHOLD = 0.75
    active_count = 0
    
    for seq_num, region in SEQUENCE_REGIONS.items():
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
        
        if gray_ratio > GRAY_THRESHOLD:
            active_count += 1
    
    return active_count

def _canonical_stat_fragment(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.lower())

def _ensure_dmg_bonus_suffix(name: str) -> str:
    if re.search(r"\bDMG\s+Bonus$", name, re.IGNORECASE):
        return name
    if re.search(r"\bDMG$", name, re.IGNORECASE):
        return f"{name} Bonus"
    return f"{name} DMG Bonus"

_CAN_MERGE_DMG_BONUS_CONTINUATIONS = not any(
    fragment.startswith(("dmg", "bonus"))
    for fragment in (_canonical_stat_fragment(name) for name in SUB_STATS)
)

def _line_with_implied_bonus(line: str, fragment: str) -> str:
    if fragment.endswith("dmg") and not fragment.startswith("crit") and "bonus" not in fragment:
        return f"{line} Bonus"
    return line

def clean_echo_substat_name_lines(lines: list[str]) -> list[str]:
    """Merge OCR-wrapped echo substat names before pairing them with values."""
    cleaned_names: list[str] = []

    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line.strip())
        if not line:
            continue

        fragment = _canonical_stat_fragment(line)
        is_wrapped_dmg_bonus_line = (
            _CAN_MERGE_DMG_BONUS_CONTINUATIONS
            and (
                fragment.startswith(("dmg", "bonus"))
                or (len(fragment) <= 8 and fuzz.ratio(fragment, "bonus") >= 75)
                or (len(fragment) <= 12 and fuzz.ratio(fragment, "dmgbonus") >= 75)
            )
        )
        if cleaned_names and is_wrapped_dmg_bonus_line:
            cleaned_names[-1] = _ensure_dmg_bonus_suffix(cleaned_names[-1])
            continue

        cleaned_names.append(_line_with_implied_bonus(line, fragment))

    return cleaned_names

def merge_wrapped_substat_names(lines: list[str]) -> list[str]:
    """clean_echo_substat_name_lines + absorb GARBAGE wrap tails for the two wrapping stats.

    Only 'Resonance Liberation DMG Bonus' (wraps 'DMG Bonus') and 'Resonance Skill DMG Bonus'
    (wraps 'Bonus') ever wrap to a second line. clean_echo_substat_name_lines merges CLEAN
    continuations; when the 2nd line OCRs as garbage (e.g. 'NIAC Rie', 'Brite', 'Do') it does
    not, leaving an extra name line that breaks name<->value count alignment. Anchor on the
    incomplete known-wrapper prefix instead of the continuation's content: if a cleaned line is
    just 'Resonance Liberation' or 'Resonance Skill[ DMG]', absorb the next line whatever it says.
    """
    def is_known_substat_name(name: str) -> bool:
        match = process.extractOne(name, _SUBSTAT_VOCAB, scorer=fuzz.WRatio)
        return bool(match and match[1] >= 80)

    cleaned = clean_echo_substat_name_lines(lines)
    out: list[str] = []
    skip = False
    for i, name in enumerate(cleaned):
        if skip:
            skip = False
            continue
        fragment = _canonical_stat_fragment(name)
        if fragment == "resonanceliberation" and i + 1 < len(cleaned):
            out.append("Resonance Liberation DMG Bonus")
            skip = True
        elif fragment in ("resonanceskill", "resonanceskilldmg") and i + 1 < len(cleaned):
            out.append("Resonance Skill DMG Bonus")
            skip = True
        elif fragment in ("resonanceliberationdmgbonus", "resonanceskilldmgbonus") and i + 1 < len(cleaned) and not is_known_substat_name(cleaned[i + 1]):
            out.append(name)
            skip = True
        else:
            out.append(name)
    return out

# --- Character and weapon asset recognition (SIFT, OCR fallback on abstain) ---
#
# Server crops a bounding region per field (server.py IMPORT_REGIONS); these
# sub-boxes locate the SIFT target and the OCR-fallback target WITHIN that region,
# so they are coupled to those server boxes and must change together:
#   character region = x[0.00, 0.32] y[0.00, 0.55]            (name strip + splash)
#   weapon region    = x[0.7542, 0.9828] y[0.3843, 0.5843]    (full weapon panel)
#
# Validated on a 500-card r2-backup slice (docs/ocr-recognition-roadmap.md): SIFT
# is more accurate than OCR (language-independent, reads non-English cards OCR
# misses) and far cheaper than RapidOCR on Railway. It abstains via conf+margin
# floors on Rover variants, look-alike weapon icons, and non-card screenshots,
# falling back to the original OCR path so accuracy never regresses.
DATA_DIR = Path(__file__).resolve().parent / "Data"

CHAR_SPLASH_SUBBOX = (0.125, 0.2545, 0.9375, 0.9455)  # splash within character region
CHAR_NAME_SUBBOX = (0.1025, 0.0135, 0.944, 0.1515)    # name strip within character region
CHAR_ELEMENT_SUBBOX = (0.018, 0.025, 0.105, 0.14)      # element badge left of name strip
CHAR_SIFT_MAX_SIDE = 150
CHAR_CONF_FLOOR = 0.10
CHAR_MARGIN_FLOOR = 0.04

WEAP_ICON_SUBBOX = (0.0, 0.0785, 0.209, 0.7285)       # square icon within weapon panel
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
    sift = SIFT_create()
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
    """Top template by SIFT good-match ratio. Returns (id, confidence, margin)."""
    sift = SIFT_create()
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
    set_id = determine_element(badge, ROVER_BADGE_SET_IDS)
    if set_id in ROVER_SET_ID_TO_ELEMENT:
        return ROVER_SET_ID_TO_ELEMENT[set_id]

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


def recognize_character_asset(region_img: np.ndarray) -> dict:
    """SIFT the character splash; OCR the name strip on abstain (Rover, junk).

    Level is not present in the splash, and cards are overwhelmingly Lv.90, so a
    SIFT accept reports level 90. The abstain path runs the original OCR, which
    still reads the true level for the rarer non-90 cards.
    """
    global _CHARACTER_FEATURES
    if _CHARACTER_FEATURES is None:
        _CHARACTER_FEATURES = _load_asset_features("Characters", CHAR_SIFT_MAX_SIDE)
    splash = _resize_max_side(_subcrop(region_img, CHAR_SPLASH_SUBBOX), CHAR_SIFT_MAX_SIDE)
    cid, conf, margin = _match_asset(splash, _CHARACTER_FEATURES)
    if cid in ROVER_GENDER_BY_ID and conf >= CHAR_CONF_FLOOR:
        rover = _rover_analysis(cid, _detect_rover_badge_element(region_img))
        if rover is not None:
            return rover
    if cid and conf >= CHAR_CONF_FLOOR and margin >= CHAR_MARGIN_FLOOR:
        return {"name": CHARACTER_ID_NAME.get(cid, ""), "id": cid, "level": 90}
    text = process_ocr("character", _subcrop(region_img, CHAR_NAME_SUBBOX))
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    parsed = parse_character_title(cleaned)
    if "rover" in re.sub(r'[^a-z]', '', parsed.get("name", "").lower()):
        rover = _rover_analysis(cid, parsed.get("element") or _detect_rover_badge_element(region_img), parsed.get("level", 90))
        if rover is not None:
            return rover
    return parsed


def recognize_weapon_asset(region_img: np.ndarray) -> dict:
    """SIFT the weapon icon; OCR the panel on abstain (look-alike icons).

    A true blank weapon panel yields ~zero SIFT confidence and unreadable OCR, so
    the result stays empty (name/id "") and the frontend signature-weapon fallback
    applies. SIFT accept reports level 90 (see recognize_character_asset).
    """
    global _WEAPON_FEATURES
    if _WEAPON_FEATURES is None:
        _WEAPON_FEATURES = _load_asset_features("Weapons", WEAP_SIFT_MAX_SIDE)
    icon = _resize_max_side(_subcrop(region_img, WEAP_ICON_SUBBOX), WEAP_SIFT_MAX_SIDE)
    wid, conf, margin = _match_asset(icon, _WEAPON_FEATURES)
    if wid and conf >= WEAP_CONF_FLOOR and margin >= WEAP_MARGIN_FLOOR:
        return {"name": WEAPON_ID_NAME.get(wid, ""), "id": wid, "level": 90}
    text = process_ocr("weapon", region_img)
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return parse_region_text("weapon", cleaned)


# --- Non-English card detection -------------------------------------------------
# On a localized (CN/JP/KR) card the substat VALUES still OCR fine (digits) but the
# NAMES are unreadable by the English engine and fuzzy-match the English vocabulary
# poorly. Paired with a "values really are present" gate -- so a wrong screenshot or a
# non-standard layout is not mislabeled as non-English -- a low name-match rate flags a
# non-English card. server.py aggregates this per-echo signal across the 5 echoes.
_SUBSTAT_VOCAB = list(SUB_STATS.keys())
_NUMERIC_VALUE_RE = re.compile(r"^\d{1,4}(\.\d)?%?$")

def echo_language_signal(cleaned_names: list[str], values: list[str]) -> dict:
    """Per-echo English-confidence: matched/total substat names + count of real values."""
    names = [n for n in cleaned_names if len(n) >= 2]
    name_good = sum(
        1 for n in names
        if (process.extractOne(n, _SUBSTAT_VOCAB, scorer=fuzz.WRatio) or (None, 0))[1] >= 80
    )
    num_values = sum(1 for v in values if _NUMERIC_VALUE_RE.match(v.strip()))
    return {"nameGood": name_good, "nameTotal": len(names), "numValues": num_values}

def process_card(image, region: str):
    """Recognize one region and return its result plus the lines it logged.

    All stdout produced during the call (top-level and nested) is captured into a
    per-thread buffer via _STDOUT and returned under "logs"; server.py emits those
    in a fixed region order. Capture is thread-safe, so regions can run on a thread
    pool (one shared process) instead of separate worker processes.
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

        for i, (name, coords) in enumerate(FORTE_REGIONS.items()):
            region_img = processed[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]]
            text = pytesseract.image_to_string(region_img).strip()
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
    elif region.startswith("echo"):
        # --- main stat: raw Tesseract read, resolved against the echo cost below ---
        main_img = _crop_region(image, ECHO_REGIONS["main"])
        main_lines = _tess_lines(preprocess_region(main_img))
        main_line = " ".join(main_lines[:2]) if len(main_lines) >= 2 else (main_lines[0] if main_lines else "")
        raw_main_name, raw_main_value = _parse_main_line(main_line)

        # --- substats: Tesseract with RapidOCR reconcile/fallback (proven path,
        # origin de011fe). The tess-only echo path (merge_wrapped_substat_names +
        # --psm 6 + upscaled repair) drops too many ATK/DEF rows to promote; see
        # docs/echo-substat-tesseract-only.md. ---
        names_img = _crop_region(image, ECHO_REGIONS["subs_names"])
        values_img = _crop_region(image, ECHO_REGIONS["subs_values"])
        names_lines = _tess_lines(preprocess_region(names_img))
        tess_values = _tess_lines(preprocess_region(values_img))
        cleaned_names, values_lines, rapid_values = reconcile_echo_substat_rows(
            names_img, values_img, names_lines, tess_values,
        )
        values = [
            choose_substat_value(name, value, rapid_values[i] if i < len(rapid_values) else None)
            for i, (name, value) in enumerate(zip(cleaned_names, values_lines[:5]))
        ]
        substats = parse_echo_substats([f"{name} {value}" for name, value in zip(cleaned_names, values)])
        lang_signal = echo_language_signal(cleaned_names, values)

        # --- identity (SIFT) + cost-aware main resolution ---
        echo_id, confidence, set_id = match_icon(image)
        if set_id is not None and set_id not in ECHO_SET_IDS.get(echo_id, []):
            set_id = None
        element_name = SET_NAME_BY_ID.get(set_id) if set_id is not None else None
        print(f"Echo identified: {echo_id} (confidence: {confidence:.2%})")
        main = resolve_echo_main(
            ECHO_COSTS.get(echo_id, 0), raw_main_name, raw_main_value,
            rapid_main=lambda: _rapid_main_line(main_img),
        )
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
        text = process_ocr(region, image)
        cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        result = parse_region_text(region, cleaned_text)

        return {
            "success": True,
            "analysis": result
        }

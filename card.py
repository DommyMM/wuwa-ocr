import cv2
import pytesseract
import re
from data import CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_ID_MAP, MAIN_STAT_NAMES, MAIN_STATS, DEFAULT_MAIN_STATS, SUB_STATS, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP, TEMPLATE_FEATURES, COST_TEMPLATES, Rapid, determine_element
import numpy as np
from rapidfuzz import fuzz, process
from typing import Tuple
from cv2 import SIFT_create, FlannBasedMatcher
from pathlib import Path
import io
import sys

# Minimum fuzz.ratio score for a weapon-name OCR read to be trusted. Real reads
# (even with OCR noise) score ~92-100; unreadable/garbled text stays under ~40.
# Below this, the weapon is reported as missing rather than guessed.
WEAPON_NAME_MIN_SCORE = 75

ROVER_ELEMENTS = ("Aero", "Spectro", "Havoc")
ROVER_ELEMENT_ALIASES = {
    "Aero": ("aero", "acro"),
    "Spectro": ("spectro", "speetro"),
    "Havoc": ("havoc", "lavoc"),
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

def reconcile_echo_substat_rows(
    names_img,
    values_img,
    names_lines: list[str],
    tess_values: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Align substat name/value rows without assuming maxed echoes have 5 rows."""
    names = clean_echo_substat_name_lines(names_lines)
    values = tess_values
    rapid_names: list[str] | None = None
    rapid_values: list[str] | None = None

    def get_rapid_names() -> list[str]:
        nonlocal rapid_names
        if rapid_names is None:
            rapid_names = clean_echo_substat_name_lines(rapid_text_lines(names_img))
        return rapid_names

    def get_rapid_values() -> list[str]:
        nonlocal rapid_values
        if rapid_values is None:
            rapid_values = rapid_text_lines(values_img)
        return rapid_values

    if len(names) != len(values):
        candidate_names = get_rapid_names()
        candidate_values = get_rapid_values()
        target_count = max(len(names), len(values))

        if len(candidate_names) > len(names) and len(candidate_names) >= target_count:
            names = candidate_names
        if len(candidate_values) > len(values) and len(candidate_values) >= len(names):
            values = candidate_values

    has_invalid_value = any(
        not is_legal_substat_value(
            value,
            validate_substat_name(name, value),
        )
        for name, value in zip(names, values)
    )
    if has_invalid_value:
        get_rapid_values()

    return names, values, rapid_values or []

def format_stat_value(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"

def max_main_stat_value(cost: int, stat_name: str) -> str | None:
    """Return the level-25 main-stat value for an echo cost/stat pair."""
    cost_key = f"{cost}cost"
    cost_stats = MAIN_STATS.get(cost_key, {})
    if stat_name in cost_stats and len(cost_stats[stat_name]) >= 2:
        return f"{format_stat_value(cost_stats[stat_name][1])}%"

    default_stat = DEFAULT_MAIN_STATS.get(cost_key)
    if default_stat and len(default_stat) >= 3 and default_stat[0] == stat_name:
        return format_stat_value(default_stat[2])

    return None

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
        case _ if name.startswith("echo"):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines:
                return []
            
            main_parts = lines[0].rsplit(' ', 1)
            if len(main_parts) == 2 and re.search(r'\d', main_parts[1]):
                main_name, main_value = main_parts
            else:
                main_name, main_value = lines[0], ""
            main_name = clean_stat_name(main_name, main_value)
            main_name = validate_stat(main_name, MAIN_STAT_NAMES)
            if main_name in ["HP", "ATK", "DEF"]:
                main_name = f"{main_name}%"
            main_value = main_value.replace('422', '22')
            
            substats = []
            for i, line in enumerate(lines[1:], 1):
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
            
            result = {
                "main": {"name": main_name, "value": main_value},
                "substats": substats
            }
            print(f"Final echo result: {result}")
            return result
            
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
    detected_element: str | None,
) -> tuple[str, float, str | None]:
    """Resolve same-body variant confusion using the visible set badge."""
    variants = _echo_family_index().get(echo_family_key(best_match), [])
    if len(variants) < 2:
        return best_match, best_conf, detected_element

    family_elements = sorted({
        element for variant in variants for element in ECHO_ELEMENTS.get(variant, [])
    })
    if len(family_elements) < 2:
        return best_match, best_conf, detected_element

    # The badge across the family's combined sets is used only to *select the
    # variant*. The element shown is recomputed by the caller from the chosen
    # variant's own legal sets, so a non-flip echo behaves exactly as before.
    badge = determine_element(element_region, family_elements)
    candidates = [
        variant
        for variant in variants
        if badge in ECHO_ELEMENTS.get(variant, [])
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

    print(f"Family badge validation: {best_match} -> {chosen} (badge {badge})")
    # Reset element so the caller recomputes it from the new identity's own sets.
    return chosen, conf_of.get(chosen, best_conf), None


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

    # When the top SIFT candidates are near-tied, the badge element disambiguates
    # look-alikes across different bodies (e.g. Chirpuff vs Gulpuff). Same-body
    # base/Nightmare confusion is resolved later by validate_echo_family_by_element.
    if len(sorted_matches) > 1 and (best_conf - sorted_matches[1][1]) < 0.1:
        close_matches = [(name, conf) for name, conf in sorted_matches if conf > 0.1]
        if len(close_matches) >= 2:
            print(f"Close matches detected: {[(n, f'{c:.4f}') for n, c in close_matches]}")

            element_region = get_element_region(image)
            candidate_elements = set()
            for name, _ in close_matches:
                candidate_elements.update(ECHO_ELEMENTS.get(name, []))
            detected_element = determine_element(element_region, list(candidate_elements))

            element_matches = [
                (name, conf)
                for name, conf in close_matches
                if detected_element in ECHO_ELEMENTS.get(name, ["Unknown"])
            ]
            # Only override SIFT when the badge points to exactly one candidate.
            if len(element_matches) == 1:
                best_match, best_conf = element_matches[0]
                print(f"-> badge {detected_element} -> '{best_match}'")

    element_region = get_element_region(image)
    return best_match, best_conf, detected_element, sorted_matches, element_region

def match_icon(image: np.ndarray) -> Tuple[str, float, str]:
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
    if cid and conf >= CHAR_CONF_FLOOR and margin >= CHAR_MARGIN_FLOOR:
        return {"name": CHARACTER_ID_NAME.get(cid, ""), "id": cid, "level": 90}
    text = process_ocr("character", _subcrop(region_img, CHAR_NAME_SUBBOX))
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return parse_character_title(cleaned)


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


def process_card(image, region: str):
    if image is None:
        return {"success": False, "error": "No image data provided"}
    
    # Create a buffer for this specific process's logs
    log_buffer = io.StringIO()
    original_stdout = sys.stdout
    
    try:
        # Redirect stdout to buffer for all regions
        sys.stdout = log_buffer
        
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
            # Process main region
            main_img = image[ECHO_REGIONS["main"]["y1"]:ECHO_REGIONS["main"]["y2"], ECHO_REGIONS["main"]["x1"]:ECHO_REGIONS["main"]["x2"]]
            main_processed = preprocess_region(main_img)
            main_lines = [l.strip() for l in pytesseract.image_to_string(main_processed).splitlines() if l.strip()]
            main_text = f"{main_lines[0]} {main_lines[1]}" if len(main_lines) >= 2 else (main_lines[0] if main_lines else "")
            
            # Process subs regions separately.
            # NOTE: this is the proven RapidOCR-fallback path (origin de011fe), put back
            # in effect for prod. The tess-only echo path (merge_wrapped_substat_names +
            # --psm 6 + upscaled repair) is still DEFINED in this file but intentionally
            # NOT wired in here: it must pass a full r2-backup regression before being
            # promoted again. See docs/echo-substat-tesseract-only.md.
            names_img = image[ECHO_REGIONS["subs_names"]["y1"]:ECHO_REGIONS["subs_names"]["y2"], ECHO_REGIONS["subs_names"]["x1"]:ECHO_REGIONS["subs_names"]["x2"]]
            values_img = image[ECHO_REGIONS["subs_values"]["y1"]:ECHO_REGIONS["subs_values"]["y2"], ECHO_REGIONS["subs_values"]["x1"]:ECHO_REGIONS["subs_values"]["x2"]]

            names_processed = preprocess_region(names_img)
            values_processed = preprocess_region(values_img)

            # Get raw lines
            names_lines = [l.strip() for l in pytesseract.image_to_string(names_processed).splitlines() if l.strip()]
            tess_values = [l.strip() for l in pytesseract.image_to_string(values_processed).splitlines() if l.strip()]

            cleaned_names, values_lines, rapid_values = reconcile_echo_substat_rows(
                names_img,
                values_img,
                names_lines,
                tess_values,
            )

            values = [
                choose_substat_value(
                    name,
                    value,
                    rapid_values[i] if i < len(rapid_values) else None,
                )
                for i, (name, value) in enumerate(zip(cleaned_names, values_lines[:5]))
            ]
            subs_text = "\n".join(f"{name} {value}" for name, value in zip(cleaned_names, values))
            cleaned_text = f"{main_text}\n{subs_text}"

            name, confidence, element_data = match_icon(image)
            print(f"Echo identified: {name} (confidence: {confidence:.2%})")
            echo_data = parse_region_text(region, cleaned_text)
            main = echo_data.get("main", {}) if isinstance(echo_data, dict) else {}
            if main:
                cost = ECHO_COSTS.get(name, 0)
                max_value = max_main_stat_value(cost, main.get("name", ""))
                if max_value:
                    old_value = main.get("value")
                    main["value"] = max_value
                    print(f"Main stat value override: {main.get('name')} {old_value!r} -> {max_value!r} (cost {cost}, assumed Lv.25)")
            print(f"Echo '{name}' -> Element: {element_data}")
            
            # Restore stdout and flush all buffered logs at once
            sys.stdout = original_stdout
            logs = log_buffer.getvalue()
            if logs:
                print(logs.rstrip(), flush=True)
            
            return {
                "success": True,
                "analysis": {
                    "name": {"name": ECHO_NAME_MAP.get(name, name), "id": name, "confidence": float(confidence)},
                    "main": echo_data.get("main", {}),
                    "substats": echo_data.get("substats", []),
                    "element": element_data
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
    except Exception as e:
        # Always restore stdout on error
        sys.stdout = original_stdout
        logs = log_buffer.getvalue()
        if logs:
            print(logs.rstrip(), flush=True)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Ensure stdout is always restored
        sys.stdout = original_stdout

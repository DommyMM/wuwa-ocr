import cv2
import pytesseract
import re
from data import CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_ID_MAP, MAIN_STAT_NAMES, MAIN_STATS, DEFAULT_MAIN_STATS, SUB_STATS, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP, TEMPLATE_FEATURES, COST_TEMPLATES, Rapid, determine_element
import numpy as np
from rapidfuzz import fuzz, process
from typing import Tuple
from cv2 import SIFT_create, FlannBasedMatcher
import io
import sys


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
    cleaned_names = clean_echo_substat_name_lines(names_lines)
    values_lines = tess_values
    rapid_values: list[str] = []

    if len(values_lines) < len(cleaned_names):
        candidate_values = rapid_text_lines(values_img)
        if len(candidate_values) >= len(cleaned_names):
            rapid_values = candidate_values
            values_lines = candidate_values

    if len(cleaned_names) < len(values_lines):
        rapid_cleaned_names = clean_echo_substat_name_lines(rapid_text_lines(names_img))
        if len(rapid_cleaned_names) >= len(values_lines):
            cleaned_names = rapid_cleaned_names

    has_invalid_tess_value = any(
        not is_legal_substat_value(
            value,
            validate_substat_name(name, value),
        )
        for name, value in zip(cleaned_names, values_lines)
    )
    if has_invalid_tess_value and not rapid_values:
        rapid_values = rapid_text_lines(values_img)

    return cleaned_names, values_lines, rapid_values

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

def parse_region_text(name, text):
    match name:
        case "character":
            parts = [p for p in text.split() if p.strip()]
            level = 1
            for part in parts:
                if "LV." in part:
                    match = re.search(r'LV\.(\d+)', part)
                    if match:
                        level = int(match.group(1))
                        parts.remove(part)
                        break
            raw_name = " ".join(parts)
            char_name = validate_character_name(raw_name)
            return {"name": char_name, "id": CHARACTER_ID_MAP.get(char_name, ""), "level": level}
            
        case "watermark":
            lines = text.split('\n')
            username = lines[0].split(':', 1)[-1].strip() if lines and ':' in lines[0] else ""
            uid = lines[1].split(':', 1)[-1].strip() if len(lines) > 1 and ':' in lines[1] else "0"
            return {
                "username": username,
                "uid": int(uid) if uid.isdigit() else 0
            }

        case "weapon":
            def validate_weapon_name(raw_name: str) -> str:
                if not WEAPON_NAMES:
                    return raw_name
                match = process.extractOne(raw_name, WEAPON_NAMES)
                return match[0] if match else raw_name
            lines = text.split('\n')
            raw_name = lines[0].strip() if lines else "Unknown"
            weapon_name = validate_weapon_name(raw_name)
            level = 1
            for line in lines[1:]:
                if "LV." in line:
                    match = re.search(r'LV\.(\d+)', line)
                    if match:
                        level = int(match.group(1))
                        break
            return {
                "name": weapon_name,
                "id": WEAPON_ID_MAP.get(weapon_name, ""),
                "level": level
            }
        case _ if name.startswith("echo"):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if not lines:
                return []
            
            main_parts = lines[0].rsplit(' ', 1)
            if len(main_parts) != 2:
                return []
            main_name, main_value = main_parts
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

def calculate_vibrancy_score(img: np.ndarray) -> float:
    """Calculate vibrancy score based on color intensity and distribution"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))

    h, s, v = cv2.split(hsv)
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]

    if len(s_masked) == 0:
        return 0.0

    avg_saturation = np.mean(s_masked)
    avg_brightness = np.mean(v_masked)

    # Vibrancy combines saturation and brightness
    return (avg_saturation * 0.6 + avg_brightness * 0.4) / 2.55

def analyze_nightmare_indicators(img: np.ndarray) -> dict:
    """Analyze image for nightmare variant indicators"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, (1, 1, 1), (255, 255, 255))

    h, s, v = cv2.split(hsv)
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]

    if len(s_masked) == 0:
        return {"avg_saturation": 0, "avg_brightness": 0, "vibrancy_score": 0, "nightmare_score": 0}

    avg_saturation = np.mean(s_masked)
    avg_brightness = np.mean(v_masked)
    vibrancy_score = calculate_vibrancy_score(img)
    high_saturation_ratio = np.sum(s_masked > 100) / len(s_masked)

    # Nightmare detection scoring (based on analysis of 18 pairs)
    nightmare_score = 0

    # Primary indicators (1 point each)
    if avg_saturation > 65.0:
        nightmare_score += 1
    if avg_brightness > 190.0:
        nightmare_score += 1
    if vibrancy_score > 35.0:
        nightmare_score += 1

    # Secondary indicators (0.5 points each)
    if high_saturation_ratio > 0.1:
        nightmare_score += 0.5
    if avg_brightness > 170 and avg_saturation > 50:
        nightmare_score += 0.5
    if vibrancy_score > 30 and avg_saturation > 45:
        nightmare_score += 0.5

    return {
        "avg_saturation": avg_saturation,
        "avg_brightness": avg_brightness,
        "vibrancy_score": vibrancy_score,
        "nightmare_score": nightmare_score
    }

def compare_icon_colors(icon_img: np.ndarray, template_name: str) -> float:
    """Enhanced color comparison focusing on nightmare vs normal detection"""
    from data import ICON_TEMPLATES

    if template_name not in ICON_TEMPLATES:
        return 0.0

    template_img = ICON_TEMPLATES[template_name]

    # Analyze both images for nightmare indicators
    icon_analysis = analyze_nightmare_indicators(icon_img)
    template_analysis = analyze_nightmare_indicators(template_img)

    # Check if this is a nightmare vs normal comparison
    is_nightmare_template = "Nightmare" in template_name

    # Calculate score similarity based on nightmare score matching
    score_diff = abs(icon_analysis["nightmare_score"] - template_analysis["nightmare_score"])
    score_similarity = max(0.0, 1.0 - score_diff / 3.0)  # Normalize by max possible difference

    if is_nightmare_template:
        # For nightmare templates, icon should also show nightmare characteristics
        if icon_analysis["nightmare_score"] >= 2.0:
            score_similarity += 0.3  # Strong bonus for matching nightmare characteristics
        else:
            score_similarity *= 0.3  # Strong penalty for normal image vs nightmare template

        # Additional saturation/vibrancy matching for nightmare variants
        sat_diff = abs(icon_analysis["avg_saturation"] - template_analysis["avg_saturation"]) / 255.0
        vib_diff = abs(icon_analysis["vibrancy_score"] - template_analysis["vibrancy_score"]) / 100.0

        similarity_score = (score_similarity * 0.7 + (1.0 - sat_diff) * 0.15 + (1.0 - vib_diff) * 0.15)

    else:
        # For normal templates, icon should NOT show strong nightmare characteristics
        if icon_analysis["nightmare_score"] < 2.0:
            score_similarity += 0.3  # Bonus for matching normal characteristics
        else:
            score_similarity *= 0.3  # Penalty for nightmare image vs normal template

        # Traditional color histogram comparison for normal variants
        icon_hsv = cv2.cvtColor(icon_img, cv2.COLOR_BGR2HSV)
        template_hsv = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)

        hist_icon = cv2.calcHist([icon_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_template = cv2.calcHist([template_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

        hist_icon = cv2.normalize(hist_icon, hist_icon).flatten()
        hist_template = cv2.normalize(hist_template, hist_template).flatten()

        hist_correlation = cv2.compareHist(hist_icon, hist_template, cv2.HISTCMP_CORREL)

        similarity_score = (score_similarity * 0.5 + hist_correlation * 0.5)

    return max(0.0, min(1.0, similarity_score))

# Minimum SIFT confidence required before the badge may promote a base echo to
# its rarer Nightmare variant. Calibrated against r2-backup: every wrong
# base->nightmare flip there sat at conf <= 0.28 (wrong-body matches), while clean
# echoes score 0.3+. Demotions (nightmare->base) are unconditional.
NIGHTMARE_PROMOTE_FLOOR = 0.30


def echo_family_key(template_id: str) -> str:
    """Group normal/nightmare variants that share the same visible echo body."""
    name = ECHO_NAME_MAP.get(template_id, template_id)
    return re.sub(r'^(?:Nightmare:\s*|Phantom\s+)', '', name).strip().lower()


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
    matches = []
    flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    detected_element = None  # Initialize to avoid duplicate element detection

    for name, (kp2, des2) in TEMPLATE_FEATURES.items():
        matches_list = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches_list if m.distance < 0.7 * n.distance]
        confidence = len(good_matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0
        matches.append((name, confidence))

    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)
    best_match, best_conf = sorted_matches[0]
    secondary_matches = [m for m in sorted_matches[1:5] if m[1] > 0.1]
    
    # Use color comparison when top matches are close (old logic) + minimum confidence filter
    if len(sorted_matches) > 1 and (best_conf - sorted_matches[1][1]) < 0.1:
        # Lowkey no idea why I chose 10% but 15% was too high so yeah
        close_matches = [(name, conf) for name, conf in sorted_matches
                        if conf > 0.1]

        # If we have multiple nightmare variants, something's wrong - filter out the weaker nightmare
        nightmare_count = sum(1 for name, conf in close_matches if "Nightmare" in name)
        if nightmare_count >= 2:
            print(f"Multiple nightmare variants detected, filtering weaker nightmare")
            # Keep the strongest nightmare and remove other nightmares
            best_nightmare = max([m for m in close_matches if "Nightmare" in m[0]], key=lambda x: x[1])
            close_matches = [m for m in close_matches if "Nightmare" not in m[0] or m == best_nightmare]

        if len(close_matches) >= 2:  # Only if there are actually multiple decent close matches
            close_scores = [(name, f"{conf:.4f}") for name, conf in close_matches]
            print(f"Close matches detected: {close_scores}")

            # Detect element for disambiguation
            element_region = get_element_region(image)

            # Get union of all possible elements for candidates
            candidate_elements = set()
            for name, _ in close_matches:
                candidate_elements.update(ECHO_ELEMENTS.get(name, []))

            # Match against candidate elements only
            detected_element = determine_element(element_region, list(candidate_elements))

            # Check which candidates match the detected element
            element_matches = []
            for name, conf in close_matches:
                possible_elements = ECHO_ELEMENTS.get(name, ["Unknown"])
                if detected_element in possible_elements:
                    element_matches.append((name, conf))

            # Use element as primary tiebreaker
            if len(element_matches) == 1:
                # Element clearly identifies only one result, use that
                best_match = element_matches[0][0]
                best_conf = element_matches[0][1]
                print(f"→ Matched {detected_element} → '{best_match}'")
            else:
                # Identified multiple candidates, tie with color, but ONLY among element-compatible candidates so we
                # don't have say Chirpuff with Law badge which we know is then Nightmare Chirpuff
                # If the element matched nothing, fall back to all close matches (this should never happen)
                color_pool = element_matches if len(element_matches) >= 2 else close_matches
                icon_img = image[0:182, 0:188]
                color_scores = []
                for name, conf in color_pool:
                    color_score = compare_icon_colors(icon_img, name)
                    color_scores.append((name, color_score))

                # Pick the best color match
                best_color_match = max(color_scores, key=lambda x: x[1])
                if best_color_match[0] != best_match:
                    print(f"→ Color-based: {best_match} -> {best_color_match[0]}")
                    best_match = best_color_match[0]
                    for name, conf in sorted_matches:
                        if name == best_match:
                            best_conf = conf
                            break
    
    if secondary_matches and (best_conf - secondary_matches[0][1]) < 0.25:
        actual_cost = get_echo_cost(image)
        print(f"Close match detected, using cost disambiguation. Actual cost: {actual_cost}")
        if actual_cost in [1, 3, 4]:
            best_cost = ECHO_COSTS.get(best_match, 0)
            if best_cost != actual_cost:
                for name, conf in secondary_matches:
                    if ECHO_COSTS.get(name, 0) == actual_cost:
                        print(f"Cost-based selection: {name} (matches cost {actual_cost})")
                        best_match = name
                        best_conf = conf
                        break

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
        elif region.startswith("echo"):
            # Process main region
            main_img = image[ECHO_REGIONS["main"]["y1"]:ECHO_REGIONS["main"]["y2"], ECHO_REGIONS["main"]["x1"]:ECHO_REGIONS["main"]["x2"]]
            main_processed = preprocess_region(main_img)
            main_lines = [l.strip() for l in pytesseract.image_to_string(main_processed).splitlines() if l.strip()]
            main_text = f"{main_lines[0]} {main_lines[1]}" if len(main_lines) >= 2 else ""
            
            # Process subs regions separately
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

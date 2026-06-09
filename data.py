from pathlib import Path
import json
import cv2
import numpy as np
import os
from typing import Dict, List, Set
from cv2 import SIFT_create, FlannBasedMatcher
from rapidocr_onnxruntime import RapidOCR


def rapid_ocr_kwargs() -> dict:
    if os.getenv("USE_GPU", "0") != "1":
        return {}

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"RapidOCR GPU requested but onnxruntime import failed: {exc}", flush=True)
        return {}

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        print(f"RapidOCR GPU requested but CUDAExecutionProvider is unavailable: {providers}", flush=True)
        return {}

    print("RapidOCR using CUDAExecutionProvider", flush=True)
    return {
        "det_use_cuda": True,
        "det_model_path": "",
        "cls_use_cuda": True,
        "cls_model_path": "",
        "rec_use_cuda": True,
        "rec_model_path": "",
    }


Rapid = RapidOCR(lang='en', **rapid_ocr_kwargs())

# Initialize empty defaults
CHARACTER_NAMES: List[str] = []
CHARACTER_ID_MAP: Dict[str, str] = {}
WEAPON_NAMES: List[str] = []
WEAPON_DATA: Dict[str, str] = {}
WEAPON_ID_MAP: Dict[str, str] = {}
MAIN_STAT_NAMES: Set[str] = set()
MAIN_STATS: Dict = {}
DEFAULT_MAIN_STATS: Dict = {}
SUB_STATS: Dict = {}
SUB_STAT_NAMES: Set[str] = set()
ECHO_NAMES: List[str] = []       # ordered list of English names (for logging/rapidfuzz)
ECHO_ELEMENTS: Dict = {}          # CDN id str → list of element/sonata-set names
ECHO_COSTS: Dict[str, int] = {}   # CDN id str → cost (1 | 3 | 4)
ECHO_NAME_MAP: Dict[str, str] = {} # CDN id str → English name (for display in response)
ICON_TEMPLATES: Dict[str, np.ndarray] = {}
TEMPLATE_FEATURES = {}
ELEMENT_TEMPLATES: Dict[str, np.ndarray] = {}
ELEMENT_FEATURES = {}
COST_TEMPLATES: Dict[int, np.ndarray] = {}

# Echoes whose CDN element list is incomplete. Hecate's in-game resonance box
# allows the 6 base elemental sonata sets on top of its default Empyrean
ECHO_ELEMENT_OVERRIDES: Dict[str, List[str]] = {
    '60000855': ['Empyrean', 'Glacio', 'Fusion', 'Electro', 'Aero', 'Spectro', 'Havoc'],
}

# Paths
DATA_DIR = Path(__file__).parent / 'Data'


def _load_from_local():
    """Load Characters, Weapons, Echoes from local Data/ files (legacy format)."""
    global CHARACTER_NAMES, CHARACTER_ID_MAP, WEAPON_NAMES, WEAPON_DATA, WEAPON_ID_MAP, ECHO_NAMES, ECHO_ELEMENTS, ECHO_COSTS, ECHO_NAME_MAP

    with open(DATA_DIR / 'Characters.json', 'r', encoding='utf-8') as f:
        characters = json.load(f)
        CHARACTER_NAMES = [c['name'] for c in characters]
        CHARACTER_ID_MAP.clear()
        for c in characters:
            name = c.get('name', '')
            cid = str(c.get('id', '')).strip()
            if name and cid:
                # Preserve first seen ID for duplicated names (Rover variants are handled in frontend fallback).
                CHARACTER_ID_MAP.setdefault(name, cid)

    WEAPON_NAMES.clear(); WEAPON_DATA.clear(); WEAPON_ID_MAP.clear()
    with open(DATA_DIR / 'Weapons.json', 'r', encoding='utf-8') as f:
        for weapon_type, weapons in json.load(f).items():
            for w in weapons:
                name = w['name']
                wid = str(w.get('id', '')).strip()
                WEAPON_NAMES.append(name)
                WEAPON_DATA[name] = weapon_type
                # Preserve first seen ID for duplicated names.
                if name and wid:
                    WEAPON_ID_MAP.setdefault(name, wid)

    ECHO_NAMES.clear(); ECHO_ELEMENTS.clear(); ECHO_COSTS.clear(); ECHO_NAME_MAP.clear()
    with open(DATA_DIR / 'Echoes.json', 'r', encoding='utf-8') as f:
        for e in json.load(f):
            eid  = str(e['id'])
            name = e['name']
            ECHO_NAMES.append(name)
            ECHO_COSTS[eid]    = e['cost']
            ECHO_ELEMENTS[eid] = e['elements']
            ECHO_NAME_MAP[eid] = name

    # Mirrors echoExtraFetterIDs in lb/internal/calc/validate.go: Hecate's in-game
    # resonance box allows the 6 base elemental sets on top of its default Empyrean.
    for eid, elements in ECHO_ELEMENT_OVERRIDES.items():
        if eid in ECHO_ELEMENTS:
            ECHO_ELEMENTS[eid] = elements

    print(f"Loaded local data: {len(CHARACTER_NAMES)} characters, {len(WEAPON_NAMES)} weapons, {len(ECHO_NAMES)} echoes")


def _read_template_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_templates(folder: str, templates: dict, features: dict, target_size: tuple = None) -> int:
    loaded_names: set[str] = set()
    sift = SIFT_create()

    template_paths = sorted(
        [
            path
            for suffix in ("*.png", "*.webp")
            for path in (DATA_DIR / folder).glob(suffix)
        ],
        key=lambda path: (path.stem, path.suffix != ".png"),
    )

    for icon_path in template_paths:
        try:
            img = _read_template_image(icon_path)
            if img is None:
                print(f"Failed to load template: {icon_path}")
                continue

            if target_size:
                img = cv2.resize(img, target_size)
            templates[icon_path.stem] = img

            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                features[icon_path.stem] = (kp, des)
                loaded_names.add(icon_path.stem)
            else:
                print(f"No features detected for: {icon_path}")

        except Exception as e:
            print(f"Error processing template {icon_path}: {e}")
    return len(loaded_names)


def load_cost_templates() -> int:
    COST_TEMPLATES.clear()
    for cost in (1, 3, 4):
        path = DATA_DIR / "Costs" / f"cost{cost}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load cost template: {path}")
            continue
        COST_TEMPLATES[cost] = img
    return len(COST_TEMPLATES)

try:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Runtime data source is local backend/Data (synced via scripts).
    _load_from_local()

    with open(DATA_DIR / 'EchoStats.json', 'r', encoding='utf-8') as f:
        echo_stats = json.load(f)
        MAIN_STATS.clear()
        MAIN_STATS.update(echo_stats.get("mainStats", {}))
        DEFAULT_MAIN_STATS.clear()
        DEFAULT_MAIN_STATS.update(echo_stats.get("defaultMainStats", {}))

        for cost_data in MAIN_STATS.values():
            for stat_name in cost_data.keys():
                if stat_name in ["HP%", "ATK%", "DEF%"]:
                    MAIN_STAT_NAMES.add(stat_name.replace("%", ""))
                else:
                    MAIN_STAT_NAMES.add(stat_name)

        SUB_STATS = echo_stats.get("subStats", {})
        SUB_STAT_NAMES = set(SUB_STATS.keys())

    echo_count = load_templates('Echoes', ICON_TEMPLATES, TEMPLATE_FEATURES, (188, 188))
    element_count = load_templates('Elements', ELEMENT_TEMPLATES, ELEMENT_FEATURES)
    cost_count = load_cost_templates()
    print(f"Loaded {echo_count} echo templates, {element_count} element templates, and {cost_count} cost templates")

except Exception as e:
    print(f"Critical error during initialization: {e}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Data directory exists: {DATA_DIR.exists()}")

# Elements that share a hue cluster — HSV alone can't separate these pairs.
# Within a cluster, fall back to SIFT. Across clusters, HSV is decisive.
# Electro (H≈135) is distinct from all clusters; no entry needed.
_HUE_CLUSTERS = [
    {'ER', 'Tidebreaking'},                                               # grayscale
    {'Trailblazing', 'Chromatic', 'Fusion', 'Flamewing', 'Flaming', 'Attack', 'Crown'},  # H≈7
    {'Pact', 'Rite', 'Spectro', 'Radiance'},                             # H≈26
    {'Halo', 'Healing'},                                                  # H≈41
    {'Sound', 'Aero', 'Gust', 'Windward'},                               # H≈77
    {'Glacio', 'Frosty', 'QuietSnow'},                                    # H≈102
    {'Law', 'Empyrean', 'Memories'},                                      # H≈109
    {'Midnight', 'Dream', 'Thread', 'Havoc'},                            # H≈161
]

def _same_cluster(candidates: list) -> bool:
    """Return True if all candidates fall within a single hue cluster."""
    cset = set(candidates)
    return any(cset <= cluster for cluster in _HUE_CLUSTERS)

def _template_color_score(image, template) -> float:
    """Compare an element crop against a resized color template."""
    tmpl = cv2.resize(template, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    channel_scores = [
        cv2.matchTemplate(image[:, :, ch], tmpl[:, :, ch], cv2.TM_CCOEFF_NORMED).max()
        for ch in range(3)
    ]
    return float(np.mean(channel_scores))

def determine_element(image, filter_elements):
    """Match element: HSV histogram first, SIFT fallback only within same hue cluster."""
    if isinstance(filter_elements, str):
        base_name = filter_elements.replace("Phantom ", "") if filter_elements.startswith("Phantom ") else filter_elements
        possible_elements = ECHO_ELEMENTS.get(base_name, ["Unknown"])
    else:
        possible_elements = filter_elements if filter_elements else ["Unknown"]

    if len(possible_elements) == 1:
        return possible_elements[0]

    # HSV histogram match
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0], mask, [36], [0, 180])
    cv2.normalize(hist, hist)

    image_colored = cv2.countNonZero(mask)

    scores = []
    for name in possible_elements:
        if name not in ELEMENT_TEMPLATES:
            continue
        tmpl = ELEMENT_TEMPLATES[name]
        t_hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
        t_mask = cv2.inRange(t_hsv, np.array([0, 40, 40]), np.array([180, 255, 255]))
        if cv2.countNonZero(t_mask) == 0:
            # Grayscale template (ER, Tidebreaking): HISTCMP_CORREL returns 1.0 for
            # all-zero histograms regardless of image content — score manually instead.
            # Use a pixel threshold to tolerate a few background/noise pixels.
            total_px = image.shape[0] * image.shape[1]
            score = 1.0 if image_colored < max(10, int(total_px * 0.03)) else -1.0
        else:
            t_hist = cv2.calcHist([t_hsv], [0], t_mask, [36], [0, 180])
            cv2.normalize(t_hist, t_hist)
            score = cv2.compareHist(hist, t_hist, cv2.HISTCMP_CORREL)
        scores.append((name, score))

    if not scores:
        return "Unknown"

    scores.sort(key=lambda x: x[1], reverse=True)
    best, second = scores[0], scores[1] if len(scores) > 1 else None

    # If top two are in the same hue cluster, HSV can't distinguish — use SIFT
    if second is not None and _same_cluster([best[0], second[0]]):
        sift = SIFT_create()
        flann = FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        kp1, des1 = sift.detectAndCompute(image, None)
        if des1 is not None:
            sift_scores = []
            for name, (kp2, des2) in ELEMENT_FEATURES.items():
                if name not in possible_elements:
                    continue
                ml = flann.knnMatch(des1, des2, k=2)
                good = [m for m, n in ml if m.distance < 0.7 * n.distance]
                sift_scores.append((name, len(good) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0))
            if sift_scores:
                best_sift = max(sift_scores, key=lambda x: x[1])
                if best_sift[1] > 0:
                    return best_sift[0]

        # Low-detail sonata crops can produce no usable SIFT matches. In that
        # case, do not let an arbitrary zero-score candidate override HSV.
        best_score = best[1]
        same_hue_candidates = [
            name
            for name, score in scores
            if _same_cluster([best[0], name]) and (best_score - score) < 0.02
        ]
        color_scores = [
            (name, _template_color_score(image, ELEMENT_TEMPLATES[name]))
            for name in same_hue_candidates
            if name in ELEMENT_TEMPLATES
        ]
        if color_scores:
            return max(color_scores, key=lambda x: x[1])[0]

    return best[0]


ECHO_REGIONS = {
    "name": {"top": 0.052, "left": 0.055, "width": 0.8, "height": 0.11},
    "level": {"top": 0.23, "left": 0.08, "width": 0.1, "height": 0.08},
    "main": {"top": 0.31, "left": 0.145, "width": 0.78, "height": 0.085},
    "sub1": {"top": 0.53, "left": 0.115, "width": 0.81, "height": 0.08},
    "sub2": {"top": 0.6, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub3": {"top": 0.685, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub4": {"top": 0.773, "left": 0.115, "width": 0.81, "height": 0.09},
    "sub5": {"top": 0.86, "left": 0.115, "width": 0.81, "height": 0.09}
}

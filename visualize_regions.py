"""
Visualize all OCR crop regions and their internal sub-regions on a card image

Usage:
  py visualize_regions.py <image_path>
  py visualize_regions.py   (uses first image in r2-backup/)

Press 's' to save as regions_debug.png, any other key to close
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Normalized 0-1 boxes on the full card
OUTER_REGIONS = {
    'character': (0.0328, 0.0074, 0.3021, 0.0833),
    'watermark':  (0.0073, 0.0741, 0.1304, 0.1370),
    'forte':      (0.4057, 0.0222, 0.7422, 0.5917),
    'sequences':  (0.0703, 0.4787, 0.3318, 0.5843),
    'weapon':     (0.7542, 0.3843, 0.9828, 0.5843),
    'echo1':      (0.0125, 0.6019, 0.2042, 0.9843),
    'echo2':      (0.2057, 0.6019, 0.3974, 0.9843),
    'echo3':      (0.4016, 0.6019, 0.5938, 0.9843),
    'echo4':      (0.5969, 0.6019, 0.7891, 0.9843),
    'echo5':      (0.7911, 0.6019, 0.9833, 0.9843),
}

OUTER_COLORS = {
    'character': (0, 255, 255),
    'watermark':  (0, 200, 255),
    'forte':      (0, 255, 0),
    'sequences':  (255, 128, 0),
    'weapon':     (255, 0, 255),
    'echo1':      (255, 80, 80),
    'echo2':      (255, 160, 80),
    'echo3':      (255, 220, 80),
    'echo4':      (80, 255, 80),
    'echo5':      (80, 180, 255),
}

# Pixel boxes inside the echo crop, from card.py's ECHO_REGIONS, icon crop, get_element_region and get_echo_cost
ECHO_SUB = {
    'icon':       (0,   0,   188, 182, (200, 200, 200)),
    'element':    (None, None, None, None, (0, 255, 200)),   # normalized, drawn from ECHO_ELEMENT_NORM
    'cost':       (302, 9,   345, 61,  (255, 255, 0)),
    'main':       (195, 66,  366, 148, (100, 200, 255)),
    'subs_names': (36,  228, 290, 400, (180, 255, 100)),
    'subs_vals':  (290, 228, 359, 400, (255, 180, 100)),
}
# element region is normalized within the echo crop
ECHO_ELEMENT_NORM = (0.654, 0.027, 0.797, 0.148)

# Pixel boxes inside the weapon crop
WEAPON_SUB = {
    'name':  (152, 25, 437, 79,  (200, 150, 255)),
    'level': (191, 79, 269, 133, (150, 200, 255)),
}

# Pixel boxes inside the forte crop
FORTE_SUB = {
    'normal':  (270, 144, 389, 204, (200, 255, 150)),
    'skill':   (48,  302, 158, 356, (200, 255, 150)),
    'circuit': (467, 296, 596, 357, (200, 255, 150)),
    'intro':   (122, 545, 247, 602, (200, 255, 150)),
    'lib':     (386, 544, 518, 601, (200, 255, 150)),
}

# Sequence node (center x, center y, width, height) in pixels inside the sequences crop
SEQ_NODES = [
    (55,  58, 30, 26),
    (130, 58, 30, 26),
    (210, 58, 30, 26),
    (290, 58, 30, 26),
    (369, 58, 30, 26),
    (449, 58, 30, 26),
]


def label(img, text, x, y, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + bl), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_rect(img, x1, y1, x2, y2, color, thickness=1, text=None):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        label(img, text, x1 + 2, y1 + 12, color)


def main():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        backup = Path(__file__).parent.parent / 'r2-backup'
        candidates = sorted(backup.glob('*.jpg'))
        if not candidates:
            print(f'No images found in {backup}')
            sys.exit(1)
        image_path = candidates[0]

    img = cv2.imread(str(image_path))
    if img is None:
        print(f'Could not load: {image_path}')
        sys.exit(1)

    H, W = img.shape[:2]
    print(f'Image: {image_path.name}  ({W}x{H})\n')
    out = img.copy()

    for name, (x1n, y1n, x2n, y2n) in OUTER_REGIONS.items():
        ox1, oy1 = round(x1n * W), round(y1n * H)
        ox2, oy2 = round(x2n * W), round(y2n * H)
        color = OUTER_COLORS[name]

        draw_rect(out, ox1, oy1, ox2, oy2, color, thickness=2, text=name)
        print(f'{name:12s}  x={ox1}-{ox2}  y={oy1}-{oy2}  ({ox2-ox1}x{oy2-oy1}px)')

        cw, ch = ox2 - ox1, oy2 - oy1

        if name.startswith('echo'):
            for sub, (sx1, sy1, sx2, sy2, sc) in ECHO_SUB.items():
                if sx1 is None:
                    continue
                draw_rect(out, ox1+sx1, oy1+sy1, ox1+sx2, oy1+sy2, sc, text=sub)
            ex1, ey1, ex2, ey2 = ECHO_ELEMENT_NORM
            draw_rect(out,
                      ox1 + round(ex1*cw), oy1 + round(ey1*ch),
                      ox1 + round(ex2*cw), oy1 + round(ey2*ch),
                      (0, 255, 200), text='element')

        elif name == 'weapon':
            for sub, (sx1, sy1, sx2, sy2, sc) in WEAPON_SUB.items():
                draw_rect(out, ox1+sx1, oy1+sy1, ox1+sx2, oy1+sy2, sc, text=sub)

        elif name == 'forte':
            for sub, (sx1, sy1, sx2, sy2, sc) in FORTE_SUB.items():
                draw_rect(out, ox1+sx1, oy1+sy1, ox1+sx2, oy1+sy2, sc, text=sub)

        elif name == 'sequences':
            for i, (cx, cy, bw, bh) in enumerate(SEQ_NODES, 1):
                draw_rect(out,
                          ox1 + cx - bw//2, oy1 + cy - bh//2,
                          ox1 + cx + bw//2, oy1 + cy + bh//2,
                          (255, 200, 0), text=f'S{i}')

    # Scale to fit screen
    scale = min(1600 / W, 900 / H, 1.0)
    display = cv2.resize(out, None, fx=scale, fy=scale) if scale < 1.0 else out

    cv2.imshow(f'Regions — {image_path.name}', display)
    print('\nPress s to save regions_debug.png, any other key to close.')
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord('s'):
        p = Path(__file__).parent / 'regions_debug.png'
        cv2.imwrite(str(p), out)
        print(f'Saved to {p}')


if __name__ == '__main__':
    main()

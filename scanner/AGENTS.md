# scanner/ — Wuthering Waves Echo Inventory Scanner

Reads a player's Echo bag from the game UI and emits canonical JSON. **Not** part of the
deployed OCR API image (excluded via `backend/.dockerignore`).

Direction, measured evidence, and the bug log live in [PLAN.md](PLAN.md). This file is the
architecture and the invariants.

## The one thing to understand first

**This is barely an OCR problem.** `Data/Stats.json` maps 20 stats onto 17 unique icons,
and every stat row renders its icon, so stat names are a template match, not text. The
only icon collisions are `HP/HP%`, `ATK/ATK%`, `DEF/DEF%`, and those three families have
**disjoint legal value sets**, so the number picks the member.

```
icon   -> the stat FAMILY   (language-independent)
number -> the member within the family
```

Everything else falls out of that:

| what | where it comes from | OCR? |
|---|---|---|
| stat names | 17 icon templates | no |
| the `%` | implied by the family | no |
| main + innate values | derived from cost (`EchoStats.json`) | no |
| echo identity / cost / set / level | the grid tile | no |
| **5 substat numbers** | **digits** | **yes** |

Digits are identical in all 9 WuWa languages, so the scanner is language-independent.

**Never infer a stat NAME from its VALUE.** That is what regressed the Tesseract-only card
path: flat `ATK 40` and flat `DEF 40` are indistinguishable that way. Here the icon fixes
the family first, and ATK and DEF have different icons.

## Tile vs panel

```
TILE (free, no click)  -> identity, cost, sonata set, level, lock, equipped
PANEL (needs a click)  -> substats, and ONLY substats
```

Census a page of 24 tiles in ~65 ms, then click only the echoes worth clicking. A
2777/3000 bag has maybe 40 levelled echoes; everything below +5 has no substats.

## Files

```
scanner/
├── AGENTS.md          # this file
├── PLAN.md            # direction, measured results, bug log
├── samples/           # 3 labelled 4K captures (JPEG q95); the regression fixtures
├── bench/
│   ├── bench_census.py   # REGRESSION: identity over a 24-tile labelled page -> 24/24
│   ├── validate_e2e.py   # REGRESSION: stat icons + substat values -> 21/21, 15/15
│   ├── bench_values.py   # OCR engine shoot-out
│   ├── engines.py        # Tesseract / RapidOCR / Paddle / EasyOCR / OneOCR / WinRT wrappers
│   └── fetch_stat_icons.py
└── wuwa_scanner/
    ├── layout.py      # proportional bounds; THREE precision regimes (read the docstring)
    ├── grid.py        # per-frame row lattice + selection detection
    ├── identify.py    # echo identity: gradient NCC + hue arbitration
    ├── stats.py       # self-locating stat rows + 17-class icon match
    ├── ocr.py         # the only OCR: read a number (WinRT primary, Tesseract fallback)
    ├── panel.py       # tile census + panel substats -> Echo record
    └── __main__.py    # CLI: `census` and `echo`
```

Run **both regressions** after any change to layout, geometry or matching. They exercise
the shipped code path, not a copy of it:

```
py bench/bench_census.py      # -> identity: 24/24
py bench/validate_e2e.py      # -> icons: 21/21   substat values: 15/15
```

The OCR engine bench needs the isolated venv (`.bench-venv`, gitignored); runtime only
needs `ocr.py`'s WinRT/Tesseract path.

## Invariants — break these and it fails silently

1. **The grid scrolls smoothly.** A fixed `TILE_ORIGIN` is only valid at scroll-top. Always
   use `grid.detect_lattice(frame)` per frame. Extrapolating rows from a constant origin
   mis-crops every tile and looks like a *matcher* bug, not a geometry bug.

2. **The panel stats box must start LEFT of the stat icons.** Generous on the left is free;
   clipping them is fatal. A 1% x-shift of a hardcoded box took accuracy from 7/7 to 1/7,
   *silently*. Only the outer box is hardcoded; the icon column, row centres, row pitch and
   value cells all self-locate.

3. **The stats block has variable height.** A substat name that wraps to two lines (only
   `Resonance Liberation DMG Bonus` and `Resonance Skill DMG Bonus` ever do) makes it
   taller and pushes the last row down. A tight y-box clips that row on exactly those
   echoes. Extend the band past any wrap and reject non-icon rows by IoU floor.

4. **Never batch the value column into one OCR pass.** It lets the engine drop a line and
   shift every row below it — the same drift `card.py::reconcile_echo_substat_rows` exists
   to survive. One cell in, one result out.

5. **A value cell must not inherit the icon's row band.** On a wrapped row the icon centres
   on the two-line block while the value stays on the first line, so the band slices the
   value. Values locate their own ink and attach to a row by overlap.

6. **Run OCR engines recognition-only.** The cell is already localised; text *detection* on
   a small crop is waste and actively fails.

7. **Never `sleep()` in navigation.** Poll for the panel to change and stabilize. Dedupe by
   fingerprint, never by scroll arithmetic.

## Identity: gradient, then hue

Match on **Sobel gradient magnitude**, not grayscale. The tile background is a soft
gradient whose colour differs from the CDN template's, and grayscale NCC is dominated by
it. A smooth background has near-zero gradient; the creature has strong edges.

But gradient is background-invariant *because it discards colour*, so same-silhouette
bodies near-tie. **Break ties with hue** (`identify.TIE_MARGIN`), a direct port of
`card.py::arbitrate_by_icon_hue`. Together: 24/24 on a labelled grid page.

## Name prefixes

- **`Phantom:`** — shares the base echo's canonical id (there is no `Phantom:` entry in
  `Echoes.json`). At tile scale the art is indistinguishable from the base, so matching to
  the base id is **correct**. Detecting the phantom *flag* needs phantom skin templates we
  do not have yet.
- **`Nightmare:`** — its own id and its own template. A normal identity problem.
- **`Reminiscence:`** — part of the official name, not a prefix family.

## Boundary with `backend/`

| Owns | Reuses from `backend/` | Doesn't touch |
|---|---|---|
| `wuwa_scanner/`, `bench/`, `samples/` | `data.py` lookups (`ECHO_NAME_MAP`, `ECHO_COSTS`, `ECHO_SET_IDS`, `SUB_STATS`, `MAIN_STATS`, `determine_element`) and `Data/` assets | `server.py`, `card.py` — the export-card pipeline |

`Data/Stats/` (17 stat icons, ~80 KB) was added for the scanner via
`bench/fetch_stat_icons.py` but is generally useful.

Little of `card.py` transfers: most of its cleverness (fuzzy matching, wrap merging,
lossy-value recovery) exists to survive *compressed Discord cards*. We read pristine native
frames. The parts that DO transfer are the ones about echo identity — hue arbitration, and
the Nightmare/Phantom family model.

## Resolution

Calibrated at 4K (3840×2160) 16:9, proportional to the game **client rect** (not the
monitor). Support fullscreen and windowed 16:9; warn on other aspect ratios rather than
silently mis-cropping. 16:10 would need its own profile (Inventory Kamera keeps separate
constants for it).

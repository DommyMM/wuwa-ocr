# Wuwa Echo Scanner — Plan & Decisions

Reads a player's Echo bag from the game UI and emits canonical JSON, for import into
[wuthering-waves-optimizer](../../wuthering-waves-optimizer/) and later wuwa.build.

Architecture and the boundary with `backend/` live in [AGENTS.md](AGENTS.md). This file
tracks **direction, decisions, and measured evidence**.

> **Status (2026-07-12):** recognition core **complete and validated**. Navigation is not
> built. Everything below is measured on 3 labelled 4K captures and one fully
> hand-labelled 24-tile grid page.

| component | result |
|---|---|
| echo identity from the tile (no click) | **24/24** @ 2.7 ms/tile (65 ms per page) |
| stat icons in the panel | **21/21** |
| substat values | **15/15** |
| grid row lattice | exact; recovers all rows from partial detections |
| tile selection (gold corners) | **3/3** |

---

## The core insight: this is barely an OCR problem

`Data/Stats.json` maps **20 stats onto 17 unique icons**, and every stat row renders its
icon. The only collisions are `HP/HP%`, `ATK/ATK%`, `DEF/DEF%`.

```
icon   -> the stat FAMILY   (17 classes, language-independent, ~0.2 ms/row)
number -> the member within the family
```

The family is fixed **before** the value is read, and the three ambiguous families have
**disjoint legal sets** (`HP%` 6.4–11.6 vs `HP` 320–580; `ATK%` 6.4–11.6 vs `ATK` 30–60;
`DEF%` 8.1–14.7 vs `DEF` 40–70), so the number alone always picks the member.

That ordering is the whole design. The Tesseract-only card path regressed because it
inferred the stat NAME from the VALUE, and flat `ATK 40` vs flat `DEF 40` are
indistinguishable that way ([roadmap](../docs/ocr-recognition-roadmap.md)). **ATK and DEF
have different icons, so that failure is structurally impossible here.**

What is left for OCR:

- stat names → icons → **no OCR**
- the `%` → implied by the family → **no OCR**
- main + innate values → derived from cost via `EchoStats.json` → **no OCR**
- echo identity / cost / set / level → the tile → **no OCR**

**Five substat numbers per echo. That is the entire OCR surface**, and digits are
identical in all 9 WuWa text languages, so the scanner is language-independent.

---

## Division of labour: tile vs panel

```
TILE (free, no click)  -> identity, cost, sonata set, level, lock, equipped
PANEL (needs a click)  -> substats, and ONLY substats
```

A 2777/3000 bag holds maybe 40 levelled echoes. Everything below +5 has no substats and
is useless to an optimizer, and the tile shows `+N` directly. So: **census the page (65 ms
for 24 tiles), then click only what is worth clicking.**

The old plan's "40 minutes for a 2000-echo bag" was a **wrong target**, not an acceptable
one.

---

## Recognition, per field

| field | method | evidence |
|---|---|---|
| **echo identity** | Sobel-gradient NCC vs `Data/Echoes`, hue arbitration on near-ties, cost prefilter | **24/24**, 2.7 ms |
| stat names | 17-class icon mask IoU | 21/21, IoU 0.78–0.94 |
| substat values | digits only, snapped to the legal set | 15/15 |
| main + innate | derived from `(cost, stat, level)` | never OCR'd |
| grid rows | gold-bar projection + lattice fit | exact |
| selection | gold **corner** bezels | 3/3 |

### Why gradient, not grayscale

The tile background is a soft gradient whose colour differs from the CDN template's
(Frostbite Coleoid sits on light blue in the tile, dark teal in the template). Grayscale
NCC is dominated by that: **2/3**. A smooth background has near-zero gradient while the
creature has strong edges, so gradient matching is background-invariant: **3/3**.

### Why hue arbitration

Gradient matching is background-invariant *because it discards colour*, so
same-silhouette bodies collapse into a near-tie: `Reminiscence: Fleurdelys` lost to
`Reminiscence: Threnodian - Leviathan` by **0.008**. One is blue/white, the other purple.
Hue separates them decisively. This is a direct port of `card.py::arbitrate_by_icon_hue`,
which exists for exactly the same reason. It fired on precisely the 2 near-ties in 24 and
took the page to **24/24**.

### Phantom / Nightmare / Reminiscence

- **Phantom** shares the base echo's canonical id (no `Phantom:` entry in `Echoes.json`),
  and at tile scale the art is indistinguishable from the base. So matching a Phantom to
  its base id is the **correct identity answer** — all 3 Phantoms in the test page passed.
  Detecting the phantom **flag** is a separate problem and needs phantom icon templates we
  do not have. *(Open: download the phantom skins as templates.)*
- **Nightmare** echoes have their own ids and templates: a normal identity problem. Both
  in the test page passed.
- **Reminiscence** is part of the official name, not a prefix family.

---

## OCR engines

Once the crops were correct, **every engine scored 5/5**. Accuracy is not a
differentiator; the earlier apparent differences were entirely crop bugs. The choice is
purely speed and packaging.

| engine | ms/echo | acc | exe cost | verdict |
|---|---:|---:|---|---|
| **WinRT** (`Windows.Media.Ocr`) | **24** | 5/5 | **zero** | primary |
| EasyOCR (rec-only, GPU) | 154 | 5/5 | ~2 GB torch | unshippable |
| Tesseract (1 spawn, N images) | 213 | 5/5 | ~10 MB | fallback |
| RapidOCR 3.x (rec-only) | 399 | 5/5 | ~50 MB | |
| PaddleOCR | crash | — | ~1 GB | redundant with RapidOCR |

Three traps, each of which changed the ranking:

1. **Run recognition-only.** The cell is already localised, so letting RapidOCR/Paddle run
   text *detection* on a 285×68 crop is waste and actively fails. RapidOCR went from 2/7
   to 7/7 the moment detection was disabled.
2. **`pytesseract` measures process spawn, not Tesseract.** 154 ms/call is startup
   reloading `eng.traineddata`. One invocation with a file list of N cells is 5–6x faster
   and still returns N separate results. (Inventory Kamera pools 8 warm in-process engines
   for the same reason; `tesserocr` has no Python 3.13 wheel.)
3. **VLM/Surya-class readers are excluded on principle** — 650M–7B params to read a
   6-digit number, contending with the game for the GPU.

### Budget

| | per echo |
|---|---:|
| tile identity | ~2.7 ms |
| stat icons | ~1.5 ms |
| 5 substat values (WinRT) | ~24 ms |
| **recognition total** | **~28 ms** |
| click + settle (UI) | ~150–250 ms |

**The scan is 100% navigation-bound.** Recognition is ~10–15% of the click cost and
disappears entirely behind it, so the engine choice barely matters and all remaining
optimization belongs in settle-time and scroll correctness. This also retires the
min-spec worry (i5-9400 / GTX 1060 / 16 GB).

---

## Geometry: three regimes

### 1. The grid ROW offset — runtime state, must be DETECTED

**The grid scrolls smoothly, not row-snapped.** A fixed `TILE_ORIGIN` is only valid at
scroll-top; every frame after a scroll has an arbitrary vertical offset. This is the one
thing no screenshot can tell us, and getting it wrong silently mis-crops every tile (it
read the bottom bar of the tile *above* as part of the tile below, which is how the first
identity bench scored 0/3 while looking like a matcher problem).

Detected per frame from the gold bottom-bar of each tile, then a **regular lattice is
fitted** through the detections — bar detection misses a row here and there, and a missing
row would shift every index below it. Image 1 recovered all 4 rows from only **2** detected
bars.

Inventory Kamera has the same problem and papers over it: fixed wheel-tick counts plus a
scroll-back every ninth page. Do not copy that.

### 2. The detail panel — 1% is fatal, so SELF-LOCATE

Glyphs are ~50 px. A **1% horizontal shift of a hardcoded box took stat-icon accuracy from
7/7 to 1/7, silently.** So exactly one box is asserted (generously) and everything finer is
derived:

```
stats box (generous)   <- the only hardcoded thing
  |- icon column       <- FIRST contiguous ink run in the column projection
  |- row centres       <- ink runs within that column
  |- row bands         <- centre +/- half the median pitch
  |- real rows         <- icon IoU >= 0.60 (rejects the "Echo Skill" heading)
  `- value cells       <- the VALUE's own ink, attached to a row by overlap
```

**Hard requirement: the box must start LEFT of the stat icons.** Generous on the left is
free; clipping them is fatal, and we then abstain (loud) rather than guess (silent).

Cross-validation: hand-measured (Photoshop) icon column x 2655..2730, pitch 89.5. The
runtime ink-projection independently found x 2653..2732, pitch 89. **Two independent
methods, ~2 px apart at 4K.**

### 3. The grid COLUMNS and tile internals — ±20 px is nothing

Tiles are 325×392, so hand-measured constants are more than adequate. All bounds in
[layout.py](wuwa_scanner/layout.py).

- unselected tile **325×392**; the **selected tile is bigger, 345×425** (scales ~6% about
  its centre). Measuring the selected tile and applying it to all of them is a
  ~15%-of-a-tile error.
- origin (334, 266), pitch **(353.2, 423)**. An earlier hand estimate of 440 for the row
  pitch was wrong; two independent detectors measured ~424, and the hand row tops
  (688 / 1112 / 1533) fit `266 + n*423` to within 2 px.
- echo art is a **292×292 square** at tile-offset (18, 11). Square matters: the CDN
  templates are square, so a square query preserves aspect through the resize.

---

## Output contract

Canonical JSON in **our** ids is the source of truth. The optimizer adapter is one
exporter: `CalculatorEchoImporter.vue` already consumes a `ParsedEcho[]`
(`{echo, set, cost, rank, mainStatLabel, substats:[{subStat, subStatValue}]}`), so we emit
that shape to a file. Its current importer is an in-browser tesseract.js parse of a
1920×1080 Discord bot card, so a full-bag JSON import is a strict upgrade.

**No upstream PR, no fork** — we ship the file; they can link the tool.

---

## Phases

- **Phase 0 — Recognition core.** ✅ Done and validated.
- **Phase 1 — Watch mode.** ⏳ Grid census by scroll + row-overlap stitching, plus passive
  capture while the *user* clicks. Live "42 of 47 levelled echoes captured" readout.
  **Zero input injection**, so zero ToS/anti-cheat exposure.
- **Phase 2 — Auto-navigate.** ⏳ Click loop as an **opt-in flag**. Producer clicks and
  polls for panel stability; a worker pool recognizes. Recognition unchanged.
- **Phase 3 — Packaging.** ⏳ Single exe, level floor, output path, progress.

**Ship language deliberately undecided**, gated on Phase 1. The recognition core needs no
SIFT, no heavy OpenCV, and possibly no Tesseract — which would make a small native binary
(Rust + `windows-capture` + egui, or C# AOT) the better ship than Python + PyInstaller.
Decide **before** writing the navigation loop and GUI.

---

## Risks

- **Synthetic input is the real risk, not the OCR.** Automating clicks is against most
  gacha ToS and WuWa ships an anti-cheat. Watch mode (Phase 1) sidesteps it entirely and
  is why it ships first.
- **UI patches move the layout.** Mitigated by self-location + loud abstain, not by
  precise constants.
- **Sample size.** 3 echoes and 1 grid page. Strong, but not validation. Every claim above
  should be re-checked against more captures — especially a cost-1 echo, an under-levelled
  echo with <5 substats, a 1920×1080 frame, and a non-English client.

### Bugs the bench caught (all structural, all silent)

Seven, and every one was found by measurement rather than reasoning. This is why the
design fails loud.

1. Row bands took height from the icon **blob extent**; the Heavy Attack glyph's faint
   chevrons fall below Otsu, so its band came out 42 px vs ~57 and clipped the value.
   → bands are centroid ± half the median **pitch**.
2. **Batching** the value column dropped a line and shifted every row below it.
   → read per row.
3. Auto-locating the icon column by **component size/shape** failed (0/49): stat icons are
   not single components (Crit DMG = a star **plus four detached arrows**).
   → the icon column is the first contiguous **ink run**.
4. A **fixed y-band clips the last row** whenever a substat name wraps, because the wrap
   makes the block taller. Silently turned a `Crit Rate` row into a bogus `Heavy Attack`
   at IoU 0.34. → extend the band, reject non-icons by IoU floor.
5. Value cells **inherited the icon's row band**. On a wrapped row the icon centres on the
   two-line block while the value stays on the first line, so the band sliced the value
   (`7.1%` read as `1770`). → values locate their own ink.
6. **Fixed `TILE_ORIGIN` ignored scroll.** Mis-cropped every tile on a scrolled frame and
   made identity look like a matcher failure. → detect the row lattice per frame.
7. Selection detected by **border brightness** picked a gold-artwork tile over the truly
   selected one. → test the **corners**, for the gold **hue**.

---

## Bench tooling (`bench/`)

| file | what it does |
|---|---|
| `bench_census.py` | **regression**: identity over a full 24-tile labelled grid page → 24/24 |
| `validate_e2e.py` | **regression**: stat icons + substat values vs gold labels → 21/21, 15/15 |
| `bench_values.py` | OCR engine shoot-out on substat value cells |
| `engines.py` | uniform wrappers: Tesseract, RapidOCR 1.x/3.x, Paddle, EasyOCR, OneOCR, WinRT |
| `fetch_stat_icons.py` | download the 17 stat icons into `Data/Stats/` |

Both regressions exercise the **shipped** code path (`wuwa_scanner/`), not a copy of it.
Run them after any change to layout, geometry or matching.

The exploratory crop-tolerance sweeps that produced the geometry findings above were
deleted once the self-locating design replaced the hardcoded one they were probing; their
results are recorded in "Geometry" and "Bugs the bench caught".

Engines live in an isolated venv (`.bench-venv`, gitignored) so the working backend env is
untouched. Only `ocr.py`'s WinRT/Tesseract path is needed at runtime.

## Fixtures

`samples/` holds 3 labelled 4K captures, stored as **JPEG q95** (~1.6 MB each rather than
~8.2 MB PNG). Both regressions stay green on the JPEGs — a free robustness result, since
real captures will not be lossless either.

| fixture | why it exists |
|---|---|
| `bag_4k_01.jpg` | the 24-tile census page: 3 Phantoms, 3 Nightmares, 5 duplicate pairs |
| `bag_4k_02.jpg` | a two-line substat wrap + a flat DEF |
| `bag_4k_03_cost3.jpg` | cost 3, **two consecutive wraps**, an ATK% substat, innate ATK 100 |

## Next

1. **Tile census fields still to wire**: level (`+25`), sonata set, lock, equipped
   portrait. Boxes are in `layout.py`; the readers are not written. These are what let us
   *skip* clicks, so they come first.
2. **Phantom flag**: download the phantom skin icons as templates.
3. **More captures**: cost-1, an under-levelled echo with <5 substats, 1920×1080 (to verify
   the proportional bounds hold), and a non-English client (to *prove* language
   independence rather than argue it).
4. Then Phase 1.

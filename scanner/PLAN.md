# Wuwa Echo Scanner — Plan & Decisions

Reads a player's Echo bag from the game UI and emits canonical JSON, for import into
[wuthering-waves-optimizer](../../wuthering-waves-optimizer/) and later wuwa.build.

Architecture and the boundary with `backend/` live in [AGENTS.md](AGENTS.md). This file
tracks **direction, decisions, and measured evidence**.

> **Status (2026-07-12):** recognition core **complete and validated**. Navigation is not
> built. Everything below is measured on 3 labelled 4K captures, one fully hand-labelled
> 24-tile grid page, and a second page carrying the hard Nightmare families.

| component | result |
|---|---|
| echo identity from the tile (no click) | **24/24** @ 1.8 ms/tile (42 ms per page) |
| sonata set from the tile | **24/24** |
| stat icons in the panel | **21/21** |
| substat values | **15/15** |
| hard Nightmare families (Crownless, Thundering Mephis, Feilian Beringal) | **5/5**, hand-confirmed in game |
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
| **echo identity** | cost prefilter → gradient NCC + hue → family-scoped badge | **24/24**, 1.8 ms |
| **sonata set** | badge, scoped to the identified echo's family | **24/24**, 0.66 ms |
| stat names | 17-class icon mask IoU | 21/21, IoU 0.78–0.94 |
| substat values | digits only, snapped to the legal set | 15/15 |
| main + innate | derived from `(cost, stat, level)` | never OCR'd |
| grid rows | gold-bar projection + lattice fit | exact |
| selection | gold **corner** bezels | 3/3 |

### The ordering is the design: identity → badge, not badge → identity

`card.py` runs **badge → identity**: its SIFT descriptors are grayscale gradients, which
literally cannot see a recolor, so it needs an outside signal to fix identity. Copying that
here would have made us *worse*. Our gradient matcher is the strong signal and the tile
badge is the weak one:

```
blind 34-way badge sweep    15/18    4.8 ms
scoped to the family        18/18    0.66 ms   (avg 1.8 candidates, not 34)
```

And the blind errors are the poisonous kind — it read Fleurdelys as **QuietSnow**, a set
Fleurdelys cannot even roll. As a hard prefilter that would have deleted the true echo from
its own candidate pool. So the dependency is **inverted**: identity leads, and that collapses
the badge's job from 34 candidates to 1.8. Fewer comparisons *is* the accuracy win — the ~32
candidates it drops are precisely the ones that generated every error.

A third of echoes have exactly one legal set, so `determine_element` short-circuits and
touches **no pixels at all** (0.001 ms). 74% are a 1-or-2-way call.

### Three signals, and no family is blind to all three

Scoring every Nightmare pair template-against-template makes the coverage structural rather
than lucky (high = the two look alike = hard):

| family | gradient | hue | badge |
|---|---|---|---|
| Viridblaze, Baby Viridblaze, Dwarf Cassowary, Baby Roseshroom | blind (0.86–0.96) | blind (0.91–0.94) | **carries it** |
| Crownless, Thundering Mephis, Inferno Rider | **carries it** (0.06–0.30) | mixed | mute (sets identical to base) |
| Feilian Beringal | blind (0.937) | **carries it alone** (−0.106) | mute (sets identical to base) |

Four families have sets **identical** to their base, so the badge is mute and `card.py` bails
outright (`if len(family_set_ids) < 2: return unchanged`). That is its known blind spot, and
gradient or hue covers every one of them. All five hard tiles — including two Phantom
Nightmare Crownless and a Nightmare Feilian Beringal — were confirmed correct in game.

### Why gradient, not grayscale

The tile background is a soft gradient whose colour differs from the CDN template's
(Frostbite Coleoid sits on light blue in the tile, dark teal in the template). Grayscale
NCC is dominated by that: **2/3**. A smooth background has near-zero gradient while the
creature has strong edges, so gradient matching is background-invariant: **3/3**.

### Why hue arbitration

Gradient is background-invariant *because it discards colour*, so same-silhouette bodies
collapse into a near-tie: `Reminiscence: Fleurdelys` lost to `Leviathan` by **0.008**. Hue
separates them decisively — a direct port of `card.py::arbitrate_by_icon_hue`.

When gradient cannot separate the top two **and** hue abstains, the answer is a coin flip
and `tile.census` says so. That abstain is what surfaced all five hard tiles instead of
quietly guessing at them.

### Why cost prefilters despite proving nothing on its own

Cost can never separate a Nightmare (variants always share their base's cost), and the speed
it buys is irrelevant against a 150–250 ms click. It was demoted to a validator on exactly
that reasoning — and then reinstated, because a washed-out **Phantom Feilian Beringal**
matched **"Zip Zap"**, a cost-1 echo from an unrelated family. The phantom shimmer flattens
the very edges gradient depends on, so every score collapsed into noise (top 0.105, top-6
smeared across 0.105–0.075). Filtering to cost 4 puts the Feilian pair back at ranks 1 and 2.
Speed was never the argument; recovering a dead signal is.

It is safe because it **abstains rather than guesses** — an unknown cost means the full
sweep, so a missed badge can never drop the true echo. It is the *only* step permitted to
remove a candidate.

### Phantom / Nightmare / Reminiscence

- **Phantom** shares the base echo's canonical id (no `Phantom:` entry in `Echoes.json`), so
  matching a Phantom to its base id is the **correct answer**, and the flag is cosmetic —
  same cost, same legal sets, same stat pools — so we do not detect it. Measured, not
  assumed: hue-vs-own-template scores a Phantom Fallacy **0.151** and its non-phantom twin
  **0.191**. No separation; the tile art is genuinely identical.

  The phantom **art** still matters, and this is the subtle part. A Phantom is a *recolor*,
  so its hue shifts away from the base template — and Feilian Beringal is separated from its
  Nightmare by **hue alone**. Comparing a phantom's shifted hue against non-phantom templates
  is precisely how a Phantom base flips to a Nightmare. So `Data/EchoPhantoms/` (38 skins) is
  loaded as a **second template under the same id**, scored best-of-variants. This removes
  the trap instead of detecting it, and it buys real margin:

  | tile | margin before | after |
  |---|---:|---:|
  | Phantom: Nightmare Crownless | 0.039 | **0.142** |
  | Phantom Reactor Husk | 0.104 | **0.447** |
  | Phantom Fallacy | 0.222 | **0.537** |

- **Nightmare** echoes have their own ids and templates. See the coverage table above.
- **Reminiscence** is part of the official name, **not** a prefix family. `_family_key` must
  strip only `Nightmare:` — stripping `Reminiscence:` too would merge
  `Reminiscence: Kronaclaw` with a future base `Kronaclaw` and let the badge flip between two
  genuinely different echoes.

### Duplicates are not a problem

Row × column **is** the identity. Two Fleurdelys in different cells are two echoes, and if
they carry identical substats they are *still* two echoes. Never dedupe scanner output by
content — key it by cell. (This concern was imported from the leaderboard, where it belongs;
here it does not.)

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
| tile identity (cost-prefiltered) | ~1.8 ms |
| sonata badge (family-scoped) | ~0.4 ms |
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
- **Sample size.** 3 echoes and 2 grid pages, one 4K resolution, one client language. Strong,
  but not validation. Specifically untested: **cost 1** (85 of 180 echoes), an under-levelled
  echo with <5 substats, 1920×1080, and a non-English client.
- **Thin margins on the hard families.** Crownless, Thundering Mephis and Feilian Beringal all
  resolve correctly but by 0.01–0.06, and hue abstains on them because the phantom shimmer
  desaturates the art hue depends on. They are flagged, not silent — but they are one UI
  patch away from flipping, and the abstain floors (`HUE_MIN_SCORE` / `HUE_MIN_MARGIN`,
  inherited from `card.py` where the crops are cleaner) have never been tuned for tile scale.

### Bugs the bench caught (all structural, all silent)

Ten, and every one was found by measurement rather than reasoning. This is why the
design fails loud.

Three of them are the same mistake in different costumes — **a confident number that was
never actually measuring the thing it claimed to.** Worth internalizing:

8. **The cost box was fitted on a page where every tile was cost 4.** "Reads 4" is then
   satisfiable by a box sitting on blank background that merely correlates with the `4`
   glyph — which is exactly what the sweep found. It scored a proud **18/18**, then went
   **0/6** on the first cost-3 row it met. Refit on a mixed-cost frame: 36/36. *Any refit
   must span at least two costs.*
9. **Tile-native cost templates scored higher and were more wrong.** Cropped from a tile,
   a template carries the artwork *behind* the digit, so it correlates on the creature:
   **0.715 mean score, 21/36 correct.** `card.py`'s clean glyph templates score **0.127**
   and are **36/36**. Gate on the margin between the top two, never on absolute score.
10. **Re-boxing the selected tile on a centred-growth model overcropped it.** The selected
    tile *is* bigger (345×425 vs 325×392) but does not grow about its centre — measured at
    (330, 250) against a column origin of 334, a 4 px shift where centred growth demands 10.
    "Fixing" it dropped identity margins 0.367 → 0.130 and 0.142 → 0.009. The unselected box
    reads every tile correctly; the 292×292 art absorbs the shift.

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
| `bench_census.py` | **regression**: identity + sonata over a 24-tile labelled page → 24/24, 24/24 |
| `validate_e2e.py` | **regression**: stat icons + substat values vs gold labels → 21/21, 15/15 |
| `bench_values.py` | OCR engine shoot-out on substat value cells |
| `engines.py` | uniform wrappers: Tesseract, RapidOCR 1.x/3.x, Paddle, EasyOCR, OneOCR, WinRT |
| `fetch_stat_icons.py` | download the 17 stat icons into `Data/Stats/` |
| `fetch_phantom_icons.py` | download the 38 phantom skins into `Data/EchoPhantoms/` |

`fetch_phantom_icons.py` must resolve URLs through the frontend's three-way `toImageUrl`
rule (`wuwabuilds/lib/echo.ts`): `/d/` → Wuthery, `/Game/` → encore, absolute → as-is.
Newly-shipped echoes are not on Wuthery yet and carry an absolute encore URL, so blindly
prefixing the CDN base silently yields `https://files.wuthery.comhttps://api.encore...`.

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

1. **Tile census fields still to wire**: level (`+25`), lock, equipped portrait. Boxes are
   in `layout.py` under the `NOT YET WIRED` marker; the readers are not written. **Level is
   the one that matters** — it is what lets us skip clicking the ~2700 echoes below +5 that
   have no substats at all.
2. **A cost-1 capture.** Every cost claim rests on cost-3 and cost-4 tiles; cost 1 is 85 of
   the 180 echoes and is completely untested.
3. **Re-sync `wuwabuilds/public/Data/Echoes.json`.** It reports no phantom skin for
   `60001945` (Reminiscence: Kronaclaw) but one exists in game, and probing the `SG_` naming
   convention across all 142 phantom-less echoes found no undocumented skins — so the table
   is stale, not the convention. A missing phantom template is a thin margin waiting to
   happen.
4. **More captures**: an under-levelled echo with <5 substats, 1920×1080 (to verify the
   proportional bounds hold), and a non-English client (to *prove* language independence
   rather than argue it).
5. Then Phase 1.

## Entry points and packaging

`__main__.py` is the debug CLI (`py -m wuwa_scanner census <frame>`) and is the right idiom
for it. **The shipped exe is not this CLI** — it is watch mode: a capture loop, a live
"42 of 47 captured" readout, and a JSON export. Different program, different entry point.

When Phase 1 lands, split it:

```
wuwa_scanner/
├── __main__.py   # 2-line shim: from .cli import main; raise SystemExit(main())
├── cli.py        # the debug commands -- importable and testable, not trapped behind -m
└── app.py        # the watcher. THIS is what PyInstaller targets.
pyproject.toml    # [project.scripts] -> a real `wuwa-scanner` command
```

Do not do it earlier; `__main__.py` is small and the split is churn until there is a second
entry point to justify it. Keep the empty `wuwa_scanner/__init__.py` — namespace packages
(PEP 420) import fine but are a known source of PyInstaller module-discovery failures, and
`setuptools.find_packages()` skips directories without one.

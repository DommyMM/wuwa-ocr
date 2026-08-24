# Wuwa Echo Scanner — Plan & Decisions

Reads a player's Echo bag from the game UI and emits canonical JSON, for import into
[wuthering-waves-optimizer](../../wuthering-waves-optimizer/) and later wuwa.build.

Architecture and the boundary with `backend/` live in [AGENTS.md](AGENTS.md). This file
tracks **direction, decisions, and measured evidence**.

> **Status (2026-08-23):** recognition core **complete and validated**. Navigation is not
> built. Measured on **5** labelled 4K captures: one fully hand-labelled 24-tile page, a
> page carrying the hard Nightmare families, and three more added since, one of which is
> the first to contain mixed costs, mixed levels, a sub-5-substat echo, and the "New"
> overlay.

| component | result |
|---|---|
| echo identity from the tile (no click) | **24/24** @ 1.6 ms/tile (38 ms per page) |
| sonata set from the tile | **24/24** |
| **cost from the tile** | **90/90** over 5 frames (15 cost-1, 12 cost-3, 63 cost-4) |
| **level from the tile** | **90/90** over 5 frames (levels 0, 15, 17, 20, 21, 22, 25) |
| stat icons in the panel | **21/21** |
| substat values | **15/15** |
| under-levelled echo (<5 substats) | correct, incl. the visible Echo Skill description |
| hard Nightmare families (Crownless, Thundering Mephis, Feilian Beringal) | **5/5**, hand-confirmed in game |
| grid row lattice | exact; recovers all rows from partial detections |
| tile selection (gold corners) | **5/5 frames**, incl. one frame with nothing selected |

Three of those rows are new, and two of them are new because the field was **broken and
passing**. Cost scored 2/18 on the first page that disagreed with its training set, and
level had never been read at all. See bugs 11-14.

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

Everything below +5 has no substats at all and is useless to an optimizer, and the tile
shows `+N` directly. So: **census the page (38 ms for 24 tiles), then click only what is
worth clicking.**

An earlier draft guessed "maybe 40 levelled echoes" in a 2777/3000 bag. **That was wrong
by an order of magnitude** — a real bag is hundreds of `+25`s, and `bag_4k_04` shows a
2668/3000 bag whose level-sorted first page is solid `+25` down to a `+15` before hitting
the `+0` fodder. The census-first design is unaffected, but the click budget is the entire
scan time, so recognition speed is not worth another minute of anyone's attention and
settle-time is worth all of it.

Sorting by level descending makes the floor an **early exit** rather than a filter: the
first tile below it ends the clicking pass, and every tile after it is below it too. We
read level from the TILE, so that exit costs no clicks at all — Inventory Kamera has to
click an item to discover it was too low.

The old plan's "40 minutes for a 2000-echo bag" was a **wrong target**, not an acceptable
one.

---

## Recognition, per field

| field | method | evidence |
|---|---|---|
| **echo identity** | cost prefilter → gradient NCC + hue → family-scoped badge | **24/24**, 1.6 ms |
| **sonata set** | badge, scoped to the identified echo's family | **24/24**, 0.66 ms |
| **cost** | gold **ink mask**, bbox-normalised, soft IoU | **90/90**, 0.15 ms |
| **level** | ink segmentation drops the `+`, then Tesseract @4x | **90/90**, 12 ms |
| stat names | 17-class icon mask IoU | 21/21, IoU 0.78–0.94 |
| substat values | digits only, snapped to the legal set | 15/15 |
| main + innate | derived from `(cost, stat, level)` | never OCR'd |
| grid rows | gold-bar projection + lattice fit | exact |
| selection | gold corner bezels, **min over the four corners** | 5/5 frames |

### Every field that failed, failed by not looking at the thing it named

Cost correlated a diamond frame. Level OCR'd a `+` as a `4`. Selection averaged four
corners and let one bright corner speak for all of them. The stat-name path in `card.py`
inferred a name from a value. Different fields, one shape of mistake, and in each case the
fix was to isolate the actual signal rather than hope the noise averaged out:

```
cost   -> mask the gold INK, discard the artwork behind it
level  -> segment the ink, DROP the leftmost blob, then read
select -> score each corner separately, take the MINIMUM
stats  -> read the icon, never the value
```

None of these were found by reasoning. All four were found by running a reader against a
frame it had not been tuned on.

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

That safety property was **false for a while**, and the failure was exactly as bad as the
design predicts. Reading a cost-3 tile as a confident "4" scoped identity to the cost-4
pool, deleted the true echo from its own candidate pool, and returned `Feilian Beringal`
at margin **0.002**. With the reader fixed the same tile returns `Spearback` at **0.251**.
Bug 11 is the post-mortem; the lesson worth keeping is that "it abstains" is a claim about
a **margin gate**, and a margin gate is worthless if the thing being measured is not the
signal. Both templates agreeing to 0.003 is not a near-tie between two readings of a
digit, it is two readings of the same diamond.

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
- **Sample size.** 5 grid pages, one 4K resolution, one client language, two accounts.
  Still untested: **1920×1080** (the resolution most friends will be on, and the one the
  other public WuWa scanner supports while we do not), **ultrawide** (more columns, so
  `GRID_COLS` has to be detected rather than fixed), and a **non-English client** — which
  would *prove* language independence rather than argue it.
- **Cost 1 is trained but not held out.** The cost masks are validated 72/72 on frames they
  never saw, but every one of those frames is cost 3 and 4. Cost 1 has 15 training
  exemplars and **zero** held-out tests, because `bag_4k_04` is the only capture containing
  one. Given that this exact field has now been "fixed" three times (bugs 8, 9, 11), that
  gap should be treated as an open failure rather than a formality.
- **Cost-1 identity is the weakest recognition surface.** Even with the prefilter correct,
  `bag_4k_04` resolves Tick Tack at margin **0.016**, Frostscourge Stalker at **0.021** and
  Baby Roseshroom at **0.030** — three of fifteen at coin-flip margins. Cost 1 is the
  largest bucket (85 of 180) and its silhouettes are the least distinctive in the game.
  They warn rather than lie, but this is where the next real accuracy work is.
- **Thin margins on the hard families.** Crownless, Thundering Mephis and Feilian Beringal all
  resolve correctly but by 0.01–0.06, and hue abstains on them because the phantom shimmer
  desaturates the art hue depends on. They are flagged, not silent — but they are one UI
  patch away from flipping, and the abstain floors (`HUE_MIN_SCORE` / `HUE_MIN_MARGIN`,
  inherited from `card.py` where the crops are cleaner) have never been tuned for tile scale.

### Bugs the bench caught (all structural, all silent)

Fourteen, and every one was found by measurement rather than reasoning. This is why the
design fails loud.

Four of them are the same mistake in different costumes — **a confident number that was
never actually measuring the thing it claimed to.** Worth internalizing, and note that
8, 9 and 11 are three attempts at the *same field*: each fix was real, each was validated
on the frames available at the time, and each left the reader looking at something other
than the digit. A field is not fixed until a frame it has never seen says so.

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

11. **The cost reader was correlating a diamond, not a digit.** `Data/Costs/*` are
    card.py's build-card templates: the digit *inside a diamond frame*. The bag tile draws
    a bare digit on artwork. The diamond is identical across all three templates, so it
    dominated the correlation and the digit contributed almost nothing — every cost-1 tile
    scored all three within **0.003–0.016** and abstained, and where the noise did clear
    the margin gate it cleared it on the wrong answer. **2/18** on the first mixed-cost
    page it ever saw, having "passed" at 36/36 on cost-3-and-4 pages. Bug #8 said a refit
    must span two costs; it did. It must span **all three**. → mask the gold ink and
    compare shapes: **90/90**, margins 0.45–0.56.
12. **`TILE_LEVEL` fit the one tile it was measured on.** The lattice carries a few pixels
    of sub-row phase, so on other frames the glyph tops crossed the box edge, and a clipped
    `+25` OCR'd as **2** — a perfectly plausible level. 17 of 90 tiles, silently. → sweep
    the bounds over every labelled tile and gate on *"does any ink touch an edge"* rather
    than *"does it read correctly"*. There is an 8 px plateau; sit in the middle of it, and
    **abstain** if ink reaches an edge anyway.
13. **Tesseract reads `+` as `4`**, so `+25` came back as 425. Inventory Kamera hit the
    identical bug and left it as a comment rather than a fix. Repairing it afterwards
    ("strip a leading 4 above 25") is keyed to one engine's quirk and corrupts a genuine 4.
    → the `+` is always the leftmost ink; drop that blob before OCR sees it. Separately,
    the digit crop needs **4x** upscale, not the 2x that suits a substat value cell — at 2x
    one tile in ninety misreads, at 4x every page-segmentation mode agrees.
14. **The "New" ribbon defeated the selection test.** A freshly-obtained echo wears an
    orange badge in its top-right corner, which is gold, and which the corner test samples.
    Pooling four corners into one mean let it score **165.9** on one corner and 0.0 on the
    other three and still pass. This is bug #7 wearing a different hat. → the ring is the
    only thing that lights **all four** corners, so take the minimum, not the mean. Real
    selections score 28–44 on their weakest corner; both classes of false positive score
    **0.0** on theirs.
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
| `bench_fields.py` | **regression**: cost + level + selection over all 5 fixtures → 90/90, 90/90, 5/5 |
| `validate_e2e.py` | **regression**: stat icons + substat values vs gold labels → 21/21, 15/15 |
| `bench_values.py` | OCR engine shoot-out on substat value cells |
| `engines.py` | uniform wrappers: Tesseract, RapidOCR 1.x/3.x, Paddle, EasyOCR, OneOCR, WinRT |
| `fit_cost_masks.py` | refit `wuwa_scanner/templates/cost_*.png` from a labelled frame |
| `fetch_stat_icons.py` | download the 17 stat icons into `Data/Stats/` |
| `fetch_phantom_icons.py` | download the 38 phantom skins into `Data/EchoPhantoms/` |

`bench_fields.py` labels are **hand-read from the tiles, never derived from identity**.
Deriving cost from the identified echo would make it a tautology that passes by
construction while the reader rots, because identity is prefiltered *by* cost. It also
grades an abstain differently from a wrong answer: an abstained cost costs a full template
sweep, a wrong one deletes the true echo from its own pool.

`fit_cost_masks.py` refuses to run on a frame that does not contain all three costs. That
refusal is bug #8 and bug #11 encoded as a precondition rather than a comment.

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
| `bag_4k_04_mixed_level.jpg` | **the one that breaks things.** Mixed cost (1/3/4), mixed level (0–25), a `+17` with only 2 substats, two "New" overlays, and a different account (2668/3000) |
| `bag_4k_05_no_selection.jpg` | **nothing is selected**, and the panel still shows a cost-1 echo that is not on the page |

`bag_4k_04` is worth its own note: it is the only fixture carrying more than one cost or
more than one level, and adding it broke the cost reader (2/18), the level reader (which
did not exist), the `TILE_LEVEL` box, and the selection test. Everything the other four
prove, they prove about the easy case. **A fixture that agrees with every reader is not
pulling its weight.**

`bag_4k_05` exists for a state the click loop has to survive: scrolling leaves the
selection behind, so the panel can describe an echo that is nowhere in the grid. A reader
that trusts the panel without confirming the selection will happily attach one echo's
substats to another's identity.

## Next

1. **Tile census fields still to wire**: **lock** and **equipped portrait**. Level is done
   (90/90). Neither box is measured yet; both are visible on every fixture:
   - lock: a small dark badge with a white padlock, at the art's right edge just above the
     cost digit, roughly tile-space `(228, 180)–(272, 222)`. A presence test, not a read.
   - equipped: a rarity-framed character head in the tile's **top-left**, roughly
     `(18, 12)–(80, 75)`, about 62×62 at 4K. Presence of the portrait is the equipped flag
     — confirmed against `bag_4k_02`, where the selected tile has no portrait and the panel
     has no "Equipped by" line. Match it the way echo art is matched (gradient + hue), not
     the way `card.py` matches character splashes: those assets are 960×696 full-body art,
     the wrong crop for a 62 px head. `Characters.json` carries `icon.iconRound`, which is
     the right one, so this needs a `bench/fetch_character_icons.py` mirroring
     `fetch_phantom_icons.py` and its three-way URL rule.
2. **A second cost-1 capture**, from anywhere in the bag. `bag_4k_04` is the only frame
   with cost-1 tiles, so those masks are trained but never held out. See Risks.
3. **Re-sync `wuwabuilds/public/Data/Echoes.json`.** It reports no phantom skin for
   `60001945` (Reminiscence: Kronaclaw) but one exists in game, and probing the `SG_` naming
   convention across all 142 phantom-less echoes found no undocumented skins — so the table
   is stale, not the convention. A missing phantom template is a thin margin waiting to
   happen.
4. **More captures**: 1920×1080 (to verify the proportional bounds hold — note the glyph
   readers normalise to their own bounding box, so they should survive, but the level crop
   drops to ~14×16 px before upscaling and nothing has tested that), ultrawide (more
   columns), and a non-English client.
5. Then Phase 1.

Closed since the last revision: level is wired, cost 1 has tiles in a fixture, and the
sub-5-substat panel case is covered (`bag_4k_04`'s `+17` reads 2 substats correctly, with
the Echo Skill description visible and correctly ignored — that description only appears
when the stat block is short enough not to push it off-screen, which no earlier fixture
managed).

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

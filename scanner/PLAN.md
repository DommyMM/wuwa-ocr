# Echo scanner

Reads a player's Echo bag from the game UI and emits canonical JSON, for import into the WutheringTools optimizer (`wuthering-waves-optimizer/`) and later wuwa.build. [AGENTS.md](AGENTS.md) holds routing, guardrails and the regression commands. This file holds the design, the evidence behind each decision, and open work.

## Core idea

Stat names come from icons, so the scanner is barely an OCR problem. `Data/Stats.json` maps 20 stats onto 17 icons and every panel row renders its icon.

- Icon gives the stat family, with no OCR and no language dependence
- Number picks the member, since only HP, ATK and DEF share an icon with their percent form and each pair's legal ranges are disjoint (flat ATK 30-60 vs ATK% 6.4-11.6)
- `%` is never read because family and number imply it
- Identity, cost and sonata set come from tile templates, not text

Never infer a stat name from its value, because flat ATK 40 and flat DEF 40 are indistinguishable that way. The card import path hit exactly this ([echo-substats.md](../docs/echo-substats.md)). ATK and DEF have different icons, so the failure cannot occur here.

The OCR surface is digits only: up to five substat values per clicked echo and one level pill per tile. Digits are identical across all 9 client languages, so recognition carries no language data.

Main and innate values need no OCR either, because `Data/EchoStats.json` fixes them from cost, main stat and level.

## Tile vs panel

- Tile, no click: identity, cost, sonata set, level, selection
- Panel, one click: substats only
- On the tile but not yet read: lock badge and equipped portrait

Census the whole visible page first, then click only echoes worth reading. Echoes below +5 have no substats and the tile shows `+N`, so the census alone decides what to click. A real bag is hundreds of +25 echoes (`bag_4k_04` is a 2668/3000 bag whose level-sorted first page runs +25 down to +15 before the +0 fodder), so clicks dominate scan time.

Sort by level descending so the level floor is an early exit: the first tile below it ends the click pass because every later tile is lower. Level comes from the tile, so the exit costs no click, where Inventory Kamera clicks an item to learn it is too low.

## Recognition per field

| field | method | code | measured |
|---|---|---|---|
| identity | cost prefilter, gradient NCC, hue on near-ties, family-scoped badge | `identify.py`, `tile.py` | 24/24 on `bag_4k_01` |
| sonata set | set badge scoped to the identified echo's family | `tile.py` | 24/24 |
| cost | gold ink mask on its own bounding box, soft IoU against harvested masks | `glyphs.py` | 90/90 over 5 frames |
| level | Otsu ink, leftmost blob (`+`) dropped, Tesseract at 4x | `glyphs.py`, `ocr.py` | 90/90, levels 0 to 25 |
| stat names | 17-class icon mask IoU | `stats.py` | 21/21 |
| substat values | digits only, snapped to the stat's legal set | `stats.py`, `panel.py` | 15/15 |
| grid rows | gold-bar projection with a lattice fit | `grid.py` | exact, recovers missed bars |
| selection | gold hue per tile corner, minimum of the four | `grid.py` | 5/5 frames, incl. none selected |

### Identity leads the badge

Identity is read first and scopes the badge to the echo's family. `card.py` runs the other way only because its grayscale SIFT cannot see a recolor. Here gradient identity is the strong signal and the badge the weak one: a blind badge sweep over all 34 sets read 15/18, while the family-scoped read went 18/18 over 1.8 candidates on average. The blind errors were the harmful kind (Fleurdelys read as QuietSnow, a set it cannot roll), which as a prefilter would have deleted the true echo. A third of echoes have one legal set, so `data.determine_element` returns without touching pixels, and 74% are a one- or two-way call.

### Three signals cover every Nightmare family

Template-against-template scores per Nightmare pair show no family is blind to all three signals (high means the pair looks alike):

| family | gradient | hue | badge |
|---|---|---|---|
| Viridblaze, Baby Viridblaze, Dwarf Cassowary, Baby Roseshroom | blind (0.86-0.96) | blind (0.91-0.94) | decides |
| Crownless, Thundering Mephis, Inferno Rider | decides (0.06-0.30) | mixed | mute, sets match base |
| Feilian Beringal | blind (0.937) | decides alone (-0.106) | mute, sets match base |

The badge is mute on the four families whose sets match their base, which is where `card.py` gives up. Gradient or hue covers each of them. The five hard-family tiles in the captures, including two Phantom Nightmare Crownless and a Nightmare Feilian Beringal, were confirmed in game by hand.

### Gradient, not grayscale

Match on Sobel gradient magnitude because tile backgrounds differ in colour from the CDN templates (Frostbite Coleoid sits on light blue in the tile, dark teal in the template). Grayscale NCC tracked the background and read 2/3. A smooth background has near-zero gradient while the creature has strong edges, so gradient read 3/3.

### Hue arbitration

Gradient discards colour, so same-silhouette bodies tie: `Reminiscence: Fleurdelys` lost to `Leviathan` by 0.008. Hue histograms break near-ties, ported from `card.py::arbitrate_by_icon_hue`. When gradient cannot separate the top two and hue abstains, `tile.census` warns instead of guessing, which is how every hard tile surfaced.

### Cost is the only step that narrows the pool

Cost prefilters identity to recover a dead signal, not for speed. A washed-out Phantom Feilian Beringal matched Zip Zap, a cost-1 echo from an unrelated family, because phantom shimmer flattens the edges gradient needs and the full sweep collapsed into noise. Filtering to cost 4 put the Feilian pair back at ranks 1 and 2. Cost can never separate a Nightmare from its base, and the time it saves is nothing next to a click.

The prefilter is safe only because it abstains: an unknown cost sweeps the full pool, while a wrong cost deletes the true echo from its own pool. No other step may remove a candidate. An abstain gate is only as good as the score behind it: the diamond-correlating cost reader read a cost-3 tile as a confident 4 and returned Feilian Beringal at margin 0.002, where the ink-mask reader returns Spearback at 0.251.

### Phantom, Nightmare, Reminiscence

- Phantom shares its base echo's id, so matching the base id is the correct answer and the phantom flag is never detected
- Phantom and base tile art do not separate by hue (Phantom Fallacy 0.151 vs its non-phantom twin 0.191 against the same template), so detection would not work anyway
- Phantom art still shifts hue, and Feilian Beringal splits from its Nightmare by hue alone, so a phantom compared only against base art can flip to the Nightmare
- `identify.py` loads each skin in `Data/EchoPhantoms/` as a second template under the base id and scores best-of-variants, which removes that trap and widens margins (Phantom Nightmare Crownless 0.039 to 0.142)
- Nightmare echoes have their own ids and templates
- Reminiscence is part of the official name, so `tile._family_key` strips only the Nightmare prefix and `Reminiscence: Kronaclaw` never merges with a future base Kronaclaw

Never dedupe scanner output by content. Row and column are the identity, so two Fleurdelys with identical substats in different cells are two echoes. Content dedup belongs to the leaderboard, not here.

## OCR engines

Pick the engine per field. On correct crops every engine that ran read the substat values 5/5, so accuracy does not separate them and the choice is speed and packaging. Earlier apparent accuracy differences were all crop bugs.

| engine | speed | exe cost | role |
|---|---|---|---|
| WinRT (`Windows.Media.Ocr`) | fastest, ~20 ms per echo | none | substat values, `ocr.default_reader` |
| Tesseract, one process for N cells | next | ~10 MB | level pills (`ocr.level_reader`), value fallback |
| RapidOCR 3.x, recognition only | slower | ~50 MB | rejected |
| EasyOCR, recognition only | slowest | ~2 GB torch | rejected, unshippable |
| PaddleOCR | fails to run | ~1 GB | rejected, same models as RapidOCR |

Level pills need Tesseract because WinRT returns no lines on a one- or two-digit crop (0/18 where Tesseract reads 18/18). `bench/bench_values.py` prints current per-engine timings.

Three traps each changed the ranking:

1. Run recognition only. Cells are already localised, and detection on a small crop fails: RapidOCR read 2/7 with detection and 7/7 without.
2. Batch Tesseract through a file list. Each `pytesseract` call spends ~154 ms spawning a process that reloads `eng.traineddata`, while one invocation over N files still returns N results. `tesserocr` would keep engines warm but has no Python 3.13 wheel.
3. Leave out Surya and VLM readers, at 650M-7B parameters to read a short number while contending with the game for the GPU.

### Budget

Scan time is navigation-bound. Census of a 24-tile page (identity, set, cost) takes tens of ms and one Tesseract pass reads its level pills in a few hundred, while a clicked echo's WinRT reads take ~20 ms against an estimated 150-250 ms click and settle. Remaining optimisation belongs in settle detection and scroll correctness, and engine speed barely matters. This also retires the min-spec worry (i5-9400, GTX 1060, 16 GB). `bench/bench_census.py` and `bench/bench_fields.py` print current per-tile timings.

## Geometry

Three regimes, split by how much error each tolerates. Bounds live in `layout.py` as fractions of the 16:9 client rect, calibrated at 4K.

### Grid rows: detect every frame

The grid scrolls smoothly without snapping to rows, so a fixed row origin holds only at scroll-top. A wrong offset silently mis-crops every tile, which once made identity score 0/3 and look like a matcher failure. `grid.detect_lattice` finds each tile's gold bottom bar and fits a regular lattice through them, because one missed bar would shift every row below it (a frame with 2 detected bars recovered all 4 rows).

Only rows whose footer clears the sort bar are censused, because the bar hides the bottom row's badge and level. That row can still be clicked.

Inventory Kamera counts wheel ticks and scrolls back every ninth page instead. Do not copy that.

### Detail panel: self-locate

Panel glyphs are ~50 px at 4K, and a 1% horizontal shift of a hardcoded box silently took stat icons from 7/7 to 1/7. So `layout.PANEL_STATS` is the only fixed panel box, drawn generously, and `stats.py` derives everything finer:

- Icon column is the first contiguous ink run in the column projection
- Row centres are ink runs within that column
- Row bands are centre ± half the median pitch
- Rows whose icon match falls below the IoU floor are dropped, which rejects the Echo Skill heading and handles echoes with fewer than five substats
- Value cells locate their own ink and attach to the row band they overlap most

`PANEL_STATS` must start left of the stat icons. Slack on the left is free, while clipping the icons makes the locator abstain (loud) rather than guess (silent).

Hand measurement (icon column x 2655-2730, pitch 89.5) and the runtime projection (x 2653-2732, pitch 89) agree within ~2 px at 4K.

### Grid columns and tile internals: fixed boxes hold

Tiles are ~325x392 px at 4K, so hand-measured proportional boxes tolerate ±20 px.

- Every tile gets the unselected box, because the larger selected tile shifts ~4 px instead of growing about its centre and the art crop absorbs that
- Echo art is cropped square because the CDN templates are square, so aspect survives the resize
- Boxes over glyphs are fitted across every labelled tile, never one (see Lessons)

## Output contract

Canonical JSON in our ids is the source of truth, keyed by grid cell. The optimizer adapter is one exporter: `CalculatorEchoImporter.vue` already consumes a `ParsedEcho[]`, so the scanner writes that shape to a file. The optimizer's current import is an in-browser tesseract.js parse of a single 1920x1080 card, so a full-bag file is a strict upgrade. No upstream PR and no fork: we ship the file and the optimizer can link the tool.

Neither exporter exists yet. `py -m wuwa_scanner census` prints per-tile census JSON and `echo` prints one `panel.Echo` record. `Echo.main`, `innate`, `locked` and `equipped_by` stay empty, because `panel.read_substats` drops the main and innate rows and the tile's lock and equipped markers are not read.

## Phases

- Phase 0, recognition core: built, under regression
- Phase 1, watch mode: grid census by scroll with row-overlap stitching, plus passive capture while the player clicks, with a live "42 of 47 levelled echoes captured" readout and no input injection
- Phase 2, auto-navigate: opt-in click loop where a producer clicks and polls for panel stability and a worker pool recognises, recognition unchanged
- Phase 3, packaging: single exe with level floor, output path and progress

Decide the ship language before writing the navigation loop and GUI, gated on Phase 1. A small native binary (Rust with `windows-capture` and egui, or C# AOT) is lighter than Python with PyInstaller, but recognition depends on Tesseract for level pills and on OpenCV SIFT inside `data.determine_element` for same-hue badges, so a native port has to replace both.

## Risks

- Synthetic input is the real risk, not OCR: click automation breaks most gacha ToS and WuWa ships an anti-cheat, which is why watch mode ships first
- UI patches move the layout, mitigated by self-location and loud abstains rather than precise constants
- Sample size is 5 grid pages at one 4K resolution, one client language and two accounts
- 1920x1080 is untested, and there the level crop drops to ~14x16 px before upscaling (the other public WuWa scanner already supports it)
- Ultrawide is untested, and more columns mean `GRID_COLS` must be detected rather than fixed
- A non-English client is untested, and would prove language independence rather than argue it
- Identity is hand-labelled only on `bag_4k_01`, so the bench never checks who the hard-family tiles on the other captures are
- Cost-1 masks are trained on `bag_4k_04`, the only frame with cost-1 tiles, so they have no held-out test on a field already fixed three times
- Cost-1 identity is the weakest surface: `bag_4k_04` resolves Tick Tack (0.016), Frostscourge Stalker (0.021) and Baby Roseshroom (0.030) at coin-flip margins, and cost 1 is the largest bucket (85 of 181 echoes)
- Hard families (Crownless, Mephis, Feilian Beringal, Inferno Rider) resolve at 0.005-0.07 with hue abstaining on phantom-desaturated art, one UI patch from flipping
- `HUE_MIN_SCORE` and `HUE_MIN_MARGIN` come from `card.py`, whose crops are cleaner, and were never tuned for tiles
## Lessons from bugs

Every failure was silent and structural, and each came from a reader not measuring the thing its field names. None were found by reasoning: all surfaced when a reader ran on a frame it had not been tuned on. A field is not fixed until a frame it has never seen says so, and a fixture that agrees with every reader is not pulling its weight.

| symptom | root cause | rule |
|---|---|---|
| cost 18/18, then 0/6 on the first cost-3 row | box fitted on an all-cost-4 page sat on background that happened to correlate with "4" | fit on a frame holding every cost |
| tile-cut cost templates scored 0.715 but read 21/36 | template carried the artwork behind the digit | gate on the top-two margin, never absolute score |
| cost 2/18 on the first mixed-cost page | `Data/Costs` templates frame the digit in a diamond that dominated the correlation | mask the gold ink and compare shapes |
| `+25` read as 2 on 17 of 90 tiles | `TILE_LEVEL` fitted on one tile, so sub-row phase clipped glyphs elsewhere | sweep bounds over every labelled tile, sit mid-plateau, abstain when ink touches an edge |
| `+25` read as 425 | Tesseract reads `+` as 4 | drop the leftmost blob before OCR, never repair digits afterwards |
| one level in 90 misread | 2x upscale too small for the pill | 4x for level crops |
| ringless tile with a "New" ribbon passed selection | mean over four corners let one orange corner carry it | minimum over the corners |
| gold-artwork tile beat the selected one | selection tested border brightness | test the corners for gold hue |
| re-boxed selected tile dropped identity margin 0.367 to 0.130 | assumed centred growth, but the tile shifts 4 px, not 10 | every tile gets the unselected box |
| every tile mis-cropped after a scroll | fixed row origin | detect the row lattice per frame |
| Heavy Attack value clipped | band height from icon blob extent, and faint chevrons fell below Otsu | band is centre ± half the median pitch |
| a value dropped and every later row shifted | value column batched into one image | read each cell as its own image |
| icon column auto-locate 0/49 | located by component size, but Crit DMG is a star plus four detached arrows | first contiguous ink run |
| Crit Rate row read as Heavy Attack at IoU 0.34 | fixed y-band clipped the last row when a name wrapped | extend the band, reject rows below the IoU floor |
| `7.1%` read as 1770 | value cell inherited the icon's band on a wrapped row | values locate their own ink |

## Bench tooling

Run from `backend/scanner/`. Engines under evaluation live in the gitignored `.bench-venv` so the backend environment stays untouched, and only `ocr.py`'s WinRT and Tesseract paths are needed at runtime.

- `bench/bench_census.py`: identity and sonata through `tile.census` on `bag_4k_01`
- `bench/bench_fields.py`: cost, level and selection on every fixture through the shipped readers
- `bench/validate_e2e.py`: stat icons and substat values on the first three fixtures
- `bench/bench_values.py`: OCR engine shoot-out on value cells, run with `.bench-venv/Scripts/python.exe`
- `bench/engines.py`: uniform wrappers for every engine in the shoot-out
- `bench/fit_cost_masks.py`: refits `wuwa_scanner/templates/cost_*.png` from `bag_4k_04`, refusing a frame without all three costs
- `bench/fetch_phantom_icons.py`: mirrors phantom skins into `Data/EchoPhantoms/` from the frontend's `public/Data/Echoes.json`, since the backend copy carries no phantom icons

`bench_fields.py` fails only on a wrong cost, not an abstained one, because an abstain costs a full template sweep while a wrong cost deletes the true echo from its pool.

## Fixtures

`samples/` holds five labelled 4K captures stored as JPEG q95 (~1.6 MB each vs ~8.2 MB PNG). Every regression passes on the JPEGs, which also stands in for lossy real captures.

| fixture | why it exists |
|---|---|
| `bag_4k_01.jpg` | identity-labelled 24-tile page with 3 Phantoms, 3 Nightmares and 8 echoes that appear twice |
| `bag_4k_02.jpg` | two-line substat wrap and a flat DEF |
| `bag_4k_03_cost3.jpg` | cost 3, two consecutive wraps, an ATK% substat and innate ATK 100 |
| `bag_4k_04_mixed_level.jpg` | only frame with mixed costs (1/3/4) and levels (0-25), a +17 with 2 substats and the Echo Skill text visible, two "New" ribbons, a second account |
| `bag_4k_05_no_selection.jpg` | nothing selected while the panel shows a cost-1 echo absent from the page, because scrolling leaves the selection behind |

`bag_4k_05` guards the click loop: a reader that trusts the panel without confirming the selection attaches one echo's substats to another's identity.

## Next

1. Read lock and equipped from the tile. Lock is a presence test for the dark padlock badge at the art's right edge just above the cost digit (roughly tile-space (228, 180)-(272, 222) at 4K). Equipped is the ~62 px round character head in the top-left (roughly (18, 12)-(80, 75)), whose presence is the flag, confirmed on `bag_4k_02` where the selected tile has no portrait and the panel no "Equipped by" line. Match the head like echo art (gradient and hue) against `iconRound` from the frontend's `Characters.json`, not against `card.py`'s full-body splashes, which needs a character-icon fetch mirroring `fetch_phantom_icons.py`.
2. Fill `Echo.main` and `Echo.innate` from cost, the main-stat icon and level via `Data/EchoStats.json`.
3. Capture a second cost-1 frame so the cost-1 masks get a held-out test.
4. Complete the phantom skins. The frontend `Echoes.json` has no phantom icon for `Reminiscence: Kronaclaw` (60001945) though one exists in game, and probing the `SG_` naming across phantom-less echoes found no other undocumented skin, so the table is stale rather than the convention. `Data/EchoPhantoms/` also lacks `Myriad Snare: Rustfire Chassis` (60002175), which the frontend table has, so rerun `fetch_phantom_icons.py`.
5. Capture 1920x1080, ultrawide and a non-English client.
6. Then Phase 1.

## Entry points and packaging

`__main__.py` is the debug CLI (`py -m wuwa_scanner census|echo <frame>`). The shipped exe is watch mode (capture loop, live readout, JSON export), a different program with its own entry point. When Phase 1 lands, split it:

```
wuwa_scanner/
├── __main__.py   # shim: from .cli import main
├── cli.py        # debug commands, importable and testable
└── app.py        # the watcher, which PyInstaller targets
pyproject.toml    # [project.scripts] wuwa-scanner
```

Do not split earlier, because it is churn until a second entry point exists. Keep the empty `wuwa_scanner/__init__.py`, because namespace packages are a known source of PyInstaller module-discovery failures and `setuptools.find_packages()` skips directories without one.

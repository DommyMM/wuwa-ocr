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

Census a page of 24 tiles in ~52 ms, then click only the echoes worth clicking. A
2777/3000 bag has maybe 40 levelled echoes; everything below +5 has no substats.

## Identity: three signals, and the order is load-bearing

```
cost      -> prefilter the pool (180 -> 42/53/85). ABSTAINS rather than guesses.
identity  -> gradient NCC + hue on near-ties.      LEADS.
scope     -> union of legal sets across the identified echo's FAMILY.
badge     -> one scoped read: names the sonata set AND picks base-vs-Nightmare.
```

**card.py runs badge → identity. We run identity → badge.** Its SIFT is weak on recolors
(grayscale descriptors cannot see a repaint) so it needs an outside signal to fix identity.
Here the gradient matcher is strong and the tile badge is weak: a blind 34-way badge sweep
scores 15/18, and its errors are poisonous — it read Fleurdelys as QuietSnow, a set
Fleurdelys cannot even roll, which as a hard filter would delete the true echo from its own
pool. Scoping the badge to the identified echo's family drops it from 34 candidates to 1.8,
which is **both faster and strictly more accurate**: the candidates it removes are precisely
the ones that produced every error. A third of echoes have one legal set, so
`determine_element` short-circuits and reads no pixels. Scoped: 18/18, 0.66 ms. Blind:
15/18, 4.8 ms.

Scoring every Nightmare pair template-against-template shows **no family is blind to all
three signals** — the coverage is structural, not luck:

| | gradient | hue | badge |
|---|---|---|---|
| Viridblaze Saurian, Baby Viridblaze, Dwarf Cassowary, Baby Roseshroom | blind (0.86–0.96) | blind (0.91–0.94) | **carries it** (disjoint sets) |
| Crownless, Thundering Mephis, Inferno Rider | **carries it** (0.06–0.30) | mixed | mute (sets identical to base) |
| Feilian Beringal | blind (0.937) | **carries it alone** (−0.106) | mute (sets identical to base) |

Cost never separates a Nightmare — variants always share their base's cost — but it is not
optional. A washed-out Phantom Feilian matched **Zip Zap**, a cost-1 echo from an unrelated
family, because the phantom shimmer flattens the edges gradient depends on and every score
collapsed to noise. Filtering to cost 4 puts the Feilian pair back at ranks 1 and 2.

## Files

```
scanner/
├── AGENTS.md          # this file
├── PLAN.md            # direction, measured results, bug log
├── samples/           # 3 labelled 4K captures (JPEG q95); the regression fixtures
├── bench/
│   ├── bench_census.py   # REGRESSION: identity + sonata over a labelled page -> 24/24, 24/24
│   ├── validate_e2e.py   # REGRESSION: stat icons + substat values -> 21/21, 15/15
│   ├── bench_values.py   # OCR engine shoot-out
│   ├── engines.py        # Tesseract / RapidOCR / Paddle / EasyOCR / OneOCR / WinRT wrappers
│   ├── fetch_stat_icons.py     # -> Data/Stats/ (17 icons)
│   └── fetch_phantom_icons.py  # -> Data/EchoPhantoms/ (38 skins)
└── wuwa_scanner/
    ├── layout.py      # proportional bounds; THREE precision regimes (read the docstring)
    ├── grid.py        # per-frame row lattice + selection detection
    ├── identify.py    # echo identity: gradient NCC + hue, phantom art as a same-id variant
    ├── tile.py        # the census ladder: cost -> identity -> family-scoped sonata badge
    ├── stats.py       # self-locating stat rows + 17-class icon match
    ├── ocr.py         # the only OCR: read a number (WinRT primary, Tesseract fallback)
    ├── panel.py       # tile census + panel substats -> Echo record
    └── __main__.py    # CLI: `census` and `echo`
```

Run **both regressions** after any change to layout, geometry or matching. They exercise
the shipped code path, not a copy of it:

```
py bench/bench_census.py      # -> identity: 24/24   sonata: 24/24
py bench/validate_e2e.py      # -> icons: 21/21      substat values: 15/15
```

`fetch_phantom_icons.py` must resolve URLs through the frontend's three-way `toImageUrl`
rule (`wuwabuilds/lib/echo.ts`): `/d/` → Wuthery, `/Game/` → encore, absolute → as-is.
Newly-shipped echoes are not on Wuthery yet and carry an absolute encore URL, so blindly
prefixing the CDN base silently produces `https://files.wuthery.comhttps://api.encore...`.

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

8. **Only cost may filter the candidate pool, and only by abstaining.** Every other signal
   confirms. A filter that guesses can delete the true echo from its own pool and the miss
   is then unrecoverable and silent — which is exactly what a blind badge sweep did
   (Fleurdelys → QuietSnow, a set it cannot roll). Cost is the one exception because an
   unknown cost degrades to the full sweep, so it can only ever *fail to help*.

9. **High template score is not correctness.** Cost templates cropped from a tile score
   0.715 mean and are wrong 15/36 times; card.py's clean glyph templates score 0.127 and are
   right 36/36. The tile-native ones carry the artwork behind the digit and correlate on the
   creature. Gate on the **margin between the top two**, never on absolute score.

## Why gradient, and why hue

Match on **Sobel gradient magnitude**, not grayscale. The tile background is a soft
gradient whose colour differs from the CDN template's, and grayscale NCC is dominated by
it. A smooth background has near-zero gradient; the creature has strong edges.

But gradient is background-invariant *because it discards colour*, so same-silhouette
bodies near-tie. **Break ties with hue** (`identify.TIE_MARGIN`), a direct port of
`card.py::arbitrate_by_icon_hue`.

When gradient cannot separate the top two **and** hue abstains, the answer is a coin flip
and `tile.census` says so in `warnings`. Do not silence that: it is the only thing standing
between a hard family and a confident lie.

## Name prefixes

- **`Phantom:`** — shares the base echo's canonical id (there is no `Phantom:` entry in
  `Echoes.json`), so matching a Phantom to its base id is **correct**, and the flag is
  cosmetic (same cost, same legal sets, same stat pools) so we do not detect it. Measured,
  not assumed: hue-vs-own-template scores a Phantom Fallacy 0.151 and its base twin 0.191 —
  no separation, and the tile art is visually identical.

  The phantom **art** still matters. A Phantom is a *recolor*, so its hue shifts away from
  the base template — and Feilian Beringal is separated from its Nightmare by hue alone.
  So `Data/EchoPhantoms/` (38 skins) is loaded as a **second template under the same id**
  and each id scores best-of-variants. This removes the trap rather than detecting it, and
  it is worth real margin: Phantom Reactor Husk went 0.104 → 0.447, Phantom Fallacy
  0.222 → 0.537, Phantom Nightmare Crownless 0.039 → 0.142.
- **`Nightmare:`** — its own id and its own template. See the coverage table above.
- **`Reminiscence:`** — part of the official name, not a prefix family.

## Duplicates

Row × column **is** the identity. Two Fleurdelys in different cells are two echoes, and if
they somehow carry identical substats they are still two echoes. Never dedupe scanner output
by content — key it by cell.

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

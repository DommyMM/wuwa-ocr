# OCR recognition roadmap

This backend should be treated as a fixed-layout recognition service, not a
general OCR service. The import image format is constrained, the crop geometry
is stable, and most labels come from small finite game-data vocabularies.

## Attempted but not adopted — Tesseract-only echo substat OCR (first attempt)

> **Superseded (2026-06).** This documents the *first* Tesseract-only attempt,
> which was rolled back because of two flat-ATK regressions (see "Why we didn't
> adopt it"). A second, improved attempt in
> [echo-substat-tesseract-only.md](echo-substat-tesseract-only.md) addresses exactly
> that dealbreaker — it fixes the `%`-suffix deterministically by stat type and
> arbitrates digit reads against the closed legal-value set instead of inferring the
> name from the value, so the flat-ATK failure mode below cannot occur — and validated
> at 97.6% identical / <1% delta on up to 800 cards. That doc is the current direction.
> Note the motivation also shifted: cost (~$7/mo) is acceptable; the point is that
> RapidOCR pins ~92% of RAM (and a latency tail) for a fallback that fires on ~25% of
> echoes. The analysis below is kept for the failure modes it documents.

A full Tesseract-only replacement for the echo substat OCR path was prototyped
end-to-end and benchmarked, then **rolled back** rather than adopted. The
production runtime was unchanged at the time. This section documents what was tried, the
exact result, and why it didn't ship, so a future attempt can either pick up
where this left off or know to skip the path entirely.

### What the prototype did

- Removed `rapidocr-onnxruntime` from `requirements.txt`; wrapped the import
  in try/except (`Rapid = None` when absent); guarded `process_ocr` so
  character and weapon fall back to Tesseract.
- Replaced the single block-OCR pass over `subs_names` / `subs_values` with
  per-row Tesseract calls in a `SUBSTAT_NAME_ROWS` / `SUBSTAT_VALUE_ROWS`
  loop. Names used PSM 6 with a best-fuzzy-match line picker against
  `SUB_STAT_NAMES` to handle the "Resonance Liberation DMG Bonus" 2-line
  wrap from Wuwa Bot cards. Values used a PSM 7→8→6 cascade on plain-gray
  (then Otsu-thresholded) input, with a regex token cleanup for trailing
  `.` artifacts and a Levenshtein-distance-≤-1 snap to the nearest legal
  value when the cascade returned an illegal one.
- Added a value-driven stat-name inference fallback: when name OCR was
  garbage (low fuzz score against any known stat) and the value was legal
  for exactly one stat, override the name. Recovered flat-HP rows where
  the small 2-char `HP` text reads as `a1` / `Aap` but the numeric value
  (`320`–`580`) uniquely identifies the stat.
- Bypassed `parse_region_text`'s `rsplit(' ', 1)` for substats (kept name
  and value as separate strings end-to-end so trailing-garbage tokens in
  the name couldn't leak into the value field).
- Dropped empty/non-numeric rows so under-leveled echoes with <5 substats
  don't fabricate fake fifth rows.

### Validation, 200-image apples-to-apples comparison

Two git worktrees side by side (old block-OCR + RapidOCR vs. the prototype)
processed the same 200 randomly-sampled English cards from `r2-backup/`:

| Outcome | Count |
|---|---:|
| Images identical between old and new | 135 / 200 (67.5%) |
| Total substat-level diffs | 145 |
| **NEW legal, OLD illegal** (improvement) | 14 |
| **NEW illegal, OLD legal** (regression) | **2** |
| Both legal but different (ambiguous) | 98 |
| Both illegal | 7 |

Net: +12 substats more accurate, but **both regressions are on flat ATK
substats** — `ATK 40` and `ATK 50` were misread as `Crit Rate=40` and
`Heavy Attack DMG Bonus=50`. Flat ATK / DEF cannot be uniquely
disambiguated by value-driven inference (`40` is legal for both `ATK` and
`DEF`), so when name OCR returns garbage with high enough fuzz confidence
to skip inference, we silently pick the wrong stat.

### Why we didn't adopt it

The current Railway cost is ~$10/month, which is acceptable. ATK and HP
are critical short stats whose accuracy directly affects every build's
calculated damage. **Any regression on short flat substats is a
dealbreaker, even when the aggregate metric improves.** A net-positive
substat count doesn't compensate for silently mis-attributing flat-ATK
rolls to other stats — the wrong value still passes downstream
calculation and shows up in builds as a confidently-wrong number.

The path back to "consider adopting" would require:
- Extending `_infer_stat_from_value` to use a fuzz-score tiebreaker
  between flat ATK / DEF when both are legal candidates.
- Re-validating the result against a hand-audited gold set (not the
  legality-only smoke test that wouldn't catch "wrong stat, legal value
  for wrong stat").
- Zero regressions on flat ATK / HP / DEF rows specifically.

The second attempt ([echo-substat-tesseract-only.md](echo-substat-tesseract-only.md)) takes
that third requirement seriously: it never *infers* the flat-ATK name from the value, so the
regression mode above cannot recur. As of 2026-06 that path is committed but **not deployed**
— RapidOCR is still loaded and called as a fallback in `data.py` / `card.py` — and a full
~12k regression is still pending before RapidOCR is actually removed.

### Other findings worth keeping

- The Wuwa Bot 2-line wrap of `Resonance Liberation DMG Bonus` /
  `Resonance Skill DMG Bonus` IS handled in the current `card.py` via
  `clean_echo_substat_name_lines`. The prototype's per-row design needed
  a separate fuzz-line-picker to reproduce that.
- Under-leveled echoes (<5 substats) are still zipped against the OCR'd
  names list in current `card.py` (`process_card` → `zip(cleaned_names,
  values_lines[:5])`), so missing rows produce fewer substats rather than
  fabricating five. The original "fabricated 5 substats" complaint no
  longer applies, but trailing-garbage rows can still slip through.
- `parse_region_text`'s `rsplit(' ', 1)` on `"<name> <value>"` strings is
  fragile to trailing-garbage tokens in the name. Worth a refactor
  independent of any OCR engine change.
- Main-stat value OCR is dead code in the current pipeline — `process_card`
  calls `max_main_stat_value(cost, main_name)` from `ECHO_COSTS` and
  overrides the OCR'd value. Could be deleted.
- The previous crash when `parse_region_text` returned `[]` for a low-
  confidence echo is now guarded by `isinstance(echo_data, dict)` before
  the `.get` calls in `process_card`. Status: shipped.

## Current finding

The Hiyuki/Frostburn reference card confirmed the key observation:

- The character splash is a fixed game asset at a fixed card position.
- The weapon icon is a fixed game asset at a fixed card position.
- Echo identity, echo cost, and echo element are already template or color
  recognition problems.
- Echo main-stat values are already overridden from `EchoStats.json` after cost
  and stat-name detection, so main-stat value OCR is dead code.
- The only truly free-form text is the watermark username. UID is digits only.

The expensive runtime path today is not a lack of crop constraints. It is that
RapidOCR and Tesseract are loaded into multiple workers to solve problems that
are mostly known-class recognition.

## Transport and endpoint direction

The original split-region browser flow made sense when the client was producing
PNG crops and the backend had to keep region work parallel. The current corpus
changes that tradeoff: import images are already overwhelmingly small JPEGs.

Local `r2-backup/` size snapshot from 11,443 images:

| Metric | Value |
|---|---:|
| JPEG/JPG files | 11,400 |
| PNG files | 43 |
| Average source size | 359,674 bytes |
| Median source size | 326,790 bytes |
| p90 source size | 456,098 bytes |
| p95 source size | 467,838 bytes |
| Files over 500 KB | 92 |
| Files over 1 MB | 70 |
| Max source size | 2,374,908 bytes |

Because the usual source file is already only ~300-450 KB, the frontend should
not crop regions into PNG payloads for the hot OCR path. In a local packaging
bench, the current cropped-PNG set averaged ~1.55 MB, while a full-card JPEG
q85/q92 averaged ~304-377 KB. Sending the original file is usually smaller than
sending all region crops.

Current import transport:

```text
browser
  `- POST original image bytes directly to OCR backend through the Cloudflare gateway

OCR backend
  |- compute content hash / training key
  |- decode once
  |- crop internally
  |- run cheap deterministic recognizers inline
  `- fan out expensive recognition work inside the service
```

Do not use R2 as the OCR transport. Uploading to R2 and then having the OCR
backend download the image adds a network hop and makes the import wait on
training storage. R2 remains useful for durable training data and issue reports,
but OCR should consume the bytes the browser already sent to the OCR request.

The cleanest future shape is a single OCR upload that also schedules the R2
training-image save. The backend can hash the raw bytes immediately, include the
training key in the response, and run the R2 write as a background/best-effort
task. If the R2 write fails, OCR should still return normally and reports can
fall back to attaching the image or retrying the save.

An acceptable transitional shape is two parallel browser requests using the
same `File`: one to OCR, one to the existing Vercel/R2 training route. This is
simple but uploads the image twice and keeps Vercel in the storage path. It is
still better than R2-as-transport because OCR does not wait for a later R2
download. This is the current production shape: OCR consumes the original file
directly, while the normal import page still saves the full image through the
existing training-image route for reports/backfill.

Do not proxy the OCR image through Vercel Functions. The OCR hot path should
stay browser to `ocr.wuwa.build` so Vercel does not pay function CPU/memory or
Fast Origin Transfer for image uploads.

Implemented endpoint shape:

- The public OCR contract is now a full-image endpoint. The old crop-header
  flow has been deleted from the frontend/gateway contract.
- `POST /api/ocr` accepts a full card image and returns the full import
  analysis.
- Accept raw binary or `multipart/form-data`, not JSON base64.
- Decode once with OpenCV, validate dimensions/layout once, then crop backend
  side.
- Preserve parallelism inside the backend. A single endpoint must not mean a
  single sequential recognition pass.
- The Cloudflare gateway no longer requires `X-OCR-Region`, caps OCR uploads at
  5 MB, and uses a full-import timeout.
- The OCR endpoint always streams `application/x-ndjson`: `meta`, per-`region`,
  and final `done` events keyed by region. Interactive import uses the region
  events for live progress; bulk import uses the same parser and consumes the
  final `done` payload.

Local architecture benchmark (`backend/bench_import_architecture.py`) compared
the old cropped-request shape with a simulated full-image endpoint while keeping
today's recognizers unchanged:

| Shape | Avg total | Avg recognition | Avg payload |
|---|---:|---:|---:|
| Current split crops | ~9.82 s | ~9.73 s | ~1.56 MB |
| Full JPEG endpoint simulation | ~9.33 s | ~9.28 s | ~382 KB |

Takeaway: endpoint packaging alone is not the main speed win. It removes waste
and simplifies orchestration, but the seconds are still in recognition. The
large win comes from replacing character/weapon OCR with finite asset matching
and later reducing echo OCR.

Backend decode/crop overhead is negligible relative to recognition. A 1,000-file
local source-image bench measured:

| Step | Average | Median | p95 |
|---|---:|---:|---:|
| Read source bytes from disk | 0.51 ms | 0.49 ms | 0.64 ms |
| OpenCV decode | 5.67 ms | 5.36 ms | 7.58 ms |
| Crop all 10 current regions | 0.02 ms | 0.02 ms | 0.03 ms |
| Decode plus all crops | 5.69 ms | 5.39 ms | 7.60 ms |

In production, network and recognition dominate. Backend-side cropping is cheap
enough to ignore in the optimization budget.

A headless Chrome benchmark (`backend/bench_frontend_canvas.mjs`) of the current
frontend-style work on 197 readable source files measured the browser cost
directly:

| Browser step | Average | Median | p95 |
|---|---:|---:|---:|
| Decode original image | 8.36 ms | 7.90 ms | 11.00 ms |
| Crop 10 regions and PNG/base64 encode | 55.64 ms | 54.20 ms | 61.50 ms |
| Build `FormData` with original `File` | 0.36 ms | 0.30 ms | 0.70 ms |
| Re-encode full image as JPEG/base64 for training upload | 18.20 ms | 18.00 ms | 20.70 ms |
| Hash original bytes in Node | 0.24 ms | 0.19 ms | 0.46 ms |

Payloads from the same browser run:

| Payload | Average | Median | p95 |
|---|---:|---:|---:|
| Original source file | 353,127 bytes | 325,705 bytes | 457,344 bytes |
| Current 10 cropped PNG payloads | 1,520,684 bytes | 1,520,517 bytes | 1,577,487 bytes |
| Full JPEG/base64 training payload | 325,820 bytes | 323,214 bytes | 338,958 bytes |

Matching backend-side measurements on the same first-200-file slice:

| Backend step | Average | Median | p95 |
|---|---:|---:|---:|
| Read source bytes from disk | 0.32 ms | 0.29 ms | 0.57 ms |
| Hash original bytes | 0.20 ms | 0.16 ms | 0.32 ms |
| OpenCV decode | 8.53 ms | 8.02 ms | 11.24 ms |
| Crop all 10 current regions | 0.27 ms | 0.23 ms | 0.45 ms |
| Decode plus all crops | 8.80 ms | 8.31 ms | 11.46 ms |

This makes the browser/backend tradeoff straightforward: moving crops backend
side removes roughly 55 ms of browser CPU and about 1.2 MB of upload payload in
the normal JPEG case, while adding less than 1 ms of backend crop work after the
decode the backend needs anyway.

## Region architecture

| Region | Target method | Notes |
|---|---|---|
| character | SIFT vs `Data/Characters/<id>.webp` | Crop `tight` `(0.04, 0.14, 0.30, 0.52)`, downscale query+templates to max_side 150. ~185 ms (dev). Conf floor ~0.10 → OCR fallback. Splash always renders incl. newest characters; language-independent, so it beats OCR on non-English cards. See Phase 1 status for the 500-card validation. |
| weapon | SIFT vs `Data/Weapons/<id>.webp` | Crop `icon` `(0.752, 0.400, 0.802, 0.530)`, downscale to max_side 120. ~143 ms (dev). Needs conf floor ~0.08 **and** margin floor ~0.03 (several icons look alike). Blank panel → conf ~0 → empty (see "Missing weapon assets"); empty is correct. |
| watermark UID | Tesseract digits-only | Use `tessedit_char_whitelist=0123456789`. |
| watermark username | Tesseract | Leave free-form and Unicode-capable for now. |
| character level | Optional Tesseract digits/template | Detect the gold LV badge by HSV in the header. Parse only plausible 1-90 values; default/null is acceptable when unreadable. |
| sequences | HSV pixel ratio | Existing method is already deterministic. |
| forte | small fixed classifier or digit templates | Five `LV.X/10` regions, classes 1-10. |
| echo icon | SIFT | Existing method. |
| echo cost | template match | Existing method. |
| echo element | HSV histogram, SIFT fallback within same hue cluster (`data.py` `determine_element`) | Existing method. Grayscale elements (ER, Tidebreaking) and same-hue pairs (e.g. Gust/Windward, Pact/Rite, Trailblazing/Chromatic/Flamewing, Midnight/Dream/Thread) resolve via SIFT. |
| echo main stat name | small classifier or template matcher | Finite class set from `EchoStats.json`. |
| echo main stat value | derive from cost and stat name | Do not OCR. |
| echo substat rows | small row classifier | Predict row `(name, value)` or two heads. |

### Missing weapon assets

Some cards render a blank weapon panel: no weapon icon and no weapon name, only
the `LV.xx` level text and the ascension stars. This is reproducible on the
current new-character cards Lucilla (`1109`), Lucy (`1511`), and Rebecca
(`1308`), whose weapon art is absent from the export while their character
splash renders normally. Verified against local reference images for those
three characters; Zani (`1507`) is the control with a fully rendered Blazing
Justice panel. Do not commit raw local/R2 image keys for these fixtures.

No recognizer can read a weapon that is not drawn. Weapon SIFT must return an
empty/no-match result for these, exactly as the OCR path now does (the weapon
name falls below the fuzz cutoff and resolves to `""`). The frontend owns the
recovery: `wuwabuilds/lib/import/convert.ts` `IMPORT_WEAPON_FALLBACKS` maps the
character id to its signature weapon (Lucilla -> Freeze Frame, Lucy -> Spectral
Trigger, Rebecca -> Skull Thrasher) and only applies when the backend reports an
empty weapon. Switching weapon recognition to SIFT does not remove this fallback
and must preserve the empty-weapon contract it keys on.

### Echo element fallback note

One player report exposed a
sonata-element edge case on echo5 (`Flora Drone`). Echo identity was correct,
but the `Rite` element crop had too few SIFT keypoints. The old fallback
accepted an all-zero SIFT result and returned an arbitrary candidate by
iteration order, producing `Memories` even though HSV had already rejected that
color family.

Current runtime behavior keeps HSV as the coarse filter, only accepts SIFT when
the best SIFT score is positive, and otherwise falls back to direct color
template comparison among close same-hue candidates. Longer term, echo elements
should move toward deterministic template/mask matching; SIFT is a better fit
for textured echo monster icons than for small circular sonata glyphs.

## Template source and format

Two CDNs are available for template assets: Wuthery (`files.wuthery.com`, PNG)
and Encore (`api-v2.encore.moe`, WebP). Testing on the live echo and element
template sets showed WebP is sufficient for SIFT and color matching — neither
needs PNG — so the standard template format is WebP, and Encore is the preferred
single source where it serves the asset.

- Echoes and elements: Encore/WebP, already proven in the running pipeline
  (`Data/Echoes` is 163 WebP templates, `Data/Elements` is 31).
- Character splash: Encore serves the `IconRolePile` splash as
  `FormationRoleCard` on the per-character **detail** endpoint
  (`/api/en/character/{id}`), e.g.
  `.../IconRolePile/T_IconRole_Pile_<codename>_UI.webp` (confirmed for Hiyuki,
  Yangyang, Lucilla). The list endpoint (`/api/en/character`) only carries
  `RoleHeadIcon` (the round 150px head), so splash fetch needs one detail call
  per character. The pile filename uses an internal codename, not the numeric
  id, so save it locally as `<id>.webp`.
- Weapon icon: Encore's list endpoint `Icon` is the full unique
  `T_IconWeapon<id>_UI.webp` — the full icon to use, not `iconMiddle`. Several
  weapon pairs share an identical `iconMiddle`, which makes SIFT ambiguous; the
  full icons are distinct. One list call covers every weapon.

The three `download_*_icons.py` scripts plus `sync_backend.py`'s element-icon
step are ~90% identical (resolve a URL from a data field, threadpool-download to
an `<id>.<ext>` file, with `--force`/`--dry-run`). They should collapse into one
declarative sync driver with a per-asset-type entry (source JSON or API, URL
resolver, destination dir). Folding it into `sync_backend.py` lets one command
refresh both the JSON vocabularies and every template set.

## Phase 1 status

Phase 1 is character and weapon asset recognition.

The primary motivation is speed: SIFT against a fixed game asset is faster and
cheaper than running Tesseract/RapidOCR on the name and weapon strips, and it
drops the dual-engine character path entirely. Robustness is a secondary bonus.
Missing-asset handling is uniform across both fields — whatever the recognizer
cannot read (a blank weapon panel, or an unreadable name strip) resolves to
empty, and the frontend fallback fills it. The character splash renders even for
the newest characters, so SIFT recovers identity where the name OCR would not;
the weapon panel can be genuinely blank (see "Missing weapon assets"), and that
empty result is correct.

**Character and weapon now recognize via SIFT** in `card.py`
(`recognize_character_asset` / `recognize_weapon_asset`, routed from
`process_card`), each with an OCR fallback on abstain. The case below was
validated on a 500-card `r2-backup` sample (`bench_char_crops.py`,
`bench_weapon_crops.py`) with the OCR result as the comparison baseline. The
local→Railway gap is part of the motivation: SIFT scales ~1x dev→Railway while
RapidOCR/onnx scales ~2.5x (8-vCPU thread oversubscription), and character OCR
was the previous import wall (~2 s on Railway). Crop+downscale overhead is ~1 ms,
negligible. Note: a SIFT accept reports character/weapon **level 90** (not in the
splash/icon); the abstain→OCR path still reads the true level for the rarer
non-90 cards.

Assets in tree: `Data/Characters/` (56 splash WebP) and `Data/Weapons/` (118 icon
WebP), Encore-sourced, committed with the backend.

### Character — validated, recommended

- Crop **`tight` `(0.04, 0.14, 0.30, 0.52)`**, downscale query+templates to
  **max_side 150**, SIFT vs `Data/Characters/<id>.webp`.
- Speed: ~185 ms median match (dev). Downscale is the whole speed lever:
  full-res ~1.9 s collapses to ~185 ms at 150px with **no accuracy loss**.
  `tight@150` beats `@120/@100` because the Luuk Herssen (`1510`) splash is
  correct at 150 but flips at the aggressive downscales (thin margin).
- 500-card agreement with OCR: **96.4%**, but this *undercounts* SIFT. Every
  consistent disagreement was reviewed: the 5 high-confidence ones are Japanese
  cards where OCR misread the name and **SIFT was correct** — Phrolova
  (フローヴァ) read by OCR as Jianxin, Denia (ダーニャ) as Hiyuki, Camellya
  (ツバキ) as Encore, Lupa (ルパ) as Lumi, plus a Cartethyia case. SIFT is
  language-independent, so it is **more accurate than the OCR baseline**.
- The remaining disagreements are low-confidence (conf < 0.06): Rover variants
  (shared banners by gender) and a non-card roster screenshot. A **confidence
  floor ~0.10 with OCR fallback** turns these into clean abstentions, not errors.

### Weapon — validated, needs a margin gate

- Crop **`icon` `(0.752, 0.400, 0.802, 0.530)`**, downscale to **max_side 120**,
  SIFT vs `Data/Weapons/<id>.webp`. ~143 ms median (dev).
- Harder than character: weapon icons are small and several look alike, so real
  matches score lower (~0.10-0.18) and a handful are genuinely ambiguous.
- 500-card: 475 OCR-read + 25 OCR-empty. Of the 25 empty, ~11 are **true blank
  panels** (SIFT conf 0.000 → correct empty), ~8 are **non-English rendered
  panels** OCR missed but SIFT read at conf 0.10-0.18 (SIFT wins again, e.g.
  Phrolova→Lethean Elegy, Lupa→Wildfire Mark), the rest low-conf ambiguous.
- 6/475 confident-ish disagreements (Emerald Sentence→Originite Type IV ×3,
  Red Spring→Hollow Mirage ×2, Commando of Conviction→Pistols#26) — but **all
  have margin < 0.02** (top-2 nearly tied). Weapon therefore needs **both a conf
  floor (~0.08) AND a margin floor (~0.03)** to abstain; conf alone is not
  enough. The empty contract holds cleanly (true blanks at conf 0.000), and the
  roster-screenshot false positive (conf 0.122) is caught by the margin gate
  (margin 0.027).

### Wiring — shipped (2026-06)

Character and weapon SIFT (`recognize_character_asset` / `recognize_weapon_asset`)
are wired into `card.py` with an OCR fallback on abstain, as of the 3.4 import work
(`r2-backup` validation against the OCR baseline). The bullets below record the
design and the floors used; the calibration note remains the tuning lever.

- Recognize character/weapon by SIFT with the crops/downscales above; on abstain
  (below conf floor, or weapon below margin floor) fall back to the existing OCR
  path, or return empty for weapon to preserve the signature-weapon fallback
  contract (see "Missing weapon assets").
- Load templates downscaled to the match max_side (scale-consistent matching is
  what makes the speed win real).
- Calibrate the floors on a larger slice before shipping; 500 cards points at
  conf ~0.10 (character) and conf ~0.08 + margin ~0.03 (weapon). Rover variants
  should abstain → OCR (or a base-Rover/element fallback) rather than guess.
- A Rover splash match identifies gender only. An explicit element suffix in
  the title is authoritative; older Aero/Spectro cards that say only `Rover`
  fall back to the saturated foreground hue of the character badge. Do not
  reuse the Sonata badge matcher here: its broader histogram mask includes the
  purple card header, which can make Aero and Spectro badges look Electro.

Phase 1 acceptance remains:

- Character agreement with current pipeline is at least 99 percent on valid
  1920x1080 cards.
- Weapon agreement with current pipeline is at least 99 percent on valid
  1920x1080 cards.
- Every disagreement is manually reviewed, because the old OCR baseline can be
  wrong too.
- Cards with a blank weapon panel count as correct when weapon SIFT returns no
  match; they must not be scored as weapon misses. See "Missing weapon assets".

## Phase 2 goal

Phase 2 optimizes crop geometry and preprocessing before training any OCR-like
model. The manually validated coordinates are good enough to ship, but they are
not proven optimal.

The optimization pass should answer:

- Which full-image import payload shape should be kept for production? Default
  answer after the latest corpus check is "send the original image file", not
  client PNG crops.
- Which crop box gives the best SIFT agreement/margin for character and weapon?
- Which echo icon crop gives the best echo ID confidence without hurting cost
  or element follow-up?
- Which watermark UID crop gives the highest exact digit-match rate?
- Whether username still needs to be OCR-read automatically, or only retained
  for report metadata/manual confirmation.
- Whether character level should be parsed at all, and if so whether the HSV
  gold-badge detector is good enough. Level is lower priority than character,
  weapon, UID, and echo correctness.
- Which main-stat-name crop is easiest to classify without reading the value?
- Which substat row crop best isolates one row for a future row model?
- Which forte digit crop isolates level text reliably enough for template or
  tiny-classifier recognition?

The RTX 5090 is most useful after the crop sweep, for training/evaluating the
main-stat, forte, and substat classifiers. The sweep itself is mostly CPU-bound
OpenCV/Tesseract work.

## Gold-label workflow

Use a small explicit gold file for crop optimization. Do not optimize only on
one screenshot.

Suggested label shape:

```json
{
  "images": [
    {
      "file": "fixtures/hiyuki-frostburn-card.jpeg",
      "character": { "id": "1108", "name": "Hiyuki" },
      "weapon": { "id": "21020086", "name": "Frostburn" },
      "watermark": { "uid": "<sample-uid>", "username": "<sample-user>" },
      "forte": [10, 10, 10, 10, 10],
      "echoes": {
        "echo1": {
          "id": "60001995",
          "element": "QuietSnow",
          "main": { "name": "ATK%", "value": "33%" },
          "substats": [
            { "name": "ATK%", "value": "10.1%" },
            { "name": "DEF%", "value": "12.8%" },
            { "name": "Crit Rate", "value": "6.9%" },
            { "name": "Crit DMG", "value": "21%" },
            { "name": "Resonance Liberation DMG Bonus", "value": "8.6%" }
          ]
        }
      }
    }
  ]
}
```

Relative `file` values are resolved against `--image-root` in
`optimize_crops.py`.

## Reference image labels

For the attached Hiyuki/Frostburn card:

- Character: `1108` / Hiyuki
- Weapon: `21020086` / Frostburn
- Watermark UID: sample UID from the fixture metadata
- Echo 1: `60001995`, QuietSnow, ATK%, 33%
- Echo 2: `60001875`, QuietSnow, Glacio DMG, 30%
- Echo 3: `60001839`, QuietSnow, ATK%, 30%
- Echo 4: `60001975`, QuietSnow, ATK%, 18%
- Echo 5: `60001965`, QuietSnow, ATK%, 18%

Substat rows from current logs are expected to be correct for this card and are
good seed labels, but they should not be the only validation data.

### Missing-weapon reference cards

These reproduce the blank weapon panel and are the gold set for the empty-weapon
contract. Keep the raw fixture keys and player UID out of committed docs:

- Lucilla missing-weapon card: character `1109`,
  weapon empty, signature `Freeze Frame` `21050086`.
- Lucy missing-weapon card: character `1511`,
  weapon empty, signature `Spectral Trigger` `21030056`.
- Rebecca missing-weapon card: character `1308`,
  weapon empty, signature `Skull Thrasher` `21030066`.
- Zani control card, panel renders:
  character `1507`, weapon `Blazing Justice` `21040036`.

## Crop sweep tool

`optimize_crops.py` is the Phase 2 harness. It intentionally does not change
runtime code. It tests candidate crop boxes against gold labels and writes
summary/debug output under `backend/benchmarks/crop_sweeps/`.

Typical commands:

```powershell
py backend/optimize_crops.py --labels labels.json --task weapon_sift --image-root .
py backend/optimize_crops.py --labels labels.json --task echo_icon --regions echo1 echo2 echo3 echo4 echo5
py backend/optimize_crops.py --labels labels.json --task watermark_uid --delta 10 --pad 8
py backend/optimize_crops.py --labels labels.json --task echo_main --regions echo1 --save-debug
py backend/optimize_crops.py --labels labels.json --task echo_substats --regions echo1 echo2 --save-debug
```

The search starts from existing crop boxes, then sweeps translations and
uniform padding. Results are sorted by exact-match rate, confidence/margin, and
low-confidence count.

## Acceptance for Phase 2

- Crop changes only land after they improve or preserve held-out accuracy.
- A one-image win is not enough. Use a curated validation set from `r2-backup`
  plus reported issue images.
- Character and weapon should remain at least 99 percent and ideally 100
  percent after crop optimization.
- Echo identity must not regress from the existing SIFT path.
- Main-stat values must be derived, not OCR-read.
- Tesseract remains available for watermark username and as a rollback while
  finite classifiers are developed.

## Phase 3 direction

Train only the small models that still need it after the crop sweep:

- Forte: 5 independent 1-10 digit/level classifiers, or template matching.
- Main stat name: around 10 classes.
- Substat rows: either one tuple class for each legal `(name, value)` pair, or
  two heads: 13-way stat name plus legal value/tier.

Do not train a general OCR model unless the finite recognizers fail the
acceptance bar.

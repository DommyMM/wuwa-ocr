# OCR recognition roadmap

This backend should be treated as a fixed-layout recognition service, not a
general OCR service. The import image format is constrained, the crop geometry
is stable, and most labels come from small finite game-data vocabularies.

## Attempted but not adopted — Tesseract-only echo substat OCR

A full Tesseract-only replacement for the echo substat OCR path was prototyped
end-to-end and benchmarked, then **rolled back** rather than adopted. The
production runtime is unchanged. This section documents what was tried, the
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

Until then, the existing RapidOCR + block-OCR pipeline stays.

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

## Region architecture

| Region | Target method | Notes |
|---|---|---|
| character | SIFT vs `Data/Characters/<id>.png` | Use splash/portrait region, not the top-left name strip. |
| weapon | SIFT vs `Data/Weapons/<id>.png` | Crop the square weapon icon or a tightly bounded weapon-icon area. |
| watermark UID | Tesseract digits-only | Use `tessedit_char_whitelist=0123456789`. |
| watermark username | Tesseract | Leave free-form and Unicode-capable for now. |
| sequences | HSV pixel ratio | Existing method is already deterministic. |
| forte | small fixed classifier or digit templates | Five `LV.X/10` regions, classes 1-10. |
| echo icon | SIFT | Existing method. |
| echo cost | template match | Existing method. |
| echo element | HSV histogram, SIFT fallback within same hue cluster (`data.py` `determine_element`) | Existing method. Grayscale elements (ER, Tidebreaking) and same-hue pairs (e.g. Gust/Windward, Pact/Rite, Trailblazing/Chromatic/Flamewing, Midnight/Dream/Thread) resolve via SIFT. |
| echo main stat name | small classifier or template matcher | Finite class set from `EchoStats.json`. |
| echo main stat value | derive from cost and stat name | Do not OCR. |
| echo substat rows | small row classifier | Predict row `(name, value)` or two heads. |

### Echo element fallback note

The Luuk Herssen report `4099f333-2e9c-4938-84fa-df08138f9f45` exposed a
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

## Phase 1 status

Phase 1 is character and weapon asset recognition.

Existing local artifacts:

- `Data/Characters/` and `Data/Weapons/` directories exist but are empty —
  template PNGs have not been generated yet. The runtime falls back to the
  existing OCR path for character/weapon until templates ship.
- `optimize_crops.py` supports `character_sift` and `weapon_sift` tasks for
  the crop sweep against gold labels.
- The standalone `eval_phase1.py`, `inspect_crops.py`, `save_debug_crops.py`,
  and `docs/phase1-sift-recognition.md` artifacts referenced in earlier
  drafts are no longer in the tree.

Phase 1 acceptance remains:

- Character agreement with current pipeline is at least 99 percent on valid
  1920x1080 cards.
- Weapon agreement with current pipeline is at least 99 percent on valid
  1920x1080 cards.
- Every disagreement is manually reviewed, because the old OCR baseline can be
  wrong too.

## Phase 2 goal

Phase 2 optimizes crop geometry and preprocessing before training any OCR-like
model. The manually validated coordinates are good enough to ship, but they are
not proven optimal.

The optimization pass should answer:

- Which crop box gives the best SIFT agreement/margin for character and weapon?
- Which echo icon crop gives the best echo ID confidence without hurting cost
  or element follow-up?
- Which watermark UID crop gives the highest exact digit-match rate?
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
      "file": "C:/Users/domin/Downloads/347056b9395d3315667093a2532b153a13d9160d.jpeg",
      "character": { "id": "1108", "name": "Hiyuki" },
      "weapon": { "id": "21020086", "name": "Frostburn" },
      "watermark": { "uid": "500006092", "username": "Dommy" },
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
- Watermark UID: `500006092`
- Echo 1: `60001995`, QuietSnow, ATK%, 33%
- Echo 2: `60001875`, QuietSnow, Glacio DMG, 30%
- Echo 3: `60001839`, QuietSnow, ATK%, 30%
- Echo 4: `60001975`, QuietSnow, ATK%, 18%
- Echo 5: `60001965`, QuietSnow, ATK%, 18%

Substat rows from current logs are expected to be correct for this card and are
good seed labels, but they should not be the only validation data.

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

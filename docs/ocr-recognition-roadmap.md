# OCR recognition roadmap

This backend should be treated as a fixed-layout recognition service, not a
general OCR service. The import image format is constrained, the crop geometry
is stable, and most labels come from small finite game-data vocabularies.

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
| echo element | HSV plus SIFT | Existing method. |
| echo main stat name | small classifier or template matcher | Finite class set from `EchoStats.json`. |
| echo main stat value | derive from cost and stat name | Do not OCR. |
| echo substat rows | small row classifier | Predict row `(name, value)` or two heads. |

## Phase 1 status

Phase 1 is character and weapon asset recognition.

Existing local artifacts:

- `docs/phase1-sift-recognition.md`
- `eval_phase1.py`
- `inspect_crops.py`
- `save_debug_crops.py`
- `Data/Characters/` and `Data/Weapons/` when templates have been generated.

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

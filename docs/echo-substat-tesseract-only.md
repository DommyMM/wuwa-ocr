# Echo Substat Reading: history of the RapidOCR-removal attempts (2026-06 → 2026-07)

Superseded. This file is kept only so the two failed attempts are not repeated.
The live design discussion moved to a row-anchored, value-first formulation; see
[ocr-recognition-roadmap.md](ocr-recognition-roadmap.md).

The benchmark artifacts (`benchmarks/echo_substats/`, gitignored) and the
scratch harnesses (`benchmark_echo_substats.py`, `prototype_geom_subs.py`,
`measure_rapid_fallback.py`, `diagnose_values.py`, `font_match.py`) were all
deleted. The numbers below are the surviving record.

## Why RapidOCR was targeted

`data.py` instantiates `Rapid = RapidOCR(...)` at module import, so **every**
worker pays its resident cost whether or not it is called. Calling it less does
not help; only deleting it frees the RAM. That made a Tesseract-only substat
path the prerequisite for removing it.

Fallback frequency, measured on an 800-card r2-backup sample:

| trigger | per-echo rate | rapid calls |
| --- | ---: | --- |
| count-mismatch (names ≠ values) | 17.6% | 2 (names + values) |
| illegal value (counts matched) | 10.8% | 1 (values) |
| any fallback | 24.6% | 0.42 / echo avg |

P(≥1 of 5 echoes hits fallback) ≈ 71% of cards.

## Attempt 1 (2026-06) — rolled back

Per-row Tesseract with a value-driven **stat-name inference** fallback: when the
name read was garbage and the value was legal for exactly one stat, the name was
overridden from the value.

200-card A/B: 135/200 identical, +14 improvements, **2 regressions**. Both
regressions were flat ATK: `ATK 40` → `Crit Rate 40`, `ATK 50` → `Heavy Attack
DMG Bonus 50`. Flat ATK and DEF share the values 40/50/60, so value-driven
inference cannot disambiguate them, and a confident-but-wrong name passes every
downstream legality check.

**Rule that came out of this and still holds: never *infer* a short flat name
from its value.** A wrong-but-legal substat is worse than a dropped one.

## Attempt 2 (2026-07) — validated at 97.6%, then rejected

Names via `image_to_string` + hardened wrap-merge; values via two cheap
Tesseract reads (`--psm 6`, and `3x cubic + tessedit_char_whitelist=0123456789.%`)
arbitrated by the closed legal-value set; `%` suffix deterministic by stat type.
No name inference, so attempt 1's failure mode could not recur.

800-card run: 97.6% of substat sets identical to prod, 0 rapid calls, −1.03%
legal substats. A July recheck on 500 panels from 100 images:

| candidate | identical to live | legal substat delta | speed |
| --- | ---: | ---: | ---: |
| Tesseract-only | 98.4% (492/500) | −12 / 2468 | 8.5% faster |
| conditional Rapid hybrid | 99.4% (497/500) | −3 / 2468 | 1.0% slower |

**Rejected.** Not because 98.4% is low, but because of *what* the 8 misses were.
All 8 disagreements from that run:

| card | region | live | tess-only |
| --- | --- | --- | --- |
| `29aa306a` | echo2 | `DEF 60` | dropped |
| `5e637268` | echo5 | `HP 470` | dropped |
| `a877bff0` | echo2 | `DEF 60` | dropped |
| `97ead2bb` | echo1 | `HP 430`, `ATK% 7.1%` | both dropped |
| `1ab8248d` | echo4 | `HP% 8.6%` | dropped |
| `8d6c11a3` | echo4 | `HP% 8.6%` + `Crit Rate 8.7%` | dropped + **8.1%** |
| `8fe69e97` | echo3 | `HP% 10.9%` + `Heavy Atk 7.9%` | dropped + **10.9%** |
| `e24e8a9f` | echo1 | 5 rows | **1 row, wrong value** |

Two things to take from this table:

1. **7 of 8 involve a short name (`HP`/`ATK`/`DEF`) or a short flat value.**
   Flat HP/ATK/DEF were 335 of 2468 rows (13.6%) in that sample, so this is a
   structural weakness, not a tail case.
2. **The last three are silent corruption, not drops.** `process_card` pairs the
   two strips with `zip(cleaned_names, values_lines[:5])`, so one missing name
   re-pairs every row below it. That amplifier, not the OCR engine, is what makes
   a dropped row dangerous.

The stated blocker was that the comparison used live output as the baseline
rather than human gold labels, so a "loss" could not be distinguished from a
correction.

## Approaches tried and rejected (do not re-attempt)

| approach | result | why rejected |
| --- | --- | --- |
| value-driven stat-name inference | 2 flat-ATK regressions | 40/50/60 are legal for both ATK and DEF |
| full-region single `image_to_data` | −159, name swaps, some cards → `[]` | reading names+values in one block degrades both |
| geometry-pair the two strips via `image_to_data` | −152, Skill↔Liberation swaps | `image_to_data` reads names worse than `image_to_string` |
| full-region values + y-slot align | −236 | y-slot alignment fragile |
| keep rapid, only repair illegal | rapid 597→801 (worse) | rapid stays loaded, so no RAM win at all |
| Lagu whole-string legal-set NCC on values | 54.4% identical, legal-but-WRONG values | whole-string NCC cannot discriminate similar legal values |
| heavier engines (EasyOCR/Paddle/Surya/TrOCR) | n/a | all heavier than RapidOCR, wrong direction on RAM |

**Key lesson:** a *value* reader must be discriminative (actually read the
digits). Template/NCC matching over candidate strings guesses and fails on
similar legal values. This does not apply to a 2-class problem such as
`ATK` vs `DEF`, where the glyphs share nothing.

## Game fonts (still relevant)

`C:\Wuthering Waves\<ver>\Fonts` ships every client font. The English stat-panel
font is **LaguSansBold.otf**, settled quantitatively by NCC against real glyphs
(mean 0.61, vs Kanit 0.50 and H7GBK-Heavy 0.46; H7 is the Chinese font). Its
correct uses are Tesseract fine-tuning and rendering labelled synthetic cells,
not whole-string template matching.

## Where RapidOCR is still called on this path

Both call sites exist only to repair list desync, in
`reconcile_echo_substat_rows`:

1. `len(names) != len(values)` → rapid on both strips.
2. `has_invalid_substat_pair(...)` → rapid on the values strip, consumed by
   `choose_substat_value`.

Neither is a judgement about glyph legibility. Both are consequences of reading
two independent block-OCR lists and pairing them positionally.

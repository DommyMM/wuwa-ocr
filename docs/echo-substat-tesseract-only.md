# Echo Substat Reading: history of the RapidOCR-removal attempts (2026-06 → 2026-07)

Superseded. This file is kept only so the two failed attempts are not repeated.
The live design discussion moved to a row-anchored, value-first formulation; see
[ocr-recognition-roadmap.md](ocr-recognition-roadmap.md). **That formulation is
Attempt 3 below, and it shipped (2026-09).**

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

## Attempt 3 (2026-09) — shipped

The row-anchored formulation, implemented as `read_substat_rows` in `card.py`.
The premise both earlier attempts missed: the drop is a *detection* failure. The
same ~78 rows per 1500 echoes vanish under fast, standard and best tessdata alike,
and the grid is deterministic to the pixel (34.0 px pitch, 15.5 px first row, zero
variance over 2392 gaps), so the fix is to never ask Tesseract to find rows at all.

What it took to pass a 6000-card gate against the live path, every difference
adjudicated:

- five fixed bands, `--psm 7` each, names and values read as separate batches
- values at 2x + digit whitelist; 3x retry only when the 2x read is not a legal
  roll (2x returns nothing for `21%`, 3x returns nothing for `9.2%`; 2x doubles
  digits on some bands where 3x is right, and vice versa — so a legal read is
  never replaced)
- the row after a wrapping stat re-read on a 10 px top, keeping whichever margin
  scores higher against the vocabulary: the spilled `DMG Bonus` fragment reads as
  `[1]` at the full band, and a tight band clips a high-sitting `DEF` to `[3`;
  both score ~0 while the real name scores ~100
- a name must score >= 65 (plain ratio) against the stat it resolves to; WRatio
  scores the fragment `ne` at 90 against `Energy Regen`, plain ratio does not
- `validate_value` snaps by numeric nearest within 0.15 (a doubled-digit `10.99`
  had been string-matched to `9` and returned as 9%)
- flat vs percent decided by magnitude, so a dropped `%` glyph is harmless
- grayscale fallback when fewer than five rows resolve (a dark upload's text is
  shredded by the fixed threshold; Rapid survived it only because it reads raw
  pixels); Otsu everywhere was measured and rejected

Result: every non-echo region 6000/6000 identical; English echoes 0.179% differ,
net +24 rows, the new reader right roughly 3:1 on adjudicated differences and its
errors almost all visible misses rather than wrong values. Uniform 3x (over-fit to
the nine bands that failed at 2x) and a uniform 10 px name top (clipped the `o` of
every `Resonance ...` row) were each tried and reverted on their own gates.

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

## Where RapidOCR was still called on this path (historical)

Before Attempt 3, both call sites existed only to repair list desync, in
`reconcile_echo_substat_rows`:

1. `len(names) != len(values)` → rapid on both strips.
2. `has_invalid_substat_pair(...)` → rapid on the values strip, consumed by
   `choose_substat_value`.

Neither was a judgement about glyph legibility. Both were consequences of reading
two independent block-OCR lists and pairing them positionally — which banding
removes by construction. As of 2026-09 RapidOCR is not called on the substat path,
the main-strip tiebreak, the weapon abstain path (Tesseract on the name strip
matched Rapid 875/875) or the character abstain hybrid (level now comes from the
`LV.` pill), and it is removed from `card.py`, `data.py` and `requirements.txt`.

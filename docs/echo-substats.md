# Echo substats

`read_substat_rows` in `card.py` reads the five substat rows of an echo panel. The design rests on one measurement: dropped rows are a layout-analysis failure, not a recognition one. Fast, standard and best tessdata all lose the same ~78 rows per 1500 echoes on a whole-block read, while the card grid is exact to the pixel (first row centred at 15.5 px, 34 px pitch, zero variance over 2392 gaps, wrapped and unwrapped alike). So Tesseract is never asked to find rows.

## Reader

| step | decision | evidence |
| --- | --- | --- |
| Bands | Each row read on its own fixed band with `--psm 7`, names and values as separate batches | A band handed to psm 7 can't be dropped |
| Values | 2x with a digit whitelist, one 3x retry only when the 2x read isn't a legal roll | 2x returns nothing for `21%` and 3x nothing for `9.2%`, and each doubles digits the other reads right, so a legal read is never replaced |
| Snapping | Nearest legal roll within 0.15 for percent stats, exact for flat | Display rounding shows DEF 11.9% for the 11.8 roll while legal rolls sit >= 0.7 apart. A string-similarity snap once turned `10.99` into `9%` |
| Flat vs percent | Decided by magnitude, not the `%` glyph | Flat ATK 30-60 and ATK% 6.4-11.6 never overlap, so a dropped `%` is harmless |
| Names | Band's first line, resolved against the closed vocabulary with WRatio | A wrapped name's clipped second line doesn't change which stat it is |
| Name floor | Plain ratio >= 65 against the resolved stat, else the row resolves to nothing | WRatio scores the fragment `ne` at 90 against `Energy Regen`. Good reads score >= 77 and garbage <= 58 |
| Resonance names | `skill` or `liberation` in the read decides the name outright | A band edge clips the `o` and `Rescnance Skill DMG` fuzzy-matches Crit DMG on the shared token |
| Row after a wrap | Re-read on a ladder of tighter tops `(12, 10, 9, 8, 7)`, keeping the most confident read, a tie keeping the primary | The wrap's second line reaches the next band's top 4 px and reads as `[1]`, resolving HP. Which top clears it isn't monotonic, and one top everywhere clipped the `o` of every Resonance row |
| Grayscale pass | Runs only when the thresholded pass resolves fewer than five rows | The fixed 140 threshold shreds dim cards, but grayscale loses ~1.2% on bright ones |
| Merge | The pass with more rows wins and the other fills only its empty bands | The passes fail on different rows. Letting the loser replace resolved rows swapped in legal but wrong values (`HP 390` for `HP 360`) |

On its 6000-card gate against the RapidOCR-backed reader, every non-echo region stayed identical and English echoes differed on 0.179%, with the banded reader right about 3:1 on adjudicated differences and its errors almost all visible misses. The post-wrap merge and top ladder then recovered 8 rows over 29,975 echoes with 0 losses and 0 value changes.

## Rules

- Never infer a short flat name from its value. Flat ATK and DEF share 40, 50 and 60, so an inferred name passes every legality check while wrong
- A wrong-but-legal row is worse than a dropped one
- A value reader must read the digits. Matching candidate strings or templates guesses and fails on similar legal values. That doesn't apply to a two-class shape problem like `ATK` vs `DEF`, where the glyphs share nothing

## Rejected approaches

| approach | result |
| --- | --- |
| Value-driven name inference | Flat `ATK 40` read as `Crit Rate 40` and `ATK 50` as `Heavy Attack DMG Bonus 50` |
| Whole-block names and values paired positionally | One missing name re-paired every row below it, turning drops into silent corruption |
| Whole-region `image_to_data` | 159 rows lost, name swaps, some cards returned nothing |
| Pairing the two strips by `image_to_data` geometry | 152 rows lost, Skill and Liberation swapped |
| Whole-region values aligned by y-slot | 236 rows lost |
| Keeping RapidOCR only to repair illegal rows | Rapid calls rose from 597 to 801 and it stayed loaded, so no RAM win |
| Whole-string template matching in the game font | 54.4% identical, with legal but wrong values |
| Uniform 3x on every value band | Lost 123 `21%` rows in 3929 cards and new rows 2x read fine |
| Heavier engines (EasyOCR, Paddle, Surya, TrOCR) | All heavier than RapidOCR, the wrong direction on RAM |

## Game font

The client ships every font under `Wuthering Waves/<ver>/Fonts`. The English stat font is `LaguSansBold.otf` (glyph NCC 0.61 against Kanit 0.50 and H7GBK-Heavy 0.46, the Chinese font). Its use is Tesseract fine-tuning and rendering labelled synthetic cells, not template matching.

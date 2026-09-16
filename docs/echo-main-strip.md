# Echo main strip

`_main_strip_lines` in `card.py` reads the echo main stat on plain grayscale at 2x with `--psm 6`, skipping the shared `preprocess_region`. Deleting that preprocessing beat all 6336 tuned alternatives, and it stopped soft uploads from importing a fabricated HP% main without any visible error.

## Root cause

`preprocess_region` ends in a fixed global threshold at 140, and the main strip is two rows at different brightness: a bright gold value row and a dimmer grey name row. One cutoff can only suit one of them, so on a soft upload the value row survives and the name row breaks into fragments. `image_to_string` then defaults to psm 3, whose layout analysis chokes on the fragments and returns nothing, not even the intact value.

| input to Tesseract | result |
| --- | --- |
| `preprocess_region`, psm 3 | `[]` |
| `preprocess_region`, psm 6 | `['Cot hE', '= 44%']` |
| raw grayscale, psm 3, 6 or 11 | `['Crit. DMG', '44%']` |

An empty read is the worst outcome, because `resolve_echo_main` then has no value anchor and falls to the cost's first legal main, which is HP% at its canonical value. Only 0.18% of clean strips returned empty, but one user's degraded capture pipeline fell through it every time and the card imported successfully with wrong data.

## Method

- Ground truth: four independent reader configs per clean crop, each snapped to the legal `(cost, name, value)` space, kept only when three or more resolve and none dissent. 1361 labels from 400 cards for tuning, 6728 from 2000 disjoint cards for validation
- Degradation: five soft-resample severities applied to the full card and re-cropped, the way a real degraded upload arrives, not tuned to any one user's upscaler
- Scoring: end to end, whether `resolve_echo_main` lands on the true `(name, value)`
- Sweep: 6336 configs over channel and mask, upscale, denoise, normalisation, sharpening, binarisation, inversion and psm, run through Tesseract list-file mode

## Result

Validated on the 2000 disjoint cards:

| level | old path | grayscale 2x psm 6 | empty reads, old to new |
| --- | --- | --- | --- |
| clean | 99.60% | 99.99% | 0.21% to 0.00% |
| mild | 90.71% | 99.99% | 6.29% to 0.00% |
| moderate | 67.24% | 99.99% | 3.78% to 0.00% |
| hard | 33.40% | 99.99% | 17.52% to 0.00% |
| severe | 7.79% | 99.99% | 55.83% to 0.00% |

Five real degraded cards went from 11/25 mains correct to 25/25, and 300 clean corpus panels changed on only one junk upload. It costs ~16 ms more per strip, since Tesseract segments real grayscale slower than a mostly blank binary image.

## Rejected axes

| axis | finding |
| --- | --- |
| `--psm 7` | 0/2112, since it forces one text line on a two-row strip |
| Any binarisation | No binarisation had the best success rate across the grid (65%), adaptive threshold the worst (18%), and Otsu lost too |
| Gold-chroma masks (`0.6R + 0.4G`, `max(R, G)`, LAB-L) | 35-39%, no better than plain luma, since the text isn't chroma-separable from the bed |
| `--psm 11` | Equal accuracy, but it returns lines in no fixed order and emitted three lines where `_parse_main_line` joins two |
| 1x instead of 2x | 99.9% against 99.99% for ~7 ms saved |

## Scope and confidence

The result covers the main strip only, so the fix lives at that call site rather than in the shared helper, which forte, the watermark and substats still use.

The 6728 labels are OCR consensus, and the chosen config sits between two of the four labelers, so the clean-level figure is partly self-fulfilling. The degradation figures aren't, since labels come from clean crops and scoring from degraded ones. A blind 24-crop check at the hard level, read by eye before seeing any label, agreed 24/24.

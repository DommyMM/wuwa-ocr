# Echo Main Stat OCR: Deleting the Preprocessing — Investigation + Result (2026-08)

Goal: **stop the echo main stat being silently fabricated as `HP%` on soft uploads.** The
lever is deleting `preprocess_region` from the main-strip read and handing Tesseract plain
grayscale.

> **Framing:** this is not a corpus-wide outage. Across 3990 strips from 798 unaltered cards
> only **0.18%** return empty under the old path. It is a thin-margin silent failure that a
> single user's degraded capture pipeline falls through, and the failure mode is the problem:
> the card imports *successfully* with wrong data, so nobody sees an error.

## Root cause (diagnosed, not guessed)

`preprocess_region` ends in a **fixed global threshold at 140**:

```python
_, thresh = cv2.threshold(sharp, 140, 255, cv2.THRESH_BINARY)
```

The main strip is two rows at different brightness: a **bright gold value row** (`44%`) and a
**dimmer gray name row** (`Crit. DMG`). A single global cutoff can only ever be right for one
of them. On a soft upload the value row clears 140 and the name row survives as broken
fragments.

Then the second half of the failure: `pytesseract.image_to_string` defaults to **psm 3**, whose
layout analysis chokes on the fragmented result and returns **nothing at all** — not even the
intact value row. Measured on the same crop:

| input to Tesseract | result |
| --- | --- |
| `preprocess_region(...)`, psm 3 | `[]` |
| `preprocess_region(...)`, psm 6 | `['Cot hE', '= 44%']` |
| **raw grayscale**, psm 3 / 6 / 11 | `['Crit. DMG', '44%']` |

An empty read is the worst possible outcome because of how `resolve_echo_main` degrades: with
no digits there is no value anchor, so the else-branch runs
`_tiebreak_main_name(list(legal), raw_name)` **without passing `rapid_main`** and falls through
to `next(iter(legal))`. `MAIN_STATS` is HP-first for every cost, so the card silently gets
**HP% at the cost's canonical value** (33% on 4cost, 22.8% on 1cost, 30% on 3cost) and no
illegal-name signal is ever raised.

## Method

**Ground truth.** 4 independent reader configs read each *clean* crop; each result is snapped
to the legal `(cost, name, value)` space; a label is kept only when >=3 resolve and none
dissent. 1361 labels from 400 cards for tuning, 6728 labels from 2000 disjoint cards for
validation. Zero reader disagreement in both passes.

**Degradation ladder.** Five soft-resample severities applied to the *full card* then
re-cropped (which is how a real degraded upload arrives), indexed by how much of the strip
survives the old 140 threshold: 632 -> 566 -> 479 -> 324 -> 208 px. Deliberately *not* tuned to
reproduce any one user's pipeline — that would overfit to one person's upscaler.

**Scoring.** End-to-end: does `resolve_echo_main` land on the ground-truth `(name, value)`.
Not raw string match.

**Sweep.** Exhaustive 6336-config grid over channel/mask, upscale, denoise, normalisation,
sharpen, binarisation, inversion and psm. Parallelised over configs; Tesseract's native
filelist mode does ~24ms/img against pytesseract's ~138ms per-call spawn, which is what makes
a sweep this size finish in ~1h.

## Result

**The winner is no preprocessing at all.** `BGR2GRAY` -> `resize x2 INTER_CUBIC` -> `--psm 6`,
validated on 2000 cards disjoint from the 400 used to tune:

| level | old (`preprocess_region` + psm 3) | new | empty reads, old -> new |
| --- | --- | --- | --- |
| clean | 99.60% | **99.99%** | 0.21% -> 0.00% |
| mild | 90.71% | **99.99%** | 6.29% -> 0.00% |
| moderate | 67.24% | **99.99%** | 3.78% -> 0.00% |
| hard | 33.40% | **99.99%** | 17.52% -> 0.00% |
| severe | 7.79% | **99.99%** | 55.83% -> 0.00% |

- **End-to-end through `process_card` on 5 real degraded cards: 25/25 mains correct, up from
  11/25 in production.**
- 300 clean corpus panels re-run before/after: **1 change**, on a junk upload (a photo of
  fabric, not a card). No real regressions. 65/65 backend tests pass.
- Preprocessing was **worse even on clean cards** (99.60 vs 99.99), so it had no upside to
  trade away.

### Axis findings (so we don't re-attempt them)

- **`--psm 7` is useless here: 0/2112.** It forces a single text line; the strip is two rows.
- **Binarising at all hurts.** `none` had the best success rate across the grid (65%);
  adaptive threshold was the *worst* (18%), below the fixed thresholds. A *lighter* threshold
  would not have fixed this — Otsu and adaptive both lose to no binarisation.
- **A gold-chroma mask buys nothing.** `0.6R + 0.4G`, `max(R,G)` and LAB-L all land within
  35-39%, indistinguishable from plain luma. The text is not chroma-separable from the bed.
- **psm 6 over psm 11** despite equal accuracy: psm 6 returned exactly two lines on 600/600
  soft strips, which is what `_parse_main_line`'s "join two lines, split the value off the
  end" assumes. psm 11 is documented as returning text in no particular order and emitted a
  3-line read where `lines[:2]` could drop the value.
- **x2 over x1** for 99.99% vs 99.9%, costing ~7ms.

### Latency

**+16ms per strip** (121ms -> 137ms). The change *adds* cost, which is counterintuitive for
deleting work: the old path was fast partly *because* the binary image it produced was mostly
blank, and Tesseract segments real grayscale more slowly. Even x1 with no preprocessing is
+9ms. The 5 echo regions run concurrently in the server pool, so wall cost is ~+16ms on a
~1500ms card. RAM is unchanged.

## Scope

Validated for the **echo main strip only**. `preprocess_region` has six call sites — character,
the generic branch (watermark/sequences), forte, echo main, substat names, substat values — and
this result covers exactly one of them. The fix therefore lives in `_main_strip_lines()` at the
main-strip call site, **not** in the shared helper. Substats in particular lean on the RapidOCR
reconcile path (see `docs/echo-substat-tesseract-only.md`) and were not measured here.

## Open items

- **`resolve_echo_main`'s empty-strip branch still fabricates HP%.** Empty reads are now 0.00%
  at every level, but the silent fallback remains one degradation away. Rapid reads these
  strips correctly (`'Crit.DMG 蒸44%'`, `'ATK X18%'`), so passing `rapid_main` in that branch is
  a one-line defence in depth. **Not done.**
- **Forte is the same shared helper and the same failure shape** — it misreads all-10 nodes as
  `0` on soft cards. Not benchmarked. Forte is not persisted in `builds`, so it corrupts the
  import UI rather than the leaderboard. Running this harness against the forte region is the
  cheaper first experiment before reaching for SIFT node matching.
- **Crop geometry, unrelated to OCR.** The single miss in 6728
  (`c2741068fe4d038b889b0eeface564bd6feb28b7bc737ae0a7355972413737c7.jpg#3`) is a card with a
  light/grey layout where the fixed `ECHO_REGIONS["main"]` x195-366 box lands wrong and clips
  `Havoc DMG` to `avoc DMG`. The old path misses it too. Worth a pass with
  `visualize_regions.py`.
- **Existing bad rows are not corrected.** The fix is forward-only; affected builds need a
  re-OCR.

## Confidence

The 6728 labels are OCR consensus, not human-read, and the chosen config sits between two of
the four labelers — so the *clean-level* figure is partly self-fulfilling. The degradation
figures are not, since labels come from clean crops and scoring happens on degraded ones. A
24-crop blind check at the `hard` level (read by eye before looking at any label or OCR output)
agreed **24/24** with both the consensus labels and the config output, which rules out
systematic label bias but is too small to separate 99.99% from 99%. Human-verified total: 61
crops.

# Echo Substat OCR: Deleting RapidOCR (Tesseract-only) — Investigation + Plan (2026-06)

Goal: **trim the latency tail and stop paying for RapidOCR's RAM footprint, without losing
accuracy.** The lever is deleting RapidOCR from the echo substat path and running 100%
Tesseract.

> **Framing (2026-06):** the absolute Railway cost (~$7/mo) is *acceptable* — this is not a
> cost-pressure exercise. The motivation is efficiency: RapidOCR is loaded in every worker and
> pins ~92% of RAM, yet the fallback that justifies it fires on only ~25% of echoes and a
> Tesseract-only path matches it within <1%. We're paying a large fixed RAM cost (and a latency
> tail) for very little. Removing it is worthwhile *if* accuracy holds — optimize-if-possible,
> not optimize-because-we-must.

## Why this is the right target (efficiency reframe)

The Railway bill for the OCR service is **~92% RAM** (`$6.35` RAM vs `$0.53` CPU of `$6.89`).
RapidOCR (onnxruntime) is loaded at import in **every** worker (`data.py` `Rapid = RapidOCR(...)`),
so it is the RAM driver whether or not it is called. Therefore:

- Calling rapid *less* barely helps; the base load + inference spikes stay.
- **Deleting RapidOCR entirely** is what frees the RAM — which requires a Tesseract-only
  path that matches accuracy.

Latency corroborates: prod recognition wall ≈ the slowest of the 5 parallel echoes, and the
RapidOCR fallback is what inflates it. Measured on r2-backup (`measure_rapid_fallback.py`,
800-card sample):

| trigger | per-echo rate | rapid calls |
| --- | ---: | --- |
| count-mismatch (names ≠ values) | 17.6% | 2 (names + values) |
| illegal value (counts matched) | 10.8% | 1 (values) |
| any fallback | 24.6% | 0.42 / echo avg |

P(≥1 of 5 echoes hits fallback) ≈ **71% of cards**, which is exactly the gap between the
~730ms no-fallback floor and the ~1237ms prod mean. Kill the fallback → kill the tail.

## Root causes of the fallback (diagnosed, not guessed)

Test card `d92aea7866c92d37b33305491102660e428b501d5b6895dcaf80ebcd46d21a8b.jpg`
together with `diagnose_values.py` made the three failure modes concrete:

1. **Name wrapping with garbage tails.** Only `Resonance Liberation DMG Bonus` (wraps
   `DMG Bonus`) and `Resonance Skill DMG Bonus` (wraps `Bonus`) ever wrap. The existing
   `clean_echo_substat_name_lines` merges *clean* continuations; when the 2nd line OCRs as
   garbage (`NIAC Rie`, `Brite`, `Do`) it doesn't, so 6-7 name lines vs 5 values → mismatch.
   This is the bulk of the 17.6%, and the **values are fine** in these cases.
2. **psm-3 drops short flat values.** Default `image_to_string` (psm 3) drops integer values
   like `430`/`60`/`40` on the narrow values strip — but **`--psm 6` recovers them**
   (`['8.6%','19.8%','430','7.5%','10.1%']`). The current code uses bare `image_to_string`.
3. **Digit misreads.** `330`→`390`, `[2`→`60`. **3x cubic upscale** recovers most of these,
   but upscale alone is not safe (it broke `7.5%`→`1.5%` elsewhere). The value is a **closed
   legal set** per stat, so legality can arbitrate between the two reads.

## The Tesseract-only path that works (`tess_only`)

Implemented + regressed in `prototype_geom_subs.py` (`tess_only_subs`), validated against the
**live card.py (with real RapidOCR) as ground truth** over an even r2-backup sample:

- **names**: `image_to_string` (unchanged, proven) + a hardened wrap-merge that absorbs the
  garbage 2nd line for the two known wrappers (`improved_merge`).
- **values**: two cheap Tesseract reads of the values strip —
  - primary `--psm 6` (recovers psm-3 drops),
  - alt `3x cubic + tessedit_char_whitelist=0123456789.%` (recovers misreads),
  - **legality arbitrates** (closed legal-value set) — the exact role RapidOCR played, but
    with a second Tesseract pass instead of the onnx model (Tesseract is already loaded → **no
    added RAM**).
- **%-suffix** is deterministic by stat type (Crit/DMG/ER/`*%` always `%`; flat HP/ATK/DEF never).

### Result (800 cards, vs live card.py)

| metric | value |
| --- | --- |
| substat sets identical to prod | **97.6%** |
| RapidOCR calls | **0** (was 0.42/echo) |
| total legal substats vs prod | −1.03% (−199 / 19331) |
| echoes where tess-only found *more* than prod | 9 |

The real accuracy delta is **< 1%**: some of the 199 "losses" are cards where prod's rapid is
itself wrong (tess-only recovered 9 echoes prod missed). Remaining true residual is digit
misreads + multi-substat drops on a handful of hard/blurry cards.

## Approaches tried and rejected (so we don't re-attempt)

| approach | result | why rejected |
| --- | --- | --- |
| full-region single `image_to_data` (`geom_anchor`) | −159, name swaps, some cards → `[]` | reading names+values in one block degrades BOTH vs dedicated strips |
| geometry-pair the two strips via `image_to_data` (`geom_pair_strips`) | −152, Skill↔Liberation swaps | `image_to_data` reads *names* worse than `image_to_string` |
| full-region values + y-slot align (`hybrid`) | −236 | y-slot alignment fragile |
| keep rapid, only repair illegal (`geom_repair`) | rapid 597→801 (worse) | doesn't cut RAM at all; rapid stays loaded |
| **Lagu whole-string legal-set NCC** (`lagu_subs`) | **54.4% identical**, legal-but-WRONG values (`10.5%`→`7.5%`, `ATK% 10.1%`→`ATK 30`) | whole-string NCC can't discriminate similar legal values; silently wrong is worse than dropped |
| heavier engines (EasyOCR/Paddle/Surya/TrOCR) | n/a | all heavier than RapidOCR → wrong way on the RAM bill |

**Key lesson:** the value reader must be **discriminative** (read the actual digits). That is
Tesseract. Template/NCC matching against candidates *guesses* and fails on similar values.

## The game fonts (found 2026-06)

`C:\Wuthering Waves\2.6.2.0\Fonts` ships every client font:

| file | script |
| --- | --- |
| **LaguSansBold.otf** | **Latin — the English stat-panel font** |
| H7GBK-Heavy.ttf | Chinese (GBK) — the CN client font, NOT English |
| MotoyaAporoStdW5.otf | Japanese |
| SourceHanSansCN-VF-2.otf | Chinese (CJK) |
| SUITE-Bold.otf | Korean |
| Kanit-Medium.ttf | Thai |

Quantitative NCC of each render vs real glyphs (`font_match.py`) settled the English font —
it is **Lagu, not H7** (H7 is the Chinese font); Lagu wins decisively on the digits:

| font | mean NCC | on values |
| --- | ---: | --- |
| **LaguSansBold** | **0.61** | **0.60–0.66** |
| Kanit-Medium | 0.50 | 0.22–0.46 |
| H7GBK-Heavy | 0.46 | 0.31–0.42 |

The font's correct use is **fine-tuning Tesseract** (a discriminative reader), not whole-string
template matching (rejected above).

## Recommendation / plan

1. **Ship `tess_only` and delete RapidOCR.** Apply the names wrap-merge + `--psm 6` values +
   upscaled second read arbitrated by the legal set + deterministic `%`. Remove
   `Rapid = RapidOCR(...)` from `data.py` and the rapid fallback in `card.py`
   (`reconcile_echo_substat_rows` / `choose_substat_value`'s rapid arm). Drop `OCR_WORKERS`
   to fit the freed RAM. **Wins: ~92% RAM cost gone, latency tail gone, < 1% accuracy delta.**
   This is deployable on the current Docker image (no new deps).
2. **Fine-tune Tesseract on the game fonts to close the residual + go multilingual.** Render
   the closed vocab (13 substat names + every legal value) through the same
   `preprocess_region`, in Lagu, across a few sizes/weights, and `tesstrain`-fine-tune
   `eng` → `wuwa.traineddata`. Same pipeline with Motoya/SourceHan/SUITE/Kanit gives custom
   per-language traineddata that beats the generic lang packs (this supersedes approach **C**
   in [multilingual-echo-investigation.md](multilingual-echo-investigation.md)). GPU does not
   help tesstrain (CPU); the RTX 5090 only matters if we ever train a custom NN, which the
   closed-set problem does not warrant.

**Open decision (user):** ship step 1 now at the < 1% delta, or fine-tune first (step 2) and
ship both together. Step 1 alone already captures the entire RAM/cost/latency win.

## Tooling (removed after the investigation)

These untracked harnesses produced the numbers above and were deleted at the end of the
session; the methodology here is enough to recreate them if needed:

- `prototype_geom_subs.py` — the A/B regression harness: every candidate above, diffed against
  the live (rapid) `card.py` as ground truth, with `PG_N` / `PG_WORKERS` / `PG_SHOW` knobs and
  per-echo latency. **Recreate this first** for any future pre-deploy regression.
- `measure_rapid_fallback.py` — fallback-frequency measurement (the 71% / 17.6% numbers).
- `diagnose_values.py` — per-config values-strip probe (the psm-6 / upscale discovery).
- `font_match.py` — quantitative font identification (the Lagu finding).

Note: regression to date is the 400/500/800-card samples (~97.1-97.6% identical, <1% delta,
0 rapid). A full ~12k (`r2-backup`) run has **not** been completed — it is ~2.5h with real
RapidOCR as ground truth — so this is validated **locally only**. The tess-only substat work
is committed to `main` (`64625dc`, unpushed) but **not deployed**, and RapidOCR has **not**
been removed yet: `data.py` still loads it and `card.py` still calls it (character level,
weapon, and substat-value fallback). Removing RapidOCR (step 1's payoff) is gated on the full
regression.

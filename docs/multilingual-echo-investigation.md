# Multilingual echo stat names

The runtime reads English cards only. SIFT made character, weapon, echo identity and set recognition language-independent, and values and forte levels are digits, so the entire multilingual surface is the five substat names per echo: a closed 13-name vocabulary in a fixed game font. This doc records the bake-off for reading them and the chosen design, which isn't built.

Substat rows render a generic bullet, not a per-stat icon, so icon matching can't replace the text (the bag screen the scanner reads is different and does show stat icons).

## Bake-off

Test set: the scout's strong candidates, 93 Japanese, 4 French and 26 Chinese cards. A name counts as read when it fuzzy-matches a localized alias in `Data/Stats.json` to a canonical stat at WRatio >= 80, spot-checked by eye. The proxy slightly over-credits reads that land on a different valid stat. Speed is the name OCR for all five echoes of one card on the dev box.

| approach | ms per card | match rate |
| --- | ---: | ---: |
| English only (current) | ~1000 | 38.8% |
| A: `eng+fra+jpn+chi_sim+chi_tra` on every card | ~3900-4500 | 84% (FR 97.5, JA 88.6, ZH 70.6) |
| B: image NCC against name-row exemplars | ~2100 | 76.9% agreement with A |
| C: detect the language, then one single-language pass | ~700-2500 | FR 99.2, JA 89.4, ZH 65.9 |

Per language under C: French with `fra` at ~700 ms, Japanese with `jpn` at ~1040 ms, Chinese with `chi_sim+chi_tra` at ~2500 ms.

## Decision

Build C when multilingual import is picked up:

1. Detect the card's language once per card (Unicode script of a quick pass, or the character-name region), not per echo
2. Run the existing substat-name pass in that language, `chi_sim+chi_tra` for Chinese, keeping values, forte, identity and sets unchanged
3. Resolve the localized name to the canonical key through an alias index built from `Data/Stats.json`
4. Repair remaining misses with the value prior: legal values in `EchoStats.json` narrow the candidates

The English path stays one spawn per echo, so English cards don't regress. The response shape doesn't change either, since the frontend localizes display labels from canonical keys.

Beyond stock language packs, the accuracy lever is `tesstrain` traineddata fine-tuned on each language's game font (Lagu for Latin, Motoya for Japanese, SourceHan for Chinese, SUITE for Korean, Kanit for Thai). It should lift the ~66% Chinese ceiling while staying pure Tesseract.

Rejected:
- A runs five languages on every card at ~780 ms per echo, roughly doubling the echo wall
- B has no accuracy edge, since variable-length text doesn't align under fixed-canvas NCC the way icon keypoints do, and whole-string font templates failed on English values too
- A trained CNN or transformer isn't warranted for a 13-way closed set that a single-language pass plus value priors covers

Building C needs the language packs added back to the Dockerfile, which installs only `tesseract-ocr-eng` today.

## Scout and validation

`find_non_english_cards.py` scans `r2-backup/` without the OCR API, reading the echo name strips with a multi-language pass and classifying cards by Unicode script and alias matches. `build_card_signal` (at least five alias hits) marks strong candidates for validation sets. Output under `forensics/non_english_ocr_scan/` is gitignored.

```powershell
py find_non_english_cards.py ..\r2-backup --workers 8 --progress-every 25
py validate_non_english_cards.py --workers 8
```

`validate_non_english_cards.py` runs candidate rows through `card.process_card` on the server's echo crops. Known localized samples that must flag:

- `f32421ba8b1f3dc03de07f879703cc23da24c8abc192ecd271e441216559bcfb.jpg` (Japanese)
- `cce1a0f29186891b20e00684e3bf5853d9ab527279b2b71ee5c83b5ef6a74ae9.jpg` (French)
- `5e17036118784d4b9e4adb2c63fbbfc03522994c5dcd972536e35e6ea1addcde.jpg` (Chinese)

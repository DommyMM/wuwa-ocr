# Multilingual Echo Stat-Name Investigation (2026-06)

The shelved multilingual OCR intake plan, re-aligned now that the backend is otherwise done,
plus the bake-off it called for. This doc absorbs the former `multilingual-ocr-plan.md`; its
alias / scout / contract reference now lives in the **Intake reference** section at the end.
The original plan reached good accuracy but was too slow, so it was never shipped: the live
runtime is English-only (`image_to_string` with no `lang=`, `RapidOCR(lang='en')`), and
`data.py` never loads `Stats.json`/`STAT_TRANSLATIONS`. Only the Dockerfile lang packs, the
copied `Stats.json`, and the scout/validate scripts remain.

This is the documented bake-off the re-alignment called for: approaches A, B, C plus other
ideas, measured on real localized cards.

> **Update (2026-06): the game fonts were found**, which upgrades approach C. The English
> work in [echo-substat-tesseract-only.md](echo-substat-tesseract-only.md) showed (a) the
> goal is deleting RapidOCR to cut the RAM bill, and (b) `C:\Wuthering Waves\...\Fonts` ships
> every client font: **Lagu** (Latin/English), **Motoya** (JP), **SourceHan** (ZH), **SUITE**
> (KR), **Kanit** (Thai). So C's "single-language Tesseract with the generic `fra`/`jpn`/`chi_*`
> lang packs" becomes **"single-language Tesseract with a custom `tesstrain` traineddata
> fine-tuned on the actual game font per language."** That should beat the generic packs
> (especially the ~66% ZH ceiling) while staying 100% Tesseract (no added RAM). Whole-string
> NCC template matching against the font was tried for English and **failed** (legal-but-wrong
> values, 54% — see the other doc); the font's value is fine-tuning the recognizer, not
> template-guessing. The numbers below stand; the implementation lever is now per-language
> traineddata, not stock lang packs.

## How SIFT shrank the problem

SIFT made character, weapon, echo-identity, and element recognition language-independent.
So the **entire** remaining multilingual surface is echo stat *text*, and it splits cleanly:

- **Values** (`9.3%`, `44%`, `320`) and **forte** (`LV.8`): language-independent digits. Already fine.
- **Stat names**: the only real problem, and it is a **closed 13-name substat vocabulary**
  (plus 13 main-stat names; union 20) in a fixed game font.

### Substats have no per-stat icon (icon-SIFT ruled out)

Stats.json carries an `icon` per stat, which suggested a fully language-independent
icon-SIFT path. Inspecting real panels killed it: **substats render a generic ✦ bullet**,
not per-stat attribute icons. Only the two *main* stats (primary + the gold secondary flat
ATK/HP) show real attribute icons, and even those share icons across flat/% variants
(HP and HP% share one icon, etc.), so the `%` in the value is still needed. The 5 substat
names are inherently a **text** problem.

The `subs_names` crop + `preprocess_region` yields crisp, evenly-spaced white-on-black rows
(verified across JA/FR/ZH), which is good for both OCR and template matching.

## Method

- Test set: the scout's strong candidates (`build_card_signal=true`): 93 JA, 4 FR, 26 ZH.
  Sampled per language; each card contributes 5 echoes x up to 5 substat names.
- Accuracy proxy: a correctly-read name fuzzy-matches a localized alias in `Stats.json`
  (200 entries across languages) to a canonical stat at WRatio >= 80/85; a mis-OCR does not.
  Reported as **match-rate** (matched names / names read). Spot-verified by eye on JA/FR/ZH
  panels. Caveat: the proxy slightly over-credits cases where OCR yields a *different* valid
  stat name (e.g. dropping a suffix: クリティカル "Crit Rate" vs クリティカルダメージ "Crit DMG").
- Speed: wall time of the substat-name OCR for all 5 echoes of a card (one Tesseract pass
  per echo over the `subs_names` strip). Harnesses: `investigate_multilingual.py` (A/C/eng),
  `investigate_multilingual_b.py` (B). Dev box 7800X3D; Linux/Railway differs but the
  *relative* ordering holds.

## Results

Per card = the 5 substat-name strips of one card (5 echoes).

| Approach | ms/card | match-rate | notes |
| --- | ---: | ---: | --- |
| **baseline eng** (current runtime) | ~1000 | 38.8% | broken on non-English; only Latin FR survives |
| **A** multilingual `eng+fra+jpn+chi_sim+chi_tra` | ~3900-4500 | 84% (FR 97.5 / JA 88.6 / ZH 70.6) | the shelved approach; ~780ms/echo is the speed regression |
| **C** single language (detected) | ~700-2500 | FR 99.2 / JA 89.4 / ZH 65.9 | same speed as today's eng pass for FR/JA; ZH needs `chi_sim+chi_tra` |
| **B** template-match rows (image NCC) | 2106 (naive) | 76.9% agree with A | no accuracy edge; see below |

Per-language C, using the right pack (`fra` / `jpn` / `chi_sim+chi_tra`):

| Lang | baseline eng | A multilingual | C single | C speed/card |
| --- | ---: | ---: | ---: | ---: |
| FR | ~90% | 97.5% | **99.2%** | ~700 ms (`fra`) |
| JA | ~9% | 88.6% | **89.4%** | ~1040 ms (`jpn`) |
| ZH | ~21% | 70.6% | 65.9% | ~2500 ms (`chi_sim+chi_tra`) |

## Analysis

**A (multilingual, shelved).** Accurate (84%) but ~780ms *per echo* just for names. Running
all five languages every time is the regression we stopped on; it would roughly double the
~600ms echo wall. Even gated to non-English, it is ~4x C on the cards it fires.

**C (detect language, then single-language pass).** The sweet spot. For FR/EN and JA it is
the **same speed as the current eng pass** (~200ms/echo) because it just swaps the language
of the one names Tesseract spawn that already runs, and it lifts name accuracy from ~39% to
~90%+. ZH is the hard case: it needs both Chinese variants (`chi_sim+chi_tra`, ~500ms/echo,
still far under A) and tops out near A's ~66-71% because dense ZH glyphs at panel resolution
are genuinely hard to OCR. ZH is also the rarest (~26 of 123 strong candidates).

**B (template-match the name rows, the "SIFT for text" idea).** Did **not** pan out like
SIFT-for-icons. Naive cross-language image NCC over the exemplar bank: 76.9% agreement with
A (so lower true accuracy, since it inherits A's ~16% error) and 2106ms/card. The speed is a
fixable artifact (brute-force Python NCC over 221 exemplars; dedup to ~40 templates +
vectorized cosine would be sub-10ms). The **accuracy** is the real problem: variable-length
text does not align under fixed-canvas NCC the way distinctive icon keypoints do, and a clean
template set would require the actual game font (which we do not have) rather than
bootstrapped exemplars. No accuracy advantage over C for materially more work.

**Other ideas considered.**
- *Icon-SIFT (I):* ruled out for substats (generic bullet, no per-stat icon).
- *Value prior (G):* the value (`%` vs flat, and legal-value sets in `EchoStats.json`)
  narrows candidates and can repair OCR ties. It cannot identify a name alone (many stats
  share value ranges) but is a cheap accuracy booster *on top of* C. Some of this already
  exists in `card.py` (`validate_value`, `is_legal_substat_value`).
- *Train a CNN/transformer:* not warranted. SIFT already handles identity; the residual is a
  13-way closed set where a single-language OCR pass plus value priors is simpler and cheaper.

## Recommendation

Ship **C**, not A or B:

1. Detect the screenshot language **once per card** (cheap: Unicode-script of a quick pass,
   or from the character-name region), not per echo.
2. Run the existing substat-name Tesseract pass in that language (`fra` / `jpn` / `eng`...);
   use `chi_sim+chi_tra` for Chinese. Keep values, forte, identity, and element exactly as
   they are. This is a surgical change to one of the three echo Tesseract spawns and does
   **not** regress the echo wall for the common (English) case.
3. Resolve the localized name to the canonical key via a `Stats.json` alias index (wire the
   `STAT_TRANSLATIONS` load that the plan specified but never merged).
4. Layer the value prior (G) to repair the remaining misses.

Expected outcome: non-English name accuracy from ~39% to ~90%+ on FR/JA at today's speed,
~66% on ZH at a modest cost on the rare ZH card, with no English-path regression. A and the
image-template B are not worth pursuing. If ZH accuracy must improve later, the lever is
per-language preprocessing or upscaling the ZH name crop, not more languages.

## Cleanup note

Independent of whether C ships, the runtime currently carries shelved scaffolding: the 4
non-eng Tesseract lang packs in the Dockerfile and the `Stats.json` copy are unused by the
live English-only path. If C is **not** built, remove them; if C **is** built, they become
load-bearing (and the Dockerfile already has them). The investigation harnesses
(`investigate_multilingual*.py`) were removed at the end of the session; this doc's method is
enough to recreate them. The scout/validate scripts (`find_non_english_cards.py`,
`validate_non_english_cards.py`) remain in `backend/`.

---

# Intake reference (for implementing C)

> Absorbed from the former `multilingual-ocr-plan.md`. This is the alias/contract/scout
> reference for whoever implements approach **C** (detect language once, run the existing
> name pass in that single language). The full multilingual pass (A) is the speed regression;
> prefer C. The font work in
> [echo-substat-tesseract-only.md](echo-substat-tesseract-only.md) supersedes C's stock lang
> packs with per-language `tesstrain` traineddata.

## Summary

WuWaBuilds import payloads stay canonical: localized screenshots may contain French, Japanese,
Simplified Chinese, or Traditional Chinese stat labels, but the backend still emits the English
stat keys the frontend and leaderboard pipeline consume. The multilingual layer lives at the
OCR parsing boundary:

```text
localized screenshot text -> translated stat alias -> canonical EchoStats key
```

`EchoStats.json` remains the source of truth for legal stat names and values. `Stats.json` is
copied into `backend/Data` and used only as an alias dictionary.

## Implementation contract

- `wuwabuilds/scripts/sync_backend.py` copies `Stats.json` alongside `EchoStats.json`, keeping
  the backend runtime self-contained.
- `backend/data.py` loads `STAT_TRANSLATIONS` from `backend/Data/Stats.json` when present.
- `backend/card.py` builds a normalized alias index from `Stats.json`, resolves localized stat
  labels to canonical keys, then runs the existing legal-value validation and max-main-stat
  override logic.
- Numeric value OCR stays constrained to digits, decimal points, and `%`.
- The API response shape does **not** change — the frontend localizes display labels from the
  canonical stat keys:

```json
{
  "main": { "name": "Crit DMG", "value": "44%" },
  "substats": [{ "name": "Energy Regen", "value": "10.8%" }]
}
```

## Non-English scout

`backend/find_non_english_cards.py` scans local `r2-backup` images without calling the OCR API.
It OCRs the high-signal stat-name strips inside the five echo panels and classifies likely
non-English cards by Unicode script plus fuzzy matches against non-English stat translations.
Triage fields per row: `alias_backed`, `hit_count`, `line_count`, and `build_card_signal`
(conservative validation-set signal, currently `hit_count >= 5`). Build parser validation sets
from `build_card_signal=true` rows first.

```powershell
$env:PYTHONPATH='<python-site-packages-path>'
python -u find_non_english_cards.py ..\r2-backup --workers 8 --progress-every 25
```

Generated forensic outputs (`backend/forensics/non_english_ocr_scan/`) are git-ignored.

## Validation

- Run `py wuwabuilds\scripts\sync_backend.py --skip-element-icons` so `backend/Data/Stats.json`
  exists, then run the scout across `r2-backup` and confirm known localized samples flag:
  `f32421ba8b1f3dc03de07f879703cc23da24c8abc192ecd271e441216559bcfb.jpg`
  (Japanese), `cce1a0f29186891b20e00684e3bf5853d9ab527279b2b71ee5c83b5ef6a74ae9.jpg`
  (French), `5e17036118784d4b9e4adb2c63fbbfc03522994c5dcd972536e35e6ea1addcde.jpg`
  (CJK).
- Run `backend/validate_non_english_cards.py` against the strong-candidate set for repeatable
  parser results. Latest local run: 123 strong candidates, 615 echoes, 615 passed the parser
  heuristic (FR 4/4, JA 93/93, ZH 26/26).
- Run `backend/test_frontend_split.py` on representative English and localized cards; echo
  outputs must stay schema-compatible with canonical English stat names, and legal value
  snapping must still come from `EchoStats.json`.

# Multilingual OCR Intake Plan

## Summary

WuWaBuilds import payloads stay canonical: localized screenshots may contain
French, Japanese, Simplified Chinese, or Traditional Chinese stat labels, but
the backend still emits the English stat keys consumed by the frontend and
leaderboard pipeline.

The multilingual layer lives at the OCR parsing boundary:

```text
localized screenshot text -> translated stat alias -> canonical EchoStats key
```

`EchoStats.json` remains the source of truth for legal stat names and values.
`Stats.json` is copied into `backend/Data` and used only as an alias dictionary.

## Implementation

- `wuwabuilds/scripts/sync_backend.py` copies `Stats.json` alongside
  `EchoStats.json`, keeping the backend runtime self-contained.
- `backend/data.py` loads `STAT_TRANSLATIONS` from `backend/Data/Stats.json`
  when present.
- `backend/card.py` builds a normalized alias index from `Stats.json`, resolves
  localized stat labels to canonical keys, then runs the existing legal-value
  validation and max-main-stat override logic.
- Echo stat-name OCR uses the multilingual Tesseract language set from
  `OCR_ECHO_TEXT_LANGS`, defaulting to `eng+fra+jpn+chi_tra+chi_sim`.
- Numeric value OCR stays constrained to digits, decimal points, and `%`.
- `backend/Dockerfile` installs the v1 production language packs:
  English, French, Japanese, Simplified Chinese, and Traditional Chinese.

## Non-English Scout

`backend/find_non_english_cards.py` scans local `r2-backup` images without
calling the OCR API. It defaults to the stat-name strips inside the five echo
panels, OCRs that smaller high-signal area, and classifies likely non-English
cards by Unicode script plus fuzzy matches against non-English stat
translations.

Each scan row includes triage fields:

- `alias_backed`: at least one localized stat translation was detected.
- `hit_count`: number of localized stat aliases detected.
- `line_count`: number of OCR text lines from the echo-name scout crop.
- `build_card_signal`: conservative validation-set signal, currently
  `hit_count >= 5`.

Use broad `candidate=true` rows to discover non-English screenshots. Build
parser validation sets from `build_card_signal=true` rows first; script-only
and low-hit candidates can include ordinary character screens or other UI.

Default output:

```text
backend/forensics/non_english_ocr_scan/
  candidates.csv
  candidates.json
  candidates.jsonl
  progress.json
  results.jsonl
  summary.json
```

Useful commands:

```powershell
$env:PYTHONPATH='C:\Users\domin\AppData\Roaming\Python\Python313\site-packages'
python -u find_non_english_cards.py ..\r2-backup --workers 8 --progress-every 25
python -u find_non_english_cards.py ..\r2-backup --workers 8 --limit 1000
python -u find_non_english_cards.py ..\r2-backup --since 2026-06-07T19:00:00-07:00
```

Generated forensic outputs are ignored by git.

## Validation

- Run `py wuwabuilds\scripts\sync_backend.py --skip-element-icons` after data
  sync so `backend/Data/Stats.json` exists.
- Run the scout across `r2-backup` and confirm known localized samples are
  flagged:
  - `f32421ba8b1f3dc0.jpg` as Japanese
  - `cce1a0f29186891b.jpg` as French
  - `5e17036118784d4b.jpg` as CJK
- Prefer `build_card_signal=true` rows when choosing the 50-100 localized
  images for parser validation.
- Run `backend/validate_non_english_cards.py` against the strong candidate set
  to write repeatable parser results, summary, and failure CSV files:

```powershell
$env:PYTHONPATH='C:\Users\domin\AppData\Roaming\Python\Python313\site-packages'
python -u backend\validate_non_english_cards.py --workers 8 --progress-every 10
```

- Run `backend/test_frontend_split.py` on representative English and localized
  cards. Echo outputs should remain schema-compatible and stat names should be
  canonical English.
- Confirm legal value snapping still comes from `EchoStats.json`.

Latest local validation:

- Scout run: `10,847` images decoded, `631` broad candidates,
  `123` strong `build_card_signal=true` candidates.
- Strong candidate parser run: `123` images, `615` echoes, `615` passed the
  current parser heuristic (`main.name` plus at least three substats).
- Strong candidate languages: French `4/4` images, Japanese `93/93`, Chinese
  `26/26`.

## Contract

The API response shape does not change:

```json
{
  "main": { "name": "Crit DMG", "value": "44%" },
  "substats": [{ "name": "Energy Regen", "value": "10.8%" }]
}
```

The frontend can continue localizing display labels from canonical stat keys.

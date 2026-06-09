# scanner/ — Wuthering Waves Echo Inventory Scanner

A dev-only tool that reads a player's Echo bag from the game UI and emits canonical JSON. Inspired by Inventory Kamera (Genshin) and HSR-Scanner (HSR). **Not** part of the deployed OCR API image — excluded via `backend/.dockerignore`.

## Status

- **Phase 1** (in progress): offline single-image scanner over panel screenshots. `py -m wuwa_scanner <image_path>` reads the right detail panel and returns JSON.
- **Phase 2** (planned): live capture via `mss`, manual hotkey to capture a frame.
- **Phase 3** (planned): auto-navigate the bag — `pydirectinput` click loop with panel-updated polling, scroll until the counter matches total.
- **Phase 4** (planned): PyInstaller single-file `.exe`. Bundles `Data/` (echo icons + lookups). Open-source code, distributable binary.

## Boundary with `backend/`

| Owns | Reuses from `backend/` | Doesn't touch |
|---|---|---|
| `wuwa_scanner/` package (layout, identify, extract, output) | `data.py` lookups (`ECHO_NAME_MAP`, `ECHO_COSTS`, `ECHO_ELEMENTS`, `SUB_STATS`, `MAIN_STATS`, `TEMPLATE_FEATURES`, `COST_TEMPLATES`, `determine_element`) | `server.py`, `card.py`, `batch_ocr.py` — those are the export-card pipeline |
| `bench_ocr.py`, `extract_regions.py` debug tools | `Data/Echoes/*.{png,webp}`, `Data/Costs/*.jpg`, `Data/Elements/*.webp` template assets | `Dockerfile`, `requirements.txt` (API service deps; scanner can add its own deps later) |
| `samples/`, `crops/`, `bench_results.txt` | | |

Imports walk up via `Path(__file__).resolve().parents[3]` to put `backend/` on `sys.path`. When the scanner becomes its own repo or `.exe`, this layer becomes a vendored copy of the lookup tables + Data/ subset.

## Pipeline (Phase 1)

The detail panel is read in four regions, each proportional to the full screen (calibrated at 3840×2160, works 1920–3840 because the bag UI is anchored to viewport corners):

| Region | Source | Engine | Notes |
|---|---|---|---|
| `echo_name_cost` | top-right of panel — name + `+25` + `COST N` | **Tesseract** | Bench-winner: cleaner `+25`/`COST` extraction than RapidOCR. Color mask (#efe4a4) tested + dropped — hurts more than helps. |
| `echo_stats` | mid-right — main + 6 substats, can include 2-line wraps | **RapidOCR** (ONNX) | 7/7 across all test images, ~300ms warm. Fuzzy-matches against `data.SUB_STATS` with legal-value snap. |
| `echo_icon` | upper-right — echo portrait | **SIFT + FLANN** | Cost-filtered against `data.TEMPLATE_FEATURES` (mirrors `card.match_icon`). 4K crop is 560×553 — well above the resolution floor where SIFT degraded in the 1080p-era bench. |
| `echo_element` | small badge under name | **HSV + SIFT** | Wraps `data.determine_element` (HSV histogram, SIFT fallback for same-hue clusters). |

Bench numbers per region in `bench_results.txt`; methodology in `bench_ocr.py`.

## Identity model (the key bit)

Echo names carry prefixes that change identity differently (confirmed against `wuwabuilds/lib/import/echoMatching.ts:31-49`):

- `Phantom: <base>` — cosmetic flag (`phantomIconUrl` on base record). **Same canonical ID.** Parser strips prefix, sets `phantom=True`, looks up base name.
- `Nightmare: <base>` — **distinct CDN record** with its own ID (e.g. Hecate `60000855` vs Nightmare: Hecate `60001155`). Don't strip — full string is the canonical name.
- `Reminiscence: <base>` — **not a prefix family**, just an official name (Reminiscence: Denia is `60002005`). Don't strip.
- bare — use as-is.

OCR is the only reliable signal for Phantom-vs-base because the silhouette is identical (SIFT can't separate them). Identity reconciliation: OCR drives, SIFT corroborates, SIFT-only as fallback when OCR fails.

## Resolution policy

- Calibrated at 3840×2160. Proportional crops cover 1920–3840 because the UI is 16:9 corner-anchored.
- Recommend fullscreen 16:9. Sub-1920 widths warned, not refused.
- The previous PLAN.md's hard 1080p floor was for the SIFT-confidence problem at 1000×550; not a concern at the resolutions players actually use.

## Data assets needed at runtime

- `backend/Data/Echoes/*.{png,webp}` — icon templates, SIFT keypoints precomputed at import time.
- `backend/Data/Elements/*.webp` — element-badge templates from Encore (`data.py` also supports PNG if needed).
- `backend/Data/Costs/cost{1,3,4}.jpg` — cost-badge templates (unused in current pipeline; cost comes from name OCR, but kept for fallback).
- `backend/Data/Echoes.json` + `EchoStats.json` — name/ID/cost/element registry and per-cost main-stat lookup.

For Phase 4 distribution: PyInstaller will bundle these via `--add-data Data;Data`.

## Sub-rules of thumb

- All proportional bounds live in `wuwa_scanner/layout.py`. Pixel anchors are recorded next to each bound — re-derive proportionals when the UI shifts.
- Run `py extract_regions.py` to dump all crops from `echo_bag/` into `crops/` for visual inspection. The fastest debugging loop when the layout drifts.
- Never `sleep()` in the navigation loop (Phase 3) — poll for a panel-updated pixel diff. Hard sleeps are the #1 flake source in Inventory Kamera.
- The `equipped_by` region was scoped out of v1 — the new UI's "Equipped by X" footer is unreliable to crop and not needed for inventory listing. Revisit if users ask.

## Files

```
scanner/
├── AGENTS.md             # this file
├── PLAN.md               # phased plan (parts stale — being rewritten as we ship)
├── .gitignore            # __pycache__, crops/, bench_results.txt
├── bench_ocr.py          # RapidOCR vs Tesseract bench on stats + name_cost
├── extract_regions.py    # debug crop tool over echo_bag/
├── samples/              # legacy 1080p reference shots
└── wuwa_scanner/
    ├── __main__.py       # CLI entry: scan_panel → JSON
    ├── layout.py         # REGIONS (proportional, calibrated at 4K)
    ├── identify/
    │   ├── sift.py       # cost-bucketed SIFT
    │   ├── element.py    # HSV+SIFT element classifier
    │   └── cost.py       # cost-badge template match (fallback, unused)
    ├── extract/
    │   ├── name.py       # Tesseract + Phantom-prefix parser
    │   ├── stats.py      # RapidOCR + row pairing + legal-value snap
    │   └── equipped.py   # (scoped out of v1)
    └── output/           # (canonical JSON writer, TBD)
```

# Wuwa Echo Scanner — Design & Plan

A standalone in-app scanner for the Wuthering Waves Echo inventory. Inspired by Inventory Kamera (Genshin) and HSR-Scanner (Honkai: Star Rail), but built on the matching pipeline already proven in `card.py` and modernized for 2025-era tooling (ONNX OCR, optional CNN fallbacks).

## Goals
- **Scope (v1):** Echo inventory only. Per-echo: identity, cost, level, sonata set(s), main stat, sub stats (1-5), locked flag, equipped-by character.
- **Resolution-agnostic** via proportional crops; minimum 1080p (1920x1080) enforced.
- **User-agnostic:** no UID assumption, no character roster assumption.
- **Pacing-agnostic:** per-step delays configurable; no hard `sleep()`.
- **Optimize accuracy AND efficiency:** modern OCR (RapidOCR ONNX), template+CNN matchers, pre-bucketed feature stores.

## What we keep from `card.py` / `data.py`
| Borrowed | Lines | Purpose |
|---|---|---|
| `data.TEMPLATE_FEATURES` | data.py:55, 178 | Pre-computed SIFT keypoints/descriptors for 162 echo templates |
| `data.ICON_TEMPLATES` | data.py:54 | 188×188 BGR templates (for color-histogram tiebreaker) |
| `data.COST_TEMPLATES` | data.py:58, 144-152 | Cost badge templates (1/3/4) — fast template match |
| `data.ECHO_COSTS` / `ECHO_ELEMENTS` / `ECHO_NAME_MAP` | data.py:51-53, 97-105 | Lookup tables — cost prefilter and ID→name |
| `data.SUB_STATS` | data.py:48 | Canonical stat names + legal value lists for snap |
| `data.MAIN_STATS` / `DEFAULT_MAIN_STATS` | data.py:49 | Main stat values derived from cost+name (deterministic, all Lv.25) |
| `data.CHARACTER_NAMES` | data.py:40 | Fuzzy-match target for "Equipped by" |
| `card.match_icon` shape | card.py:406-508 | SIFT+FLANN with ratio 0.7, nightmare-variant tiebreak, color disambig |
| `card.choose_substat_value` | card.py:138-151 | Dual-OCR fallback when value is illegal for stat |
| `card.max_main_stat_value` | card.py:162-173 | OCR'd main value discarded; looked up by (cost, stat) |
| `determine_element` (HSV+SIFT) | data.py:207-267 | Sonata icon → element with hue-cluster SIFT fallback |

## What's new
- **Capture loop** (`mss`) and **input driver** (`pydirectinput` on Windows).
- **Detail-panel crop schema** (proportional, see `extract_regions.REGIONS`).
- **Grid-cell crop schema** (per visible cell: icon, cost badge, level "+N", lock icon).
- **State machine** for inventory navigation: enumerate page → click each cell → wait for panel update → read → next.
- **RapidOCR-first** for free-form stats; Tesseract kept as fallback (no per-request server, single warm process).
- **Cost-bucketed templates** — pre-grouped at load so cost-filtered SIFT iterates only ~40 of 162.

## Bench results (samples: `bag_view_01.png` 1920×1080, `bag_view_02.jpg` 1000×550)

| Step | Sample 1 (1080p) | Sample 2 (1000×550) |
|---|---|---|
| Cost badge match | 1ms, **OK** (cost=1) | 0.5ms, **OK** (cost=4) |
| SIFT full (162 templates) | 490ms, **OK** Devotee's Flesh @ 0.099 (next @ 0.081) | 870ms, **WRONG** (Dreamless at #12) |
| SIFT cost-filtered (39-72 templates) | 210ms, **OK** @ 0.095 (next @ 0.068) | 214ms, **WRONG** (Dreamless at #2, conf 0.052 vs Sentry Construct 0.101) |
| SIFT upscale 384×384 | n/a | 307ms, **WRONG** — upscale doesn't help (info already lost in source) |
| Stats RapidOCR | 333ms | 503ms |
| Stats parser | **7/7 rows** | **7/7 rows** |
| Echo name Tesseract+fuzzy | 124ms, fuzzy 100.0 | 103ms, fuzzy 100.0 |
| Equipped-by RapidOCR | 1020ms, **OK** (low conf) | 1059ms, **OK** (100) |

Cold-start: RapidOCR init ≈ 260ms (one-time per process).

**Key takeaway:** SIFT is reliable when the icon preview is ≥ ~250×200 px (native 1080p+). Below that, JPG/aliasing kills keypoint distinctiveness and similar silhouettes (Sentry Construct vs Dreamless) collide. **Recommendation: enforce 1080p+ native game resolution, like Kamera and HSR-Scanner.**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Scanner Process (single, long-running)                          │
│                                                                 │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Capture  │→ │ Anchor &   │→ │ Per-cell │→ │ Per-panel    │   │
│  │ (mss)    │  │ Layout     │  │ classify │  │ extract      │   │
│  └──────────┘  │ (proport.) │  │ (cost,   │  │ (SIFT,       │   │
│                └────────────┘  │  level)  │  │  RapidOCR,   │   │
│                                └─────┬────┘  │  HSV element)│   │
│                                      │       └──────┬───────┘   │
│                                      ▼              │           │
│                                ┌──────────┐         │           │
│                                │ Input    │◄────────┘           │
│                                │ driver   │ (click next cell)   │
│                                │ (pydirectinput)                │
│                                └──────────┘                     │
│                                                                 │
│                          Outputs:                               │
│                          ├─ echoes.json  (canonical)            │
│                          └─ debug/      (failed crops)          │
└─────────────────────────────────────────────────────────────────┘
```

## Module layout (proposed)
```
wuwa_scanner/
  __main__.py             # CLI entry
  capture.py              # mss screen grab, monitor selection
  anchors.py              # detect inventory panel bounds (template match)
  layout.py               # REGIONS dict (proportional), grid cell rects
  navigate.py             # click loop, scroll, page-end detection
  identify/
    cost.py               # template match (3 classes)  ← reuses COST_TEMPLATES
    sift.py               # cost-bucketed SIFT          ← reuses TEMPLATE_FEATURES
    element.py            # HSV + SIFT for sonata       ← reuses determine_element
    color_tiebreak.py     # histogram for close matches ← reuses compare_icon_colors
  extract/
    name.py               # Tesseract single-line + fuzzy
    stats.py              # RapidOCR + row pairing parser (legal-value snap)
    equipped.py           # RapidOCR + fuzzy vs CHARACTER_NAMES
    level.py              # small digit classifier or Tesseract digits-only
  output/
    canonical.py          # dataclass → dict
    writer.py             # json/yaml dump
  config.py               # delays, resolution policy, hotkeys
data/                     # symlink to existing Data/
```

## Resolution policy
- **Auto-detect** monitor and game-window resolution at startup.
- **Hard floor: 1920×1080 native (windowed or fullscreen).** Below that, refuse with a clear error.
- Aspect tolerance: 16:9 (1.77) ± 5%. Out-of-range warns but proceeds with reduced confidence threshold.
- All region coords are proportional (0-1) — `extract_regions.REGIONS` already does this; tested across 1920×1080 and 1000×550 with same bounds working for crops.

## Pacing policy
Default delays (configurable in `config.py`):
- `click_settle_ms`: 150 — between mouse-down and reading panel
- `panel_redraw_ms`: 250 — after click before OCR
- `scroll_settle_ms`: 400 — after a page scroll
- `inter_cell_ms`: 50

Per-cell budget at defaults ≈ 450ms (delay) + 350ms (OCR+SIFT cost-filtered) ≈ **800ms/cell**, so 2000 echoes ≈ 27 min. Tunable down for fast machines, up for slow.

**No hard `sleep`** beyond settle delays — panel-updated detection samples a known-changing pixel after click (e.g. the cost badge color or the echo name text region) and polls until it differs from the prior cell.

## Identity strategy (the hard part)
A single channel isn't reliable across resolutions. Voting:

```
For each cell:
  1. cost_badge_match    → cost ∈ {1,3,4}          [3 templates, ~ms]
  2. sift_cost_filtered  → top-K candidates        [~200ms cost-bucketed]
  3. if top1.conf > 2× top2.conf  → accept top1
     elif top1 and top2 in same element cluster → element disambig via sonata icons
     elif still ambiguous → color_histogram tiebreak (card.py:350-404)
     else                → mark UNCERTAIN, dump crop to debug/
```

**Failure mode handled:** the Sample-2 case (Dreamless at #2 with 2× gap behind Sentry Construct) would currently mark UNCERTAIN. Honest call — don't silently emit wrong data. v2: add a small 162-class CNN fallback (~5MB ResNet-18 trained on templates + augmentations) for cases SIFT can't separate.

## Stats strategy
- **RapidOCR primary** on the unified `stats_block` crop (main stat + substats together).
- Row parser pairs lines → fuzzy-match against `SUB_STATS` keys, with wrap-continuation heuristic.
- Apply **legal-value snap** (`SUB_STATS["ATK"] = [30,40,50,60]`) — if OCR value differs from nearest legal by >2.0, snap.
- Main stat value: **discard OCR'd value, look up from (cost, stat) in `MAIN_STATS`** — deterministic since all echoes are Lv.25.
- **Tesseract fallback** ONLY for the substat row whose value fails legal-snap (mirrors card.py:138-151).

## Equipped-by strategy
- Crop the bottom footer band of the detail panel.
- Strip "Equipped by" prefix (regex), fuzzy match against `CHARACTER_NAMES`.
- If score < 70 → mark UNKNOWN.

## Output schema (canonical)
```json
{
  "version": "1.0",
  "captured_at": "2026-05-27T...",
  "resolution": [1920, 1080],
  "echoes": [
    {
      "id": "60001105",
      "name": "Devotee's Flesh",
      "cost": 1,
      "level": 25,
      "locked": false,
      "sonata_set": "Empyrean Anthem",
      "main_stat": {"name": "HP%", "value": 22.8},
      "substats": [
        {"name": "HP", "value": 2280},
        {"name": "Crit Rate", "value": 6.9},
        ...
      ],
      "equipped_by": "Cartethyia",
      "confidence": {
        "id": 0.93,
        "stats": 1.0
      }
    }
  ]
}
```

## Phasing

**Phase 0 — Recon (current):** crops calibrated, RapidOCR+Tesseract+SIFT benched on 2 samples. ✅

**Phase 1 — Offline scanner (no automation):**
- Process a folder of screenshots (same shape as `batch_ocr.py`)
- Emit canonical JSON
- Confidence-flagged uncertain rows surface in a debug folder for manual review
- ~1 week

**Phase 2 — Live capture (manual trigger):**
- `mss` capture on hotkey
- User scrolls; tool reads each visible page when triggered
- ~3 days on top of Phase 1

**Phase 3 — Auto-navigate:**
- Anchor-detect inventory panel and grid origin every frame
- `pydirectinput` click loop with panel-updated polling
- Scroll/page-end detection (counter OCR "342/2000")
- ~1 week

**Phase 4 — Hardening:**
- Color-histogram tiebreaker integration (lift `compare_icon_colors`)
- Small CNN classifier for low-res icon disambiguation (~5MB, optional)
- 4K/1440p test coverage
- Packaging: PyInstaller single-file .exe

## Open questions (when you wake up)
1. **Distribution target:** standalone .exe (PyInstaller, ~80MB) or pip-installable? Affects whether we bundle the 200MB Data/ folder or fetch it on first run from R2.
   - **Decided: single .exe via PyInstaller**, Data/ bundled. Phase 4 packaging targets ~80MB binary.
2. **Output consumer:** is there a frontend that'll eat the JSON, or is this raw export for now? If the existing wuwa-ocr-api response shape is the consumer, we should match it.
3. **Sort dependency:** UI offers Sort by Level / Sort by Cost. Do we lock sort to a known order before scanning (recommended; otherwise scroll-back-to-top detection is harder)?
4. **Lock-state read:** the lock icon on the grid cell is binary — do you want it scanned? (cheap, 1 template match per cell).
5. **Equipped-by reliability:** at 1080p the footer is 14-16px tall — Tesseract works but it's tight. Would you accept "unknown" when uncertain rather than guessing?
6. **Resolution policy:** sub-1080p captures — hard floor, CNN fallback, or mark uncertain?
   - **Decided: hard 1080p floor.** Scanner refuses below 1920×1080 native at startup. Matches Kamera / HSR-Scanner. Removes the Dreamless-class failure without needing a CNN. CNN deferred indefinitely.

# Wuwa Echo Scanner — Plan & Decisions

A dev-only tool that scans a player's Echo inventory in Wuthering Waves. Inspired by Inventory Kamera (Genshin) and HSR-Scanner (HSR). Distributed as a single `.exe` (PyInstaller, Phase 4) bundling the matching pipeline already proven in the OCR API.

Architecture, boundary with `backend/`, file layout, and the prefix model live in [AGENTS.md](AGENTS.md). This file tracks **direction, decisions, and progress** — what's done and what's next.

## Goal (v1)

Scan the Echo bag, emit canonical JSON per echo: `id, name, phantom, cost, level, element, main_stat, sub_stats[]`, plus per-field confidence. Run end-to-end on a folder of screenshots (Phase 1), then live capture (Phase 2), then auto-navigate the bag (Phase 3), then ship a `.exe` (Phase 4).

**Out of scope for v1:** equipped-by character, lock state, sort-order discovery, character/weapon/echo-set scanning.

## Decisions

- **Calibrate proportional crops at 4K (3840×2160).** Same bounds work down to 1920×1080 because the bag UI is 16:9 corner-anchored. **No hard resolution floor** — recommend fullscreen + 16:9 at startup, warn below 1920px width, refuse only if aspect ratio is out of 16:9 ±5%.
  - Reverses the old 1080p hard-floor decision, which was driven by SIFT confusion at 1000×550. At resolutions players actually use (1080p–4K) the icon preview is 280–560 px wide and SIFT is reliable.
- **Per-region OCR engines, picked by bench (see [bench_ocr.py](bench_ocr.py) + [bench_results.txt](bench_results.txt)):**
  - `echo_stats` → RapidOCR (raw BGR). 7/7 across 3 test images, ~300ms warm.
  - `echo_name_cost` → Tesseract (raw BGR). Cleaner `+25` and `COST` extraction than RapidOCR. ~265ms warm.
  - Color-mask preprocess (#efe4a4) **tested and dropped** — fragments antialiased strokes, hurts accuracy.
- **OCR-driven identity, SIFT corroborates.** OCR is the only signal that captures the `Phantom: ` prefix (silhouette is identical, SIFT can't disambiguate). When OCR resolves to a canonical name, that wins; SIFT is fallback and a sanity check (`ocr_vs_sift_agree` flag in the output).
- **Prefix model** (per `wuwabuilds/lib/import/echoMatching.ts:31-49`): `Phantom: ` strip + set `phantom=True`; `Nightmare:` / `Reminiscence:` are part of the canonical name, **don't strip**. Documented in detail in [AGENTS.md](AGENTS.md#identity-model-the-key-bit).
- **Stats parsing borrowed wholesale** from the OCR API: RapidOCR rows → fuzzy-match `data.SUB_STATS` → legal-value snap. Wrap-continuation handles two-line substats (`Resonance Liberation DMG Bonus`). Main stat is the first row; the OCR'd value is kept (not looked up from `(cost, stat)` since UI shows it directly).
- **Reuse, don't fork.** Scanner imports `backend/data.py` (`ECHO_NAME_MAP`, `ECHO_COSTS`, `ECHO_ELEMENTS`, `SUB_STATS`, `TEMPLATE_FEATURES`, `determine_element`) and `backend/Data/` assets via `sys.path` walk-up. At Phase 4 packaging, these get vendored into the `.exe`.
- **No CNN fallback.** The Phase-4 plan to ship a small classifier for low-res icon disambiguation is deferred indefinitely — the resolution policy removes the failure mode it was meant to fix.
- **No equipped-by, no lock-state, no grid-scan in v1.** Cheaper to click every cell and read the panel than to risk grid-level misreads. Per-cell budget (~1.2s) keeps a 2000-echo bag at ~40 min, acceptable for a one-shot import flow.

## Bench results (4K, on echo_bag/ ground-truth set, end-to-end)

3 screenshots × 1 echo each. See [bench_results.txt](bench_results.txt) for raw OCR output and `wuwa_scanner/__main__.py` for the integration.

| Echo | ID | Phantom | Cost | Level | Element | Stats | OCR=SIFT? |
|---|---|---|---|---|---|---|---|
| Phantom: Sigillum | `60001915` | **true** | 4 | 25 | Trailblazing | **7/7** | ✓ |
| Nightmare: Hecate | `60001155` | false | 4 | 25 | Dream | **7/7** | ✗ — SIFT picked base Hecate `60000855`; OCR correctly resolved Nightmare variant |
| Reminiscence: Denia | `60002005` | false | 4 | 25 | Chromatic | **7/7** | ✓ |

The Hecate row validates the OCR-driven design. Without the prefix-aware text signal, SIFT misidentifies Nightmare variants as their base echo (silhouettes are too similar). The reconciliation rule + `ocr_vs_sift_agree=false` flag surfaces the mismatch in the output rather than silently shipping wrong data.

**Per-region warm latency (RapidOCR cold ≈ 790ms, Tesseract cold ≈ 1180ms; warmed once per process):**

| Region | Engine | Latency | Notes |
|---|---|---|---|
| `echo_stats` | RapidOCR | ~300ms | 7/7 rows across all 3 test images |
| `echo_name_cost` | Tesseract | ~265ms | name + `+25` + `COST` all extracted |
| `echo_icon` | SIFT (cost-filtered) | not benched at 4K yet | works in end-to-end runs; Hecate Nightmare-vs-base is the known confusion case |
| `echo_element` | HSV + SIFT | trivial | reuses `data.determine_element` |

**Per-echo OCR budget:** ~565ms (Tesseract name + RapidOCR stats) + SIFT (~200ms est.) ≈ **~800ms compute**, plus click+settle in Phase 3.

## Phases

- **Phase 0 — Recon.** ✅ Crops calibrated at 4K. OCR engines benched. Prefix model confirmed against frontend.
- **Phase 1 — Offline scanner (folder of screenshots).** ✅ `py -m wuwa_scanner <image>` returns canonical JSON. 3/3 echo_bag screenshots verified end-to-end.
- **Phase 2 — Live capture (manual hotkey).** ⏳ `mss` grab on hotkey, scan the currently-shown detail panel. User does the scrolling. Inherits everything from Phase 1; only the capture frontend is new.
- **Phase 3 — Auto-navigate.** ⏳ `pydirectinput` click loop, panel-updated polling (sample a pixel that's guaranteed to change between echoes — e.g. the echo_name_cost text region or the icon centroid — and poll until it differs from the previous read; never `sleep()`). Scroll/page-end via counter OCR ("X/3000") or a sentinel "Echoes 1227/3000" string match. Modal/dialog guard.
- **Phase 4 — Packaging.** ⏳ PyInstaller `--add-data Data;Data` → single `.exe` (~80MB target). Open-source code, distributable binary.

## Open items

- **Sort lock-in for Phase 3.** Should the scanner enforce a known sort order (e.g. "Sort by Level") before scanning so scroll-back-to-top is detectable? Recommended yes.
- **Output consumer.** Is the canonical JSON eaten by the existing `wuwabuilds/lib/import/` flow, or is it raw export? If the former, match the shape (this is a quick alignment once the importer side is identified).
- **4K SIFT bench.** Hecate's OCR/SIFT disagreement is expected for Nightmare variants, but we haven't quantified SIFT confidence margins across the full 162-echo set at 4K. A dedicated bench would let us tune the `is_confident` ratio floor (currently 2.0) and decide whether to fall back to OCR earlier when SIFT is borderline.
- **Lock state.** Cheap to scan (1 template match per cell). Worth adding to v1 if downstream tooling cares; skip otherwise.
- **Equipped-by.** Footer text is OCR-readable but currently scoped out. Re-enable if the output consumer asks for it.

## What's done in `scanner/`

Tracked here as a stable summary; AGENTS.md has the per-file map.

- **Crops** — `wuwa_scanner/layout.py` with 4 proportional regions calibrated against the 4K echo_bag screenshots; pixel anchors recorded next to each entry. `extract_regions.py` dumps all crops from any source dir into `crops/` for visual inspection.
- **OCR engine bench** — `bench_ocr.py` runs RapidOCR vs Tesseract on `echo_stats` + `echo_name_cost` (raw + #efe4a4 mask), scores against hard-coded ground truth, dumps raw output to `bench_results.txt`.
- **Name+cost+level** — `wuwa_scanner/extract/name.py` (Tesseract on raw crop, prefix-aware: Phantom strip + flag, Nightmare/Reminiscence keep). Returns `NameRead(raw_text, raw_name_line, name, phantom, cost, level)`.
- **Stats** — `wuwa_scanner/extract/stats.py` (RapidOCR, row pairing, fuzzy-match `SUB_STATS`, legal-value snap, wrap-continuation for 2-line substats).
- **Echo SIFT identification** — `wuwa_scanner/identify/sift.py` (cost-bucketed SIFT against `data.TEMPLATE_FEATURES`, returns ranked candidates with confidence).
- **Element classification** — `wuwa_scanner/identify/element.py` (thin wrapper over `data.determine_element` — HSV histogram + SIFT fallback for same-hue clusters).
- **End-to-end CLI** — `wuwa_scanner/__main__.py` ties the four together, reconciles OCR vs SIFT identity (OCR wins, SIFT corroborates), returns canonical JSON. `py -m wuwa_scanner <image>`.
- **Docs** — `AGENTS.md` (architecture + boundary + reuse map + prefix model), this `PLAN.md`. `.dockerignore` updated to exclude `scanner/`, `echo_bag/`, `r2-backup/` from the API image.

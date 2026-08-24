# scanner — Echo Inventory Scanner

Reads a player's in-game Echo inventory and emits canonical JSON. It is excluded
from the deployed OCR API image.

## Start Here

`PLAN.md` contains the architecture, measured evidence, decisions, and bug log.
Use it for behavioral context; keep this file to routing and easy-to-break
constraints.

| Area | Location |
| --- | --- |
| Scanner implementation | `wuwa_scanner/` |
| Regression and benchmark tools | `bench/` |
| Labelled fixtures | `samples/` |
| Shared game data and lookups | parent `Data/` and `data.py` |

The scanner owns `wuwa_scanner/`, `bench/`, and `samples/`. It reuses backend
data but does not modify the export-card pipeline in `server.py` or `card.py`.

## Working Model

- Tile census provides identity, cost, set, level, lock, and equipped state.
  The detail panel provides substats only. Lock and equipped are not wired yet.
- Stat names come from icon matching. Never infer a stat name from its value.
  OCR reads one localized substat-number cell at a time in recognition-only
  mode.
- Read a glyph by isolating its ink, not by correlating the crop it sits in.
  The cost digit is masked by gold hue because it sits on artwork; the level
  pill is Otsu-thresholded because it sits on flat chrome. Both then normalise
  to the ink's own bounding box.
- The OCR engine is chosen per FIELD. `ocr.default_reader()` for substat values
  (WinRT, 5/5), `ocr.level_reader()` for level pills (Tesseract at 4x upscale;
  WinRT returns nothing at all on a two-digit crop).
- Echo identification order is cost prefilter, gradient match, hue arbitration,
  then a family-scoped set badge. Only cost may narrow the candidate pool, and
  it must abstain to the full pool when uncertain.

## Guardrails

- Detect the scrolling grid lattice on every frame; a fixed row origin is valid
  only at scroll-top.
- Self-locate panel rows and value ink. Do not reuse icon row bands for wrapped
  values or batch the value column into one OCR pass.
- Poll for panel changes and stability; do not use fixed sleeps or scroll
  arithmetic for navigation.
- Preserve ambiguity warnings and gate template decisions on the margin between
  candidates, not an absolute score. A margin gate only protects you if the
  score is measuring the signal: the old cost reader abstained on a 0.003
  margin between two readings of the same diamond frame.
- Any change to the cost or level reader must be measured on every fixture, not
  a convenient one. `bench_fields.py` labels are hand-read from the tiles;
  never derive them from the identified echo, since identity is prefiltered by
  cost and the check would pass by construction.
- Fixed boxes over text are a re-check, not a constant. Sweep the bounds across
  every labelled tile and gate on "does ink touch an edge", then sit in the
  middle of the plateau.
- Key inventory entries by row and column; never deduplicate them by content.
- Keep geometry relative to the 16:9 game client rect. Warn on unsupported
  aspect ratios rather than silently mis-cropping.
- `bench/fetch_phantom_icons.py` must follow the frontend's
  `wuwabuilds/lib/echo.ts` URL-resolution rules, including local `/assets/`
  mirrors and already-absolute URLs.

## Verification

After changing layout, geometry, matching, or OCR behavior, run all three
shipped regressions from `backend/scanner/`:

```text
py bench/bench_census.py     # identity + sonata            24/24, 24/24
py bench/bench_fields.py     # cost + level + selection     90/90, 90/90, 5/5
py bench/validate_e2e.py     # stat icons + substat values  21/21, 15/15
```

`bench/fit_cost_masks.py --write` regenerates `wuwa_scanner/templates/cost_*.png`
and is only needed when the bag UI changes. It refuses to fit on a frame that
does not contain all three costs.

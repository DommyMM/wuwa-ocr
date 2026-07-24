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
  The detail panel provides substats only.
- Stat names come from icon matching. Never infer a stat name from its value.
  OCR reads one localized substat-number cell at a time in recognition-only
  mode.
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
  candidates, not an absolute score.
- Key inventory entries by row and column; never deduplicate them by content.
- Keep geometry relative to the 16:9 game client rect. Warn on unsupported
  aspect ratios rather than silently mis-cropping.
- `bench/fetch_phantom_icons.py` must follow the frontend's
  `wuwabuilds/lib/echo.ts` URL-resolution rules, including local `/assets/`
  mirrors and already-absolute URLs.

## Verification

After changing layout, geometry, matching, or OCR behavior, run both shipped
regressions from `backend/scanner/`:

```text
py bench/bench_census.py
py bench/validate_e2e.py
```

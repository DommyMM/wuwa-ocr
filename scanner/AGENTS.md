# Echo inventory scanner

Reads a player's in-game Echo inventory and emits canonical JSON. Excluded from the deployed OCR image.

## Start here

`PLAN.md` holds the architecture, measured evidence, decisions, bug lessons and open work. This file keeps to routing and easy-to-break constraints.

| area | location |
| --- | --- |
| Scanner implementation | `wuwa_scanner/` |
| Regression and benchmark tools | `bench/` |
| Labelled fixtures | `samples/` |
| Shared game data and lookups | parent `Data/` and `data.py` |

The scanner owns `wuwa_scanner/`, `bench/` and `samples/`. It reuses backend data but never modifies the export-card pipeline in `server.py` or `card.py`.

## Working model

- Tile census gives identity, cost, set, level and selection, and the detail panel gives substats only. Lock and equipped show on the tile but aren't read yet
- Stat names come from icon matching. Never infer a stat name from its value
- OCR reads only localized number cells (substat values, tile level pills), each as its own image, in recognition-only mode
- Read a glyph by isolating its ink, not by correlating the crop it sits in. The cost digit is masked by gold hue since it sits on artwork, the level pill is Otsu-thresholded since it sits on flat chrome, and both normalise to the ink's own bounding box
- The OCR engine is chosen per field: `ocr.default_reader()` (WinRT) for substat values, `ocr.level_reader()` (Tesseract at 4x) for level pills, since WinRT returns nothing on a two-digit crop
- Echo identification runs cost prefilter, gradient match, hue arbitration, then a family-scoped set badge. Only cost may narrow the candidate pool, and it must abstain to the full pool when unsure

## Guardrails

- Detect the scrolling grid lattice on every frame, since a fixed row origin holds only at scroll-top
- Self-locate panel rows and value ink. Never reuse icon row bands for wrapped values or batch the value column into one OCR pass
- Poll for panel changes and stability rather than fixed sleeps or scroll arithmetic
- Keep ambiguity warnings and gate template decisions on the margin between candidates, not an absolute score. A margin gate only protects you if the score measures the signal: the old cost reader abstained on a 0.003 margin between two readings of the same diamond frame
- Measure any cost or level reader change on every fixture, not a convenient one. `bench_fields.py` labels are hand-read from the tiles, never derived from the identified echo, since identity is prefiltered by cost and the check would pass by construction
- Fixed boxes over text are a re-check, not a constant. Sweep the bounds across every labelled tile, gate on whether ink touches an edge, then sit in the middle of the plateau
- Key inventory entries by row and column, never deduplicate them by content
- Keep geometry relative to the 16:9 game client rect, and warn on unsupported aspect ratios rather than mis-cropping silently
- `bench/fetch_phantom_icons.py` must follow the frontend's `wuwabuilds/lib/echo.ts` URL resolution, including local `/assets/` mirrors and already-absolute URLs

## Verification

After changing layout, geometry, matching or OCR behavior, run all three regressions from `backend/scanner/`. `validate_e2e.py` reads values with its own Tesseract loop, so it doesn't cover `panel.read_substats` or WinRT.

```text
py bench/bench_census.py     # identity + sonata            24/24, 24/24
py bench/bench_fields.py     # cost + level + selection     90/90, 90/90, 5/5
py bench/validate_e2e.py     # stat icons + substat values  21/21, 15/15
```

`bench/fit_cost_masks.py --write` regenerates `wuwa_scanner/templates/cost_*.png` and is only needed when the bag UI changes. It refuses to fit on a frame that doesn't contain all three costs.

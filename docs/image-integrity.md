# Build Card Image Integrity

This document captures the current plan for detecting invalid or manipulated
build-card screenshots before they are trusted as OCR/training data.

## Scope

The goal is not immediate automatic deletion. The first production target should
be a review queue with evidence: key, verdict, reason, per-panel metrics, and
overlay/debug artifacts.

## Observed Evidence

In one confirmed manipulated KuroBot-format card, the last three echo panels
were visually edited. The strongest measured signal was not generic "gold looks
different"; it was a **position-aware dark row-run outlier** in echo panels 3-5.
In a scan of 4,419 local root images:

| Panel | Suspect dark-row avg | Local percentile | Local median | Local p95 |
| --- | ---: | ---: | ---: | ---: |
| echo1 | 0.852 | 18.6% | 0.877 | 0.915 |
| echo2 | 0.913 | 94.2% | 0.874 | 0.914 |
| echo3 | 0.411 | 99.75% | 0.152 | 0.170 |
| echo4 | 0.315 | 99.84% | 0.00496 | 0.00732 |
| echo5 | 0.315 | 99.84% | 0.00392 | 0.00552 |

This matters because the five panels are not interchangeable. Panels 1-2 can be
naturally dark in clean images, while panels 4-5 normally have almost no long
near-black row strips. Integrity checks should compare each echo crop to the
historical distribution for the same panel position.

One known non-KuroBot card from a different build-card generator failed a cheap
shape gate before subtle forensics were needed: it was a large JPG with a
non-16:9 aspect ratio. Existing KuroBot frontend crop coordinates landed on the
wrong content and OCR returned low-confidence nonsense.

JPG-only local backup stats:

| Extension | Count | Min | Median | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `.jpg` | 4,376 | 181,439 | 329,025 | 464,154 | 871,550 |

There are also 43 local `.png` files, all 1920x1080 and much larger
(~1.5-2.4 MB). Size checks must be format-aware. For current JPG uploads, a
large size is a useful review hint, but dimensions/aspect are cleaner than size.

## Current Scripts

- `scan_image_integrity.py`
  - Primary harness for day-to-day use.
  - Accepts either an image directory or a single image file.
  - Builds position-aware panel baselines from the scanned directory, or from
    `--baseline <dir>` when scanning a single file.
  - Writes `integrity_scan.csv`, `panel_metrics.csv`, and `review_queue.json`.
  - Verdicts:
    - `ok`: no current rule fired.
    - `review`: non-fatal review hints, such as non-JPG local artifacts.
    - `suspect`: echo-row integrity outlier.
    - `reject`: decode/shape/layout-level failure.

- `review_integrity_gui.py`
  - Browser-based review UI for `scan_image_integrity.py` output.
  - Shows the image, verdict, reasons, and panel metrics on one page.
  - Buttons and keyboard shortcuts mark `keep`, `delete`, or `review`.
  - Writes `review_decisions.json` next to the queue and exports delete
    decisions to `backend/invalid_images.json`, which is the input consumed
    by `clean_invalid.py`.

- `forensics_echo_integrity.py`
  - Crops the five frontend echo regions.
  - Writes panel crops, gold/row overlays, edge maps, ELA heatmaps, `metrics.csv`,
    and `report.json`.
  - Debug-only. Use when a flagged card needs visual artifacts for inspection.

- `baseline_echo_row_darkness.py`
  - Scans a local image directory.
  - Computes row darkness and longest dark-run ratios per panel.
  - Reports panel-position p99.5 thresholds and top outliers for panels 3-5.
  - Debug-only. Its production-relevant behavior is now folded into
    `scan_image_integrity.py`.

Example commands:

```powershell
py scan_image_integrity.py `
  "..\r2-backup" `
  --out "..\forensics\integrity_scan"

py scan_image_integrity.py `
  "..\r2-backup\<image-key>.jpg" `
  --baseline "..\r2-backup" `
  --out "..\forensics\one_image_scan"

py review_integrity_gui.py `
  "..\forensics\integrity_scan\review_queue.json" `
  "..\r2-backup" `
  --verdict suspect

py clean_invalid.py
py clean_invalid.py --run

py forensics_echo_integrity.py `
  "..\r2-backup\<image-key>.jpg" `
  --reference "..\trusted-reference\<reference-card>.jpg" `
  --out "..\forensics\<image-key>"

py baseline_echo_row_darkness.py `
  "..\r2-backup" `
  --suspect "..\r2-backup\<image-key>.jpg"
```

## Recommended Gates

Run gates from cheapest to most expensive.

### 1. Decode and Shape Gate

Reject or review when:

- File magic does not match an accepted image format.
- Decode fails.
- Aspect ratio is not close to 16:9.
- Resolution is too small for OCR.
- The current JPG upload is unusually large and also has bad dimensions/aspect.

Current evidence:

- The known non-KuroBot sample is the only aspect/resolution outlier in the
  local root image scan.
- File size alone should not reject PNGs or future high-quality images.

### 2. KuroBot Layout Gate

Check fixed visual anchors before OCR:

- Expected character/watermark region exists.
- Echo-panel band exists where the frontend crop coordinates expect it.
- QR/card/weapon regions roughly match the KuroBot build-card layout.

This catches rival-site cards where the image is valid but not our card format.

### 3. OCR Sanity Gate

Review when OCR output is structurally implausible:

- Missing UID/character.
- Fewer than 3-5 plausible echo analyses.
- Very low echo-template confidence across most panels.
- Non-stat values in main/substat slots after parsing.

OCR sanity should be a review signal, not the only detector, because edited cards
can still OCR cleanly.

### 4. Echo Integrity Gate

Use position-aware baselines:

- Compare `echo1` to historical `echo1`, `echo2` to historical `echo2`, etc.
- Start with dark-row averages and longest dark-run ratios.
- Mark review if at least two of `echo3`, `echo4`, `echo5` exceed p99.5 for
  their panel position.

For the confirmed manipulated image, all three of `echo3`, `echo4`, and `echo5`
crossed the p99.5 dark-row threshold. `echo1` and `echo2` did not form the same
evidence pattern.

### 5. Manual Review Queue

Recommended review record:

```json
{
  "key": "<image-key>.jpg",
  "buildId": "<optional-build-id>",
  "verdict": "suspect",
  "reason": "echo_row_darkness_outlier",
  "panels": {
    "echo3": { "darkAvg": 0.411, "percentile": 99.75 },
    "echo4": { "darkAvg": 0.315, "percentile": 99.84 },
    "echo5": { "darkAvg": 0.315, "percentile": 99.84 }
  },
  "artifacts": {
    "overlay": "forensics/<image-key>/overlay.png",
    "report": "forensics/<image-key>/report.json"
  }
}
```

Do not auto-delete initially. Generate a queue, review the top outliers, then
promote high-confidence rules to automated rejection only after false positives
are understood.

## Upload-Time vs Cron

Use both.

Upload-time:

- Run decode/shape/layout checks synchronously.
- Run the cheap echo-row integrity check after image upload.
- If suspicious, store the image/report and mark as `needs_review` instead of
  trusting it immediately.

Cron/backfill:

- Rescan R2 or `r2-backup` nightly.
- Recompute baselines after UI changes.
- Produce a review CSV/JSON and artifact overlays.
- Backfill verdicts for existing keys.

## Machine Learning Option

An ML classifier is viable, but should be a second phase.

The easy ML task is **KuroBot card vs non-KuroBot card**:

- Positives: trusted 16:9 KuroBot cards from R2.
- Negatives: rival-site cards, random screenshots, malformed images, old invalid
  submissions.
- Model: small image classifier on downscaled full-card images or layout crops.
- Deployment: ONNX or lightweight PyTorch model in `backend`.
- Output: `kurobot_probability`; route low scores to review/reject.

The harder ML task is **edited KuroBot card vs clean KuroBot card**:

- We currently have one confirmed manipulated image, not enough labeled examples.
- Synthetic negatives can be generated by pasting altered row strips/text into
  clean cards, but thresholds from deterministic OpenCV checks should remain the
  first line of defense.
- ML should be additive evidence, not the sole deletion trigger.

Recommended ML rollout:

1. Build deterministic gates first.
2. Save review decisions as labels.
3. Train a small binary layout classifier for non-KuroBot rejection.
4. Later train an edited-card classifier after enough confirmed edits exist.

## Next Implementation Steps

1. Turn `baseline_echo_row_darkness.py` into a reusable module function.
2. Add a CLI that emits review JSON for all images, not just console summaries.
3. Add a lightweight upload-time `validate_image_integrity(image)` entry point.
4. Store verdict metadata next to OCR issue reports or in a small local review
   manifest.
5. Add artifact generation for suspicious images only.
6. Review the current top outlier cluster before deleting or blocking anything.

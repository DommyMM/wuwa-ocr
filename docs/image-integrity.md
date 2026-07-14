# Build Card Image Integrity

The upload path validates KuroBot build cards before trusting them as build or
training data. It is a deterministic layout check, not a generic AI detector.

## Production flow

`image_integrity.py` runs after decode and returns one of three verdicts:

- `ok`: a 1920x1080 card with the expected Discord QR/layout anchors. R2 upload
  and region OCR run concurrently.
- `suspect`: an anchor is missing or lower stat rows are unusual. Region OCR
  runs first; R2 storage starts only if the OCR structure is plausible.
- `reject`: wrong dimensions or a high-confidence modified-row signature. R2
  storage and OCR are both skipped and the client receives a specific message.

The suspicious OCR gate requires a recognized character and weapon, a hidden
or nine-digit UID, at least four structurally complete echoes, and at least two
confident echo-template matches. This rejects unrelated screenshots, Discord
screenshots, and other card generators without storing them. A suspicious but
valid transformed KuroBot card can still pass.

## Evidence and thresholds

The fast checks were evaluated against 17,938 canonical local images. The
absolute lower-row rule initially produced 95 candidates, including broadly
dark but otherwise genuine-looking transformations. Requiring the signature in
at least two of echo panels 3-5 while limiting whole-image near-black coverage
narrowed direct rejection to six files; review found all six invalid and no
genuine card among them.

Brightness-relative row-deficit limits are deliberately escalation-only. In a
1,500-card sample, observed p99.9 values were below 0.099, 0.088, and 0.087 for
echo3, echo4, and echo5. Production limits are 0.105, 0.100, and 0.095. This
signal continued to flag the generated example after recompression, blur,
brightness, gamma, and noise transformations. A blurred dark-run signal makes
the check resilient to pixel noise, while an unusual 1st-percentile tone floor
escalates whole-image brightness/gamma shifts.

The expected QR anchor is also escalation-only because a small number of valid
historical cards do not decode reliably. In a 300-card sample, 296 decoded an
accepted Discord host and four did not decode. A Discord screenshot containing
an embedded card can retain the QR, which is why OCR structure remains the
second gate.

On a 1,500-image local run, 1,440 were accepted immediately, 56 were escalated,
and four known-invalid files were directly rejected. The fast pass measured
18.6 ms median and 21.2 ms p95. Only suspicious inputs pay the additional OCR
validation/storage sequencing cost; normal OCR wall time remains concurrent
with R2.

## Offline tools

- `scan_image_integrity.py`: corpus scan and JSON/CSV review queue.
- `review_integrity_gui.py`: manual keep/delete/review decisions.
- `forensics_echo_integrity.py`: panel crops and diagnostic overlays.
- `baseline_echo_row_darkness.py`: position-specific threshold analysis.
- `clean_invalid.py`: dry-run or apply reviewed deletions.

Do not auto-delete new statistical outliers. Direct rejection should remain
limited to rules validated against the corpus; novel cases belong in review
until enough labeled examples exist to justify another production rule.

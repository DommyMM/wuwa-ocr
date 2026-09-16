# Build card image integrity

Uploads are validated as KuroBot build cards before they become build or training data. Every check keys on an invariant a genuine card can't vary. Past false positives all came from keying on something a real player does vary (progression, language, image quality), so none of that is measured.

Three phases ask different questions, and only the first gates today.

| phase | question | where | status |
| --- | --- | --- | --- |
| A | Is this a KuroBot card at all? | `image_integrity.validate_header_dimensions`, `validate_image_integrity` | Rejects uploads |
| B | Was a stat cell pasted onto the echo bed? | `image_integrity.echo_bed_score` | Logged only |
| C | Were substat rows re-rendered in an editor? | `forensics_card_render.render_consistency` | Offline tool, ELA unreliable across the corpus |

## Phase A: card gate

- Dimensions are read from the PNG or JPEG header before `cv2.imdecode`, which allocates `width * height * 3` for whatever the header declares. A few-hundred-KiB PNG inside the 5 MiB cap can claim 60000x60000 and ask for ~10 GB
- Anything but exactly 1920x1080 is rejected. `validate_image_integrity` repeats the dimension check so the invariant holds for every caller, not just ingest
- The decoded card's fixed chrome (frame, labels, forte pentagon, everything above the echo band) is compared against a median reference under a mask. Per-card median normalisation ignores tint and exposure, and a shared blur lets soft and sharp scans converge while a wrong layout doesn't
- The mask excludes the weapon panel, since a distinctive weapon deviated from the average-weapon blur enough to flag clean cards
- Genuine English cards score <= 2.4 and AI-generated fakes and non-cards >= 4.0, so `OCR_CHROME_REJECT` defaults to 3.5 in the gap. Non-English cards can score near it and pass, which is intended: they get the language error downstream instead of "not a build card"
- There is no suspect tier. A card passes and runs OCR and storage concurrently, or is rejected before either, and the client gets only the message, since a live score would give cheaters a reference
- A missing reference asset fails open: the chrome check is skipped rather than rejecting every upload

## Phase B: echo bed score

A genuine panel's substat bed is a gradient that varies only with x, and a pasted stat cell carries its own background level. It can't gate, because wrapped substat names ("Resonance Liberation DMG Bonus") break the same assumption. `server.py` logs `echo_bed_observed` only at or above `OCR_BED_OBSERVE_FLOOR` (2.5), since genuine cards sit ~1.6 and labelled pastes start at 3.2.

## Phase C: re-rendered substat rows

A row re-typed in an image editor is brighter than its neighbours and has a lower error level, since it never went through the original JPEG quantization. `render_consistency` sorts the 25 substat label cells by brightness, splits at the largest gap leaving at least three rows per side, and compares error level across the split. `ela_delta` ignores background level, so the wrapped names that blind Phase B don't move it.

The two confirmed forgeries were Hiyuki with 8 of 25 rows redrawn and Aemeath with 14. Every forged value was a legal roll, since `lb` already enforces the exact roll table, so stored CV matched the forged pixels exactly.

A sample stratified at 300 or more cards per upload month looked clean: `combined` (gap x ela_delta) peaked at 2.00 on 2649 genuine cards against 40.8 and 64.4 for the forgeries, and `gap >= 5` fired on 3 of them. The full-corpus sweep overturned that. It flagged 40 of 25,551 cards with the forgeries ranked only 18th and 24th, and the flag rate tracks encoder era rather than content (15.4% of `dqt 1.0` files, 0.22% of canvas re-encodes, 0 of 4757 original-byte uploads, 0.06% of PNG). ELA measures the storage pipeline, so don't tune it per era.

Tone spread, the max minus min text tone over the 25 label cells, is the era-stable signal. A genuine card is one render pass with one text tone, while a forgery goes bimodal:

| population | median | p90 | p99 | max |
| --- | --- | --- | --- | --- |
| Canvas re-encodes (249) | 10.3 | 11.8 | 17.6 | 139.2 |
| Original bytes (248) | 9.8 | 11.2 | 11.8 | 12.2 |
| PNG (249) | 9.8 | 11.2 | 12.6 | 72.3 |
| Both forgeries | ~33 | | | |

It's also cheaper, one grayscale pass with no re-encode. Absolute tone isn't comparable across eras, so re-rendered rows aren't reliably brighter, and `gap` or `tone_sd` alone flag genuine highlight-band cards (gap 41, tone_sd 18). `render_consistency` doesn't output tone spread yet.

### JPEG encoder signature

`encoder_signature` is for triage and logging, never grounds for rejection. The forgeries carry `DQT[0] mean 9.25`, 4:4:4 chroma and a JFIF header, which looks decisive against recent uploads but is only the older encoder era:

| era | signature |
| --- | --- |
| Frontend canvas re-encodes | `dqt 9.25` with APP0 dominant, ~10% 4:4:4 |
| Original upload bytes | `dqt 23.08`, 4:2:0, no APP0, plus PNG |

A stale cached client would look identical to a forgery.

## Offline tools

- `scan_image_integrity.py` scans a corpus and writes a JSON and CSV review queue
- `review_integrity_gui.py` records keep, delete and review decisions and exports `invalid_images.json`
- `clean_invalid.py` dry-runs or applies the reviewed deletions from R2 and `r2-backup/`
- `forensics_card_render.py` runs Phase C with the encoder signature over a corpus
- `forensics_echo_integrity.py` writes panel crops and diagnostic overlays for one suspect
- `baseline_echo_row_darkness.py` measures position-specific row darkness thresholds

Never auto-delete a statistical outlier. Direct rejection stays limited to rules validated against the corpus, and novel cases go to review until enough labelled examples justify another production rule. Two labelled forgeries don't.

## Open work

1. Add tone spread to `render_consistency`, then eyeball ~10 cards from the canvas and PNG tails (up to 139 and 72, absent in original bytes) to learn whether they're highlight-band layouts or more forgeries before choosing any threshold
2. Review corpus candidates by hand: `py sync_r2.py --run`, then `py forensics_card_render.py ../r2-backup --out ../forensics/card_render`
3. A structural detector needs no pixels: builds with no `scanId` or source image and most substats at max roll, and UIDs one digit from a blacklisted UID treated as the same actor
4. Promote a detector to a review queue only once it is era-stable across canvas, original-byte and PNG uploads, never as a user-facing rejection

The payload-side attack (builds submitted with no image and out-of-range scalars) is an `lb` problem with its own fixes in `lb/docs/submit-hardening-plan.md`.

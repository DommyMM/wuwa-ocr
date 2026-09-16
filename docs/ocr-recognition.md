# OCR recognition

The backend is a fixed-layout recognizer, not a general OCR service. Import cards are one 1920x1080 layout, crop geometry is stable, and almost every label comes from a small finite vocabulary. So a region is recognized as a known class wherever it can be, and Tesseract reads only digits, the watermark username and short stat names that resolve against a closed list.

## Pipeline

1. `server.py` gates dimensions from the image header before decoding, since `cv2.imdecode` allocates whatever the header declares, then checks the card chrome (see [image-integrity.md](image-integrity.md))
2. A rejected image stops there, so it reaches neither R2 nor OCR
3. An accepted image starts its R2 upload and splits into ten fixed regions (`IMPORT_REGIONS`) at the same time
4. Regions run on a process pool with one worker per heavy region (five echoes and forte), so those six finish in one wave and the light regions follow
5. Each region streams back as it completes, then `done` carries the merged analysis (contract in [README.md](../README.md))

Recognition code lives in `card.py`, vocabularies and templates load in `data.py`, and `Data/` is refreshed from the frontend by `wuwabuilds/scripts/sync_backend.py`.

## Per region

| region | method | key decision |
| --- | --- | --- |
| character | SIFT on the splash, abstain to a letters-only read of the name strip | Splash is language-independent, so it identifies Japanese cards the name read misses |
| character level | Gold `LV.` pill located by hue, inverted, read at 3x then 2x | Pill floats with name length, so no fixed box holds it, and its right 22% is decorative stripes cropped before reading |
| Rover | Splash gives gender only, then title element, then badge hue | Sonata badge matcher can't be reused since its mask includes the purple header |
| weapon | SIFT on the square icon with confidence and margin floors, abstain to the name strip | Several icons look alike, so confidence alone passed near-ties |
| weapon name | Name strip read thresholded, then plain grayscale in the same spawn | Dark uploads carry gold text under the 140 threshold |
| weapon level | Its own box, read on both paths | ~6% of weapons aren't level 90, so the old SIFT-accept default was wrong about one build in seventeen |
| watermark | UID line at 2x with a digit whitelist, corroborated by a free-form 2x read of the whole strip | A wrong 9-digit UID is permanent, so two disagreeing valid reads write no UID |
| sequences | HSV pale-pixel ratio per node | Deterministic, never needed OCR |
| forte | Five nodes in one batched Tesseract spawn | Template matching on `LV.1`-`LV.10` was measured and the classes don't separate |
| echo identity | Cost badge prefilter, SIFT, badge then icon hue on near ties, family badge for Nightmare variants | Cost narrows the sweep but abstains to all templates when unsure, and promotion to a Nightmare needs SIFT confidence |
| echo set | HSV histogram, SIFT only inside a hue cluster, color template when SIFT scores zero | Grayscale sets and same-hue sets can't be split by hue alone |
| echo main | Plain grayscale at 2x, `--psm 6`, resolved against the legal mains for the echo's cost | Fixed threshold shreds the name row, see [echo-main-strip.md](echo-main-strip.md) |
| echo substats | Fixed row bands, see [echo-substats.md](echo-substats.md) | Row drops are a layout failure, so Tesseract is never asked to find rows |

`lb` forces character and weapon level to 90 in the damage calc, so read levels change what a profile and the editor show, not any board score.

## Contracts and rules

- A blank weapon panel resolves to an empty name and id, and the frontend's `IMPORT_WEAPON_FALLBACKS` fills the signature weapon only then. Weapon recognition must keep returning empty for a panel that draws no weapon
- Echo main values come from `EchoStats.json` once cost and stat are known, never from OCR
- A short flat substat name is never inferred from its value, since flat ATK and DEF share 40, 50 and 60
- A dropped row beats a wrong-but-legal row, since a visible gap gets fixed and a plausible number corrupts CV silently
- Non-English cards still identify through SIFT, while their substat names drop below the confidence floor, and `detect_unsupported_language` flags them out of auto-submit

## Runtime and dependencies

- Production runs `tessdata_best`, pinned by commit and sha256 in the Dockerfile. Debian's package ships `tessdata_fast`, which read a valid UID on 228 of 300 cards where best read all 300
- A pytesseract call spawns a process that reloads the model, ~78 ms on Railway against under 5 ms of OCR on a small crop. `tess_batch` runs a list file through one process, byte-identical to separate calls
- Tesseract is deterministic under load, so parallel captures and sequential re-runs agree read for read. FLANN is the one nondeterministic step (see open items)
- `requirements.txt` pins every direct dependency to what the production image resolved. `opencv-python-headless`, `numpy` and `Pillow` decide the pixels Tesseract sees, so bumping any of them is an accuracy change that needs a gate in the production image, not a green test suite
- The process pool is deliberate. A `ThreadPoolExecutor` pushed production median latency from ~690 ms to ~3.9 s

## Gating a recognition change

One percent of cards is hundreds of leaderboard entries, so a change is measured on as much of `r2-backup/` as practical, never a sample of a few hundred.

1. Freeze both trees with `git archive` of the baseline and candidate commits, so a later edit can't leak into a running capture
2. Capture every region's analysis per card from each tree over a seeded shuffle of the corpus, so any run's first N files are a prefix of any other's
3. Diff region by region: exact equality on forte, sequences, watermark, weapon id and level, character id and level, and echo identity, main, set and substats separately. Split English from non-English echoes by the language signal
4. Adjudicate every English difference by eye on a rendered contact sheet, bucketed by which side is right
5. Re-run every disputed case sequentially from both frozen trees before believing the diff, and check a capture's mtime against its tree

Run captures in the production image, not on the dev box. The image reproduces production (129 of 130 logged echo regions matched), while Windows differs in OpenCV, numpy, leptonica and libjpeg-turbo even with an identical `eng.traineddata`.

```
docker build -t wuwa-ocr:local backend/
docker run --rm -v <r2-backup>:/data:ro -v <scratch>:/work:ro -v <scratch>/out:/out \
  -e OMP_THREAD_LIMIT=1 -e LIST=/work/list.txt -e OUT=/out/base.json -e N=6000 \
  -e WORKERS=12 --entrypoint python wuwa-ocr:local /work/capture.py
```

Mount a candidate `card.py` over `/app/card.py` for the second run instead of rebuilding. Bind-mount paths must be Windows-style, since a POSIX `$PWD` from Git Bash doesn't mount. A 6000-card substat A/B takes about 45 minutes on the dev box.

`optimize_crops.py` sweeps crop boxes offline against a gold-label JSON (shape in its docstring) and never changes runtime coordinates.

## Rejected approaches

| approach | why rejected |
| --- | --- |
| RapidOCR as a fallback reader | Every worker paid its RAM whether called or not. Banded Tesseract, the pill reader and the name-strip reads replaced every call site with no accuracy loss on the gates |
| Otsu in the shared `preprocess_region` | Forte's circuit node read 0 on most cards and substats lost 597 rows per 30,000 |
| Forte by digit templates | Scores for `LV.1` to `LV.10` don't separate |
| Accepting 16:9 inputs below 1920 wide | Text that reads at native 1920 doesn't survive the 720p round trip |
| Client-side region crops as the upload | Ten PNG crops averaged ~1.5 MB against a ~350 KB original, and cost ~55 ms of browser CPU for under 1 ms of backend cropping |
| R2 as the OCR transport | Adds a network hop and makes OCR wait on storage, so OCR consumes the bytes the browser already sent |
| Proxying the image through Vercel Functions | Vercel would pay function time and origin transfer on every upload |
| Scraping wuwaflex | The UID exists only in the image, so rows can't be attributed without running this pipeline anyway, and its copies are 720p |

Substat and main-strip rejections live in their own docs.

## Template assets

`sync_backend.py` writes every template set as WebP, which is enough for SIFT and color matching.

- Character splash is Encore's `FormationRoleCard` from the per-character detail endpoint, saved as `<id>.webp` since the pile filename uses an internal codename
- Weapon icon is Encore's full `Icon`, not `iconMiddle`, since several weapon pairs share an identical `iconMiddle`
- Echo and element templates come from the frontend's mirrored assets and Encore FetterGroup icons

Missing-weapon reference cards for the empty-weapon contract: Lucilla `1109` (signature Freeze Frame `21050086`), Lucy `1511` (Spectral Trigger `21030056`), Rebecca `1308` (Skull Thrasher `21030066`), with Zani `1507` and Blazing Justice `21040036` as the rendered control. Keep raw image keys and player UIDs out of committed docs.

## Open items

- Residual substat misreads, each a handful per 30,000 echoes: values highlighted as a max roll (the bright box defeats the threshold), `21%` read as `219`, and rows where both render scales return empty. The lever is fine-tuning tessdata on the game font LaguSansBold, not more heuristics
- `resolve_echo_main` with no readable value and no matching name falls to the cost's first legal main, which is HP%. Empty reads are near zero since the grayscale read, but that fallback stays one degradation away from fabricating a stat
- FLANN's randomized kd-tree wobbles near-tie confidence between runs, so echo identity flips about twice per 30,000 echoes below confidence ~0.07. Treat a lone low-confidence identity diff in a gate as noise. A deterministic matcher or an abstain floor would end it
- Forte uses the shared thresholded preprocess and misreads 10 as 0 on soft cards. It isn't persisted in builds, so it corrupts the import UI rather than boards. Running the main-strip harness on forte is the cheap first experiment
- A light or grey card layout can clip the echo main box (`Havoc DMG` read as `avoc DMG`)
- Some builds' `source_image_key` points at the wrong image, and at least one all-black image backs a build row. `lb/cmd/sourceimageaudit` and `cmd/sourceimagematch` exist for this class
- Stored panel order can permute against read order (44 of 109 disputed panels matched another echo index better), so anything joining `echoPanels[i]` to `echo{i+1}` must tolerate that
- Latency: with `OCR_WORKERS=10` the wall is one echo region at ~790 ms and nothing else, so the next lever is inside the echo region (the SIFT sweep and its Tesseract batches)
- 24 transitive dependencies stay unpinned. A full `pip freeze` lock would close that, though none of them touch the OCR path

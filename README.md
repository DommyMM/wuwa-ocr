# WuWaBuilds OCR backend

FastAPI service that reads KuroBot build cards for wuwa.build imports, hosted at `https://ocr.wuwa.build` behind the Cloudflare gateway.

The service takes the original screenshot, gates it as a genuine 1920x1080 card, crops ten fixed regions and recognizes them on a process pool, streaming each region back as it finishes. The original bytes are stored in R2 under a content-addressed key while recognition runs. Game vocabularies and templates load from `Data/` at import time (`data.py`).

| doc | covers |
| --- | --- |
| [docs/ocr-recognition.md](docs/ocr-recognition.md) | Recognition design per region, decisions, rejected approaches, gating a change, open items |
| [docs/echo-substats.md](docs/echo-substats.md) | Banded substat reader and its evidence |
| [docs/echo-main-strip.md](docs/echo-main-strip.md) | Why the echo main strip skips preprocessing |
| [docs/image-integrity.md](docs/image-integrity.md) | Card gate, pasted-cell score, forgery forensics, review tools |
| [docs/multilingual-echo-investigation.md](docs/multilingual-echo-investigation.md) | Non-English stat names: bake-off and chosen design |
| [docs/railway-observability.md](docs/railway-observability.md) | Railway logs, metrics, memory and cost |
| [scanner/AGENTS.md](scanner/AGENTS.md) | Echo inventory scanner, excluded from the deployed image |

## Run

```bash
py server.py              # port 5000, or PORT
py -m pytest tests -q     # fake S3 clients and in-memory images, no network
```

## `POST /api/ocr`

Send one card as `multipart/form-data` with an `image` file field, or as a raw image body. The response is an `application/x-ndjson` stream, one JSON event per line, every event carrying the same `scanId`:

```json
{"type":"meta","scanId":"00000000-0000-4000-8000-000000000000","sourceImageKey":"0123…cdef.jpg","image":{"width":1920,"height":1080,"bytes":271977,"mediaType":"image/jpeg"}}
{"type":"region","scanId":"00000000-0000-4000-8000-000000000000","region":"watermark","status":"done","analysis":{"username":"Player","uid":123456789},"elapsedMs":180.2}
{"type":"done","success":true,"scanId":"00000000-0000-4000-8000-000000000000","analysis":{},"progress":{},"regionErrors":{},"timings":{},"storage":{"result":"stored","elapsedMs":86.4},"sourceImageKey":"0123…cdef.jpg","trainingImageKey":"0123…cdef.jpg","unsupportedLanguage":false}
```

- `region` events arrive in completion order with `status` `done` or `error`. Interactive import renders them live, and bulk import reads only `done`
- A card that fails the integrity gate, an unsupported format, a decode failure, a recognition exception or the `OCR_TIMEOUT` deadline ends the stream with one `{"type":"error","success":false,"scanId":…,"error":…}` event. A rejected card reaches neither R2 nor recognition, and the error carries only a message
- `sourceImageKey` is optimistic: it names the object even while the upload is still running, and it is null when storage is disabled or the upload failed. Builds use it for provenance without waiting on R2
- `trainingImageKey` is confirmation-only, set once R2 reports `stored` or `already_present`. Issue reports use it
- `storage.result` is `stored`, `already_present`, `pending` (recognition finished first and the upload continues in the process), `failed`, `timed_out` or `disabled`. A storage problem never fails recognition
- `unsupportedLanguage` flags a real build card whose substat names don't read as English, so the frontend keeps it out of auto-submit

Errors before the stream starts are JSON: `400` for a missing image, `413` over 5 MiB, `429` over the per-IP rate limit (with `Retry-After`), `500` for an unexpected failure.

## `POST /api/report-ocr-issue`

Stores one import issue report as JSON in R2, attached to its card image. The body is `multipart/form-data` with a `report` field (JSON, at most 256 KiB) and optionally an `image` file (at most 5 MiB).

- The report's `trainingImageKey` from the `done` event is the normal image source and is trusted without another R2 round trip, since this service minted and stored it
- Without a key, the fallback `image` is stored under the same content-addressed key, reusing an existing object. A key wins when both are sent
- `route` must be `/import` and `reason` one of `illegal_echo`, `ocr_error`, `validation_error`, `manual_report`. `scanId` is null or a canonical UUID, and image keys must be canonical, since a key becomes an R2 object name
- Validation stops there on purpose. Unknown top-level fields are dropped rather than rejected and unknown `progress` regions are kept, since a report about unexpected client state is the one worth keeping

Success is `201` with `{"success":true,"reportId":…,"reportKey":"reports/YYYY/MM/DD/<id>.json","trainingImageKey":…,"imageStorage":"referenced|stored|already_present"}`. Errors are `{"success":false,"reason":…}` with `400`, `403` (not from the gateway while `INTERNAL_API_KEY` is set), `413`, `415`, `429` or `503` (storage not configured or unavailable), and never include R2 details.

`GET /health` returns `{"status":"ok"}` and `GET /` returns endpoint metadata.

## R2 persistence

Only JPEG and PNG are accepted, detected from file magic rather than the filename or `Content-Type`. The key is `<sha256 of the exact request bytes>.<jpg|png>` at the bucket root. Before writing, a `HEAD` reuses an existing object whose size and stored digest match. A new `PUT` carries the original bytes, MIME type, SHA-256 checksum and digest metadata. Use a token scoped to the bucket with Object Read & Write, and never expose these credentials to the browser.

## Environment

| variable | default | purpose |
| --- | --- | --- |
| `PORT` | `5000` | Listen port |
| `OCR_WORKERS` | `6` | Process pool size, one worker per heavy region. Production runs `10`, and each worker holds its own templates, so this is the main lever on the RAM bill |
| `OCR_OPENCV_THREADS` | `1` | `cv2.setNumThreads` per worker |
| `OCR_TIMEOUT` | `60` | Seconds before recognition times out |
| `OCR_RATE_LIMIT` | `10` | OCR requests per minute per client IP. Raise it locally for batch imports |
| `OCR_REPORT_RATE_LIMIT` | `5` | Issue reports per minute per client IP |
| `INTERNAL_API_KEY` | unset | Gateway key: trusts the forwarded client IP and, when set, is required on the issue-report route |
| `OCR_CHROME_REJECT` | `3.5` | Chrome score at or above which a card is rejected |
| `OCR_BED_OBSERVE_FLOOR` | `2.5` | Echo-bed score worth a log line |
| `OCR_NONENGLISH_NAME_FLOOR` | `0.35` | English substat-name match rate below which a card is flagged non-English |
| `OCR_NONENGLISH_MIN_VALUES` | `15` | Readable values a card needs before the language flag applies |
| `OCR_R2_UPLOAD_ENABLED` | `0` | Enables R2 persistence (`1/0`, `true/false`, `yes/no`, `on/off`), and startup fails if a credential below is missing |
| `OCR_R2_TIMEOUT_SECONDS` | `5` | Deadline for the R2 `HEAD` and `PUT` result |
| `CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` | unset | R2 endpoint and credentials |
| `TESS_BATCH_DIR` | `/dev/shm` when present | Temp directory for batched Tesseract images |
| `OMP_THREAD_LIMIT` | `1` in the Dockerfile | Keeps each Tesseract process single-threaded while workers parallelize |

## Data sync

The backend never fetches game data at runtime. From `wuwabuilds/scripts`, run the data sync (`py sync_all.py`, see `wuwabuilds/docs/sync-sources.md`), then `py sync_backend.py`, which writes the vocabulary JSONs, copies `EchoStats.json` and `Stats.json`, and refreshes the character, weapon, echo and element templates as WebP. From `backend/`, `py regress_echo_webp.py --limit 500` checks a full echo-template swap before committing it.

## Local helpers

- `sync_r2.py` mirrors the R2 bucket into `r2-backup/` and stamps original upload times on file mtimes
- `r2_date_summary.py --since <iso time>` counts local screenshots in a patch window
- `stage_r2_backfill.py --since <iso time> --clean` stages a filtered folder for the frontend `/bulk-import` page
- `visualize_regions.py [image]` draws every crop region and sub-box on a card

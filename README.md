# Wuthering Waves OCR Backend

FastAPI OCR service for WuWaBuilds import scans.  
Hosted at `https://ocr.wuwa.build`.

## Runtime Model

- Single OCR mode: English full-card import processing. The API receives the original
  screenshot, decodes it once, crops fixed regions server-side, then fans out
  region recognition through `card.py`.
- Legacy full-screen mode (`char.py` / `echo.py`) has been removed
- Data is loaded from local `backend/Data/*.json` and image templates at startup import time (`data.py`)
- Echo, character, and weapon OCR results include IDs for robust frontend matching
- Echo and element templates may be PNG or WebP; current element templates are WebP-only.
- The original JPEG/PNG request bytes are content-addressed and persisted to
  Cloudflare R2 concurrently with recognition when `OCR_R2_UPLOAD_ENABLED=1`.

## Start

```bash
py server.py
```

Default port is `5000` (`PORT` env var supported).

## API

### `POST /api/ocr`

Process one full build-card screenshot.

Request body can be `multipart/form-data`:

```http
image=<file>
```

or a raw image body with an image `Content-Type`.

The response is always an `application/x-ndjson` stream. Each line is one JSON
event: `meta`, zero or more per-region `region` events, then a final `done`
event that contains the merged import analysis, per-region status, and timing
data.

```json
{"type":"meta","scanId":"76078ac4-9ac5-4b52-a933-4fb724f62659","image":{"width":1920,"height":1080,"bytes":271977,"mediaType":"image/jpeg"}}
{"type":"region","scanId":"76078ac4-9ac5-4b52-a933-4fb724f62659","region":"watermark","status":"done","analysis":{"username":"Player","uid":500000000},"elapsedMs":180.2}
{"type":"region","scanId":"76078ac4-9ac5-4b52-a933-4fb724f62659","region":"echo1","status":"done","analysis":{"main":{"name":"ATK%","value":"18%"}},"elapsedMs":1111.0}
{"type":"done","success":true,"scanId":"76078ac4-9ac5-4b52-a933-4fb724f62659","analysis":{},"progress":{},"timings":{"r2Ms":86.4,"storageWaitMs":0.02},"storage":{"result":"stored","elapsedMs":86.4},"trainingImageKey":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.jpg"}
```

Interactive import consumes the `region` events for live UI updates. Bulk import
uses the same stream parser and only consumes the final `done` event.

`meta` intentionally never contains `trainingImageKey`: storage is still in
progress at that point. The final `done.trainingImageKey` is a confirmed,
root-level R2 object key, or `null` if storage was disabled, failed, or exceeded
its deadline. A storage problem does not fail otherwise-successful OCR.

The final `storage.result` is one of:

- `stored` — this request wrote the object;
- `already_present` — identical exact bytes were already stored;
- `failed` — R2 returned an error;
- `timed_out` — the configured storage deadline elapsed;
- `disabled` — backend persistence is turned off.

Every event includes the same `scanId`, which can be passed to downstream build
submission and used to correlate frontend, OCR, and leaderboard logs.

### `POST /api/report-ocr-issue`

Persist one import issue report and attach it to the original training image.
The request must be `multipart/form-data` with:

- one `report` text field containing at most 256 KiB of JSON; and
- an image source: either a canonical `trainingImageKey` inside the report JSON
  or one `image` file containing at most 5 MiB of original JPEG/PNG bytes.

The normal path uses the `trainingImageKey` from the final OCR `done` event and
does not send the image again. That key was minted and stored by this service,
so it is taken at face value rather than re-confirmed with an extra R2 HEAD.
When OCR could not return a key, the fallback sends the original file once; the
service derives the same content-addressed key and reuses an object that already
exists before attempting a write. If both are supplied, the key wins.

```text
report={"schemaVersion":1,"route":"/import","reason":"manual_report",...}
image=<optional fallback file>
```

`reason` must be `illegal_echo`, `ocr_error`, `validation_error`, or
`manual_report`, and `route` must be `/import`. `scanId` may be null when a
network or gateway failure happened before the first OCR event; otherwise it
must be a canonical UUID. Non-canonical image keys are rejected, because a key
becomes an R2 object name.

Validation deliberately stops there. A report is diagnostic material, so unknown
fields and unrecognized `progress` regions are stored rather than rejected: a
report about an unexpected client state is exactly the report worth keeping, and
a 400 would discard it at the moment it is most useful.

A successful response is `201 application/json`:

```json
{"success":true,"reportId":"870de1eb-eed6-49cb-9f75-ce4b23bedca3","reportKey":"reports/2026/07/11/870de1eb-eed6-49cb-9f75-ce4b23bedca3.json","trainingImageKey":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.jpg","imageStorage":"referenced"}
```

`imageStorage` is `referenced`, `stored`, or `already_present`. Error responses
use the stable shape `{"success":false,"reason":"..."}` and never include R2
or credential details. When `INTERNAL_API_KEY` is configured, this write route
only accepts requests from the trusted gateway.

## R2 image persistence

Only JPEG and PNG inputs are accepted, determined from file magic rather than
the multipart filename or request `Content-Type`. The canonical key is:

```text
<64 lowercase hexadecimal SHA-256 of the exact request bytes>.<jpg|png>
```

Keys live at the bucket root; there is no `training-images/` prefix. Before
writing, the service performs `HEAD` and reuses an existing object with the
same key and byte length. A new `PUT` carries the original bytes, detected MIME
type, SHA-256 checksum, and digest metadata. R2 work begins alongside region
recognition, and only the final NDJSON event waits for its bounded result.

### Other Endpoints

- `GET /health` -> health check
- `GET /` -> API status metadata
- `GET /ocr-results` -> serves `../ocr_results.json` (output of `batch_ocr.py`); 404 when missing

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | HTTP listen port |
| `INTERNAL_API_KEY` | — | Trusted proxy key (gateway → OCR); selects the forwarded client IP and is required by the issue-report write route when configured. |
| `OCR_WORKERS` | `8` | `ProcessPoolExecutor` size — parallel Tesseract processes. Match to CPU thread count locally (e.g. `16` for 7800X3D). Railway is capped at `8` vCPU. |
| `OCR_RATE_LIMIT` | `10` | Requests per minute per IP. Set to `10000` locally to disable effective limiting during batch import. |
| `OCR_REPORT_RATE_LIMIT` | `5` | Issue reports per minute per client IP. Independent of OCR admission. |
| `OCR_TIMEOUT` | `60` | Seconds before a single OCR request times out. |
| `OCR_OPENCV_THREADS` | `1` | Per-worker `cv2.setNumThreads` value. |
| `OCR_R2_UPLOAD_ENABLED` | `0` | Enable backend-owned R2 persistence. Accepts `1/0`, `true/false`, `yes/no`, or `on/off`. |
| `OCR_R2_TIMEOUT_SECONDS` | `5` | End-to-end deadline for the asynchronous R2 `HEAD`/`PUT` result. Must be positive. |
| `CLOUDFLARE_ACCOUNT_ID` | — | Cloudflare account used to form the R2 S3 endpoint; required when upload is enabled. |
| `R2_ACCESS_KEY_ID` | — | R2 S3 access-key ID; required when upload is enabled. |
| `R2_SECRET_ACCESS_KEY` | — | R2 S3 secret; required when upload is enabled. |
| `R2_BUCKET_NAME` | — | Destination bucket; required when upload is enabled. |
| `OMP_THREAD_LIMIT` | `1` in Dockerfile | Keeps each Tesseract subprocess single-threaded while the service parallelizes across regions/workers. |
| `USE_GPU` | `1` locally, `0` on Railway | Enables RapidOCR CUDA providers in `data.py` when `onnxruntime-gpu` is available. |
| `RAILWAY_ENVIRONMENT_NAME` | — | Auto-set on Railway; toggles GPU default and is used to log environment context. |

## Limits and Errors

- OCR rate limit: `OCR_RATE_LIMIT` requests/minute per IP (default `10`)
- Issue-report rate limit: `OCR_REPORT_RATE_LIMIT` requests/minute per IP (default `5`)
- Image body: at most 5 MiB after multipart extraction
- Issue-report metadata: at most 256 KiB; fallback image at most 5 MiB; full
  multipart envelope at most 5,570,560 bytes
- Timeout: `OCR_TIMEOUT` per OCR request (default `60s`)
- Common statuses:
  - `400` invalid image/region/request
  - `408` processing timeout
  - `429` rate limit exceeded
  - `500` internal server error

Use an R2 token scoped to the destination bucket with Object Read & Write
permission: the service needs `HEAD` for image deduplication and `PUT` for new
image and report objects.
When R2 upload is enabled, startup fails fast if any required setting is absent
or the timeout value is invalid. Do not expose these credentials to the browser.

## Tests

```bash
py -m unittest discover -s tests -v
```

The ingest tests use fake S3 clients and local in-memory images. They do not
contact R2, Railway, or any production service.

## Railway Operations

Production runs in Railway project `wuwa-backend` as service `WuWa OCR`.
Operational commands, latency/cost snapshots, and log
query examples live in [`docs/railway-observability.md`](docs/railway-observability.md).

## Data Sync Expectations

The backend does not fetch runtime game data from production frontend URLs.  
Keep `backend/Data` synchronized from `wuwabuilds/scripts`:

1. From `wuwabuilds/scripts`, run the data sync (`py sync_all.py`, or the targeted Encore merge path documented in `wuwabuilds/docs/sync-sources.md`).
2. Run `py sync_backend.py` to refresh backend JSONs and Encore element badge templates.
3. Run `py download_echo_icons.py --clean --force` only when echo icon templates need a full refresh. The downloader stores backend echo templates as WebP by default, converting PNG source assets when needed.
4. To validate a full echo-template WebP swap without committing converted assets, run `py backend\regress_echo_webp.py --limit 500` from the workspace root after installing `backend/requirements.txt`.

## Local Backfill Helpers

- `py backend\r2_date_summary.py --since 2026-06-07T19:00:00-07:00` counts local R2 screenshots in a patch window.
- `py backend\stage_r2_backfill.py --since 2026-06-07T19:00:00-07:00 --clean` stages a filtered folder for the frontend `/bulk-import` page.
